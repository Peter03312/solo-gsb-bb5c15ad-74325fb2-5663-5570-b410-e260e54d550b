"""Unit tests for the server-configurable conflict-tree node budget."""

import pytest

from app.resolver import parse_index, parse_locks, parse_requirements, resolve
from app.serialize import (
    CONFLICT_NODE_BUDGET_ENV,
    conflict_summary,
    parse_node_budget,
    serialize_conflict,
)


def solve(packages, root, locks=None, **kwargs):
    index, errors = parse_index(packages)
    assert not errors, errors
    roots = parse_requirements(root, "root")
    parsed_locks, lock_errors = parse_locks(locks or {})
    assert not lock_errors, lock_errors
    return resolve(index, roots, parsed_locks, **kwargs)


# Wide fan-out: top composite with three direct leaf terminals.
#
#   root(exhausted)
#     leaf a
#     leaf b
#     leaf c
def wide_fanout():
    packages = {
        "root": {
            "3.0": {"dependencies": {"a": ">=2.0"}},
            "2.0": {"dependencies": {"b": ">=2.0"}},
            "1.0": {"dependencies": {"c": ">=2.0"}},
        },
        "a": {"1.0": {}},
        "b": {"1.0": {}},
        "c": {"1.0": {}},
    }
    return solve(packages, {"root": ""})


# Nested failure tree (6 nodes pre-order, 3 unique terminals):
#
#   root(exhausted)
#     leaf ghost            (root 2.0 directly needs absent "ghost")
#     mid(exhausted)        (root 1.0 needs mid)
#       leaf midleaf        (mid 2.0 needs midleaf>=2.0; only 1.0 exists)
#       deepdeep(exhausted) (mid 1.0 needs deepdeep)
#         leaf ddghost      (deepdeep 1.0 needs absent "ddghost")
def nested_tree():
    packages = {
        "root": {
            "2.0": {"dependencies": {"ghost": ""}},
            "1.0": {"dependencies": {"mid": ">=1.0"}},
        },
        "mid": {
            "2.0": {"dependencies": {"midleaf": ">=2.0"}},
            "1.0": {"dependencies": {"deepdeep": ">=1.0"}},
        },
        "deepdeep": {"1.0": {"dependencies": {"ddghost": ""}}},
        "midleaf": {"1.0": {}},
    }
    return solve(packages, {"root": ""})


def test_parse_node_budget():
    assert parse_node_budget(None) is None
    assert parse_node_budget("") is None
    assert parse_node_budget("   ") is None
    assert parse_node_budget("1") == 1
    assert parse_node_budget(" 250 ") == 250
    for bad in ("0", "-1", "abc", "1.5"):
        with pytest.raises(ValueError):
            parse_node_budget(bad)


def test_unlimited_budget_matches_full_renderer():
    result = nested_tree()
    summary = conflict_summary(result, node_budget=None)
    assert "truncated" not in summary
    assert summary["tree"] == serialize_conflict(result.conflict)
    # The default (server env unset) renders the identical full summary.
    assert conflict_summary(result) == summary


def test_wide_fanout_full_shape():
    result = wide_fanout()
    summary = conflict_summary(result, node_budget=None)
    tree = summary["tree"]
    assert tree["kind"] == "composite"
    assert len(tree["failed_branches"]) == 3
    assert "truncated" not in tree
    assert {c["package"] for c in summary["terminal_conflicts"]} == {"a", "b", "c"}


def test_budget_cuts_remaining_branches_in_order():
    result = wide_fanout()
    summary = conflict_summary(result, node_budget=3)
    tree = summary["tree"]
    # Root + the first two leaves fit; the third branch is not expanded.
    assert tree["kind"] == "composite"
    assert [c["package"] for c in tree["failed_branches"]] == ["a", "b"]
    assert tree["truncated"] is True
    assert tree["omitted_branches"] == 1
    assert summary["truncated"] is True
    assert summary["truncation_reason"] == "node_budget_exhausted"
    assert summary["node_budget"] == 3
    assert summary["retained_nodes"] == 3
    # Only terminals under retained nodes are listed; branch order preserved.
    assert [c["package"] for c in summary["terminal_conflicts"]] == ["a", "b"]


def test_budget_of_one_keeps_only_root():
    result = wide_fanout()
    summary = conflict_summary(result, node_budget=1)
    assert summary["tree"]["failed_branches"] == []
    assert summary["tree"]["omitted_branches"] == 3
    assert summary["terminal_conflicts"] == []
    assert summary["retained_nodes"] == 1


def test_budget_larger_than_tree_is_not_truncated():
    result = wide_fanout()
    summary = conflict_summary(result, node_budget=100)
    assert "truncated" not in summary
    assert len(summary["tree"]["failed_branches"]) == 3


def test_nested_tree_full_shape():
    result = nested_tree()
    summary = conflict_summary(result, node_budget=None)
    root = summary["tree"]
    assert root["package"] == "root"
    assert [c["kind"] for c in root["failed_branches"]] == ["leaf", "composite"]
    mid = root["failed_branches"][1]
    assert mid["package"] == "mid"
    assert [c["package"] for c in mid["failed_branches"]] == ["midleaf", "deepdeep"]
    assert sorted(t["package"] for t in summary["terminal_conflicts"]) == [
        "ddghost",
        "ghost",
        "midleaf",
    ]


def test_nested_budget_cuts_deepest_subtree_first():
    result = nested_tree()
    summary = conflict_summary(result, node_budget=4)
    # Nodes 1-4: root, ghost, mid, midleaf.  The deepdeep subtree is never
    # entered; the root itself keeps both of its children.
    root = summary["tree"]
    assert root.get("truncated") is not True
    assert [c["kind"] for c in root["failed_branches"]] == ["leaf", "composite"]
    mid = root["failed_branches"][1]
    assert mid["truncated"] is True
    assert mid["omitted_branches"] == 1
    assert [c["package"] for c in mid["failed_branches"]] == ["midleaf"]
    assert summary["truncated"] is True
    assert summary["retained_nodes"] == 4
    assert sorted(t["package"] for t in summary["terminal_conflicts"]) == [
        "ghost",
        "midleaf",
    ]


def test_nested_budget_cut_between_root_children():
    result = nested_tree()
    summary = conflict_summary(result, node_budget=2)
    root = summary["tree"]
    assert root["truncated"] is True
    assert root["omitted_branches"] == 1
    assert [c["package"] for c in root["failed_branches"]] == ["ghost"]
    assert [t["package"] for t in summary["terminal_conflicts"]] == ["ghost"]


def test_retained_terminal_paths_are_complete():
    result = nested_tree()
    summary = conflict_summary(result, node_budget=4)
    for terminal in summary["terminal_conflicts"]:
        for entry in terminal["active_constraints"]:
            assert entry["path"][0]["name"] == "<root>"
            assert entry["path"][-1]["name"]
            # Every retained path is rendered end-to-end with versions.
            assert all("name" in node for node in entry["path"])


def test_budget_does_not_change_solver_result():
    result = nested_tree()
    full_attempts = result.attempts
    full_status = result.status
    full_locks = dict(result.lock_notes)
    summary = conflict_summary(result, node_budget=1)
    # Serialization only: choices, attempts and lock notes are untouched.
    assert result.attempts == full_attempts
    assert result.status == full_status
    assert dict(result.lock_notes) == full_locks
    assert summary["truncated"] is True


def test_duplicate_leaves_under_budget_are_deduplicated_once_retained():
    # The same pending constraint (root 1.0 -> absent "rootghost") is
    # re-evaluated under every candidate of "mid", producing identical leaf
    # signatures; the summary must keep only one, like the full renderer.
    packages = {
        "root": {
            "2.0": {"dependencies": {"ghost": ""}},
            "1.0": {"dependencies": {"mid": ">=1.0", "rootghost": ""}},
        },
        "mid": {"2.0": {}, "1.0": {}},
    }
    result = solve(packages, {"root": ""})
    tree = result.conflict
    mid = tree.children[1]
    raw_rootghost = [c for c in mid.children if c.package == "rootghost"]
    assert len(raw_rootghost) == 2  # two identical leaves in the failure tree
    summary = conflict_summary(result, node_budget=100)
    duplicate_terminals = [
        t for t in summary["terminal_conflicts"] if t["package"] == "rootghost"
    ]
    assert len(duplicate_terminals) == 1

    # A tight budget that retains just the first duplicate lists it once too.
    truncated = conflict_summary(result, node_budget=4)
    assert truncated["terminal_conflicts"][-1]["package"] == "rootghost"
    assert sum(
        1 for t in truncated["terminal_conflicts"] if t["package"] == "rootghost"
    ) == 1


def test_server_env_budget_is_applied(monkeypatch):
    result = wide_fanout()
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "2")
    summary = conflict_summary(result)
    assert summary["truncated"] is True
    assert summary["node_budget"] == 2
    assert len(summary["tree"]["failed_branches"]) == 1


def test_server_env_unset_or_empty_is_unlimited(monkeypatch):
    result = wide_fanout()
    monkeypatch.delenv(CONFLICT_NODE_BUDGET_ENV, raising=False)
    assert "truncated" not in conflict_summary(result)
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "  ")
    assert "truncated" not in conflict_summary(result)


def test_server_env_invalid_value_falls_back_to_unlimited(monkeypatch):
    result = wide_fanout()
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "nonsense")
    summary = conflict_summary(result)
    assert "truncated" not in summary
    assert len(summary["terminal_conflicts"]) == 3
