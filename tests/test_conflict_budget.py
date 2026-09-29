"""Regression tests for the server-configurable conflict-tree node budget.

The solver itself (candidate choice, attempt counting, lock disposition) is
untouched by the budget: it only bounds how much of the already-computed
failure tree is rendered into the 409 body.  These tests cover

* budget parsing / server configuration (env override, 0, invalid fallback);
* unbudgeted output remaining byte-identical to the historical shape;
* pre-order truncation: retained nodes keep fields, branch order and
  constraint paths; cut composites and the summary are marked;
* ``terminal_conflicts`` listing terminals of retained nodes only;
* the same 409 response over ``/resolve`` (and the shared helper used by the
  analysis endpoints), with attempts and locks unchanged.
"""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

import app.serialize as serialize
from app.main import app
from app.resolver import parse_index, parse_requirements, resolve
from app.serialize import (
    CONFLICT_NODE_BUDGET_ENV,
    DEFAULT_CONFLICT_NODE_BUDGET,
    conflict_summary,
    parse_conflict_node_budget,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_budget_cache():
    """Keep env-driven configuration from leaking between tests."""
    yield
    serialize._configured_budget = serialize._UNSET


# ---------------------------------------------------------------------------
# Inputs that produce sizable failure trees
# ---------------------------------------------------------------------------


def _wide_packages(width: int = 8) -> dict:
    # Every app candidate pins its own missing package: the root composite
    # gets exactly ``width`` leaf branches, in solver (highest-first) order.
    packages = {
        "app": {
            f"{i}.0": {"dependencies": {f"ghost{i}": "==9.9"}}
            for i in range(1, width + 1)
        },
    }
    packages.update({f"ghost{i}": {"1.0": {}} for i in range(1, width + 1)})
    return packages


def _wide_result(width: int = 8, max_attempts: int = 100_000):
    index, errors = parse_index(_wide_packages(width))
    assert not errors
    return resolve(
        index,
        parse_requirements(["app"], "root requirements"),
        {},
        max_attempts=max_attempts,
    )


def _merged_packages():
    # Three app branches all explore the same three lib versions, and every
    # lib version fails identically on ``ghost``: nine leaf nodes but only
    # three distinct terminal signatures after the historical dedupe.
    return {
        "app": {
            "3.0": {"dependencies": {"lib": ""}},
            "2.0": {"dependencies": {"lib": ""}},
            "1.0": {"dependencies": {"lib": ""}},
        },
        "lib": {
            "3.0": {"dependencies": {"ghost": "==9.9"}},
            "2.0": {"dependencies": {"ghost": "==9.9"}},
            "1.0": {"dependencies": {"ghost": "==9.9"}},
        },
        "ghost": {"1.0": {}},
    }


def _merged_result():
    index, errors = parse_index(_merged_packages())
    assert not errors
    return resolve(index, parse_requirements(["app"], "root requirements"), {})


def _node_count(node: dict) -> int:
    if node["kind"] == "leaf":
        return 1
    return 1 + sum(_node_count(child) for child in node["failed_branches"])


# ---------------------------------------------------------------------------
# Parsing / server configuration
# ---------------------------------------------------------------------------


def test_parse_node_budget_values():
    assert parse_conflict_node_budget(None) == DEFAULT_CONFLICT_NODE_BUDGET
    assert parse_conflict_node_budget("") == DEFAULT_CONFLICT_NODE_BUDGET
    assert parse_conflict_node_budget("   ") == DEFAULT_CONFLICT_NODE_BUDGET
    assert parse_conflict_node_budget("0") is None  # unlimited
    assert parse_conflict_node_budget("42") == 42
    assert parse_conflict_node_budget(42) == 42
    with pytest.raises(ValueError):
        parse_conflict_node_budget("abc")
    with pytest.raises(ValueError):
        parse_conflict_node_budget("-1")


def test_configured_budget_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv(CONFLICT_NODE_BUDGET_ENV, raising=False)
    serialize._configured_budget = serialize._UNSET
    assert serialize.get_configured_node_budget() == DEFAULT_CONFLICT_NODE_BUDGET

    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "7")
    serialize._configured_budget = serialize._UNSET
    assert serialize.get_configured_node_budget() == 7

    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "0")
    serialize._configured_budget = serialize._UNSET
    assert serialize.get_configured_node_budget() is None


def test_invalid_configuration_warns_and_keeps_service_usable(monkeypatch):
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "not-a-number")
    serialize._configured_budget = serialize._UNSET
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert serialize.get_configured_node_budget() == DEFAULT_CONFLICT_NODE_BUDGET
    assert caught and CONFLICT_NODE_BUDGET_ENV in str(caught[0].message)
    # Startup must remain available despite the bad value.
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Unbudgeted output is unchanged
# ---------------------------------------------------------------------------


def test_unlimited_budget_keeps_historical_shape():
    result = _wide_result()
    for budget in (None, 0):
        summary = conflict_summary(result, node_budget=budget)
        assert set(summary.keys()) == {"tree", "terminal_conflicts"}
        assert "truncated" not in summary["tree"]
    # Explicit large budget and the implicit configured default agree when
    # the tree fits.
    full = conflict_summary(result, node_budget=None)
    assert conflict_summary(result, node_budget=10_000) == full
    serialize._configured_budget = 10_000
    assert conflict_summary(result) == full


def test_default_configuration_does_not_truncate_small_trees():
    status, body = _resolve_http(_wide_packages(8))
    assert status == 409
    assert "truncated" not in body["conflict"]
    assert len(body["conflict"]["terminal_conflicts"]) == 8


# ---------------------------------------------------------------------------
# Truncation semantics (unit level)
# ---------------------------------------------------------------------------


def test_budget_cuts_tree_preorder_and_marks_it():
    result = _wide_result()
    full = conflict_summary(result, node_budget=None)
    summary = conflict_summary(result, node_budget=4)

    # Root plus the first three solver-order leaves fit; nothing else.
    assert summary["retained_nodes"] == 4
    assert _node_count(summary["tree"]) == 4
    assert summary["node_budget"] == 4
    assert summary["omitted_nodes"] == serialize._count_nodes(result.conflict) - 4
    assert summary["omitted_branches"] == 5
    assert summary["truncated"] is True

    root = summary["tree"]
    assert root["truncated"] is True
    assert root["omitted_branches"] == 5
    assert len(root["failed_branches"]) == 3
    # Branch order and content of the kept prefix are exactly the full ones.
    full_root = full["tree"]
    assert root["failed_branches"] == full_root["failed_branches"][:3]
    # Composite fields and constraint paths survive untouched.
    for key in (
        "kind", "package", "reason", "message",
        "attempted_versions", "available_versions", "active_constraints",
    ):
        assert root[key] == full_root[key]


def test_terminal_conflicts_only_cover_retained_nodes():
    result = _wide_result()
    full = conflict_summary(result, node_budget=None)
    summary = conflict_summary(result, node_budget=4)

    # The solver explores app 8..1, so leaves ghost8/7/6 survive; after the
    # historical package sort that yields ghost6..ghost8.
    kept_packages = {terminal["package"] for terminal in summary["terminal_conflicts"]}
    assert kept_packages == {"ghost6", "ghost7", "ghost8"}

    full_by_package = {t["package"]: t for t in full["terminal_conflicts"]}
    for terminal in summary["terminal_conflicts"]:
        # Same payload (paths included) as in the complete report.
        assert terminal == full_by_package[terminal["package"]]
    assert len(summary["terminal_conflicts"]) == 3
    # The five cut branches' terminals must not leak through.
    assert "ghost1" not in kept_packages


def test_nested_cuts_are_marked_inside_the_tree():
    result = _merged_result()
    # Tree: app composite -> three lib composites -> each three ghost leaves
    # (13 nodes).  Budget 6 keeps the app node, the first lib composite and
    # its three leaves, plus the second lib composite frame (whose branches
    # are all cut); the third lib subtree is skipped wholesale at the app
    # level.
    assert serialize._count_nodes(result.conflict) == 13
    summary = conflict_summary(result, node_budget=6)

    app_node = summary["tree"]
    assert app_node["truncated"] is True
    assert app_node["omitted_branches"] == 1
    assert len(app_node["failed_branches"]) == 2
    first_lib, second_lib = app_node["failed_branches"]
    assert len(first_lib["failed_branches"]) == 3
    assert "truncated" not in first_lib
    assert second_lib["truncated"] is True
    assert second_lib["omitted_branches"] == 3
    assert summary["omitted_branches"] == 4
    assert summary["retained_nodes"] == 6


def test_terminal_dedupe_runs_over_retained_nodes_only():
    result = _merged_result()
    # The full tree collapses nine identical leaves to three terminals.
    full = conflict_summary(result, node_budget=None)
    assert {t["package"] for t in full["terminal_conflicts"]} == {"ghost"}
    assert len(full["terminal_conflicts"]) == 3

    # Budget 3 keeps only the first lib composite's first leaf node.
    summary = conflict_summary(result, node_budget=3)
    assert len(summary["terminal_conflicts"]) == 1
    terminal = summary["terminal_conflicts"][0]
    assert terminal["package"] == "ghost"
    constraint = terminal["active_constraints"][0]
    assert constraint["required_by"] == "lib"
    assert constraint["required_by_version"] == "3.0"


def test_exact_fit_does_not_truncate():
    result = _wide_result()
    total = serialize._count_nodes(result.conflict)  # root + 8 leaves = 9
    summary = conflict_summary(result, node_budget=total)
    assert "truncated" not in summary
    assert len(summary["terminal_conflicts"]) == 8


def test_invalid_budget_argument_is_rejected():
    result = _wide_result()
    with pytest.raises(ValueError):
        conflict_summary(result, node_budget=-1)
    with pytest.raises(ValueError):
        conflict_summary(result, node_budget="4")
    with pytest.raises(ValueError):
        conflict_summary(result, node_budget=True)  # bool is not a budget


# ---------------------------------------------------------------------------
# HTTP delivery through /resolve
# ---------------------------------------------------------------------------


def _resolve_http(packages, **extra):
    response = client.post("/resolve", json={"packages": packages, "root": ["app"], **extra})
    return response.status_code, response.json()


def test_resolve_409_is_truncated_under_small_server_budget(monkeypatch):
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "4")
    serialize._configured_budget = serialize._UNSET

    status, body = _resolve_http(_wide_packages(8))
    assert status == 409
    summary = body["conflict"]
    assert summary["truncated"] is True
    assert summary["node_budget"] == 4
    assert summary["retained_nodes"] == 4
    assert summary["omitted_nodes"] == 5
    assert summary["omitted_branches"] == 5
    assert len(summary["tree"]["failed_branches"]) == 3
    assert {t["package"] for t in summary["terminal_conflicts"]} == {
        "ghost6", "ghost7", "ghost8"
    }


def test_truncation_changes_only_the_report_not_solving(monkeypatch):
    packages = _wide_packages(8)
    status_full, body_full = _resolve_http(packages)
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "4")
    serialize._configured_budget = serialize._UNSET
    status_cut, body_cut = _resolve_http(packages)

    assert status_full == status_cut == 409
    # Solver-owned fields are identical; only the conflict payload shrinks.
    for key in ("status", "attempts", "locks", "allow_prereleases"):
        assert body_full[key] == body_cut[key]
    assert body_cut["conflict"]["truncated"] is True
    assert "truncated" not in body_full["conflict"]


def test_search_limit_409_is_also_budgeted(monkeypatch):
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "2")
    serialize._configured_budget = serialize._UNSET
    # Eight failing candidates; stop after three attempts so the limit
    # composite carries three leaf branches.
    status, body = _resolve_http(_wide_packages(8), max_attempts=3)
    assert status == 409
    assert body["status"] == "limit"
    assert body["error"]["code"] == "search_limit_exceeded"
    summary = body["conflict"]
    assert summary["truncated"] is True
    assert summary["retained_nodes"] == 2
    assert summary["omitted_branches"] == 2
    assert len(summary["tree"]["failed_branches"]) == 1
    assert len(summary["terminal_conflicts"]) == 1


def test_budget_is_shared_by_analysis_endpoints(monkeypatch):
    # The analysis endpoints render conflicts with the same helper, so a
    # tight server budget truncates them identically.  avoid-vulnerabilities
    # keeps the full candidate fan-out (the advisory excludes no version).
    monkeypatch.setenv(CONFLICT_NODE_BUDGET_ENV, "3")
    serialize._configured_budget = serialize._UNSET
    response = client.post(
        "/resolve/avoid-vulnerabilities",
        json={
            "packages": _wide_packages(8),
            "root": ["app"],
            "advisories": [
                {"id": "ADV-1", "package": "ghost1", "affected": ">=99"}
            ],
        },
    )
    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "conflict"
    summary = body["conflict"]
    assert summary["truncated"] is True
    assert summary["retained_nodes"] == 3
    assert summary["omitted_branches"] == 6
