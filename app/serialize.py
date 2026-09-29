"""Serialization of solver results into the public JSON response shape."""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional

from .resolver import (
    ROOT,
    CompositeConflict,
    Constraint,
    LeafConflict,
    SolveResult,
)

# ---------------------------------------------------------------------------
# Server-side node budget
# ---------------------------------------------------------------------------
#
# An exhaustive failure tree can fan out into far more branches than are
# useful (or affordable) in a 409 response.  The budget caps the number of
# conflict-tree *nodes* (composite + leaf) rendered into ``conflict.tree``;
# traversal is the same pre-order depth-first walk as before, so while the
# budget holds the node fields, branch order and constraint paths are
# identical to the unbudgeted report.  When it runs out, the tree is cut in
# place and only terminals belonging to retained nodes are listed.
#
# Configured once per server process from the environment; ``0`` means
# unlimited.  A missing/empty value uses the default, and an invalid value
# never prevents the service from starting: a warning is emitted and the
# default is used.

DEFAULT_CONFLICT_NODE_BUDGET = 10_000
CONFLICT_NODE_BUDGET_ENV = "RESOLVER_CONFLICT_NODE_BUDGET"


def parse_conflict_node_budget(raw: Any) -> Optional[int]:
    """Parse a raw node-budget value.

    ``None``/empty -> :data:`DEFAULT_CONFLICT_NODE_BUDGET`; ``0`` -> ``None``
    (unlimited); any other non-negative integer is returned as-is.
    """
    if raw is None or str(raw).strip() == "":
        return DEFAULT_CONFLICT_NODE_BUDGET
    text = str(raw).strip()
    try:
        value = int(text)
    except ValueError:
        raise ValueError(
            f"{CONFLICT_NODE_BUDGET_ENV} must be a non-negative integer, got {text!r}"
        ) from None
    if value < 0:
        raise ValueError(
            f"{CONFLICT_NODE_BUDGET_ENV} must be a non-negative integer, got {text!r}"
        )
    return None if value == 0 else value


def _load_conflict_node_budget() -> Optional[int]:
    raw = os.environ.get(CONFLICT_NODE_BUDGET_ENV, "")
    try:
        return parse_conflict_node_budget(raw)
    except ValueError as exc:
        warnings.warn(
            f"{exc}; falling back to {DEFAULT_CONFLICT_NODE_BUDGET}",
            stacklevel=2,
        )
        return DEFAULT_CONFLICT_NODE_BUDGET


_UNSET = object()
# Process-wide cache of the environment-configured budget.  Tests reset it to
# ``_UNSET`` together with the environment variable.
_configured_budget: Any = _UNSET


def get_configured_node_budget() -> Optional[int]:
    """The server-wide conflict node budget (cached after first lookup)."""
    global _configured_budget
    if _configured_budget is _UNSET:
        _configured_budget = _load_conflict_node_budget()
    return _configured_budget  # type: ignore[return-value]


class _NodeBudget:
    """Mutable pre-order traversal budget shared by the recursive renderer."""

    def __init__(self, limit: Optional[int]) -> None:
        # ``None`` means unlimited.
        self.limit = limit
        self.used = 0
        # Terminal leaves whose node actually made it into the rendered tree.
        self.kept_leaves: list[LeafConflict] = []
        # Immediate child branches skipped at cut composites, summed over
        # every composite in the tree.
        self.omitted_branches = 0

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    def take(self) -> bool:
        """Reserve one node. Returns ``False`` once the budget is spent."""
        if self.exhausted:
            return False
        self.used += 1
        return True


# ---------------------------------------------------------------------------
# Node rendering
# ---------------------------------------------------------------------------


def _path_of(constraint: Constraint) -> list[dict[str, str]]:
    """Render the dependency path that introduced a constraint.

    ``path_nodes`` holds the hops leading to the requiring package; the
    requiring package itself is appended.  Root requirements render as a
    single ``<root>`` node.
    """
    path: list[dict[str, str]] = [{"name": ROOT, "version": ""}]
    for name, version in constraint.path_nodes:
        path.append({"name": name, "version": version})
    if constraint.source != ROOT:
        path.append({"name": constraint.source, "version": constraint.source_version or ""})
    return path


def _constraint_entry(constraint: Constraint, target: str) -> dict[str, Any]:
    spec = constraint.specifier_text or ""
    return {
        "requirement": f"{target}{spec}" if spec else target,
        "specifier": spec,
        "required_by": constraint.source,
        "required_by_version": constraint.source_version or "",
        "path": _path_of(constraint),
    }


def _all_constraints(leaf: LeafConflict) -> list[Constraint]:
    """Every constraint participating in the contradiction.

    ``rejected`` constraints are the ones that could not coexist with an
    already selected version; together with the active constraints they
    show the mutually-exclusive requirements.  Deduplicated because the
    same (source, version, specifier) can be observed on multiple branches.
    """
    combined = list(leaf.constraints) + list(leaf.rejected)
    unique: dict[tuple[str, str, str, tuple], Constraint] = {}
    for c in combined:
        key = (
            c.source,
            c.source_version or "",
            c.specifier_text,
            c.path_nodes,
        )
        unique.setdefault(key, c)
    return sorted(
        unique.values(),
        key=lambda c: (c.source, c.source_version or "", c.specifier_text),
    )


def _leaf_node(leaf: LeafConflict) -> dict[str, Any]:
    package = leaf.package
    return {
        "kind": "leaf",
        "package": package,
        "reason": leaf.reason,
        "message": leaf.detail,
        "available_versions": leaf.available,
        "attempted_versions": leaf.attempted,
        "active_constraints": [_constraint_entry(c, package) for c in _all_constraints(leaf)],
        "rejected_by": [_constraint_entry(c, package) for c in sorted(
            leaf.rejected,
            key=lambda c: (c.source, c.source_version or "", c.specifier_text),
        )],
    }


def _terminal_entry(leaf: LeafConflict) -> dict[str, Any]:
    return {
        "package": leaf.package,
        "reason": leaf.reason,
        "message": leaf.detail,
        "available_versions": leaf.available,
        "attempted_versions": leaf.attempted,
        "active_constraints": [
            _constraint_entry(c, leaf.package)
            for c in _all_constraints(leaf)
        ],
        "rejected_by": [
            _constraint_entry(c, leaf.package)
            for c in sorted(
                leaf.rejected,
                key=lambda c: (c.source, c.source_version or "", c.specifier_text),
            )
        ],
    }


def _render_leaf(leaf: LeafConflict, budget: _NodeBudget) -> Optional[dict[str, Any]]:
    if not budget.take():
        return None
    budget.kept_leaves.append(leaf)
    return _leaf_node(leaf)


def _render_composite(
    composite: CompositeConflict,
    depth: int,
    seen: set[str],
    budget: _NodeBudget,
) -> Optional[dict[str, Any]]:
    if not budget.take():
        return None
    package = composite.package
    children: list[dict[str, Any]] = []
    omitted_here = 0
    # Branch order is the solver's own order; once the budget is exhausted a
    # branch is either skipped wholesale here or comes back already marked
    # truncated from deeper inside it.
    for child in composite.children:
        if budget.exhausted:
            omitted_here += 1
            budget.omitted_branches += 1
            continue
        # ``take`` succeeded for this frame, so a child always renders at
        # least its own root node while the budget is not already exhausted.
        node = _render_conflict(child, depth + 1, seen, budget)
        children.append(node)
    node = {
        "kind": "composite",
        "package": package,
        "reason": composite.reason,
        "message": (
            f"every eligible version of {package} failed: "
            + (", ".join(composite.attempted) if composite.attempted else "none tried")
        ),
        "attempted_versions": composite.attempted,
        "available_versions": composite.available,
        "active_constraints": [
            _constraint_entry(c, package)
            for c in sorted(
                composite.constraints,
                key=lambda c: (c.source, c.source_version or "", c.specifier_text),
            )
        ],
        "failed_branches": children,
    }
    if omitted_here:
        node["truncated"] = True
        node["omitted_branches"] = omitted_here
    return node


def _render_conflict(
    conflict: Any,
    depth: int,
    seen: set[str],
    budget: _NodeBudget,
) -> Optional[dict[str, Any]]:
    if isinstance(conflict, LeafConflict):
        return _render_leaf(conflict, budget)
    return _render_composite(conflict, depth, seen, budget)


def serialize_conflict(conflict: Any, depth: int = 0, seen: set[str] | None = None) -> dict[str, Any]:
    """Render a conflict tree without a node budget (unlimited)."""
    node = _render_conflict(conflict, depth, seen if seen is not None else set(), _NodeBudget(None))
    assert node is not None
    return node


def _count_nodes(conflict: Any) -> int:
    """Total node count of the in-memory failure tree."""
    if isinstance(conflict, LeafConflict):
        return 1
    return 1 + sum(_count_nodes(child) for child in conflict.children)


def _leaf_signature(leaf: LeafConflict) -> tuple[str, str]:
    return (
        leaf.package,
        "|".join(
            sorted(
                f"{c.source}:{c.source_version or ''}:{c.specifier_text}"
                for c in leaf.constraints + leaf.rejected
            )
        ),
    )


def conflict_summary(
    result: SolveResult,
    node_budget: Any = _UNSET,
) -> dict[str, Any]:
    """Build the ``conflict`` object for a 409 response.

    ``node_budget`` selects the traversal budget (number of rendered tree
    nodes): unset -> the server-wide :data:`CONFLICT_NODE_BUDGET_ENV`
    configuration; ``None`` or ``0`` -> unlimited; a positive integer ->
    that many nodes.  With the budget sufficient the output is byte-for-byte
    the historical shape (same fields, branch order and constraint paths).
    When the budget is exhausted, ``tree`` is cut in pre-order and marked in
    place, and ``terminal_conflicts`` lists only terminals of retained nodes.
    """
    assert result.conflict is not None
    if node_budget is _UNSET:
        limit = get_configured_node_budget()
    elif node_budget is None or node_budget == 0:
        limit = None
    else:
        if not isinstance(node_budget, int) or isinstance(node_budget, bool) or node_budget < 0:
            raise ValueError("node_budget must be a non-negative integer, 0, or None")
        limit = node_budget

    budget = _NodeBudget(limit)
    tree = _render_conflict(result.conflict, 0, set(), budget)
    assert tree is not None  # every non-negative budget retains the root

    # Deduplicate terminals the same way the unbudgeted report did, but only
    # over leaves that actually survived traversal; ordering still follows
    # the historical package sort.
    unique_leaves: dict[tuple[str, str], LeafConflict] = {}
    for leaf in budget.kept_leaves:
        unique_leaves.setdefault(_leaf_signature(leaf), leaf)
    leaves = sorted(unique_leaves.values(), key=lambda l: l.package)

    summary: dict[str, Any] = {
        "tree": tree,
        "terminal_conflicts": [_terminal_entry(leaf) for leaf in leaves],
    }

    total_nodes = _count_nodes(result.conflict)
    omitted_nodes = total_nodes - budget.used
    if omitted_nodes > 0 or budget.omitted_branches:
        summary["truncated"] = True
        summary["node_budget"] = limit
        summary["retained_nodes"] = budget.used
        summary["omitted_nodes"] = omitted_nodes
        summary["omitted_branches"] = budget.omitted_branches
    return summary
