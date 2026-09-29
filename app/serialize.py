"""Serialization of solver results into the public JSON response shape."""

from __future__ import annotations

import os
from typing import Any, Optional

from .resolver import (
    ROOT,
    CompositeConflict,
    Constraint,
    LeafConflict,
    SolveResult,
)

#: Environment variable holding the server-side conflict-tree node budget.
CONFLICT_NODE_BUDGET_ENV = "RESOLVER_CONFLICT_NODE_BUDGET"

#: Sentinel selecting the server-configured (environment) node budget.
_SERVER_DEFAULT: Any = object()


def parse_node_budget(raw: Optional[str]) -> Optional[int]:
    """Parse a configured conflict-tree node budget.

    ``None`` / empty mean *unlimited* (the historical behaviour); otherwise a
    positive integer is required.  Malformed or non-positive values raise
    ``ValueError`` so the server can reject the configuration at startup.
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(
            f"{CONFLICT_NODE_BUDGET_ENV}={text!r}: node budget must be a positive integer"
        ) from exc
    if value < 1:
        raise ValueError(
            f"{CONFLICT_NODE_BUDGET_ENV}={text!r}: node budget must be a positive integer"
        )
    return value


def get_node_budget() -> Optional[int]:
    """Read the conflict-tree node budget from the environment per call.

    The configuration is validated once when the server starts; if a malformed
    value reaches the serializer through direct library use, the unlimited
    default is used rather than failing the request.
    """
    try:
        return parse_node_budget(os.environ.get(CONFLICT_NODE_BUDGET_ENV))
    except ValueError:
        return None


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


def _composite_base(composite: CompositeConflict) -> dict[str, Any]:
    """Composite node with the same fields/order as the unlimited renderer.

    ``failed_branches`` starts empty and is filled in by the caller, so both
    the full and the budgeted traversal serialize identical node shapes.
    """
    package = composite.package
    return {
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
        "failed_branches": [],
    }


def _composite_node(
    composite: CompositeConflict, depth: int, seen: set[str]
) -> dict[str, Any]:
    node = _composite_base(composite)
    node["failed_branches"] = [
        serialize_conflict(child, depth + 1, seen) for child in composite.children
    ]
    return node


def serialize_conflict(conflict: Any, depth: int = 0, seen: set[str] | None = None) -> dict[str, Any]:
    """Serialize the *complete* failure tree without any node budget."""
    if seen is None:
        seen = set()
    if isinstance(conflict, LeafConflict):
        return _leaf_node(conflict)
    return _composite_node(conflict, depth, seen)


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


def _collect_leaves(conflict: Any, out: list[LeafConflict], seen: set[tuple[str, str]]) -> None:
    if isinstance(conflict, LeafConflict):
        signature = _leaf_signature(conflict)
        if signature not in seen:
            seen.add(signature)
            out.append(conflict)
        return
    for child in conflict.children:
        _collect_leaves(child, out, seen)


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


class _NodeBudget:
    """Reservation counter for serialized conflict-tree nodes.

    ``limit`` of ``None`` means unlimited.  Once ``used`` reaches the limit
    every further reservation fails and ``exhausted`` latches true, so the
    traversal stops expanding the remaining failure branches.
    """

    __slots__ = ("limit", "used", "exhausted")

    def __init__(self, limit: Optional[int]) -> None:
        self.limit = limit
        self.used = 0
        self.exhausted = False

    def take(self) -> bool:
        if self.limit is None:
            self.used += 1
            return True
        if self.used >= self.limit:
            self.exhausted = True
            return False
        self.used += 1
        return True


def _render_budgeted(
    conflict: Any,
    budget: _NodeBudget,
    leaves_out: list[LeafConflict],
) -> Optional[dict[str, Any]]:
    """Pre-order DFS that emits at most ``budget.limit`` tree nodes.

    Branch order and every retained node's fields are identical to
    :func:`serialize_conflict`.  A subtree whose root cannot be reserved is
    dropped entirely; a composite whose children are cut short gets
    ``truncated`` / ``omitted_branches`` markers.  Only leaves rendered into
    the retained tree are appended to ``leaves_out``.
    """
    if not budget.take():
        return None
    if isinstance(conflict, LeafConflict):
        leaves_out.append(conflict)
        return _leaf_node(conflict)

    node = _composite_base(conflict)
    retained_branches: list[dict[str, Any]] = []
    omitted = 0
    for child in conflict.children:
        rendered = _render_budgeted(child, budget, leaves_out)
        if rendered is None:
            omitted += 1
        else:
            retained_branches.append(rendered)
    node["failed_branches"] = retained_branches
    if omitted:
        node["truncated"] = True
        node["omitted_branches"] = omitted
    return node


def _terminal_conflicts(leaves: list[LeafConflict]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], LeafConflict] = {}
    for leaf in leaves:
        unique.setdefault(_leaf_signature(leaf), leaf)
    return [_terminal_entry(leaf) for leaf in sorted(unique.values(), key=lambda l: l.package)]


def conflict_summary(
    result: SolveResult,
    node_budget: Optional[int] = _SERVER_DEFAULT,
) -> dict[str, Any]:
    """Render the conflict summary, optionally bounded by a node budget.

    With an unlimited budget (the default, controlled server-side via the
    ``RESOLVER_CONFLICT_NODE_BUDGET`` environment variable) the full ``tree``
    and ``terminal_conflicts`` are emitted exactly as before.  When the budget
    is finite and the failure tree is larger, traversal stops in deterministic
    branch order: ``tree`` keeps only the reserved nodes (each cut marked with
    ``truncated`` / ``omitted_branches``), ``terminal_conflicts`` lists only
    the terminals under those retained nodes, and the summary itself carries
    ``truncated`` metadata.  Solver choices, attempts and lock dispositions
    are unaffected -- only this serialization is bounded.
    """
    assert result.conflict is not None
    if node_budget is _SERVER_DEFAULT:
        node_budget = get_node_budget()

    if node_budget is None:
        leaves: list[LeafConflict] = []
        _collect_leaves(result.conflict, leaves, set())
        return {
            "tree": serialize_conflict(result.conflict),
            "terminal_conflicts": _terminal_conflicts(leaves),
        }

    budget = _NodeBudget(node_budget)
    retained_leaves: list[LeafConflict] = []
    tree = _render_budgeted(result.conflict, budget, retained_leaves)
    summary: dict[str, Any] = {
        "tree": tree,
        "terminal_conflicts": _terminal_conflicts(retained_leaves),
    }
    if budget.exhausted:
        summary["truncated"] = True
        summary["truncation_reason"] = "node_budget_exhausted"
        summary["node_budget"] = node_budget
        summary["retained_nodes"] = budget.used
    return summary
