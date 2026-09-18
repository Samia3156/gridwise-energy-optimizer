"""
GridWise Energy Optimizer - Deterministic guardrails.

The LLM in `app/llm.py` is the operator-note interpretation path, but the
LLM is NEVER trusted blindly. This module validates every LLM-emitted
DirectiveInterpretation before it is allowed to reach the optimizer.

We fail SAFE: on any rule violation we coerce the result to
{
    "applies": false,
    "directive_type": "no_op",
    "structured_adjustment": null
}
and surface the reason in logs. The optimizer must never see
unvalidated LLM output.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas import DirectiveInterpretation, OptimizeRequest

log = logging.getLogger("gridwise.guardrails")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NO_OP: dict[str, Any] = {
    "applies": False,
    "directive_type": "no_op",
    "structured_adjustment": None,
    "explanation": "no_op fallback",
}

_VALID_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_interpretations(
    raw: list[dict[str, Any]],
    request: OptimizeRequest,
) -> list[DirectiveInterpretation]:
    """
    Validate all LLM-produced directive interpretations.

    The returned list:
    - has exactly the same length as operator_notes
    - preserves the same order
    - contains only validated DirectiveInterpretation objects
    - converts invalid LLM output into safe no-op directives
    """

    n_notes = len(request.operator_notes)

    if len(raw) != n_notes:
        log.warning(
            "LLM returned %d interpretations for %d notes; coercing.",
            len(raw),
            n_notes,
        )

        raw = list(raw)

        if len(raw) < n_notes:
            raw.extend(
                [_NO_OP.copy() for _ in range(n_notes - len(raw))]
            )

        raw = raw[:n_notes]

    validated: list[DirectiveInterpretation] = []

    for idx, item in enumerate(raw):
        note = request.operator_notes[idx]

        safe = _validate_one(
            item,
            note=note,
            index=idx,
        )

        validated.append(safe)

    return validated


def validate_single(
    raw: dict[str, Any],
    *,
    note: str,
    index: int = 0,
) -> DirectiveInterpretation:
    """Validate one LLM interpretation. Exposed for tests."""

    return _validate_one(
        raw,
        note=note,
        index=index,
    )


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_one(
    raw: Any,
    *,
    note: str,
    index: int,
) -> DirectiveInterpretation:

    # 1. Result must be a dictionary.
    if not isinstance(raw, dict):
        log.warning(
            "note[%d]: LLM result is not a dict -> no_op.",
            index,
        )

        return _safe_no_op(index=index)

    # 2. applies must be a boolean.
    applies_raw = raw.get("applies")

    if not isinstance(applies_raw, bool):
        log.warning(
            "note[%d]: 'applies' must be bool -> no_op.",
            index,
        )

        return _safe_no_op(index=index)

    # 3. directive_type must be supported.
    dtype_raw = raw.get("directive_type")

    if not isinstance(dtype_raw, str) or dtype_raw not in _VALID_TYPES:
        log.warning(
            "note[%d]: unsupported directive_type %r -> no_op.",
            index,
            dtype_raw,
        )

        return _safe_no_op(index=index)

    directive_type: str = dtype_raw

    # 4. Read structured adjustment.
    adj_raw = raw.get("structured_adjustment")

    # 5. applies=false MUST always become no_op.
    if not applies_raw:

        if directive_type != "no_op" or adj_raw is not None:
            log.warning(
                "note[%d]: applies=false requires "
                "no_op + null adjustment -> coerced.",
                index,
            )

        return _safe_no_op(index=index)

    # 6. applies=true cannot use no_op.
    if directive_type == "no_op":
        log.warning(
            "note[%d]: applies=true with no_op is contradictory -> no_op.",
            index,
        )

        return _safe_no_op(index=index)

    # 7. Validate structured adjustment.
    safe_adj = _validate_adjustment(
        directive_type,
        adj_raw,
        index=index,
    )

    if safe_adj is _INVALID_ADJUSTMENT:
        log.warning(
            "note[%d] (%s): invalid structured_adjustment "
            "-> no_op. note=%r",
            index,
            directive_type,
            note,
        )

        return _safe_no_op(index=index)

    # 8. Explanation is informational only.
    explanation = _coerce_explanation(
        raw.get("explanation")
    )

    # 9. IMPORTANT:
    #    note_index is required by DirectiveInterpretation.
    return DirectiveInterpretation(
        note_index=index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=safe_adj,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Safe fallback
# ---------------------------------------------------------------------------

def _safe_no_op(
    *,
    index: int,
) -> DirectiveInterpretation:
    """
    Construct a schema-valid safe no-op.

    IMPORTANT:
    DirectiveInterpretation requires `note_index`, which was the source
    of the previous 500 Internal Server Error.
    """

    return DirectiveInterpretation(
        note_index=index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation="",
    )


# ---------------------------------------------------------------------------
# Explanation helper
# ---------------------------------------------------------------------------

def _coerce_explanation(value: Any) -> str:
    """Trim free-text explanation to a string."""

    if not isinstance(value, str):
        return ""

    return value.strip()


# ---------------------------------------------------------------------------
# Adjustment validation
# ---------------------------------------------------------------------------

_INVALID_ADJUSTMENT = object()


def _validate_adjustment(
    directive_type: str,
    adj: Any,
    *,
    index: int,
) -> Any:
    """
    Validate and sanitize structured_adjustment.

    Returns:
        sanitized dict
        OR
        _INVALID_ADJUSTMENT
    """

    if not isinstance(adj, dict):
        return _INVALID_ADJUSTMENT

    # ---------------------------------------------------------
    # solar_reduction
    # ---------------------------------------------------------

    if directive_type == "solar_reduction":

        hours = _validate_hours_list(
            adj.get("hours"),
            index=index,
        )

        if hours is _INVALID_ADJUSTMENT:
            return _INVALID_ADJUSTMENT

        factor = adj.get("factor")

        if not _is_number(factor):
            return _INVALID_ADJUSTMENT

        factor_float = float(factor)

        if not (0.0 <= factor_float <= 1.0):
            return _INVALID_ADJUSTMENT

        return {
            "hours": hours,
            "factor": factor_float,
        }

    # ---------------------------------------------------------
    # minimum_battery_reserve
    # ---------------------------------------------------------

    if directive_type == "minimum_battery_reserve":

        hours = _validate_hours_list(
            adj.get("hours"),
            index=index,
        )

        if hours is _INVALID_ADJUSTMENT:
            return _INVALID_ADJUSTMENT

        min_kwh = adj.get("minimum_energy_kwh")

        if not _is_number(min_kwh):
            return _INVALID_ADJUSTMENT

        min_kwh_float = float(min_kwh)

        if min_kwh_float < 0:
            return _INVALID_ADJUSTMENT

        return {
            "hours": hours,
            "minimum_energy_kwh": min_kwh_float,
        }

    # ---------------------------------------------------------
    # no_charge_window / no_discharge_window
    # ---------------------------------------------------------

    if directive_type in {
        "no_charge_window",
        "no_discharge_window",
    }:

        hours = _validate_hours_list(
            adj.get("hours"),
            index=index,
        )

        if hours is _INVALID_ADJUSTMENT:
            return _INVALID_ADJUSTMENT

        return {
            "hours": hours,
        }

    # ---------------------------------------------------------
    # max_grid_window
    # ---------------------------------------------------------

    if directive_type == "max_grid_window":

        hours = _validate_hours_list(
            adj.get("hours"),
            index=index,
        )

        if hours is _INVALID_ADJUSTMENT:
            return _INVALID_ADJUSTMENT

        max_grid = adj.get("max_grid_kwh")

        if not _is_number(max_grid):
            return _INVALID_ADJUSTMENT

        max_grid_float = float(max_grid)

        if max_grid_float < 0:
            return _INVALID_ADJUSTMENT

        return {
            "hours": hours,
            "max_grid_kwh": max_grid_float,
        }

    # Defensive fallback.
    return _INVALID_ADJUSTMENT


# ---------------------------------------------------------------------------
# Hours validation
# ---------------------------------------------------------------------------

def _validate_hours_list(
    value: Any,
    *,
    index: int,
) -> Any:
    """
    Validate official hours list.

    Requirements:
    - must be a list
    - must not be empty
    - every value must be a plain int
    - bool is rejected
    - every hour must be 0..23
    - values must be unique
    - values must be sorted ascending
    """

    if not isinstance(value, list):
        return _INVALID_ADJUSTMENT

    if len(value) == 0:
        return _INVALID_ADJUSTMENT

    seen: set[int] = set()
    out: list[int] = []

    last = -1

    for h in value:

        if not _is_int(h):
            return _INVALID_ADJUSTMENT

        h_int = int(h)

        if not (0 <= h_int <= 23):
            return _INVALID_ADJUSTMENT

        if h_int in seen:
            return _INVALID_ADJUSTMENT

        if h_int <= last:
            return _INVALID_ADJUSTMENT

        seen.add(h_int)
        out.append(h_int)
        last = h_int

    return out


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    """
    True for finite ints/floats.

    bool is explicitly rejected.
    NaN and infinity are rejected.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return value == value and value not in (
        float("inf"),
        float("-inf"),
    )


def _is_int(value: Any) -> bool:
    """True for ints, but not bools."""

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
    )
