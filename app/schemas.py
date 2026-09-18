"""
GridWise Energy Optimizer - Pydantic request/response schemas.

These schemas mirror the official BUP CSE Fest 2026 Preliminary Hackathon
problem statement and the public sample case pack v2.0.

REQUEST (POST /optimize-energy)
-------------------------------
{
  "scenario_id": "<string>",
  "hours": [
    { "hour": 0..23, "demand_kwh": >=0, "solar_kwh": >=0, "tariff_bdt_per_kwh": >=0 },
    ... exactly 24 entries, hours 0..23 in ascending order ...
  ],
  "battery": {
    "capacity_kwh": >0,
    "initial_energy_kwh": >=0,
    "minimum_energy_kwh": >=0,
    "max_charge_kwh_per_hour": >=0,
    "max_discharge_kwh_per_hour": >=0
  },
  "operator_notes": [ "<non-empty string>", ... ]   # 1..3 entries
}

RESPONSE
--------
{
  "scenario_id": "<string>",
  "directive_interpretation": [
    {
      "note_index": int,
      "applies": bool,
      "directive_type": "<one of six supported strings>",
      "structured_adjustment": object | null,
      "explanation": "<free-text>"
    },
    ... one entry per operator note, in note_index order ...
  ],
  "hourly_plan": [
    {
      "hour": 0..23,
      "grid_kwh": >=0,
      "solar_used_kwh": >=0,
      "battery_action": "charge" | "discharge" | "idle",
      "battery_kwh": >=0,
      "battery_energy_after_kwh": >=0
    },
    ... exactly 24 entries, hours 0..23 in ascending order ...
  ],
  "total_grid_kwh": >=0,
  "total_cost_bdt": >=0,
  "peak_grid_kwh": >=0,
  "plan_summary": "<short prose summary>"
}

DIRECTIVE STRUCTURED_ADJUSTMENT (LLM-emitted, guardrail-validated)
-----------------------------------------------------------------
solar_reduction:
    { "hours": [int, ...], "factor": 0.0..1.0 }      # usable fraction remaining
minimum_battery_reserve:
    { "hours": [int, ...], "minimum_energy_kwh": >=0 }
no_charge_window:
    { "hours": [int, ...] }
no_discharge_window:
    { "hours": [int, ...] }
max_grid_window:
    { "hours": [int, ...], "max_grid_kwh": >=0 }
no_op:
    structured_adjustment MUST be null and applies MUST be false.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants (kept in one place so the LLM and optimizer can reuse them).
# ---------------------------------------------------------------------------

HOURS_IN_DAY = 24
MIN_OPERATOR_NOTES = 1
MAX_OPERATOR_NOTES = 3

SUPPORTED_DIRECTIVE_TYPES: set[str] = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

BATTERY_ACTIONS: set[str] = {"charge", "discharge", "idle"}


# ---------------------------------------------------------------------------
# Request: POST /optimize-energy
# ---------------------------------------------------------------------------

class HourEntry(BaseModel):
    """One hour of the 24-hour grid snapshot (official field names)."""

    hour: int = Field(ge=0, le=23, description="Hour of the day, 0..23.")
    demand_kwh: float = Field(ge=0, description="Demand this hour (kWh).")
    solar_kwh: float = Field(ge=0, description="Forecast solar this hour (kWh).")
    tariff_bdt_per_kwh: float = Field(ge=0, description="Grid tariff this hour (BDT/kWh).")


class BatterySpec(BaseModel):
    """Static battery parameters (official field names)."""

    capacity_kwh: float = Field(
        gt=0,
        description="Maximum usable battery capacity (kWh).",
    )
    initial_energy_kwh: float = Field(
        ge=0,
        description="Battery energy at the start of hour 0 (kWh).",
    )
    minimum_energy_kwh: float = Field(
        ge=0,
        description="Minimum allowed battery energy at any time (kWh).",
    )
    max_charge_kwh_per_hour: float = Field(
        ge=0,
        description="Max energy that may be charged in a single hour (kWh).",
    )
    max_discharge_kwh_per_hour: float = Field(
        ge=0,
        description="Max energy that may be discharged in a single hour (kWh).",
    )


class OptimizeRequest(BaseModel):
    """Body of POST /optimize-energy (official format)."""

    scenario_id: str = Field(
        ...,
        min_length=1,
        description="Caller-supplied scenario identifier.",
    )
    hours: List[HourEntry] = Field(
        ...,
        min_length=HOURS_IN_DAY,
        max_length=HOURS_IN_DAY,
        description="24 per-hour entries (hours 0..23, ascending, unique).",
    )
    battery: BatterySpec
    operator_notes: List[str] = Field(
        ...,
        min_length=MIN_OPERATOR_NOTES,
        max_length=MAX_OPERATOR_NOTES,
        description="1..3 non-empty natural-language directives.",
    )

    # ------------------------------------------------------------------ validators
    @field_validator("scenario_id")
    @classmethod
    def _scenario_id_non_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("scenario_id must be a non-empty string")
        return v

    @field_validator("hours")
    @classmethod
    def _hours_sequential_unique(cls, entries: List[HourEntry]) -> List[HourEntry]:
        seen: set[int] = set()
        last = -1
        for idx, e in enumerate(entries):
            if e.hour in seen:
                raise ValueError(f"hours[{idx}].hour={e.hour} is duplicated.")
            if e.hour != idx:
                raise ValueError(
                    f"hours[{idx}].hour must equal {idx}, got {e.hour}."
                )
            if e.hour <= last:
                raise ValueError(f"hours are not strictly ascending at index {idx}.")
            seen.add(e.hour)
            last = e.hour
        return entries

    @field_validator("operator_notes")
    @classmethod
    def _non_empty_strings(cls, values: List[str]) -> List[str]:
        for idx, note in enumerate(values):
            if not isinstance(note, str) or not note.strip():
                raise ValueError(
                    f"operator_notes[{idx}] must be a non-empty string"
                )
        return values

    @model_validator(mode="after")
    def _battery_invariants(self) -> "OptimizeRequest":
        b = self.battery
        if b.minimum_energy_kwh > b.capacity_kwh:
            raise ValueError(
                f"battery.minimum_energy_kwh ({b.minimum_energy_kwh}) "
                f"> capacity_kwh ({b.capacity_kwh})."
            )
        if not (b.minimum_energy_kwh <= b.initial_energy_kwh <= b.capacity_kwh):
            raise ValueError(
                "battery.initial_energy_kwh must lie within "
                "[minimum_energy_kwh, capacity_kwh]."
            )
        return self


# ---------------------------------------------------------------------------
# Response: POST /optimize-energy (official format)
# ---------------------------------------------------------------------------

class DirectiveInterpretation(BaseModel):
    """One LLM-interpreted operator note (official response shape)."""

    note_index: int = Field(
        ge=0,
        description="Index of the operator note in the request (0-based).",
    )
    applies: bool = Field(
        ...,
        description="True if the directive affects the optimizer.",
    )
    directive_type: str = Field(
        ...,
        description=(
            "One of: solar_reduction, minimum_battery_reserve, "
            "no_charge_window, no_discharge_window, max_grid_window, no_op."
        ),
    )
    structured_adjustment: Optional[dict] = Field(
        default=None,
        description="Deterministic parameters for the optimizer, or null.",
    )
    explanation: str = Field(
        default="",
        description="Short free-text rationale.",
    )


class HourlyPlanItem(BaseModel):
    """One hour of the official 24-hour plan."""

    hour: int = Field(ge=0, le=23, description="Hour of the day, 0..23.")
    grid_kwh: float = Field(ge=0, description="Energy drawn from the grid this hour (kWh).")
    solar_used_kwh: float = Field(ge=0, description="Solar energy used this hour (kWh).")
    battery_action: str = Field(
        ...,
        description='"charge" | "discharge" | "idle".',
    )
    battery_kwh: float = Field(
        ge=0,
        description=(
            "Energy moved into (charge) or out of (discharge) the battery this "
            "hour (kWh). MUST be 0 when battery_action == 'idle'."
        ),
    )
    battery_energy_after_kwh: float = Field(
        ge=0,
        description="Battery state-of-charge at the END of this hour (kWh).",
    )

    @field_validator("battery_action")
    @classmethod
    def _battery_action_valid(cls, v: str) -> str:
        if v not in BATTERY_ACTIONS:
            raise ValueError(
                f"battery_action must be one of {sorted(BATTERY_ACTIONS)}, got {v!r}."
            )
        return v


class OptimizeResponse(BaseModel):
    """Body of the POST /optimize-energy response (official shape)."""

    scenario_id: str = Field(..., description="Echo of the request scenario_id.")
    directive_interpretation: List[DirectiveInterpretation] = Field(
        ...,
        description="One entry per operator note, in note_index order.",
    )
    hourly_plan: List[HourlyPlanItem] = Field(
        ...,
        min_length=HOURS_IN_DAY,
        max_length=HOURS_IN_DAY,
        description="24 per-hour plan entries.",
    )
    total_grid_kwh: float = Field(ge=0, description="Sum of grid_kwh across all hours.")
    total_cost_bdt: float = Field(ge=0, description="Sum of grid_kwh * tariff across all hours.")
    peak_grid_kwh: float = Field(ge=0, description="Max grid_kwh over all 24 hours.")
    plan_summary: str = Field(
        default="",
        description="Short prose summary of the schedule.",
    )
