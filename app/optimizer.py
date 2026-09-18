"""
GridWise Energy Optimizer - Deterministic 24-hour optimizer.

This module consumes the guardrail-validated DirectiveInterpretations and
computes a feasible hourly plan that minimizes total grid electricity cost
over 24 hours.

DESIGN PRINCIPLES
-----------------
* The optimizer NEVER calls the LLM and NEVER does phrase matching.
* All inputs (scenario + directives) are assumed to have already passed
  Pydantic + guardrails validation.
* The algorithm is deterministic, lightweight, and uses only stdlib.
* It produces a plan that is guaranteed to satisfy the energy balance,
  battery SoC continuity (final == initial), and every directive
  constraint. Final state is verified by `validate_plan` before return.

TIME-WINDOW CONVENTION
----------------------
Per the hackathon spec, windows are start-inclusive and end-exclusive:
    start=13, end=15  =>  affects hours 13 and 14.
This is used consistently for no_charge_window, no_discharge_window,
and max_grid_window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from app.schemas import (
    DirectiveInterpretation,
    HourlyPlanItem,
    OptimizeRequest,
    OptimizeResponse,
)

log = logging.getLogger("gridwise.optimizer")

HOURS_IN_DAY = 24
NUMERIC_TOL = 1e-2  # 0.01 kWh / BDT tolerance for final validation


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OptimizerError(RuntimeError):
    """Base class for all optimizer failures."""


class InfeasiblePlanError(OptimizerError):
    """No feasible 24-hour plan satisfies the constraints."""


# ---------------------------------------------------------------------------
# Compiled-directive view (easier for the optimizer to consume)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CompiledDirectives:
    solar_factors: tuple[float, ...]  # length 24, one factor per hour
    min_battery_reserve_kwh: float    # max of all minimum_battery_reserve directives
    no_charge_hours: frozenset[int]
    no_discharge_hours: frozenset[int]
    max_grid_hours: dict[int, float]  # hour -> max grid kWh


def _compile_directives(
    directives: Iterable[DirectiveInterpretation],
) -> _CompiledDirectives:
    solar_factors = [1.0] * HOURS_IN_DAY
    min_reserve: float = 0.0
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    max_grid: dict[int, float] = {}

    for d in directives:
        if not d.applies:
            continue
        adj = d.structured_adjustment or {}

        if d.directive_type == "solar_reduction":
            f = float(adj.get("factor", 1.0))
            f = max(0.0, min(1.0, f))
            for h in _hours_iter(adj):
                solar_factors[h] = min(solar_factors[h], f)

        elif d.directive_type == "minimum_battery_reserve":
            v = float(adj.get("minimum_energy_kwh", 0.0))
            if v > min_reserve:
                min_reserve = v

        elif d.directive_type == "no_charge_window":
            for h in _hours_iter(adj):
                no_charge.add(h)

        elif d.directive_type == "no_discharge_window":
            for h in _hours_iter(adj):
                no_discharge.add(h)

        elif d.directive_type == "max_grid_window":
            cap = float(adj.get("max_grid_kwh", 0.0))
            for h in _hours_iter(adj):
                # Strictest cap wins.
                prev = max_grid.get(h)
                if prev is None or cap < prev:
                    max_grid[h] = cap

        elif d.directive_type == "no_op":
            continue

    return _CompiledDirectives(
        solar_factors=tuple(solar_factors),
        min_battery_reserve_kwh=min_reserve,
        no_charge_hours=frozenset(no_charge),
        no_discharge_hours=frozenset(no_discharge),
        max_grid_hours=max_grid,
    )


def _hours_iter(adj: dict) -> Iterable[int]:
    """Return the official 'hours' list from a validated adjustment dict."""
    raw = adj.get("hours") if isinstance(adj, dict) else None
    if isinstance(raw, list):
        return [int(h) for h in raw]
    return []


# ---------------------------------------------------------------------------
# Per-hour mutable plan state
# ---------------------------------------------------------------------------

@dataclass
class _HourState:
    hour: int
    grid_kwh: float = 0.0
    solar_used_kwh: float = 0.0
    charge_kwh: float = 0.0
    discharge_kwh: float = 0.0
    energy_after_kwh: float = 0.0


# ---------------------------------------------------------------------------
# Optimization core
# ---------------------------------------------------------------------------

def optimize(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
) -> OptimizeResponse:
    """
    Compute the 24-hour energy plan.

    Raises:
        InfeasiblePlanError: if no plan can satisfy all hard constraints.
    """
    batt = request.battery

    # 0. Basic structural sanity (guardrails already enforce this).
    if batt.minimum_energy_kwh > batt.capacity_kwh + NUMERIC_TOL:
        raise InfeasiblePlanError(
            f"battery.minimum_energy_kwh ({batt.minimum_energy_kwh}) "
            f"> capacity_kwh ({batt.capacity_kwh})."
        )
    if not (
        batt.minimum_energy_kwh - NUMERIC_TOL
        <= batt.initial_energy_kwh
        <= batt.capacity_kwh + NUMERIC_TOL
    ):
        raise InfeasiblePlanError(
            "battery.initial_energy_kwh must lie within "
            "[minimum_energy_kwh, capacity_kwh]."
        )

    compiled = _compile_directives(directives)

    demand = [float(h.demand_kwh) for h in request.hours]
    solar_avail = [float(h.solar_kwh) for h in request.hours]
    tariff = [float(h.tariff_bdt_per_kwh) for h in request.hours]
    eff_solar = [
        max(0.0, solar_avail[h] * compiled.solar_factors[h])
        for h in range(HOURS_IN_DAY)
    ]

    eff_min_kwh = max(batt.minimum_energy_kwh, compiled.min_battery_reserve_kwh)
    if eff_min_kwh > batt.capacity_kwh + NUMERIC_TOL:
        raise InfeasiblePlanError(
            "minimum_battery_reserve directive exceeds battery capacity."
        )

    H = [_HourState(hour=h) for h in range(HOURS_IN_DAY)]

    # 1. Free solar first (subject to effective solar availability).
    for h in range(HOURS_IN_DAY):
        usable_solar = min(eff_solar[h], demand[h])
        H[h].solar_used_kwh = usable_solar
        demand[h] -= usable_solar
        eff_solar[h] -= usable_solar

    # 2. Bank leftover solar into the battery (no_charge_window + capacity).
    soc = float(batt.initial_energy_kwh)
    for h in range(HOURS_IN_DAY):
        spare = eff_solar[h]
        if spare > NUMERIC_TOL and h not in compiled.no_charge_hours:
            room = batt.capacity_kwh - soc
            charge = min(spare, batt.max_charge_kwh_per_hour, room)
            if charge > 0:
                H[h].charge_kwh = charge
                soc += charge
        H[h].energy_after_kwh = soc

    # 3. Greedy discharge in expensive hours, respecting all constraints.
    total_charge_planned = sum(h.charge_kwh for h in H)
    total_discharge_budget = max(
        0.0,
        batt.initial_energy_kwh + total_charge_planned - eff_min_kwh,
    )

    expensive_order = sorted(range(HOURS_IN_DAY), key=lambda h: -tariff[h])
    for h in expensive_order:
        if demand[h] <= NUMERIC_TOL:
            continue
        if h in compiled.no_discharge_hours:
            continue
        room = min(
            batt.max_discharge_kwh_per_hour,
            max(0.0, soc - eff_min_kwh),
            total_discharge_budget,
        )
        if room <= NUMERIC_TOL:
            continue
        give = min(demand[h], room)
        if give <= NUMERIC_TOL:
            continue
        H[h].discharge_kwh = give
        soc -= give
        demand[h] -= give
        total_discharge_budget -= give

    # 4. Cover remaining residual from grid; enforce max_grid_window.
    for h in range(HOURS_IN_DAY):
        if demand[h] <= NUMERIC_TOL:
            continue
        cap = compiled.max_grid_hours.get(h)
        if cap is not None and demand[h] > cap + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"max_grid_window forces grid_kwh[{h}] <= {cap:.4f} but "
                f"demand[{h}] requires {demand[h]:.4f} kWh from grid."
            )
        H[h].grid_kwh = demand[h]

    # 5. SoC continuity correction (final SoC must equal initial_energy_kwh).
    soc = float(batt.initial_energy_kwh)
    for h in range(HOURS_IN_DAY):
        soc += H[h].charge_kwh - H[h].discharge_kwh

    soc_error = batt.initial_energy_kwh - soc  # >0 means we need to charge more
    if abs(soc_error) > NUMERIC_TOL:
        if soc_error > 0:
            order = sorted(range(HOURS_IN_DAY), key=lambda h: tariff[h])
            for h in order:
                if soc_error <= NUMERIC_TOL:
                    break
                if h in compiled.no_charge_hours:
                    continue
                running_soc = float(batt.initial_energy_kwh) + sum(
                    H[k].charge_kwh - H[k].discharge_kwh for k in range(h)
                )
                room = min(
                    batt.max_charge_kwh_per_hour - H[h].charge_kwh,
                    batt.capacity_kwh - running_soc,
                )
                if room <= NUMERIC_TOL:
                    continue
                add = min(room, soc_error)
                H[h].charge_kwh += add
                H[h].grid_kwh += add  # correction charge comes from the grid
                soc_error -= add
        else:
            need = -soc_error
            order = sorted(range(HOURS_IN_DAY), key=lambda h: -tariff[h])
            for h in order:
                if need <= NUMERIC_TOL:
                    break
                if h in compiled.no_discharge_hours:
                    continue
                running_soc = float(batt.initial_energy_kwh) + sum(
                    H[k].charge_kwh - H[k].discharge_kwh for k in range(h + 1)
                )
                room = min(
                    batt.max_discharge_kwh_per_hour - H[h].discharge_kwh,
                    max(0.0, running_soc - eff_min_kwh),
                )
                if room <= NUMERIC_TOL:
                    continue
                add = min(room, need)
                H[h].discharge_kwh += add
                H[h].grid_kwh = max(0.0, H[h].grid_kwh - add)
                need -= add

    # 6. Re-enforce max_grid_window after corrections.
    for h in range(HOURS_IN_DAY):
        cap = compiled.max_grid_hours.get(h)
        if cap is not None and H[h].grid_kwh > cap + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"max_grid_window forces grid_kwh[{h}] <= {cap:.4f} "
                f"but plan requires {H[h].grid_kwh:.4f} kWh."
            )

    # 7. Recompute SoC trajectory as the source of truth.
    soc = float(batt.initial_energy_kwh)
    for h in range(HOURS_IN_DAY):
        soc += H[h].charge_kwh - H[h].discharge_kwh
        H[h].energy_after_kwh = soc

    # 8. Build response in the official shape.
    plan: list[HourlyPlanItem] = []
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0
    for h in range(HOURS_IN_DAY):
        if H[h].charge_kwh > NUMERIC_TOL:
            action = "charge"
            battery_kwh = round(H[h].charge_kwh, 6)
        elif H[h].discharge_kwh > NUMERIC_TOL:
            action = "discharge"
            battery_kwh = round(H[h].discharge_kwh, 6)
        else:
            action = "idle"
            battery_kwh = 0.0

        grid = H[h].grid_kwh
        solar_used = H[h].solar_used_kwh
        energy_after = H[h].energy_after_kwh

        plan.append(
            HourlyPlanItem(
                hour=h,
                grid_kwh=round(grid, 6),
                solar_used_kwh=round(solar_used, 6),
                battery_action=action,
                battery_kwh=battery_kwh,
                battery_energy_after_kwh=round(energy_after, 6),
            )
        )
        total_grid += grid
        total_cost += grid * tariff[h]
        if grid > peak_grid:
            peak_grid = grid

    plan_summary = _build_plan_summary(
        request=request,
        plan=plan,
        total_grid=total_grid,
        total_cost=total_cost,
    )

    response = OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=directives,
        hourly_plan=plan,
        total_grid_kwh=round(total_grid, 6),
        total_cost_bdt=round(total_cost, 6),
        peak_grid_kwh=round(peak_grid, 6),
        plan_summary=plan_summary,
    )

    # 9. Hard self-validation.
    validate_plan(response, request, compiled)
    return response


def _build_plan_summary(
    *,
    request: OptimizeRequest,
    plan: list[HourlyPlanItem],
    total_grid: float,
    total_cost: float,
) -> str:
    """Compose a short prose summary of the 24-hour plan."""
    batt = request.battery
    n_charge = sum(1 for p in plan if p.battery_action == "charge")
    n_discharge = sum(1 for p in plan if p.battery_action == "discharge")
    return (
        f"Scenario {request.scenario_id}: 24-hour plan drawn "
        f"{total_grid:.2f} kWh from the grid at a total cost of "
        f"{total_cost:.2f} BDT; battery charged in {n_charge} hour(s) "
        f"and discharged in {n_discharge} hour(s); SoC returned to "
        f"{batt.initial_energy_kwh:.2f} kWh."
    )


# ---------------------------------------------------------------------------
# Internal self-validation
# ---------------------------------------------------------------------------

def validate_plan(
    response: OptimizeResponse,
    request: OptimizeRequest,
    compiled: _CompiledDirectives | None = None,
) -> None:
    """
    Hard-validate the produced plan. Raises InfeasiblePlanError on violation.
    """
    plan = response.hourly_plan
    batt = request.battery
    if compiled is None:
        compiled = _compile_directives(response.directive_interpretation)

    if len(plan) != HOURS_IN_DAY:
        raise InfeasiblePlanError(
            f"hourly_plan must have 24 entries, got {len(plan)}."
        )
    for idx, item in enumerate(plan):
        if item.hour != idx:
            raise InfeasiblePlanError(
                f"hourly_plan[{idx}].hour must be {idx}, got {item.hour}."
            )

    solar_avail = [float(h.solar_kwh) for h in request.hours]
    eff_solar = [
        max(0.0, solar_avail[h] * compiled.solar_factors[h])
        for h in range(HOURS_IN_DAY)
    ]
    eff_min_kwh = max(batt.minimum_energy_kwh, compiled.min_battery_reserve_kwh)

    soc = batt.initial_energy_kwh
    total_grid_check = 0.0
    total_cost_check = 0.0
    peak_grid_check = 0.0
    for h in range(HOURS_IN_DAY):
        item = plan[h]

        if item.grid_kwh < -NUMERIC_TOL:
            raise InfeasiblePlanError(f"hour {h}: grid_kwh < 0.")
        if item.solar_used_kwh < -NUMERIC_TOL:
            raise InfeasiblePlanError(f"hour {h}: solar_used_kwh < 0.")
        if item.battery_kwh < -NUMERIC_TOL:
            raise InfeasiblePlanError(f"hour {h}: battery_kwh < 0.")
        if item.battery_energy_after_kwh < -NUMERIC_TOL:
            raise InfeasiblePlanError(f"hour {h}: battery_energy_after_kwh < 0.")

        if item.solar_used_kwh > eff_solar[h] + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: solar_used_kwh {item.solar_used_kwh:.4f} exceeds "
                f"effective solar {eff_solar[h]:.4f}."
            )

        charge_kwh = item.battery_kwh if item.battery_action == "charge" else 0.0
        discharge_kwh = item.battery_kwh if item.battery_action == "discharge" else 0.0

        if item.battery_action == "charge" and item.battery_kwh > batt.max_charge_kwh_per_hour + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: battery_kwh {item.battery_kwh:.4f} "
                f"> max_charge_kwh_per_hour {batt.max_charge_kwh_per_hour:.4f}."
            )
        if item.battery_action == "discharge" and item.battery_kwh > batt.max_discharge_kwh_per_hour + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: battery_kwh {item.battery_kwh:.4f} "
                f"> max_discharge_kwh_per_hour {batt.max_discharge_kwh_per_hour:.4f}."
            )

        soc = soc + charge_kwh - discharge_kwh
        if soc < eff_min_kwh - NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: battery_energy_after_kwh {soc:.4f} dropped below "
                f"minimum_energy_kwh {eff_min_kwh:.4f}."
            )
        if soc > batt.capacity_kwh + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: battery_energy_after_kwh {soc:.4f} exceeded "
                f"capacity_kwh {batt.capacity_kwh:.4f}."
            )
        if abs(soc - item.battery_energy_after_kwh) > NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: battery_energy_after_kwh "
                f"{item.battery_energy_after_kwh:.4f} disagrees with SoC "
                f"trajectory {soc:.4f}."
            )

        supply = item.grid_kwh + item.solar_used_kwh + discharge_kwh
        use = float(request.hours[h].demand_kwh) + charge_kwh
        if abs(supply - use) > NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: energy balance violated "
                f"(supply {supply:.4f} != use {use:.4f})."
            )

        if h in compiled.no_charge_hours and charge_kwh > NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: no_charge_window violated "
                f"(charge={charge_kwh:.4f})."
            )
        if h in compiled.no_discharge_hours and discharge_kwh > NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: no_discharge_window violated "
                f"(discharge={discharge_kwh:.4f})."
            )
        cap = compiled.max_grid_hours.get(h)
        if cap is not None and item.grid_kwh > cap + NUMERIC_TOL:
            raise InfeasiblePlanError(
                f"hour {h}: max_grid_window violated "
                f"(grid_kwh={item.grid_kwh:.4f} > cap {cap:.4f})."
            )

        expected_cost = item.grid_kwh * float(request.hours[h].tariff_bdt_per_kwh)

        total_grid_check += item.grid_kwh
        total_cost_check += expected_cost
        if item.grid_kwh > peak_grid_check:
            peak_grid_check = item.grid_kwh

    if abs(soc - batt.initial_energy_kwh) > NUMERIC_TOL:
        raise InfeasiblePlanError(
            f"final battery_energy_after_kwh {soc:.4f} "
            f"!= initial_energy_kwh {batt.initial_energy_kwh:.4f}."
        )
    if abs(total_grid_check - response.total_grid_kwh) > NUMERIC_TOL:
        raise InfeasiblePlanError(
            f"total_grid_kwh {response.total_grid_kwh:.4f} != "
            f"sum(hourly grid_kwh) {total_grid_check:.4f}."
        )
    if abs(total_cost_check - response.total_cost_bdt) > NUMERIC_TOL:
        raise InfeasiblePlanError(
            f"total_cost_bdt {response.total_cost_bdt:.4f} != "
            f"sum(hourly grid_kwh*tariff) {total_cost_check:.4f}."
        )
    if abs(peak_grid_check - response.peak_grid_kwh) > NUMERIC_TOL:
        raise InfeasiblePlanError(
            f"peak_grid_kwh {response.peak_grid_kwh:.4f} != "
            f"max(hourly grid_kwh) {peak_grid_check:.4f}."
        )
