"""
GridWise Energy Optimizer - Deterministic 24-hour optimizer.

Consumes guardrail-validated DirectiveInterpretations and computes
a feasible hourly plan that minimizes total grid electricity cost
over 24 hours.

The optimizer is deterministic and never calls the LLM.
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
NUMERIC_TOL = 1e-2


class OptimizerError(RuntimeError):
    """Base class for optimizer failures."""


class InfeasiblePlanError(OptimizerError):
    """No feasible 24-hour plan satisfies the constraints."""


@dataclass(frozen=True)
class _CompiledDirectives:
    solar_factors: tuple[float, ...]
    min_battery_reserve_kwh: float
    no_charge_hours: frozenset[int]
    no_discharge_hours: frozenset[int]
    max_grid_hours: dict[int, float]


def _hours_iter(adj: dict) -> Iterable[int]:
    raw = adj.get("hours") if isinstance(adj, dict) else None

    if isinstance(raw, list):
        return [int(h) for h in raw]

    return []


def _compile_directives(
    directives: Iterable[DirectiveInterpretation],
) -> _CompiledDirectives:

    solar_factors = [1.0] * HOURS_IN_DAY
    min_reserve = 0.0
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    max_grid: dict[int, float] = {}

    for d in directives:

        if not d.applies:
            continue

        adj = d.structured_adjustment or {}

        if d.directive_type == "solar_reduction":

            factor = float(
                adj.get("factor", 1.0)
            )

            factor = max(
                0.0,
                min(1.0, factor),
            )

            for h in _hours_iter(adj):

                if 0 <= h < HOURS_IN_DAY:
                    solar_factors[h] = min(
                        solar_factors[h],
                        factor,
                    )

        elif d.directive_type == "minimum_battery_reserve":

            value = float(
                adj.get(
                    "minimum_energy_kwh",
                    0.0,
                )
            )

            min_reserve = max(
                min_reserve,
                value,
            )

        elif d.directive_type == "no_charge_window":

            for h in _hours_iter(adj):

                if 0 <= h < HOURS_IN_DAY:
                    no_charge.add(h)

        elif d.directive_type == "no_discharge_window":

            for h in _hours_iter(adj):

                if 0 <= h < HOURS_IN_DAY:
                    no_discharge.add(h)

        elif d.directive_type == "max_grid_window":

            cap = float(
                adj.get(
                    "max_grid_kwh",
                    0.0,
                )
            )

            for h in _hours_iter(adj):

                if 0 <= h < HOURS_IN_DAY:

                    previous = max_grid.get(h)

                    if (
                        previous is None
                        or cap < previous
                    ):
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


@dataclass
class _HourState:

    hour: int

    grid_kwh: float = 0.0

    solar_used_kwh: float = 0.0

    charge_kwh: float = 0.0

    discharge_kwh: float = 0.0

    energy_after_kwh: float = 0.0


def _recompute_soc(
    states: list[_HourState],
    initial_soc: float,
) -> None:

    soc = float(initial_soc)

    for state in states:

        soc += (
            state.charge_kwh
            - state.discharge_kwh
        )

        state.energy_after_kwh = soc


def _soc_before(
    states: list[_HourState],
    initial_soc: float,
    hour: int,
) -> float:

    soc = float(initial_soc)

    for h in range(hour):

        soc += (
            states[h].charge_kwh
            - states[h].discharge_kwh
        )

    return soc


def optimize(
    request: OptimizeRequest,
    directives: list[DirectiveInterpretation],
) -> OptimizeResponse:

    batt = request.battery

    # ------------------------------------------------------------------
    # 0. Basic validation
    # ------------------------------------------------------------------

    if batt.minimum_energy_kwh > (
        batt.capacity_kwh + NUMERIC_TOL
    ):
        raise InfeasiblePlanError(
            "battery.minimum_energy_kwh "
            f"({batt.minimum_energy_kwh}) > "
            f"capacity_kwh ({batt.capacity_kwh})."
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

    if len(request.hours) != HOURS_IN_DAY:
        raise InfeasiblePlanError(
            f"Expected exactly 24 hours, "
            f"got {len(request.hours)}."
        )

    compiled = _compile_directives(
        directives
    )

    demand = [
        float(h.demand_kwh)
        for h in request.hours
    ]

    solar_avail = [
        float(h.solar_kwh)
        for h in request.hours
    ]

    tariff = [
        float(h.tariff_bdt_per_kwh)
        for h in request.hours
    ]

    effective_solar = [

        max(
            0.0,
            solar_avail[h]
            * compiled.solar_factors[h],
        )

        for h in range(HOURS_IN_DAY)
    ]

    minimum_soc = max(
        batt.minimum_energy_kwh,
        compiled.min_battery_reserve_kwh,
    )

    if minimum_soc > (
        batt.capacity_kwh + NUMERIC_TOL
    ):
        raise InfeasiblePlanError(
            "Minimum battery reserve directive "
            "exceeds battery capacity."
        )

    H = [
        _HourState(hour=h)
        for h in range(HOURS_IN_DAY)
    ]

    # ------------------------------------------------------------------
    # 1. Use solar directly for demand.
    # ------------------------------------------------------------------

    residual_demand = demand.copy()
    solar_surplus = [0.0] * HOURS_IN_DAY

    for h in range(HOURS_IN_DAY):

        direct_solar = min(
            effective_solar[h],
            residual_demand[h],
        )

        H[h].solar_used_kwh = direct_solar

        residual_demand[h] -= direct_solar

        solar_surplus[h] = max(
            0.0,
            effective_solar[h]
            - direct_solar,
        )

    # ------------------------------------------------------------------
    # 2. Charge battery from solar surplus.
    # ------------------------------------------------------------------

    soc = float(
        batt.initial_energy_kwh
    )

    for h in range(HOURS_IN_DAY):

        if (
            solar_surplus[h] > NUMERIC_TOL
            and h not in compiled.no_charge_hours
        ):

            room = max(
                0.0,
                batt.capacity_kwh - soc,
            )

            charge = min(
                solar_surplus[h],
                batt.max_charge_kwh_per_hour,
                room,
            )

            if charge > NUMERIC_TOL:

                H[h].charge_kwh = charge

                soc += charge

    # ------------------------------------------------------------------
    # 3. Grid must cover residual demand.
    #
    # Battery discharge is intentionally planned AFTER solar charging.
    # ------------------------------------------------------------------

    for h in range(HOURS_IN_DAY):

        H[h].grid_kwh = residual_demand[h]

    # ------------------------------------------------------------------
    # 4. Discharge battery at expensive hours.
    # ------------------------------------------------------------------

    expensive_order = sorted(
        range(HOURS_IN_DAY),
        key=lambda h: (-tariff[h], h),
    )

    for h in expensive_order:

        if (
            residual_demand[h]
            <= NUMERIC_TOL
        ):
            continue

        if h in compiled.no_discharge_hours:
            continue

        # Never charge and discharge in
        # the same hour.

        if (
            H[h].charge_kwh
            > NUMERIC_TOL
        ):
            continue

        running_soc = _soc_before(
            H,
            batt.initial_energy_kwh,
            h,
        )

        available_soc = max(
            0.0,
            running_soc - minimum_soc,
        )

        discharge = min(
            residual_demand[h],
            batt.max_discharge_kwh_per_hour,
            available_soc,
        )

        if discharge <= NUMERIC_TOL:
            continue

        H[h].discharge_kwh = discharge

        H[h].grid_kwh -= discharge

        residual_demand[h] -= discharge

    # ------------------------------------------------------------------
    # 5. Restore final SoC to initial SoC.
    #
    # If solar charged the battery, discharge the surplus later.
    # If the optimizer discharged too much, charge from grid.
    # ------------------------------------------------------------------

    _recompute_soc(
        H,
        batt.initial_energy_kwh,
    )

    final_soc = H[-1].energy_after_kwh

    soc_error = (
        batt.initial_energy_kwh
        - final_soc
    )

    # Need additional discharge.
    if soc_error < -NUMERIC_TOL:

        remaining = -soc_error

        order = sorted(
            range(HOURS_IN_DAY),
            key=lambda h: (-tariff[h], h),
        )

        for h in order:

            if remaining <= NUMERIC_TOL:
                break

            if h in compiled.no_discharge_hours:
                continue

            if (
                H[h].charge_kwh
                > NUMERIC_TOL
            ):
                continue

            running_soc = _soc_before(
                H,
                batt.initial_energy_kwh,
                h,
            )

            available = max(
                0.0,
                running_soc - minimum_soc,
            )

            rate_room = max(
                0.0,
                batt.max_discharge_kwh_per_hour
                - H[h].discharge_kwh,
            )

            room = min(
                available,
                rate_room,
            )

            if room <= NUMERIC_TOL:
                continue

            add = min(
                room,
                remaining,
            )

            H[h].discharge_kwh += add

            H[h].grid_kwh = max(
                0.0,
                H[h].grid_kwh - add,
            )

            remaining -= add

        if remaining > NUMERIC_TOL:

            raise InfeasiblePlanError(
                "Could not restore battery to its "
                f"initial energy. Remaining discharge "
                f"needed: {remaining:.4f} kWh."
            )

    # Need additional charging.
    elif soc_error > NUMERIC_TOL:

        remaining = soc_error

        order = sorted(
            range(HOURS_IN_DAY),
            key=lambda h: (tariff[h], h),
        )

        for h in order:

            if remaining <= NUMERIC_TOL:
                break

            if h in compiled.no_charge_hours:
                continue

            if (
                H[h].discharge_kwh
                > NUMERIC_TOL
            ):
                continue

            running_soc = _soc_before(
                H,
                batt.initial_energy_kwh,
                h,
            )

            capacity_room = max(
                0.0,
                batt.capacity_kwh
                - running_soc,
            )

            rate_room = max(
                0.0,
                batt.max_charge_kwh_per_hour
                - H[h].charge_kwh,
            )

            room = min(
                capacity_room,
                rate_room,
            )

            if room <= NUMERIC_TOL:
                continue

            add = min(
                room,
                remaining,
            )

            H[h].charge_kwh += add

            # Additional charging comes from grid.
            H[h].grid_kwh += add

            remaining -= add

        if remaining > NUMERIC_TOL:

            raise InfeasiblePlanError(
                "Could not restore battery to its "
                f"initial energy. Remaining charge "
                f"needed: {remaining:.4f} kWh."
            )

    # ------------------------------------------------------------------
    # 6. Recompute SoC.
    # ------------------------------------------------------------------

    _recompute_soc(
        H,
        batt.initial_energy_kwh,
    )

    # ------------------------------------------------------------------
    # 7. Re-enforce grid caps.
    # ------------------------------------------------------------------

    for h in range(HOURS_IN_DAY):

        cap = compiled.max_grid_hours.get(h)

        if (
            cap is not None
            and H[h].grid_kwh
            > cap + NUMERIC_TOL
        ):
            raise InfeasiblePlanError(
                f"max_grid_window forces "
                f"grid_kwh[{h}] <= {cap:.4f}, "
                f"but plan requires "
                f"{H[h].grid_kwh:.4f} kWh."
            )

    # ------------------------------------------------------------------
    # 8. Build response.
    # ------------------------------------------------------------------

    plan: list[HourlyPlanItem] = []

    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in range(HOURS_IN_DAY):

        if (
            H[h].charge_kwh
            > NUMERIC_TOL
        ):

            action = "charge"

            battery_kwh = round(
                H[h].charge_kwh,
                6,
            )

        elif (
            H[h].discharge_kwh
            > NUMERIC_TOL
        ):

            action = "discharge"

            battery_kwh = round(
                H[h].discharge_kwh,
                6,
            )

        else:

            action = "idle"
            battery_kwh = 0.0

        grid = max(
            0.0,
            H[h].grid_kwh,
        )

        solar_used = max(
            0.0,
            H[h].solar_used_kwh,
        )

        energy_after = H[
            h
        ].energy_after_kwh

        plan.append(
            HourlyPlanItem(
                hour=h,

                grid_kwh=round(
                    grid,
                    6,
                ),

                solar_used_kwh=round(
                    solar_used,
                    6,
                ),

                battery_action=action,

                battery_kwh=battery_kwh,

                battery_energy_after_kwh=round(
                    energy_after,
                    6,
                ),
            )
        )

        total_grid += grid

        total_cost += (
            grid * tariff[h]
        )

        peak_grid = max(
            peak_grid,
            grid,
        )

    # ------------------------------------------------------------------
    # 9. Summary.
    # ------------------------------------------------------------------

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

        total_grid_kwh=round(
            total_grid,
            6,
        ),

        total_cost_bdt=round(
            total_cost,
            6,
        ),

        peak_grid_kwh=round(
            peak_grid,
            6,
        ),

        plan_summary=plan_summary,
    )

    # ------------------------------------------------------------------
    # 10. Hard validation.
    # ------------------------------------------------------------------

    validate_plan(
        response,
        request,
        compiled,
    )

    return response


def _build_plan_summary(
    *,
    request: OptimizeRequest,
    plan: list[HourlyPlanItem],
    total_grid: float,
    total_cost: float,
) -> str:

    batt = request.battery

    n_charge = sum(
        1
        for p in plan
        if p.battery_action == "charge"
    )

    n_discharge = sum(
        1
        for p in plan
        if p.battery_action == "discharge"
    )

    return (
        f"Scenario {request.scenario_id}: "
        f"24-hour plan drew "
        f"{total_grid:.2f} kWh from the grid "
        f"at a total cost of "
        f"{total_cost:.2f} BDT; "
        f"battery charged in {n_charge} hour(s) "
        f"and discharged in {n_discharge} hour(s); "
        f"SoC returned to "
        f"{batt.initial_energy_kwh:.2f} kWh."
    )


def validate_plan(
    response: OptimizeResponse,
    request: OptimizeRequest,
    compiled: _CompiledDirectives | None = None,
) -> None:

    plan = response.hourly_plan

    batt = request.battery

    if compiled is None:

        compiled = _compile_directives(
            response.directive_interpretation
        )

    if len(plan) != HOURS_IN_DAY:

        raise InfeasiblePlanError(
            f"hourly_plan must have 24 entries, "
            f"got {len(plan)}."
        )

    for idx, item in enumerate(plan):

        if item.hour != idx:

            raise InfeasiblePlanError(
                f"hourly_plan[{idx}].hour "
                f"must be {idx}, "
                f"got {item.hour}."
            )

    effective_solar = [

        max(
            0.0,
            float(
                request.hours[h].solar_kwh
            )
            * compiled.solar_factors[h],
        )

        for h in range(HOURS_IN_DAY)
    ]

    minimum_soc = max(
        batt.minimum_energy_kwh,
        compiled.min_battery_reserve_kwh,
    )

    soc = float(
        batt.initial_energy_kwh
    )

    total_grid_check = 0.0

    total_cost_check = 0.0

    peak_grid_check = 0.0

    for h in range(HOURS_IN_DAY):

        item = plan[h]

        if item.grid_kwh < -NUMERIC_TOL:

            raise InfeasiblePlanError(
                f"hour {h}: grid_kwh < 0."
            )

        if item.solar_used_kwh < -NUMERIC_TOL:

            raise InfeasiblePlanError(
                f"hour {h}: solar_used_kwh < 0."
            )

        if item.battery_kwh < -NUMERIC_TOL:

            raise InfeasiblePlanError(
                f"hour {h}: battery_kwh < 0."
            )

        if (
            item.solar_used_kwh
            > effective_solar[h]
            + NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: solar_used_kwh "
                f"{item.solar_used_kwh:.4f} exceeds "
                f"effective solar "
                f"{effective_solar[h]:.4f}."
            )

        if item.battery_action not in {
            "idle",
            "charge",
            "discharge",
        }:

            raise InfeasiblePlanError(
                f"hour {h}: invalid "
                f"battery_action "
                f"{item.battery_action!r}."
            )

        charge_kwh = (
            item.battery_kwh
            if item.battery_action
            == "charge"
            else 0.0
        )

        discharge_kwh = (
            item.battery_kwh
            if item.battery_action
            == "discharge"
            else 0.0
        )

        # --------------------------------------------------------------
        # Rate limits
        # --------------------------------------------------------------

        if (
            charge_kwh
            > batt.max_charge_kwh_per_hour
            + NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: charge "
                f"{charge_kwh:.4f} > "
                f"max charge rate "
                f"{batt.max_charge_kwh_per_hour:.4f}."
            )

        if (
            discharge_kwh
            > batt.max_discharge_kwh_per_hour
            + NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: discharge "
                f"{discharge_kwh:.4f} > "
                f"max discharge rate "
                f"{batt.max_discharge_kwh_per_hour:.4f}."
            )

        # --------------------------------------------------------------
        # SoC
        # --------------------------------------------------------------

        soc += (
            charge_kwh
            - discharge_kwh
        )

        if (
            soc
            < minimum_soc - NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: battery SoC "
                f"{soc:.4f} below minimum "
                f"{minimum_soc:.4f}."
            )

        if (
            soc
            > batt.capacity_kwh
            + NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: battery SoC "
                f"{soc:.4f} exceeds capacity "
                f"{batt.capacity_kwh:.4f}."
            )

        if (
            abs(
                soc
                - item.battery_energy_after_kwh
            )
            > NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: battery_energy_after_kwh "
                f"{item.battery_energy_after_kwh:.4f} "
                f"disagrees with SoC trajectory "
                f"{soc:.4f}."
            )

        # --------------------------------------------------------------
        # Energy balance
        #
        # solar + grid + discharge
        # =
        # demand + charge
        #
        # This correctly handles solar surplus charging.
        # --------------------------------------------------------------

        supply = (
            item.grid_kwh
            + item.solar_used_kwh
            + discharge_kwh
        )

        use = (
            float(
                request.hours[h]
                .demand_kwh
            )
            + charge_kwh
        )

        if (
            abs(
                supply - use
            )
            > NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: energy balance "
                f"violated "
                f"(supply {supply:.4f} "
                f"!= use {use:.4f})."
            )

        # --------------------------------------------------------------
        # Directives
        # --------------------------------------------------------------

        if (
            h in compiled.no_charge_hours
            and charge_kwh
            > NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: no_charge_window "
                f"violated."
            )

        if (
            h in compiled.no_discharge_hours
            and discharge_kwh
            > NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: no_discharge_window "
                f"violated."
            )

        cap = compiled.max_grid_hours.get(h)

        if (
            cap is not None
            and item.grid_kwh
            > cap + NUMERIC_TOL
        ):

            raise InfeasiblePlanError(
                f"hour {h}: max_grid_window "
                f"violated "
                f"(grid={item.grid_kwh:.4f}, "
                f"cap={cap:.4f})."
            )

        # --------------------------------------------------------------
        # Totals
        # --------------------------------------------------------------

        tariff = float(
            request.hours[h]
            .tariff_bdt_per_kwh
        )

        total_grid_check += (
            item.grid_kwh
        )

        total_cost_check += (
            item.grid_kwh
            * tariff
        )

        peak_grid_check = max(
            peak_grid_check,
            item.grid_kwh,
        )

    # ------------------------------------------------------------------
    # Final SoC
    # ------------------------------------------------------------------

    if (
        abs(
            soc
            - batt.initial_energy_kwh
        )
        > NUMERIC_TOL
    ):

        raise InfeasiblePlanError(
            f"final battery SoC "
            f"{soc:.4f} != initial "
            f"{batt.initial_energy_kwh:.4f}."
        )

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------

    if (
        abs(
            total_grid_check
            - response.total_grid_kwh
        )
        > NUMERIC_TOL
    ):

        raise InfeasiblePlanError(
            f"total_grid_kwh "
            f"{response.total_grid_kwh:.4f} != "
            f"sum hourly grid "
            f"{total_grid_check:.4f}."
        )

    if (
        abs(
            total_cost_check
            - response.total_cost_bdt
        )
        > NUMERIC_TOL
    ):

        raise InfeasiblePlanError(
            f"total_cost_bdt "
            f"{response.total_cost_bdt:.4f} != "
            f"sum hourly cost "
            f"{total_cost_check:.4f}."
        )

    if (
        abs(
            peak_grid_check
            - response.peak_grid_kwh
        )
        > NUMERIC_TOL
    ):

        raise InfeasiblePlanError(
            f"peak_grid_kwh "
            f"{response.peak_grid_kwh:.4f} != "
            f"max hourly grid "
            f"{peak_grid_check:.4f}."
        )