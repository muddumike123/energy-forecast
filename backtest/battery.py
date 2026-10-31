"""Daily battery schedule optimisation as a small MILP."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from config import (
    BATTERY_CYCLES_PER_DAY,
    BATTERY_ENERGY_MWH,
    BATTERY_POWER_MW,
    BATTERY_ROUND_TRIP_EFFICIENCY,
)


@dataclass(frozen=True)
class Battery:
    power_mw: float = BATTERY_POWER_MW
    energy_mwh: float = BATTERY_ENERGY_MWH
    round_trip_efficiency: float = BATTERY_ROUND_TRIP_EFFICIENCY
    cycles_per_day: float = BATTERY_CYCLES_PER_DAY


def optimise_day(prices: np.ndarray, battery: Battery) -> tuple[np.ndarray, np.ndarray]:
    """Charge/discharge schedule (grid-side MW per hour) maximising sum(price * (discharge - charge)).

    - Losses are split evenly: sqrt(efficiency) on the way in and on the way out.
    - The battery starts and ends the day empty (no overnight carry-over).
    - Stored energy discharged per day is capped at cycles_per_day * energy_mwh.
    - A binary per hour prevents charging and discharging at the same time.
    Works for 23/24/25-hour days.
    """
    p = np.asarray(prices, dtype=float)
    hours = len(p)
    eta = np.sqrt(battery.round_trip_efficiency)
    power, energy = battery.power_mw, battery.energy_mwh

    # Variable layout: [charge, discharge, state_of_charge, is_charging], each `hours` long.
    charge, discharge, soc, is_charging = (slice(i * hours, (i + 1) * hours) for i in range(4))
    n = 4 * hours
    rows, lower, upper = [], [], []

    def constrain(row: np.ndarray, lo: float, hi: float) -> None:
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for t in range(hours):
        row = np.zeros(n)  # soc[t] = soc[t-1] + eta * charge[t] - discharge[t] / eta
        row[soc.start + t] = 1
        if t > 0:
            row[soc.start + t - 1] = -1
        row[charge.start + t] = -eta
        row[discharge.start + t] = 1 / eta
        constrain(row, 0, 0)

        row = np.zeros(n)  # charge[t] <= power * is_charging[t]
        row[charge.start + t] = 1
        row[is_charging.start + t] = -power
        constrain(row, -np.inf, 0)

        row = np.zeros(n)  # discharge[t] <= power * (1 - is_charging[t])
        row[discharge.start + t] = 1
        row[is_charging.start + t] = power
        constrain(row, -np.inf, power)

    row = np.zeros(n)  # daily cycle limit on stored energy discharged
    row[discharge] = 1 / eta
    constrain(row, 0, battery.cycles_per_day * energy)

    row = np.zeros(n)  # empty at the end of the day
    row[soc.start + hours - 1] = 1
    constrain(row, 0, 0)

    cost = np.zeros(n)
    cost[charge] = p
    cost[discharge] = -p
    upper_bounds = np.concatenate([np.full(hours, power), np.full(hours, power), np.full(hours, energy), np.ones(hours)])
    integrality = np.zeros(n)
    integrality[is_charging] = 1

    result = milp(
        cost,
        constraints=LinearConstraint(np.array(rows), lower, upper),
        bounds=Bounds(np.zeros(n), upper_bounds),
        integrality=integrality,
    )
    if not result.success:
        raise RuntimeError(f"Battery optimisation failed: {result.message}")
    x = np.round(result.x, 6)
    return x[charge], x[discharge]
