"""Run and visualize a Backward Euler versus Crank-Nicolson comparison.

Run from the repository root with the documentation dependency group available:

    uv run --group docs python benchmarks/example_compare_time_integrators.py
"""

from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmarks.compare_time_integrators import create_comparison_figures, run_comparison
from cable_thermal_model import CableLayer, CablePosition, CircuitType, StaticEnvSoil
from cable_thermal_model.cable.schemas.circuit_schemas import CircuitInSoilFromCableIdInputSchema
from cable_thermal_model.model.model_soil import ModelSoil

START = pd.Timestamp("2026-01-01")
DURATION = timedelta(days=3)
LOAD_STEP_TIME = START + timedelta(days=1)
CIRCUIT_NAME = "benchmark_circuit"


def build_environment() -> StaticEnvSoil:
    """Create a representative single-circuit soil environment."""
    environment = StaticEnvSoil()
    environment.add_circuit_from_cable_id(
        CircuitInSoilFromCableIdInputSchema(
            x=0.0,
            y=-1.0,
            circuit_name=CIRCUIT_NAME,
            cable_id="YMeKrvaslqwd 12/20kV 1x630 Alrm + as50",
            circuit_type=CircuitType.Trefoil,
        )
    )
    return environment


def build_scenario(timestep_seconds: float, profile: str) -> pd.DataFrame:
    """Create identical continuous forcing sampled at the requested timestep."""
    index = pd.date_range(
        start=START,
        end=START + DURATION,
        freq=pd.to_timedelta(timestep_seconds, unit="s"),
    )
    elapsed_hours = (index - START).total_seconds().to_numpy() / 3600
    if profile == "constant":
        load = np.full(len(index), 400.0)
    elif profile == "cyclic":
        load = 400.0 + 200.0 * np.sin(2 * np.pi * elapsed_hours / 24)
    elif profile == "step":
        load = np.where(index < LOAD_STEP_TIME, 100.0, 600.0)
    else:
        raise ValueError(f"Unknown load profile: {profile}")

    return pd.DataFrame(
        {
            f"load_{CIRCUIT_NAME}": load,
            "ambient_temperature": 15.0 + 2.0 * np.sin(2 * np.pi * elapsed_hours / 24),
            "soil_thermal_resistivity": 0.75,
            "soil_thermal_capacity": 2.0e6,
        },
        index=index,
    )


def main(output_directory: Path, repetitions: int) -> None:
    """Execute the comparison and write machine-readable results and figures."""
    environment = build_environment()
    timesteps = (900.0, 1800.0, 3600.0)
    reference_timestep = 300.0
    output_directory.mkdir(parents=True, exist_ok=True)
    reports = []
    raw_timings = []
    for profile in ("constant", "cyclic", "step"):
        profile_directory = output_directory / profile
        profile_directory.mkdir(exist_ok=True)
        result = run_comparison(
            model_factory=lambda: ModelSoil(environment),
            scenarios={timestep: build_scenario(timestep, profile) for timestep in timesteps},
            reference_scenario=build_scenario(reference_timestep, profile),
            signal=(CIRCUIT_NAME, CablePosition.TrefoilTop, CableLayer.Conductor),
            repetitions=repetitions,
            run_options={"temperature_dependent_electric_resistance": False},
        )

        profile_timings = result.timings.assign(scenario=profile)
        profile_report = result.kpis.merge(result.timing_summary, on=["method", "timestep_seconds"]).assign(
            scenario=profile
        )
        raw_timings.append(profile_timings)
        reports.append(profile_report)

        figures = create_comparison_figures(
            result,
            timestep_seconds=max(timesteps),
            stability_start=LOAD_STEP_TIME - timedelta(hours=4),
            stability_end=LOAD_STEP_TIME + timedelta(hours=12),
        )
        for name, figure in figures.items():
            figure.savefig(profile_directory / f"{name}.png", dpi=160, bbox_inches="tight")
            plt.close(figure)

    report = pd.concat(reports, ignore_index=True)
    pd.concat(raw_timings, ignore_index=True).to_csv(output_directory / "raw_timings.csv", index=False)
    report.to_csv(output_directory / "comparison_summary.csv", index=False)
    print(report.to_string(index=False))
    print(f"\nResults written to {output_directory.resolve()}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/time_integrators"))
    parser.add_argument("--repetitions", type=int, default=3)
    arguments = parser.parse_args()
    main(output_directory=arguments.output, repetitions=arguments.repetitions)