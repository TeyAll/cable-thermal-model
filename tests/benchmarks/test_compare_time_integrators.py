from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.compare_time_integrators import (
    calculate_accuracy_kpis,
    create_comparison_figures,
    run_comparison,
    summarize_timings,
)
from cable_thermal_model.model.schemas.run_options import SolutionMethod

SIGNAL = ("circuit", "single", "Conductor")


class FakeModel:
    def run(self, scenario, run_options):
        method = SolutionMethod(run_options["solution_method"])
        elapsed_seconds = (scenario.index - scenario.index[0]).total_seconds().to_numpy()
        timestep_seconds = (scenario.index[1] - scenario.index[0]).total_seconds()
        method_factor = 0.5 if method is SolutionMethod.CrankNicolson else 1.0
        values = 20.0 + elapsed_seconds / 3600 + method_factor * timestep_seconds / 3600
        result = pd.DataFrame(values, index=scenario.index, columns=pd.MultiIndex.from_tuples([SIGNAL]))
        return SimpleNamespace(result=result)


def make_scenario(timestep_seconds: float) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=int(3600 / timestep_seconds) + 1, freq=f"{timestep_seconds}s")
    return pd.DataFrame({"load": 1.0}, index=index)


def test_calculate_accuracy_kpis_aligns_fine_reference():
    reference = pd.Series([0.0, 1.0, 2.0], index=pd.date_range("2026-01-01", periods=3, freq="30min"))
    solution = pd.Series([0.0, 3.0], index=reference.index[[0, 2]])

    kpis = calculate_accuracy_kpis(solution, reference)

    assert kpis["maximum_absolute_error"] == 1.0
    assert kpis["mean_absolute_error"] == 0.5
    assert kpis["rmse"] == pytest.approx(np.sqrt(0.5))


def test_run_comparison_returns_outputs_kpis_and_timing_summary():
    comparison = run_comparison(
        model_factory=FakeModel,
        scenarios={1800.0: make_scenario(1800.0), 3600.0: make_scenario(3600.0)},
        reference_scenario=make_scenario(900.0),
        signal=SIGNAL,
        repetitions=2,
        warmup=False,
    )

    assert len(comparison.solutions) == 4
    assert len(comparison.timings) == 8
    assert len(comparison.kpis) == 4
    assert set(comparison.timing_summary["method"]) == {method.value for method in SolutionMethod}
    assert comparison.reference_timestep_seconds == 900.0
    assert comparison.timing_summary["standard_deviation_seconds"].notna().all()
    assert np.all(comparison.timing_summary["standard_deviation_seconds"] >= 0.0)

    figures = create_comparison_figures(comparison, timestep_seconds=3600.0)
    assert set(figures) == {
        "temperature_trajectories",
        "method_difference",
        "accuracy_vs_timestep",
        "runtime_vs_timestep",
        "accuracy_runtime_tradeoff",
        "stability_overshoot",
    }
    for figure in figures.values():
        figure.clear()


def test_summarize_timings_requires_complete_records():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_timings(pd.DataFrame({"method": ["BackwardEuler"]}))