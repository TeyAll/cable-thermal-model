"""Accuracy and performance comparison for soil-model time integrators."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

import numpy as np
import pandas as pd

from cable_thermal_model.model.schemas.run_options import SolutionMethod

TemperatureColumn = tuple[Any, Any, Any]
RunKey = tuple[SolutionMethod, float]


class ModelRunner(Protocol):
    """Structural interface required by the comparison harness."""

    def run(self, scenario: pd.DataFrame, run_options: Mapping[str, Any]) -> Any:
        """Run a scenario and return an object with a temperature ``result``."""
        ...


@dataclass(frozen=True)
class ComparisonResult:
    """Raw model outputs and derived measurements from one comparison."""

    timings: pd.DataFrame
    timing_summary: pd.DataFrame
    kpis: pd.DataFrame
    solutions: Mapping[RunKey, pd.DataFrame]
    reference: pd.DataFrame
    signal: TemperatureColumn
    reference_method: SolutionMethod
    reference_timestep_seconds: float


def run_comparison(
    model_factory: Callable[[], ModelRunner],
    scenarios: Mapping[float, pd.DataFrame],
    reference_scenario: pd.DataFrame,
    signal: TemperatureColumn,
    *,
    repetitions: int = 5,
    warmup: bool = True,
    methods: Sequence[SolutionMethod] = tuple(SolutionMethod),
    reference_method: SolutionMethod = SolutionMethod.BackwardEuler,
    run_options: Mapping[str, Any] | None = None,
) -> ComparisonResult:
    """Run accuracy and timing comparisons across methods and timestep sizes.

    ``scenarios`` is keyed by timestep in seconds. Supplying scenarios explicitly avoids
    hidden interpolation assumptions for discontinuous loads and ambient conditions.
    The reference scenario must span the same interval at a finer timestep.
    """
    _validate_comparison_inputs(scenarios, reference_scenario, repetitions, methods)

    shared_run_options = dict(run_options or {})
    reference_output = model_factory().run(
        reference_scenario,
        run_options={**shared_run_options, "solution_method": reference_method},
    ).result
    _validate_signal(reference_output, signal)

    timings: list[dict[str, float | int | str]] = []
    solutions: dict[RunKey, pd.DataFrame] = {}
    kpis: list[dict[str, float | str]] = []

    for timestep_seconds, scenario in sorted(scenarios.items()):
        for method in methods:
            method_run_options = {**shared_run_options, "solution_method": method}
            if warmup:
                model_factory().run(scenario, run_options=method_run_options)

            output: pd.DataFrame | None = None
            for repetition in range(repetitions):
                model = model_factory()
                start = perf_counter()
                output = model.run(scenario, run_options=method_run_options).result
                elapsed_seconds = perf_counter() - start
                timings.append(
                    {
                        "method": method.value,
                        "timestep_seconds": timestep_seconds,
                        "repetition": repetition,
                        "elapsed_seconds": elapsed_seconds,
                        "steps": len(scenario) - 1,
                    }
                )

            if output is None:
                raise RuntimeError("No model output was produced")
            _validate_signal(output, signal)
            solutions[(method, timestep_seconds)] = output
            kpis.append(
                {
                    "method": method.value,
                    "timestep_seconds": timestep_seconds,
                    **calculate_accuracy_kpis(output[signal], reference_output[signal]),
                }
            )

    timing_records = pd.DataFrame.from_records(timings)
    return ComparisonResult(
        timings=timing_records,
        timing_summary=summarize_timings(timing_records),
        kpis=pd.DataFrame.from_records(kpis),
        solutions=solutions,
        reference=reference_output,
        signal=signal,
        reference_method=reference_method,
        reference_timestep_seconds=_infer_timestep_seconds(reference_scenario),
    )


def calculate_accuracy_kpis(solution: pd.Series, reference: pd.Series) -> dict[str, float]:
    """Calculate temperature-error KPIs after aligning the reference in time."""
    aligned_reference = _reference_at_solution_times(reference, solution.index)
    error = solution.to_numpy(dtype=float) - aligned_reference.to_numpy(dtype=float)
    absolute_error = np.abs(error)
    peak_index = int(np.argmax(solution.to_numpy(dtype=float)))
    reference_peak_index = int(np.argmax(aligned_reference.to_numpy(dtype=float)))

    return {
        "maximum_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "final_error": float(error[-1]),
        "maximum_overshoot": float(max(np.max(error), 0.0)),
        "maximum_undershoot": float(min(np.min(error), 0.0)),
        "peak_temperature_error": float(solution.iloc[peak_index] - aligned_reference.iloc[reference_peak_index]),
        "peak_time_error_seconds": _index_distance_seconds(
            solution.index[peak_index], aligned_reference.index[reference_peak_index]
        ),
        "maximum_step_change": (
            float(np.max(np.abs(np.diff(solution.to_numpy(dtype=float))))) if len(solution) > 1 else 0.0
        ),
    }


def summarize_timings(records: pd.DataFrame) -> pd.DataFrame:
    """Return robust runtime statistics grouped by method and timestep."""
    required_columns = {"method", "timestep_seconds", "elapsed_seconds", "steps"}
    missing_columns = required_columns.difference(records.columns)
    if missing_columns:
        raise ValueError(f"Benchmark records are missing columns: {sorted(missing_columns)}")

    summary = (
        records.groupby(["method", "timestep_seconds"], as_index=False)
        .agg(
            median_seconds=("elapsed_seconds", "median"),
            minimum_seconds=("elapsed_seconds", "min"),
            maximum_seconds=("elapsed_seconds", "max"),
            standard_deviation_seconds=("elapsed_seconds", "std"),
            steps=("steps", "first"),
        )
        .sort_values(["timestep_seconds", "method"])
    )
    summary["seconds_per_step"] = summary["median_seconds"] / summary["steps"].clip(lower=1)
    summary["standard_deviation_seconds"] = summary["standard_deviation_seconds"].fillna(0.0)
    return summary


def plot_temperature_trajectories(result: ComparisonResult, timestep_seconds: float):
    """Plot both methods and the reference temperature trajectory."""
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(result.reference.index, result.reference[result.signal], color="black", linewidth=1.5, label="Reference")
    for method in SolutionMethod:
        solution = result.solutions.get((method, timestep_seconds))
        if solution is not None:
            axis.plot(solution.index, solution[result.signal], marker=".", label=method.value)
    _style_time_axis(axis, "Temperature trajectories", "Temperature [°C]")
    return figure


def plot_method_difference(result: ComparisonResult, timestep_seconds: float):
    """Plot Crank-Nicolson minus Backward Euler over time."""
    plt = _get_pyplot()
    backward_euler = result.solutions[(SolutionMethod.BackwardEuler, timestep_seconds)][result.signal]
    crank_nicolson = result.solutions[(SolutionMethod.CrankNicolson, timestep_seconds)][result.signal]
    difference = crank_nicolson - backward_euler
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(difference.index, difference, color="#b33c2f")
    _style_time_axis(axis, "Crank-Nicolson minus Backward Euler", "Temperature difference [°C]")
    return figure


def plot_accuracy_vs_timestep(result: ComparisonResult):
    """Plot RMSE and maximum absolute error against timestep size."""
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, records in result.kpis.groupby("method"):
        records = records.sort_values("timestep_seconds")
        axis.loglog(records["timestep_seconds"], records["rmse"], marker="o", label=f"{method} RMSE")
        axis.loglog(
            records["timestep_seconds"],
            records["maximum_absolute_error"],
            marker="s",
            linestyle="--",
            label=f"{method} maximum",
        )
    _style_numeric_axis(axis, "Accuracy versus timestep", "Timestep [s]", "Temperature error [°C]")
    return figure


def plot_runtime_vs_timestep(result: ComparisonResult):
    """Plot median end-to-end runtime against timestep size."""
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, records in result.timing_summary.groupby("method"):
        records = records.sort_values("timestep_seconds")
        axis.loglog(records["timestep_seconds"], records["median_seconds"], marker="o", label=method)
    _style_numeric_axis(axis, "Runtime versus timestep", "Timestep [s]", "Median runtime [s]")
    return figure


def plot_accuracy_runtime_tradeoff(result: ComparisonResult):
    """Plot accuracy against runtime, annotated by timestep."""
    plt = _get_pyplot()
    records = result.kpis.merge(result.timing_summary, on=["method", "timestep_seconds"])
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, method_records in records.groupby("method"):
        axis.loglog(method_records["median_seconds"], method_records["rmse"], marker="o", label=method)
        for record in method_records.itertuples():
            axis.annotate(f"{record.timestep_seconds:g}s", (record.median_seconds, record.rmse))
    _style_numeric_axis(axis, "Accuracy-runtime trade-off", "Median runtime [s]", "RMSE [°C]")
    return figure


def plot_stability_window(
    result: ComparisonResult,
    timestep_seconds: float,
    start: Any | None = None,
    end: Any | None = None,
):
    """Plot a detailed transient window to inspect oscillation and overshoot."""
    plt = _get_pyplot()
    figure, axis = plt.subplots(figsize=(10, 5))
    for method in SolutionMethod:
        solution = result.solutions.get((method, timestep_seconds))
        if solution is not None:
            signal = solution[result.signal].loc[start:end]
            axis.plot(signal.index, signal, marker="o", label=method.value)
    reference = result.reference[result.signal].loc[start:end]
    axis.plot(reference.index, reference, color="black", linewidth=1.5, label="Reference")
    _style_time_axis(axis, "Transient stability and overshoot", "Temperature [°C]")
    return figure


def create_comparison_figures(
    result: ComparisonResult,
    timestep_seconds: float,
    *,
    stability_start: Any | None = None,
    stability_end: Any | None = None,
) -> dict[str, Any]:
    """Create all standard comparison figures."""
    return {
        "temperature_trajectories": plot_temperature_trajectories(result, timestep_seconds),
        "method_difference": plot_method_difference(result, timestep_seconds),
        "accuracy_vs_timestep": plot_accuracy_vs_timestep(result),
        "runtime_vs_timestep": plot_runtime_vs_timestep(result),
        "accuracy_runtime_tradeoff": plot_accuracy_runtime_tradeoff(result),
        "stability_overshoot": plot_stability_window(
            result, timestep_seconds, start=stability_start, end=stability_end
        ),
    }


def _validate_comparison_inputs(
    scenarios: Mapping[float, pd.DataFrame],
    reference_scenario: pd.DataFrame,
    repetitions: int,
    methods: Sequence[SolutionMethod],
) -> None:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if not scenarios:
        raise ValueError("scenarios must contain at least one timestep")
    if not methods:
        raise ValueError("methods must contain at least one solution method")
    for timestep_seconds, scenario in scenarios.items():
        if timestep_seconds <= 0:
            raise ValueError("scenario timestep keys must be positive")
        inferred_timestep = _infer_timestep_seconds(scenario)
        if not np.isclose(timestep_seconds, inferred_timestep):
            raise ValueError(
                f"Scenario key {timestep_seconds:g}s does not match its inferred timestep {inferred_timestep:g}s"
            )
        if scenario.index[0] != reference_scenario.index[0] or scenario.index[-1] != reference_scenario.index[-1]:
            raise ValueError("All scenarios must span the same interval as the reference scenario")


def _validate_signal(result: pd.DataFrame, signal: TemperatureColumn) -> None:
    if signal not in result.columns:
        raise KeyError(f"Temperature signal {signal!r} is not present in the model output")


def _infer_timestep_seconds(scenario: pd.DataFrame) -> float:
    if len(scenario.index) < 2:
        raise ValueError("Scenarios must contain at least two timestamps")
    differences = pd.Series(scenario.index[1:] - scenario.index[:-1])
    if differences.nunique() != 1:
        raise ValueError("Scenarios must use a constant timestep")
    return float(differences.iloc[0].total_seconds())


def _reference_at_solution_times(reference: pd.Series, solution_index: pd.Index) -> pd.Series:
    missing_timestamps = solution_index.difference(reference.index)
    if not missing_timestamps.empty:
        raise ValueError("The reference scenario must contain every compared solution timestamp")
    return reference.loc[solution_index]


def _index_distance_seconds(left: Any, right: Any) -> float:
    difference = left - right
    return float(difference.total_seconds())


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Plotting benchmark results requires the project's docs dependency group") from error
    return plt


def _style_time_axis(axis: Any, title: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("Time")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    axis.figure.autofmt_xdate()


def _style_numeric_axis(axis: Any, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25, which="both")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)