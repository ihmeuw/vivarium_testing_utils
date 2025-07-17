from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Literal

import pandas as pd
from matplotlib.figure import Figure

from vivarium_testing_utils.automated_validation.comparison import Comparison, FuzzyComparison
from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation import (
    measures,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_bundle import (
    RatioMeasureDataBundle,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    CategoricalRelativeRisk,
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.visualization import plot_utils


class ValidationContext:
    def __init__(self, results_dir: str | Path, scenario_columns: Collection[str] = ()):
        self._data_loader = DataLoader(Path(results_dir))
        self.comparisons: dict[str, Comparison] = {}
        self.age_groups = self._get_age_groups()
        self.scenario_columns = scenario_columns

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets available in the given simulation output directory."""
        return self._data_loader.get_sim_outputs()

    def get_artifact_keys(self) -> list[str]:
        """Get a list of the artifact keys available to compare against."""
        return self._data_loader.get_artifact_keys()

    def get_raw_data(self, data_key: str, source: str) -> Any:
        """Return a copy of the data for manual inspection."""
        return self._data_loader.get_data(data_key, DataSource.from_str(source))

    def upload_custom_data(
        self, data_key: str, data: pd.DataFrame | pd.Series[float]
    ) -> None:
        """Upload a custom DataFrame or Series to the context given by a data key."""
        if isinstance(data, pd.Series):
            data = data.to_frame(name="value")
        self._data_loader.upload_custom_data(data_key, data)

    def add_comparison(
        self,
        measure_key: str,
        test_source: str,
        ref_source: str,
        test_scenarios: dict[str, str] = {},
        ref_scenarios: dict[str, str] = {},
        stratifications: list[str] = [],
    ) -> None:
        """Add a comparison to the context given a measure key and data sources."""
        if measure_key.endswith(".relative_risk"):
            raise ValueError(
                f"For relative risk measures, use 'add_relative_risk_comparison' instead. "
                f"Got measure_key='{measure_key}'"
            )

        measure = measures.get_measure_from_key(measure_key, list(self.scenario_columns))
        self._add_comparison_with_measure(
            measure, test_source, ref_source, test_scenarios, ref_scenarios, stratifications
        )

    def add_relative_risk_comparison(
        self,
        risk_factor: str,
        affected_entity: str,
        affected_measure: str,
        test_source: str,
        ref_source: str,
        test_scenarios: dict[str, str] = {},
        ref_scenarios: dict[str, str] = {},
        stratifications: list[str] = [],
        risk_stratification_column: str | None = None,
        risk_state_mapping: dict[str, str] | None = None,
    ) -> None:
        """Add a relative risk comparison to the context.

        Parameters
        ----------
        risk_factor
            The risk factor name (e.g., 'child_stunting')
        affected_entity
            The entity affected by the risk factor (e.g., 'cause.diarrheal_diseases')
        affected_measure
            The measure to calculate (e.g., 'excess_mortality_rate', 'incidence_rate')
        risk_stratification_column
            The column to use for stratifying the risk factor in simulation data (e.g., 'risk_factor')
        test_source
            Source for test data ('sim', 'artifact', or 'custom')
        ref_source
            Source for reference data ('sim', 'artifact', or 'custom')
        test_scenarios
            Dictionary of scenario filters for test data
        ref_scenarios
            Dictionary of scenario filters for reference data
        stratifications
            List of stratification columns
        """

        measure = CategoricalRelativeRisk(
            risk_factor,
            affected_entity,
            affected_measure,
            risk_stratification_column,
            risk_state_mapping,
        )
        self._add_comparison_with_measure(
            measure, test_source, ref_source, test_scenarios, ref_scenarios, stratifications
        )

    def _add_comparison_with_measure(
        self,
        measure: Measure,
        test_source: str,
        ref_source: str,
        test_scenarios: dict[str, str] = {},
        ref_scenarios: dict[str, str] = {},
        stratifications: list[str] = [],
    ) -> None:
        """Internal method to add a comparison with a pre-constructed measure."""

        test_source_enum = DataSource.from_str(test_source)
        ref_source_enum = DataSource.from_str(ref_source)

        if not test_source_enum == DataSource.SIM:
            raise NotImplementedError(
                f"Comparison for {test_source} source not implemented. Must be SIM."
            )

        # Check if the measure is a RatioMeasure for FuzzyComparison
        if not isinstance(measure, RatioMeasure):
            raise NotImplementedError(
                f"Measure {measure.measure_key} is not a RatioMeasure. Only RatioMeasures are currently supported for comparisons."
            )

        for source, scenarios in (
            (test_source_enum, test_scenarios),
            (ref_source_enum, ref_scenarios),
        ):
            if source == DataSource.SIM and set(scenarios.keys()) != set(
                self.scenario_columns
            ):
                raise ValueError(
                    f"Each simulation comparison subject must choose a specific scenario. "
                    f"You are missing scenarios for: {set(self.scenario_columns) - set(scenarios.keys())}."
                )
        test_data = RatioMeasureDataBundle(
            measure=measure,
            source=test_source_enum,
            data_loader=self._data_loader,
            age_group_df=self.age_groups,
            scenarios=test_scenarios,
        )
        ref_data = RatioMeasureDataBundle(
            measure=measure,
            source=ref_source_enum,
            data_loader=self._data_loader,
            age_group_df=self.age_groups,
            scenarios=ref_scenarios,
        )

        comparison = FuzzyComparison(
            measure=measure,
            test_data=test_data,
            reference_data=ref_data,
            stratifications=stratifications,
        )
        self.comparisons[measure.measure_key] = comparison

    def verify(self, comparison_key: str, stratifications: Collection[str] = ()):  # type: ignore[no-untyped-def]
        self.comparisons[comparison_key].verify(stratifications)

    def metadata(self, comparison_key: str) -> pd.DataFrame:
        return self.comparisons[comparison_key].metadata

    def get_frame(
        self,
        comparison_key: str,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "",
        ascending: bool = False,
        aggregate_draws: bool = False,
    ) -> pd.DataFrame:
        """Get a DataFrame of the comparison data, with naive comparison of the test and reference.

        Parameters:
        -----------
        comparison_key
            The key of the comparison for which to get the data
        stratifications
            The stratifications to use for the comparison
        num_rows
            The number of rows to return. If "all", return all rows.
        sort_by
            The column to sort by. Default is "percent_error" for non-aggregated data, and no sorting for aggregated data.
        ascending
            Whether to sort in ascending order. Default is False.
        aggregate_draws
            If True, aggregate over draws to show means and 95% uncertainty intervals.
        Returns:
        --------
        A DataFrame of the comparison data.
        """
        if not aggregate_draws and not sort_by:
            sort_by = "percent_error"

        if (isinstance(num_rows, int) and num_rows > 0) or num_rows == "all":
            return self.comparisons[comparison_key].get_frame(
                stratifications, num_rows, sort_by, ascending, aggregate_draws
            )
        else:
            raise ValueError("num_rows must be a positive integer or literal 'all'")

    def plot_comparison(
        self, comparison_key: str, type: str, condition: dict[str, Any] = {}, **kwargs: Any
    ) -> Figure | list[Figure]:
        return plot_utils.plot_comparison(
            self.comparisons[comparison_key], type, condition, **kwargs
        )

    def generate_comparisons(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def verify_all(self):  # type: ignore[no-untyped-def]
        for comparison in self.comparisons.values():
            comparison.verify()

    def plot_all(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_results(self, verbose: bool = False):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    # TODO MIC-6047 Let user pass in custom age groups
    def _get_age_groups(self) -> pd.DataFrame:
        """Get the age groups from the given DataFrame or from the artifact."""
        from vivarium.framework.artifact.artifact import ArtifactException

        try:
            age_groups: pd.DataFrame = self._data_loader.get_data(
                "population.age_bins", DataSource.ARTIFACT
            )
        # If we can't find the age groups in the artifact, get them directly from vivarium inputs
        except ArtifactException:
            from vivarium_inputs import get_age_bins

            age_groups = get_age_bins()

        # mypy wants this to do type narrowing
        if age_groups is None:
            raise ValueError(
                "No age groups found. Please provide a DataFrame or use the artifact."
            )
            # relabel index level age_group_name to age_group

        return age_groups.rename_axis(index={"age_group_name": "age_group"})
