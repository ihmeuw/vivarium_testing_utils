from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Literal

import pandas as pd
from matplotlib.figure import Figure

from vivarium_testing_utils.automated_validation.comparison import Comparison, FuzzyComparison
from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    resolve_age_groups,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    MEASURE_KEY_MAPPINGS,
    Measure,
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

    def get_raw_dataset(self, dataset_key: str, source: str) -> pd.DataFrame:
        """Return a copy of the dataset for manual inspection."""
        return self._data_loader.get_dataset(dataset_key, DataSource.from_str(source))

    def upload_custom_data(
        self, dataset_key: str, data: pd.DataFrame | pd.Series[float]
    ) -> None:
        """Upload a custom DataFrame or Series to the context given by a dataset key."""
        if isinstance(data, pd.Series):
            data = data.to_frame(name="value")
        self._data_loader.upload_custom_data(dataset_key, data)

    def add_comparison(
        self,
        measure_key: str,
        test_source: str,
        ref_source: str,
        stratifications: list[str] = [],
    ) -> None:
        """Add a comparison to the context given a measure key and data sources."""
        entity_type, entity, measure_name = measure_key.split(".")
        measure = MEASURE_KEY_MAPPINGS[entity_type][measure_name](entity)

        test_source_enum = DataSource.from_str(test_source)

        if not test_source_enum == DataSource.SIM:
            raise NotImplementedError(
                f"Comparison for {test_source} source not implemented. Must be SIM."
            )
        test_raw_datasets = self._get_raw_datasets_from_source(measure, test_source_enum)
        test_raw_datasets = {
            dataset_name: resolve_age_groups(dataset, self.age_groups)
            for dataset_name, dataset in test_raw_datasets.items()
        }
        test_datasets = measure.get_ratio_datasets_from_sim(
            **test_raw_datasets,
        )

        ref_source_enum = DataSource.from_str(ref_source)
        ref_raw_datasets = self._get_raw_datasets_from_source(measure, ref_source_enum)
        ref_raw_datasets = {
            dataset_name: resolve_age_groups(dataset, self.age_groups)
            for dataset_name, dataset in ref_raw_datasets.items()
        }
        ref_data = measure.get_measure_data(ref_source_enum, **ref_raw_datasets)

        comparison = FuzzyComparison(
            measure=measure,
            test_source=test_source_enum,
            test_datasets=test_datasets,
            reference_source=ref_source_enum,
            reference_data=ref_data,
            scenario_cols=self.scenario_columns,
            stratifications=stratifications,
        )
        self.comparisons[measure_key] = comparison

    def verify(self, comparison_key: str, stratifications: Collection[str] = ()):  # type: ignore[no-untyped-def]
        self.comparisons[comparison_key].verify(stratifications)

    def metadata(self, comparison_key: str) -> pd.DataFrame:
        return self.comparisons[comparison_key].metadata

    def get_frame(
        self,
        comparison_key: str,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "percent_error",
        ascending: bool = False,
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
            The column to sort by. Default is "percent_error".
        ascending
            Whether to sort in ascending order. Default is False.
        Returns:
        --------
        A DataFrame of the comparison data.
        """
        if (isinstance(num_rows, int) and num_rows > 0) or num_rows == "all":
            return self.comparisons[comparison_key].get_diff(
                stratifications, num_rows, sort_by, ascending
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
            age_groups = self._data_loader.get_dataset(
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

    def _get_raw_datasets_from_source(
        self, measure: Measure, source: DataSource
    ) -> dict[str, pd.DataFrame]:
        """Get the raw datasets from the given source."""
        return {
            dataset_name: self._data_loader.get_dataset(dataset_key, source)
            for dataset_name, dataset_key in measure.get_required_datasets(source).items()
        }
