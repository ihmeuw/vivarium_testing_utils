from __future__ import annotations

from pathlib import Path

import pandas as pd

from vivarium_testing_utils.automated_validation import plot_utils
from vivarium_testing_utils.automated_validation.comparison import Comparison, FuzzyComparison
from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    resolve_age_groups,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    MEASURE_KEY_MAPPINGS,
    Measure,
)


class ValidationContext:
    def __init__(self, results_dir: str | Path):
        self._data_loader = DataLoader(Path(results_dir))
        self.comparisons: dict[str, Comparison] = {}
        self.age_groups = self._get_age_groups()

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
        test_data = measure.get_ratio_data_from_sim(
            **test_raw_datasets,
        )

        ref_source_enum = DataSource.from_str(ref_source)
        ref_raw_datasets = self._get_raw_datasets_from_source(measure, ref_source_enum)
        ref_data = measure.get_measure_data(ref_source_enum, **ref_raw_datasets)

        test_data = resolve_age_groups(test_data, self.age_groups)
        ref_data = resolve_age_groups(ref_data, self.age_groups)
        comparison = FuzzyComparison(
            measure,
            test_data,
            ref_data,
            stratifications,
        )
        self.comparisons[measure_key] = comparison

    def verify(self, comparison_key: str, stratifications: list[str] = []):  # type: ignore[no-untyped-def]
        self.comparisons[comparison_key].verify(stratifications)

    def summarize(self, comparison_key: str, stratifications: list[str] = []):  # type: ignore[no-untyped-def]
        return self.comparisons[comparison_key].summarize(stratifications)

    def heads(self, comparison_key: str, stratifications: list[str] = []):  # type: ignore[no-untyped-def]
        self.comparisons[comparison_key].heads(stratifications)

    def plot_comparison(self, comparison_key: str, type: str, **kwargs):  # type: ignore[no-untyped-def]
        return plot_utils.plot_comparison(self.comparisons[comparison_key], type, kwargs)

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
            import vivarium_inputs as vi

            age_groups = vi.get_age_bins()

        # mypy wants this to do type narrowing
        if age_groups is None:
            raise ValueError(
                "No age groups found. Please provide a DataFrame or use the artifact."
            )
        return age_groups

    def _get_raw_datasets_from_source(
        self, measure: Measure, source: DataSource
    ) -> dict[str, pd.DataFrame]:
        """Get the raw datasets from the given source."""
        return {
            dataset_name: self._data_loader.get_dataset(dataset_key, source)
            for dataset_name, dataset_key in measure.get_required_datasets(source).items()
        }
