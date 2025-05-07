from pathlib import Path

import pandas as pd

from vivarium_testing_utils.automated_validation import plot_utils
from vivarium_testing_utils.automated_validation.comparison import Comparison, FuzzyComparison
from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    MEASURE_KEY_MAPPINGS,
    Measure,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    align_indexes,
    resolve_age_groups,
)
import vivarium_inputs as vi

class ValidationContext:

    def __init__(self, results_dir: str | Path, age_groups: pd.DataFrame | None = None):
        self._data_loader = DataLoader(results_dir)
        self.comparisons: dict[str, Comparison] = {}
        self.age_groups = self._get_age_groups(age_groups)

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets available in the given simulation output directory."""
        return self._data_loader.get_sim_outputs()

    def get_artifact_keys(self) -> list[str]:
        """Get a list of the artifact keys available to compare against."""
        return self._data_loader.get_artifact_keys()

    def get_raw_dataset(self, dataset_key: str, source: str) -> pd.DataFrame:
        """Return a copy of the dataset for manual inspection."""
        return self._data_loader.get_dataset(dataset_key, DataSource.from_str(source))

    def upload_custom_data(self, dataset_key: str, data: pd.DataFrame | pd.Series) -> None:
        """Upload a custom DataFrame or Series to the context given by a dataset key."""
        self._data_loader.upload_custom_data(dataset_key, data)

    def add_comparison(
        self, measure_key: str, test_source: str, ref_source: str, stratifications: list[str]
    ) -> None:
        """Add a comparison to the context given a measure key and data sources."""
        entity_type, entity, measure = measure_key.split(".")
        measure = MEASURE_KEY_MAPPINGS[entity_type][measure](entity)

        test_source = DataSource.from_str(test_source)

        if not test_source == DataSource.SIM:
            raise NotImplementedError(
                f"Fuzzy Comparison for {test_source} source not implemented. Must be SIM."
            )
        test_raw_datasets = self._get_raw_datasets_from_source(measure, test_source)
        test_data = measure.get_ratio_data_from_sim(
            **test_raw_datasets,
        )

        ref_source = DataSource.from_str(ref_source)
        ref_raw_datasets = self._get_raw_datasets_from_source(measure, ref_source)
        ref_data = measure.get_measure_data(ref_source, **ref_raw_datasets)
        test_data, ref_data = self.align_datasets(test_data, ref_data)
        comparison = FuzzyComparison(
            measure,
            test_source,
            test_data,
            ref_source,
            ref_data,
            stratifications,
        )
        self.comparisons.update({measure_key: comparison})

    def verify(self, comparison_key: str, stratifications: list[str] = []):
        self.comparisons[comparison_key].verify(stratifications)

    def summarize(self, comparison_key: str):
        self.comparisons[comparison_key].summarize()

    def heads(self, comparison_key: str, stratifications: list[str] = []):
        self.comparisons[comparison_key].heads(stratifications)

    def plot_comparison(self, comparison_key: str, type: str, **kwargs):
        return plot_utils.plot_comparison(self.comparisons[comparison_key], type, kwargs)

    def generate_comparisons(self):
        raise NotImplementedError

    def verify_all(self):
        for comparison in self.comparisons.values():
            comparison.verify()

    def plot_all(self):
        raise NotImplementedError

    def get_results(self, verbose: bool = False):
        raise NotImplementedError

    def _get_age_groups(self, age_groups: pd.DataFrame | None) -> pd.DataFrame:
        """Get the age groups from the given DataFrame or from the artifact."""
        if age_groups is None:
            try:
                age_groups = self._data_loader.get_dataset(
                    "population.age_bins", DataSource.SIM
                )
            except KeyError:
                age_groups = vi.get_age_bins()
        return age_groups

    def _get_raw_datasets_from_source(
        self, measure: Measure, source: DataSource
    ) -> dict[str, pd.DataFrame]:
        """Get the raw datasets from the given source."""
        return {
            dataset_name: self._data_loader.get_dataset(dataset_key, source)
            for dataset_name, dataset_key in measure.get_required_datasets(source).items()
        }

    def align_datasets(
        self, test_data: pd.DataFrame, ref_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align the test and reference datasets on the same index."""
        test_data = resolve_age_groups(test_data, self.age_bins)
        ref_data = resolve_age_groups(ref_data, self.age_bins)
        test_data, ref_data = align_indexes([test_data, ref_data])
        return test_data, ref_data
