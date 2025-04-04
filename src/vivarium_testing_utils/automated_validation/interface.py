from pathlib import Path

import pandas as pd
from layered_config_tree import LayeredConfigTree

from vivarium_testing_utils.automated_validation import plot_utils
from vivarium_testing_utils.automated_validation.comparison import Comparison
from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.measures import MEASURE_KEY_MAPPINGS


class ValidationContext:
    def __init__(self, results_dir: str | Path, age_groups: pd.DataFrame | None):
        self._data_loader = DataLoader(results_dir)
        self.comparisons = LayeredConfigTree()

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets available in the given simulation output directory."""
        return self._data_loader.sim_outputs()

    def get_artifact_keys(self) -> list[str]:
        """Get a list of the artifact keys available to compare against."""
        return self._data_loader.artifact_keys()

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
        test_source = DataSource.from_str(test_source)
        ref_source = DataSource.from_str(ref_source)

        entity_type, entity, measure = measure_key.split(".")
        if test_source == DataSource.SIM:
            measure = MEASURE_KEY_MAPPINGS[entity_type][measure](entity)
            test_raw_datasets = {
                dataset_name: self._data_loader.get_dataset(dataset_name, test_source)
                for dataset_name in measure.required_data_keys
            }
            test_data = measure.process_raw_sim_data(**test_raw_datasets)

    def verify(self, comparison_key: str, stratifications: list[str] = []):
        self.comparisons[comparison_key].verify(stratifications)

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
