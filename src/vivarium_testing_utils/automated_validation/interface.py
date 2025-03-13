from pathlib import Path

import pandas as pd
from layered_config_tree import LayeredConfigTree

from vivarium_testing_utils.automated_validation import plot_utils
from vivarium_testing_utils.automated_validation.comparison import Comparison
from vivarium_testing_utils.automated_validation.data_loader import DataManager


class ValidationContext:
    def __init__(self, results_dir: str | Path, age_groups: pd.DataFrame | None):
        self.data_loader = DataManager(results_dir)
        self.comparisons = LayeredConfigTree()

    def sim_outputs(self):
        return self.data_loader.sim_outputs()

    def artifact_keys(self):
        return self.data_loader.artifact_keys()

    def add_comparison(
        self, measure_key: str, test_source: str, ref_source: str, stratifications: list[str]
    ) -> None:
        test_data = self.data_loader.get_dataset(measure_key, test_source)
        ref_data = self.data_loader.get_dataset(measure_key, ref_source)
        self.comparisons.update(
            [measure_key], Comparison(measure_key, test_data, ref_data, stratifications)
        )

    def verify(self, comparison_key: str, stratifications: list[str] = []):
        self.comparisons[comparison_key].verify(stratifications)

    def plot_comparison(self, comparison_key: str, type: str, kwargs):
        return plot_utils.plot_comparison(self.comparisons[comparison_key], type, kwargs)

    def generate_comparisons(self):
        pass

    def verify_all(self):
        for comparison in self.comparisons.values():
            comparison.verify()

    def plot_all(self):
        pass

    def get_results(self, verbose: bool = False):
        pass
