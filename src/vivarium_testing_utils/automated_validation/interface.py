from pathlib import Path

import pandas as pd
from layered_config_tree import LayeredConfigTree

from vivarium_testing_utils.automated_validation import plot_utils
from vivarium_testing_utils.automated_validation.comparison import Comparison
from vivarium_testing_utils.automated_validation.data_loader import DataLoader


class ValidationContext:
    def __init__(self, results_dir: str | Path, age_groups: pd.DataFrame | None):
        self.data_loader = DataLoader(results_dir)
        self.comparisons = LayeredConfigTree()

    def sim_outputs(self):
        return self.data_loader.sim_outputs()

    def artifact_keys(self):
        return self.data_loader.artifact_keys()

    def add_comparison(
        self, measure_key: str, test_source: str, ref_source: str, stratifications: list[str]
    ) -> None:
        self.comparisons.update(
            [measure_key], Comparison(measure_key, test_source, ref_source, stratifications)
        )

    def verify_comparison(self, comparison_key: str):
        self.comparisons[comparison_key].verify()

    def plot_comparison(self, comparison_key: str, type: str, kwargs):
        return plot_utils.plot_comparison(self.comparisons[comparison_key], type, kwargs)

    def generate_comparisons(self):
        pass

    def verify_all(self):
        pass

    def plot_all(self):
        pass

    def get_results(self):
        pass
