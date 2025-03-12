import pandas as pd

from vivarium_testing_utils.automated_validation.calculations import compute_metric


class Comparison:
    def __init__(
        self,
        measure_key: str,
        test_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        stratifications: list[str] = [],
    ):
        self.measure = measure_key
        self.test_data = test_data
        self.reference_data = reference_data
        self.computed_comparison = compute_metric(
            self.test_data, self.reference_data, self.measure
        )
        # you need to marginalize out the non-stratified columns as well

    def verify(self, stratifications: list[str]):
        pass

    def summarize(self, stratifications: list[str]):
        pass

    def heads(self, stratifications: list[str]):
        pass
