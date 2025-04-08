import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    ArtifactData,
    MeasureData,
    RatioData,
    SimOutputData,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
)


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
        raise NotImplementedError

    def summarize(self, stratifications: list[str]):
        raise NotImplementedError

    def heads(self, stratifications: list[str]):
        raise NotImplementedError
