from abc import ABC, abstractmethod

import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    RatioData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)


class Comparison(ABC):
    """A Comparison is the basic testing unit to compare two datasets, a "test" dataset and a
    "reference" dataset. The test dataset is the one that is being validated, while the reference
    dataset is the one that is used as a benchmark. The comparison operates on a *measure* of the two datasets,
    typically a derived quantity of the test data such as incidence rate or prevalence."""

    measure: Measure
    test_data: pd.DataFrame
    reference_data: pd.DataFrame
    stratifications: list[str]

    @abstractmethod
    def verify(self, stratifications: list[str] = []):
        pass

    @abstractmethod
    def summarize(self, stratifications: list[str] = []):
        pass

    @abstractmethod
    def heads(self, stratifications: list[str] = []):
        pass


class FuzzyComparison(Comparison):
    def __init__(
        self,
        measure: RatioMeasure,
        test_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        stratifications: list[str] = [],
    ):
        self.measure = measure
        self.test_data = test_data
        self.reference_data = reference_data
        self.stratifications = stratifications

    def verify(self, stratifications: list[str] = []):
        raise NotImplementedError

    def summarize(self, stratifications: list[str] = []):
        raise NotImplementedError

    def heads(self, stratifications: list[str] = []):
        raise NotImplementedError
