from abc import ABC, abstractmethod

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    MeasureData,
    RatioData,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
)


class Comparison(ABC):
    @abstractmethod
    def verify(self, stratifications: list[str]):
        pass

    @abstractmethod
    def summarize(self, stratifications: list[str]):
        pass

    @abstractmethod
    def heads(self, stratifications: list[str]):
        pass


class FuzzyComparison:
    def __init__(
        self,
        measure: RatioMeasure,
        test_data: RatioData,
        reference_data: MeasureData,
        stratifications: list[str] = [],
    ):
        self.measure = measure
        self.test_data = test_data
        self.reference_data = reference_data
        self.stratifications = stratifications
        # you need to marginalize out the non-stratified columns as well

    def verify(self, stratifications: list[str]):
        raise NotImplementedError

    def summarize(self, stratifications: list[str]):
        raise NotImplementedError

    def heads(self, stratifications: list[str]):
        raise NotImplementedError
