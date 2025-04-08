from abc import ABC, abstractmethod

import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    align_indexes,
    ratio,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    ArtifactData,
    MeasureData,
    RatioData,
    SimOutputData,
)
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
    SimDataFormatter,
    TransitionCounts,
)
from vivarium_testing_utils.automated_validation.data_loader import DataSource


class Measure(ABC):
    """Base class for all measures."""

    required_datasets: dict[str, str]

    @abstractmethod
    def get_measure_data_from_artifact(self, *args, **kwargs) -> MeasureData:
        """Process artifact data into a format suitable for calculations."""
        pass

    @abstractmethod
    def get_measure_data_from_sim(self, *args, **kwargs) -> MeasureData:
        """Process raw simulation data into a format suitable for calculations."""
        pass

    def get_measure_data(self, source: DataSource, *args, **kwargs) -> MeasureData:
        """Process data from the specified source into a format suitable for calculations."""
        if source == DataSource.SIM:
            return self.get_measure_data_from_sim(*args, **kwargs)
        elif source == DataSource.ARTIFACT:
            return self.get_measure_data_from_artifact(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported data source: {source}")


class RatioMeasure(Measure):
    """A Measure that calculates ratio data from simulation data."""

    cause: str
    numerator: SimDataFormatter
    denominator: SimDataFormatter

    @property
    def required_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        return {
            "numerator": self.numerator.data_key,
            "denominator": self.denominator.data_key,
        }

    @abstractmethod
    def get_ratio_data_from_sim(
        self, numerator_data: SimOutputData, denominator_data: SimOutputData
    ) -> RatioData:
        """Process raw simulation data into a format suitable for calculations."""
        pass

    def get_measure_data_from_artifact(self, artifact_data: ArtifactData) -> MeasureData:
        return artifact_data

    def get_ratio_data_from_artifact(self, artifact_data: ArtifactData) -> RatioData:
        raise NotImplementedError("Artifact data cannot be processed as a ratio.")

    def get_measure_data_from_ratio(self, ratio_data: RatioData) -> MeasureData:
        """Compute final measure data from split data."""
        return ratio(
            ratio_data,
            numerator=self.numerator.renamed_column,
            denominator=self.denominator.renamed_column,
        )

    def get_measure_data_from_sim(self, *args, **kwargs) -> MeasureData:
        """Process raw simulation data into a format suitable for calculations."""
        return self.get_measure_data_from_ratio(self.get_ratio_data_from_sim(*args, **kwargs))

    def get_ratio_data_from_sim(
        self, numerator_data: SimOutputData, denominator_data: SimOutputData
    ) -> RatioData:
        """Process raw incidence data into a format suitable for calculations."""
        aligned_datasets = align_indexes([numerator_data, denominator_data])
        return pd.concat(aligned_datasets, axis=0)


class Incidence(RatioMeasure):
    """Class to compute incidence from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.numerator = TransitionCounts(cause, f"susceptible_to_{self.cause}", cause)
        self.denominator = PersonTime(cause, f"susceptible_to_{self.cause}")


class Prevalence(RatioMeasure):
    """Computes prevalence from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.numerator = PersonTime(cause, cause)
        self.denominator = PersonTime(cause)


class Remission(RatioMeasure):
    """Computes remission from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.numerator = TransitionCounts(cause, cause, f"susceptible_to_{self.cause}")
        self.denominator = PersonTime(cause, f"susceptible_to_{self.cause}")


MEASURE_KEY_MAPPINGS = {
    "cause": {
        "incidence": Incidence,
        "prevalence": Prevalence,
        "remission": Remission,
    }
}
