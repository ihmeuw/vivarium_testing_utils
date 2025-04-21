from abc import ABC, abstractmethod

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    align_indexes,
    ratio,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    RatioData,
    SimOutputData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
    SimDataFormatter,
    TransitionCounts,
)


class Measure(ABC):
    """A Measure contains key information and methods to take raw data from a DataSource
    and process it into an epidemiological measure suitable for use in a Comparison."""

    sim_datasets: dict[str, str]
    artifact_datasets: dict[str, str]

    @abstractmethod
    def get_measure_data_from_artifact(
        self, *args, **kwargs
    ) -> DataFrame[SingleNumericColumn]:
        """Process artifact data into a format suitable for calculations."""
        pass

    @abstractmethod
    def get_measure_data_from_sim(self, *args, **kwargs) -> DataFrame[SingleNumericColumn]:
        """Process raw simulation data into a format suitable for calculations."""
        pass

    @pa.check_types
    def get_measure_data(
        self, source: DataSource, *args, **kwargs
    ) -> Series[SingleNumericColumn]:
        """Process data from the specified source into a format suitable for calculations."""
        if source == DataSource.SIM:
            return self.get_measure_data_from_sim(*args, **kwargs)
        elif source == DataSource.ARTIFACT:
            return self.get_measure_data_from_artifact(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported data source: {source}")

    def get_required_datasets(self, source: DataSource) -> dict[str, str]:
        """Return a dictionary of required datasets for the specified source."""
        if source == DataSource.SIM:
            return self.sim_datasets
        elif source == DataSource.ARTIFACT:
            return self.artifact_datasets
        else:
            raise ValueError(f"Unsupported data source: {source}")


class RatioMeasure(Measure, ABC):
    """A Measure that calculates ratio data from simulation data."""

    measure_key: str
    numerator: SimDataFormatter
    denominator: SimDataFormatter

    @property
    def sim_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        return {
            "numerator_data": self.numerator.data_key,
            "denominator_data": self.denominator.data_key,
        }

    @property
    def artifact_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        return {
            "artifact_data": self.measure_key,
        }

    @abstractmethod
    def get_ratio_data_from_sim(
        self,
        numerator_data: DataFrame[SimOutputData],
        denominator_data: DataFrame[SimOutputData],
    ) -> DataFrame[RatioData]:
        """Process raw simulation data into a format suitable for calculations."""
        pass

    @pa.check_types
    def get_measure_data_from_artifact(
        self, artifact_data: DataFrame[SingleNumericColumn]
    ) -> Series[SingleNumericColumn]:
        return artifact_data

    @pa.check_types
    def get_measure_data_from_ratio(
        self, ratio_data: DataFrame[RatioData]
    ) -> Series[SingleNumericColumn]:
        """Compute final measure data from split data."""
        return ratio(
            ratio_data,
            numerator=self.numerator.new_value_column_name,
            denominator=self.denominator.new_value_column_name,
        )

    @pa.check_types
    def get_measure_data_from_sim(self, *args, **kwargs) -> Series[SingleNumericColumn]:
        """Process raw simulation data into a format suitable for calculations."""
        return self.get_measure_data_from_ratio(self.get_ratio_data_from_sim(*args, **kwargs))

    @pa.check_types
    def get_ratio_data_from_sim(
        self,
        numerator_data: DataFrame[SimOutputData],
        denominator_data: DataFrame[SimOutputData],
    ) -> DataFrame[RatioData]:
        """Process raw incidence data into a format suitable for calculations."""
        numerator_data, denominator_data = align_indexes([numerator_data, denominator_data])
        numerator_data = self.numerator.format_dataset(numerator_data)
        denominator_data = self.denominator.format_dataset(denominator_data)
        return pd.concat([numerator_data, denominator_data], axis=1)


class Incidence(RatioMeasure):
    """Computes Susceptible Population Incidence Rate."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.incidence_rate"
        self.numerator = TransitionCounts(cause, f"susceptible_to_{cause}", cause)
        self.denominator = PersonTime(cause, f"susceptible_to_{cause}")


class Prevalence(RatioMeasure):
    """Computes Prevalence of cause in the population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.prevalence"
        self.numerator = PersonTime(cause, cause)
        self.denominator = PersonTime(cause)


class SIRemission(RatioMeasure):
    """Computes (SI) remission rate among infected population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.remission_rate"
        self.numerator = TransitionCounts(cause, cause, f"susceptible_to_{cause}")
        self.denominator = PersonTime(cause, cause)


MEASURE_KEY_MAPPINGS = {
    "cause": {
        "incidence_rate": Incidence,
        "prevalence": Prevalence,
        "remission_rate": SIRemission,
    }
}
