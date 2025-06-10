from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import ratio
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    Deaths,
    SimDataFormatter,
    StatePersonTime,
    TotalPopulationPersonTime,
    TransitionCounts,
)
from vivarium_testing_utils.automated_validation.data_transformation.utils import check_io


class Measure(ABC):
    """A Measure contains key information and methods to take raw data from a DataSource
    and process it into an epidemiological measure suitable for use in a Comparison."""

    measure_key: str

    @property
    @abstractmethod
    def sim_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        pass

    @property
    @abstractmethod
    def artifact_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        pass

    @abstractmethod
    def get_measure_data_from_artifact(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Process artifact data into a format suitable for calculations."""
        pass

    @abstractmethod
    def get_measure_data_from_sim(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Process raw simulation data into a format suitable for calculations."""
        pass

    @check_io(out=SingleNumericColumn)
    def get_measure_data(self, source: DataSource, *args: Any, **kwargs: Any) -> pd.DataFrame:
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

    @check_io(artifact_data=SingleNumericColumn, out=SingleNumericColumn)
    def get_measure_data_from_artifact(self, artifact_data: pd.DataFrame) -> pd.DataFrame:
        return artifact_data

    @check_io(
        numerator_data=SingleNumericColumn,
        denominator_data=SingleNumericColumn,
        out=SingleNumericColumn,
    )
    def get_measure_data_from_ratio(
        self, numerator_data: pd.DataFrame, denominator_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute final measure data from separate numerator and denominator data."""
        return ratio(numerator_data, denominator_data)

    @check_io(out=SingleNumericColumn)
    def get_measure_data_from_sim(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Process raw simulation data into a format suitable for calculations."""
        return self.get_measure_data_from_ratio(
            **self.get_ratio_datasets_from_sim(*args, **kwargs)
        )

    @check_io(
        numerator_data=SimOutputData,
        denominator_data=SimOutputData,
    )
    def get_ratio_datasets_from_sim(
        self,
        numerator_data: pd.DataFrame,
        denominator_data: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Process raw simulation data and return numerator and denominator DataFrames separately."""
        numerator_data = self.numerator.format_dataset(numerator_data)
        denominator_data = self.denominator.format_dataset(denominator_data)
        return {"numerator_data": numerator_data, "denominator_data": denominator_data}


class Incidence(RatioMeasure):
    """Computes Susceptible Population Incidence Rate."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.incidence_rate"
        self.numerator = TransitionCounts(cause, f"susceptible_to_{cause}", cause)
        self.denominator = StatePersonTime(cause, f"susceptible_to_{cause}")


class Prevalence(RatioMeasure):
    """Computes Prevalence of cause in the population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.prevalence"
        self.numerator = StatePersonTime(cause, cause)
        self.denominator = StatePersonTime(cause)


class SIRemission(RatioMeasure):
    """Computes (SI) remission rate among infected population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.remission_rate"
        self.numerator = TransitionCounts(cause, cause, f"susceptible_to_{cause}")
        self.denominator = StatePersonTime(cause, cause)


class CauseSpecificMortalityRate(RatioMeasure):
    """Computes cause-specific mortality rate in the population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.cause_specific_mortality_rate"
        self.numerator = Deaths(cause)  # Deaths due to specific cause
        self.denominator = StatePersonTime()  # Total person time


class ExcessMortalityRate(RatioMeasure):
    """Computes excess mortality rate among those with the disease compared to the general population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = f"cause.{cause}.excess_mortality_rate"
        self.numerator = Deaths(cause)  # Deaths due to specific cause
        self.denominator = StatePersonTime(
            cause, cause
        )  # Person time among those with the disease


class PopulationStructure(RatioMeasure):
    """Compares simulation population structure against artifact population structure.

    This measure aggregates person time data by age groups and sex to match
    the population structure format from the artifact. It's useful for validating
    that the simulation maintains realistic demographic distributions.
    """

    def __init__(self, scenario_columns: list[str] = None):
        """Initialize PopulationStructure measure.

        Parameters
        ----------
        scenario_columns : list[str], optional
            Column names for scenario stratification. Defaults to an empty list.
        """
        self.measure_key = "population.structure"
        self.numerator = StatePersonTime()
        self.denominator = TotalPopulationPersonTime(scenario_columns)

    @check_io(artifact_data=SingleNumericColumn, out=SingleNumericColumn)
    def get_measure_data_from_artifact(self, artifact_data: pd.DataFrame) -> pd.DataFrame:
        return artifact_data / artifact_data.sum()


MEASURE_KEY_MAPPINGS = {
    "cause": {
        "incidence_rate": Incidence,
        "prevalence": Prevalence,
        "remission_rate": SIRemission,
        "cause_specific_mortality_rate": CauseSpecificMortalityRate,
        "excess_mortality_rate": ExcessMortalityRate,
    },
    "population": {
        "structure": PopulationStructure,
    },
}


def get_measure_from_key(measure_key: str, scenario_columns: list[str]) -> Measure:
    """Get a measure instance from a measure key string.

    Parameters
    ----------
    measure_key : str
        The measure key in format 'entity_type.entity.measure_name' or 'entity_type.measure_name'
    scenario_columns : list[str], optional
        Column names for scenario stratification. Used by some measures like PopulationStructure.

    Returns
    -------
    Measure
        The instantiated measure object
    """
    parts = measure_key.split(".")
    if len(parts) == 3:
        entity_type, entity, measure_name = parts
        return MEASURE_KEY_MAPPINGS[entity_type][measure_name](entity)
    elif len(parts) == 2:
        entity_type, measure_name = parts
        # Special case for PopulationStructure which needs scenario_columns
        if entity_type == "population" and measure_name == "structure":
            return MEASURE_KEY_MAPPINGS[entity_type][measure_name](scenario_columns)
        else:
            return MEASURE_KEY_MAPPINGS[entity_type][measure_name]()
    else:
        raise ValueError(f"Invalid measure key format: {measure_key}")
