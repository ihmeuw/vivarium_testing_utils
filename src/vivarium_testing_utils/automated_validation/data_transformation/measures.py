from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import pandera as pa

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
    Deaths,
    RiskStatePersonTime,
    RiskTotalPersonTime,
    SimDataFormatter,
    StatePersonTime,
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

    @check_io(ratio_data=RatioData, out=SingleNumericColumn)
    def get_measure_data_from_ratio(self, ratio_data: pd.DataFrame) -> pd.DataFrame:
        """Compute final measure data from split data."""
        return ratio(
            ratio_data,
            numerator=self.numerator.new_value_column_name,
            denominator=self.denominator.new_value_column_name,
        )

    @check_io(out=SingleNumericColumn)
    def get_measure_data_from_sim(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Process raw simulation data into a format suitable for calculations."""
        return self.get_measure_data_from_ratio(self.get_ratio_data_from_sim(*args, **kwargs))

    @check_io(
        numerator_data=SimOutputData,
        denominator_data=SimOutputData,
        out=RatioData,
    )
    def get_ratio_data_from_sim(
        self,
        numerator_data: pd.DataFrame,
        denominator_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process raw simulation data into a RatioData frame with count columns to be divided later."""
        numerator_data = self.numerator.format_dataset(numerator_data)
        denominator_data = self.denominator.format_dataset(denominator_data)
        numerator_data, denominator_data = align_indexes([numerator_data, denominator_data])
        return pd.concat([numerator_data, denominator_data], axis=1)


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


class RiskExposure(RatioMeasure):
    """Computes risk factor exposure levels in the population.

    This measure calculates exposure prevalence from state-specific person time data.
    For categorical risk factors (e.g., child wasting, stunting), exposure is computed
    as the proportion of person time spent in each risk state.

    Numerator: Person time in specific risk state
    Denominator: Total person time across all risk states
    """

    def __init__(self, risk_factor: str) -> None:
        self.measure_key = f"risk_factor.{risk_factor}.exposure"
        self.risk_factor = risk_factor

        # Create custom formatters for risk exposure
        self.numerator = RiskStatePersonTime(risk_factor)
        self.denominator = RiskTotalPersonTime(risk_factor, sum_all=True)


MEASURE_KEY_MAPPINGS = {
    "cause": {
        "incidence_rate": Incidence,
        "prevalence": Prevalence,
        "remission_rate": SIRemission,
        "cause_specific_mortality_rate": CauseSpecificMortalityRate,
        "excess_mortality_rate": ExcessMortalityRate,
    },
    "risk_factor": {
        "exposure": RiskExposure,
    },
}
