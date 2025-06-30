from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    ratio,
    filter_data,
    marginalize,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    Deaths,
    RiskStatePersonTime,
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
    artifact_key: str

    def __str__(self) -> str:
        return self.measure_key

    @property
    def title(self) -> str:
        """Return a formatted title for the measure."""
        return _format_title(self.measure_key)

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
    artifact_key: str
    numerator: SimDataFormatter
    denominator: SimDataFormatter

    @property
    def sim_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        return {
            "numerator_data": self.numerator.raw_dataset_name,
            "denominator_data": self.denominator.raw_dataset_name,
        }

    @property
    def artifact_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        return {
            "artifact_data": self.artifact_key,
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
        numerator_data, denominator_data = _align_indexes(numerator_data, denominator_data)
        return {"numerator_data": numerator_data, "denominator_data": denominator_data}


class Incidence(RatioMeasure):
    """Computes Susceptible Population Incidence Rate."""

    def __init__(self, cause: str) -> None:
        self.measure_key = self.artifact_key = f"cause.{cause}.incidence_rate"
        self.numerator = TransitionCounts(cause, f"susceptible_to_{cause}", cause)
        self.denominator = StatePersonTime(cause, f"susceptible_to_{cause}")


class Prevalence(RatioMeasure):
    """Computes Prevalence of cause in the population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = self.artifact_key = f"cause.{cause}.prevalence"
        self.numerator = StatePersonTime(cause, cause)
        self.denominator = StatePersonTime(cause)


class SIRemission(RatioMeasure):
    """Computes (SI) remission rate among infected population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = self.artifact_key = f"cause.{cause}.remission_rate"
        self.numerator = TransitionCounts(cause, cause, f"susceptible_to_{cause}")
        self.denominator = StatePersonTime(cause, cause)


class CauseSpecificMortalityRate(RatioMeasure):
    """Computes cause-specific mortality rate in the population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = self.artifact_key = f"cause.{cause}.cause_specific_mortality_rate"
        self.numerator = Deaths(cause)  # Deaths due to specific cause
        self.denominator = StatePersonTime()  # Total person time


class ExcessMortalityRate(RatioMeasure):
    """Computes excess mortality rate among those with the disease compared to the general population."""

    def __init__(self, cause: str) -> None:
        self.measure_key = self.artifact_key = f"cause.{cause}.excess_mortality_rate"
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

    def __init__(self, scenario_columns: list[str]):
        """Initialize PopulationStructure measure.

        Parameters
        ----------
        scenario_columns
            Column names for scenario stratification. Defaults to an empty list.
        """
        self.measure_key = self.artifact_key = "population.structure"
        self.numerator = StatePersonTime()
        self.denominator = TotalPopulationPersonTime(scenario_columns)

    @check_io(artifact_data=SingleNumericColumn, out=SingleNumericColumn)
    def get_measure_data_from_artifact(self, artifact_data: pd.DataFrame) -> pd.DataFrame:
        return artifact_data / artifact_data.sum()

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


class RiskExposure(RatioMeasure):
    """Computes risk factor exposure levels in the population.

    This measure calculates exposure prevalence from state-specific person time data.
    For categorical risk factors (e.g., child wasting, stunting), exposure is computed
    as the proportion of person time spent in each risk state.

    Numerator: Person time in specific risk state
    Denominator: Total person time across all risk states
    """

    def __init__(self, risk_factor: str) -> None:
        self.measure_key = self.artifact_key = f"risk_factor.{risk_factor}.exposure"
        self.risk_factor = risk_factor

        # Create custom formatters for risk exposure
        self.numerator = RiskStatePersonTime(risk_factor)
        self.denominator = RiskStatePersonTime(risk_factor, sum_all=True)


class CategoricalRelativeRisk(RatioMeasure):
    """Computes relative risk of a categorical variable."""

    def __init__(
        self,
        risk_factor: str,
        affected_entity: str,
        affected_measure: str,
        risk_stratification_column: str,
        risk_state_mapping: dict[str, str] | None,
    ) -> None:
        self.risk_factor = risk_factor
        self.measure_key = (
            f"risk_factor.{risk_factor}.relative_risk.{affected_entity}.{affected_measure}"
        )
        self.artifact_key = f"risk_factor.{risk_factor}.relative_risk"
        self.affected_entity = affected_entity
        self.affected_measure_name = affected_measure
        self.affected_measure: RatioMeasure = MEASURE_KEY_MAPPINGS["cause"][affected_measure](
            affected_entity
        )
        self.numerator = self.affected_measure.numerator
        self.denominator = self.affected_measure.denominator
        self.risk_stratification_column = risk_stratification_column
        self.risk_state_mapping = risk_state_mapping

    @property
    def title(self) -> str:
        """Return a human-readable title for the measure."""
        format_str = lambda x: x.replace("_", " ").title()
        return f"Effect of {format_str(self.risk_factor)} on {format_str(self.affected_entity)} {format_str(self.affected_measure_name)}"

    @property
    def artifact_datasets(self) -> dict[str, str]:
        """Return a dictionary of required datasets for this measure."""
        return {
            "relative_risks": self.artifact_key,
            "affected_data": self.affected_measure.measure_key,
        }

    @check_io(
        relative_risks=SingleNumericColumn,
        affected_data=SingleNumericColumn,
        out=SingleNumericColumn,
    )
    def get_measure_data_from_artifact(
        self, relative_risks: pd.DataFrame, affected_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Multiply relative risks by affected data to get final measure data."""
        relative_risks = filter_data(
            relative_risks,
            filter_cols={
                "affected_entity": self.affected_entity,
                "affected_measure": self.affected_measure_name,
            },
        )
        ## multiply relative risks by affected data being sure to broadcast unequal index levels
        risk_stratified_measure_data = relative_risks * affected_data
        if self.risk_state_mapping:
            # Map level 'parameter' values to risk states given by risk_state_mapping
            risk_stratified_measure_data = risk_stratified_measure_data.rename(
                index=self.risk_state_mapping, level="parameter"
            ).rename_axis(index={"parameter": self.risk_stratification_column})
            return risk_stratified_measure_data

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
        ratio_datasets = self.affected_measure.get_ratio_datasets_from_sim(
            numerator_data=numerator_data,
            denominator_data=denominator_data,
        )
        # for dataset in ratio_datasets.values():
        #     if not self.risk_stratification_column in dataset.index.names:
        #         raise ValueError(
        #             f"Risk stratification column '{self.risk_stratification_column}' not found in dataset index names."
        #         )
        return ratio_datasets


MEASURE_KEY_MAPPINGS: dict[str, dict[str, Callable[..., Measure]]] = {
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
    "risk_factor": {
        "exposure": RiskExposure,
    },
}


def get_measure_from_key(measure_key: str, scenario_columns: list[str]) -> Measure:
    """Get a measure instance from a measure key string.

    Parameters
    ----------
    measure_key
        The measure key in format 'entity_type.entity.measure_key' or 'entity_type.measure_key'
    scenario_columns
        Column names for scenario stratification. Used by some measures like PopulationStructure.

    Returns
    -------
        The instantiated measure object
    """
    parts = measure_key.split(".")
    if len(parts) == 3:
        entity_type, entity, measure_key = parts
        return MEASURE_KEY_MAPPINGS[entity_type][measure_key](entity)
    elif len(parts) == 2:
        entity_type, measure_key = parts
        # Special case for PopulationStructure which needs scenario_columns
        if entity_type == "population" and measure_key == "structure":
            return MEASURE_KEY_MAPPINGS[entity_type][measure_key](scenario_columns)
        else:
            return MEASURE_KEY_MAPPINGS[entity_type][measure_key]()
    else:
        raise ValueError(
            f"Invalid measure key format: {measure_key}. Expected format is two or three period-delimited strings e.g. 'population.structure' or 'cause.deaths.excess_mortality_rate'."
        )


def _align_indexes(
    numerator: pd.DataFrame, denominator: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile indexes between numerator and denominator DataFrames. Dataframes can have unique columns given by the numerator_only_indexes and denominator_only_indexes.
    All other index levels must be summed over."""
    numerator_index_levels = set(numerator.index.names)
    denominator_index_levels = set(denominator.index.names)

    for level in numerator_index_levels - denominator_index_levels:
        numerator = marginalize(numerator, [level])
    for level in denominator_index_levels - numerator_index_levels:
        denominator = marginalize(denominator, [level])
    return (numerator, denominator)


def _format_title(measure_key: str) -> str:
    """Convert a measure key to a more readable format.

    For example, "cause.disease.incidence_rate" becomes "Disease Incidence Rate".
    """
    parts = measure_key.split(".")
    if len(parts) > 2:
        parts = parts[1:]
    title = " ".join(parts)
    title = title.replace("_", " ")
    title = " ".join([word.capitalize() for word in title.split()])
    return title
