from vivarium_testing_utils.automated_validation.calculations import (
    ratio,
    aggregate_sum,
    stratify,
    marginalize,
    linear_combination,
    filter_data,
    align_indexes,
)
from vivarium_testing_utils.automated_validation.data_schema import (
    SimOutputData,
    ArtifactData,
    CustomData,
    IntermediateData,
    ProcessedData,
)
import pandas as pd
from abc import ABC, abstractmethod


class Measure(ABC):

    @abstractmethod
    def process_raw_sim_data(self, *args, **kwargs) -> pd.DataFrame:
        """Process raw simulation data into a format suitable for calculations."""
        pass

    @abstractmethod
    def compute_metric(self, intermediate_data: pd.DataFrame) -> pd.Series:
        """Compute the metric from the processed simulation data."""
        pass


class Incidence(Measure):
    """Class to compute incidence from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.required_data_keys = {
            f"transition_count_{self.cause}",
            f"person_time_{self.cause}",
        }
        self.transition_string = f"susceptible_to_{self.cause}_TO_{self.cause}"
        self.state_string = f"susceptible_to_{self.cause}"
        self.numerator_column = f"{self.transition_string}_transition_count"
        self.denominator_column = f"{self.state_string}_person_time"

    def process_raw_sim_data(
        self, transition_count_data: pd.DataFrame, person_time_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Process raw incidence data into a format suitable for calculations."""
        # get cause from cause index column. There should be a single value
        transition_count_data = drop_redundant_column(
            transition_count_data,
            "cause",
            self.cause,
        )
        person_time_data = drop_redundant_column(
            person_time_data,
            "cause",
            self.cause,
        )
        # filter for susceptible to infected transitions
        transition_count_data = filter_data(
            transition_count_data,
            {
                f"{self.cause}_transition": [
                    self.transition_string,
                ]
            },
        )
        # rename value column to incident cases
        transition_count_data = transition_count_data.rename(
            columns={"value": self.numerator_column}
        )
        # filter person time for susceptible person time
        person_time_data = filter_data(
            person_time_data,
            {
                f"{self.cause}_state": [
                    self.state_string,
                ]
            },
        )
        # rename value column to person_time
        person_time_data = person_time_data.rename(columns={"value": self.denominator_column})
        return combine_sim_output_data(
            [
                transition_count_data,
                person_time_data,
            ]
        )

    def compute_metric(self, intermediate_data: pd.DataFrame) -> pd.Series:
        """Compute incidence from processed simulation data."""
        # Compute the incidence using the transition count and person time
        return ratio(
            intermediate_data,
            numerator=self.numerator_column,
            denominator=self.denominator_column,
        )


class Prevalence(Measure):
    """Class to compute prevalence from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.required_data_keys = {
            f"person_time_{self.cause}",
        }
        self.state_string = f"{self.cause}"
        self.denominator_column = f"{self.state_string}_person_time"
        self.numerator_column = f"{self.cause}_prevalence"

    def process_raw_sim_data(self, person_time_data: pd.DataFrame) -> pd.DataFrame:
        """Process raw prevalence data into a format suitable for calculations."""
        # get cause from cause index column. There should be a single value
        person_time_data = drop_redundant_column(
            person_time_data,
            "cause",
            self.cause,
        )

        # Get the person time for the cause
        cause_person_time_data = filter_data(
            person_time_data,
            {
                f"{self.cause}_state": [
                    self.state_string,
                ]
            },
        )
        # get denominator dataframe for total person time
        total_person_time_data = marginalize(
            person_time_data,
            [f"{self.cause}_state"],
        )
        # rename value column to person_time
        total_person_time_data = total_person_time_data.rename(
            columns={"value": "total_person_time"}
        )
        return combine_sim_output_data(
            [
                cause_person_time_data,
                total_person_time_data,
            ]
        )

    def compute_metric(self, intermediate_data: pd.DataFrame) -> pd.Series:
        """Compute prevalence from processed simulation data."""
        # Compute the prevalence using the person time
        return ratio(
            intermediate_data,
            numerator=self.numerator_column,
            denominator=self.denominator_column,
        )


class Remission(Measure):
    """Class to compute remission from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.required_data_keys = {
            f"transition_count_{self.cause}",
            f"person_time_{self.cause}",
        }
        self.transition_string = f"{self.cause}_TO_susceptible_to_{self.cause}"
        self.state_string = f"{self.cause}"
        self.numerator_column = f"{self.transition_string}_transition_count"
        self.denominator_column = f"{self.state_string}_person_time"

    def process_raw_sim_data(
        self, transition_count_data: pd.DataFrame, person_time_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Process raw remission data into a format suitable for calculations."""
        # get cause from cause index column. There should be a single value
        transition_count_data = drop_redundant_column(
            transition_count_data,
            "cause",
            self.cause,
        )
        person_time_data = drop_redundant_column(
            person_time_data,
            "cause",
            self.cause,
        )

        # filter for infected to susceptible transitions
        transition_count_data = filter_data(
            transition_count_data,
            {
                f"{self.cause}_transition": [
                    self.transition_string,
                ]
            },
        )
        # rename value column to incident cases
        transition_count_data = transition_count_data.rename(
            columns={"value": self.numerator_column}
        )
        # filter person time for susceptible person time
        person_time_data = filter_data(
            person_time_data,
            {
                f"{self.cause}_state": [
                    self.state_string,
                ]
            },
        )
        # rename value column to person_time
        person_time_data = person_time_data.rename(columns={"value": self.denominator_column})
        return combine_sim_output_data(
            [
                transition_count_data,
                person_time_data,
            ]
        )

    def compute_metric(self, intermediate_data: pd.DataFrame) -> pd.Series:
        """Compute remission from processed simulation data."""
        # Compute the remission using the transition count and person time
        return ratio(
            intermediate_data,
            numerator=self.numerator_column,
            denominator=self.denominator_column,
        )


def combine_sim_output_data(raw_datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Process raw simulation data into a format suitable for calculations."""
    # Align the indexes of the datasets
    aligned_datasets = align_indexes(raw_datasets)

    # Combine the aligned datasets into a single DataFrame
    combined_data = pd.concat(aligned_datasets, axis=0)

    return combined_data


def drop_redundant_column(data: pd.DataFrame, column_name: str, column_value: str) -> None:
    """Validate that the column is singular-valued, then drop it"""

    if not all(data.index.get_level_values(column_name) == column_value):
        raise ValueError(
            f"Cause {data.index.get_level_values('cause').unique()} in data does not match expected cause {column_name}"
        )
    data = data.drop(columns=[column_name])
    return data


MEASURE_KEY_MAPPINGS = {
    "cause": {
        "incidence": Incidence,
        "prevalence": Prevalence,
        "remission": Remission,
    }
}
