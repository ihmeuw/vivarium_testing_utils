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
    RatioData,
    MeasureData,
)
import pandas as pd
from abc import ABC, abstractmethod


class Transition:
    """structured container to hold info about Transition Counts"""

    def __init__(self, cause, start_state, end_state):
        self.type = "transition_count"
        self.cause = cause
        self.data_key = f"{self.type}_{self.cause}"
        self.start_state = start_state
        self.end_state = end_state
        self.transition_string = f"{self.start_state}_TO_{self.end_state}"
        self.groupby_column = f"{self.cause}_transition"
        self.renamed_column = f"{self.transition_string}_{self.type}"

    def format_dataset(self, dataset: SimOutputData) -> SimOutputData:
        """Clean up cause column, filter for the transition, and rename the value column."""
        dataset = drop_redundant_column(
            dataset,
            "cause",
            self.cause,
        )
        dataset = filter_data(dataset, {self.groupby_column: [self.transition_string]})
        dataset = dataset.rename(columns={"value": self.renamed_column})
        return dataset


class PersonTime:
    """structured container to hold info about Person Time"""

    def __init__(self, cause, state=None):
        self.type = "person_time"
        self.cause = cause
        self.data_key = f"{self.type}_{self.cause}"
        self.state = state if state else "total"
        self.groupby_column = f"{self.cause}_state"
        self.renamed_column = f"{self.state}_{self.type}"

    def format_dataset(self, dataset: SimOutputData) -> SimOutputData:
        """Clean up cause column, filter for the state, and rename the value column."""
        dataset = drop_redundant_column(
            dataset,
            "cause",
            self.cause,
        )
        if self.state == "total":
            dataset = marginalize(
                dataset,
                [self.groupby_column],
            )
        else:
            dataset = filter_data(dataset, {self.groupby_column: [self.state]})
        dataset = dataset.rename(columns={"value": self.renamed_column})
        return dataset


class Measure(ABC):

    @abstractmethod
    def get_measure_data_from_artifact(self, *args, **kwargs) -> MeasureData:
        """Process artifact data into a format suitable for calculations."""
        pass

    @abstractmethod
    def get_measure_data_from_sim(self, *args, **kwargs) -> MeasureData:
        """Process raw simulation data into a format suitable for calculations."""
        pass


class RatioMeasure(Measure):
    """A Measure that calculates ratio data from simulation data."""

    cause: str
    numerator: Transition | PersonTime
    denominator: Transition | PersonTime

    @abstractmethod
    def get_ratio_data_from_sim(
        self, numerator_data: SimOutputData, denominator_data: SimOutputData
    ) -> RatioData:
        """Process raw simulation data into a format suitable for calculations."""
        pass

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
        # get cause from cause index column. There should be a single value
        return combine_sim_output_data(
            [
                self.numerator.format_dataset(numerator_data),
                self.denominator.format_dataset(denominator_data),
            ]
        )


class Incidence(RatioMeasure):
    """Class to compute incidence from simulation data."""

    def __init__(self, cause: str):
        self.cause = cause
        self.numerator = Transition(cause, f"susceptible_to_{self.cause}", cause)
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
        self.numerator = Transition(cause, cause, f"susceptible_to_{self.cause}")
        self.denominator = PersonTime(cause, f"susceptible_to_{self.cause}")


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
