from abc import ABC, abstractmethod

import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    filter_data,
    marginalize,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
)


class SimDataFormatter(ABC):
    """A SimDataFormatter contains information about how to format particular kinds of
    simulaton data for use in a measure calculation. For example, incidence relies on
    both transition counts and person time data, which require different formatting/ operations
    on assumed columns in the simulation data."""

    type: str
    cause: str
    data_key: str
    groupby_column: str
    renamed_column: str

    @abstractmethod
    def format_dataset(self, dataset: SimOutputData) -> SimOutputData:
        """Format the dataset for the specific measure."""
        pass


class TransitionCounts(SimDataFormatter):
    """Formatter for simulation data that contains transition counts."""

    def __init__(self, cause: str, start_state: str, end_state: str) -> None:
        self.type = "transition_count"
        self.cause = cause
        self.data_key = f"{self.type}_{self.cause}"
        self.start_state = start_state
        self.end_state = end_state
        self.transition_string = f"{self.start_state}_to_{self.end_state}"
        self.groupby_column = "sub_entity"
        self.renamed_column = f"{self.transition_string}_{self.type}"

    def format_dataset(self, dataset: SimOutputData) -> SimOutputData:
        """Clean up redundant columns, filter for the transition, and rename the value column."""
        dataset = drop_redundant_index(
            dataset,
            "measure",
            self.type,
        )
        dataset = drop_redundant_index(
            dataset,
            "entity_type",
            "cause",
        )
        dataset = drop_redundant_index(
            dataset,
            "entity",
            self.cause,
        )
        dataset = filter_data(dataset, {self.groupby_column: [self.transition_string]})
        dataset = dataset.rename(columns={"value": self.renamed_column})
        return dataset


class PersonTime(SimDataFormatter):
    """Formatter for simulation data that contains person time."""

    def __init__(self, cause: str, state=None) -> None:
        self.type = "person_time"
        self.cause = cause
        self.data_key = f"{self.type}_{self.cause}"
        self.state = state if state else "total"
        self.groupby_column = f"sub_entity"
        self.renamed_column = f"{self.state}_{self.type}"

    def format_dataset(self, dataset: SimOutputData) -> SimOutputData:
        """Clean up redundant columns, filter for the state, and rename the value column."""
        dataset = drop_redundant_index(
            dataset,
            "measure",
            self.type,
        )
        dataset = drop_redundant_index(
            dataset,
            "entity_type",
            "cause",
        )
        dataset = drop_redundant_index(
            dataset,
            "entity",
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


def drop_redundant_index(
    data: pd.DataFrame, idx_column_name: str, idx_column_value: str
) -> None:
    """Validate that a DataFrame column is singular-valued, then drop it from the index."""
    if not (data.index.get_level_values(idx_column_name) == idx_column_value).all():
        raise ValueError(
            f"Cause {data.index.get_level_values(idx_column_name).unique()} in data does not match expected cause {idx_column_name}"
        )
    data = data.droplevel([idx_column_name])
    return data
