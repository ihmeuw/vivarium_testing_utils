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
    """Validate that the column is singular-valued, then drop it"""
    if not (data.index.get_level_values(idx_column_name) == idx_column_value).all():
        raise ValueError(
            f"Cause {data.index.get_level_values(idx_column_name).unique()} in data does not match expected cause {idx_column_name}"
        )
    data = data.droplevel([idx_column_name])
    return data
