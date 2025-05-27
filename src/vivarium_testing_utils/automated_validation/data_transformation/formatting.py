import pandas as pd
from typing import Optional

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    filter_data,
    marginalize,
)


class SimDataFormatter:
    """A SimDataFormatter contains information about how to format particular kinds of
    simulaton data for use in a measure calculation. For example, incidence relies on
    both transition counts and person time data, which require different formatting/ operations
    on assumed columns in the simulation data."""

    def __init__(
        self, measure: str, entity_type: str, entity: str, filter_value: str
    ) -> None:
        self.measure = measure
        self.entity = entity
        self.data_key = f"{self.measure}_{self.entity}"
        self.redundant_columns = {
            "measure": self.measure,
            "entity_type": entity_type,
            "entity": self.entity,
        }
        self.filter_column = "sub_entity"
        self.filter_value = filter_value
        self.new_value_column_name = f"{self.filter_value}_{self.measure}"

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clean up redundant columns, filter for the state, and rename the value column."""
        for column, value in self.redundant_columns.items():
            dataset = _drop_redundant_index(
                dataset,
                column,
                value,
            )
        if self.filter_value == "total":
            dataset = marginalize(dataset, [self.filter_column])
        else:
            dataset = filter_data(dataset, {self.filter_column: [self.filter_value]})
        dataset = dataset.rename(columns={"value": self.new_value_column_name})
        return dataset


class TransitionCounts(SimDataFormatter):
    """Formatter for simulation data that contains transition counts."""

    def __init__(self, cause: str, start_state: str, end_state: str) -> None:
        super().__init__(
            measure="transition_count",
            entity_type="cause",
            entity=cause,
            filter_value=f"{start_state}_to_{end_state}",
        )


class PersonTime(SimDataFormatter):
    """Formatter for simulation data that contains person time."""

    def __init__(self, cause: str, state: str | None = None) -> None:
        super().__init__(
            measure="person_time",
            entity_type="cause",
            entity=cause,
            filter_value=state or "total",
        )


class TotalPersonTime(SimDataFormatter):
    """
    Formatter for retrieving total person time across all states in the population.
    """

    def __init__(self) -> None:
        """Initialize the TotalPersonTime formatter with population-level settings."""
        self.measure = "person_time"
        self.entity = "population"
        self.data_key = f"{self.measure}_{self.entity}"
        self.new_value_column_name = f"{self.entity}_{self.measure}"

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.rename(columns={"value": self.new_value_column_name})


def _drop_redundant_index(
    data: pd.DataFrame, idx_column_name: str, idx_column_value: str
) -> pd.DataFrame:
    """Validate that a DataFrame column is singular-valued, then drop it from the index."""
    # TODO: Make sure we handle this case appropriately when we
    # want to automatically add many comparisons
    if not (data.index.get_level_values(idx_column_name) == idx_column_value).all():
        raise ValueError(
            f"Cause {data.index.get_level_values(idx_column_name).unique()} in data does not match expected cause {idx_column_name}"
        )
    data = data.droplevel([idx_column_name])
    return data
