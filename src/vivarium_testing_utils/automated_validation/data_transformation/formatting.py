import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    filter_data,
    marginalize,
)


class SimDataFormatter:
    """A SimDataFormatter contains information about how to format particular kinds of
    simulaton data for use in a measure calculation. For example, incidence relies on
    both transition counts and person time data, which require different formatting/ operations
    on assumed columns in the simulation data."""

    def __init__(self, type: str, cause: str, filter_value: str) -> None:
        self.type = type
        self.cause = cause
        self.data_key = f"{self.type}_{self.cause}"
        self.unused_cols = [
            "measure",
            "entity_type",
            "entity",
        ]
        self.filter_column = "sub_entity"
        self.filter_value = filter_value
        self.new_value_column_name = f"{self.filter_value}_{self.type}"

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clean up redundant columns, filter for the state, and rename the value column."""
        dataset = marginalize(dataset, self.unused_cols)
        if self.filter_value == "total":
            dataset = marginalize(dataset, [self.filter_column])
        else:
            dataset = filter_data(dataset, {self.filter_column: [self.filter_value]})
        dataset = dataset.rename(columns={"value": self.new_value_column_name})
        return dataset


class TransitionCounts(SimDataFormatter):
    """Formatter for simulation data that contains transition counts."""

    def __init__(self, cause: str, start_state: str, end_state: str) -> None:
        super().__init__("transition_count", cause, f"{start_state}_to_{end_state}")


class PersonTime(SimDataFormatter):
    """Formatter for simulation data that contains person time."""

    def __init__(self, cause: str, state: str | None = None) -> None:
        super().__init__("person_time", cause, state or "total")
