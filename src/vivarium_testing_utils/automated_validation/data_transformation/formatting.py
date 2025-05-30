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

    def __init__(self, measure: str, entity: str, filter_value: str) -> None:
        self.measure = measure
        self.entity = entity
        self.data_key = f"{self.measure}_{self.entity}"
        self.unused_columns = [
            "measure",
            "entity_type",
            "entity",
        ]
        self.filters = {"sub_entity": [filter_value]}
        self.filter_value = filter_value
        self.new_value_column_name = f"{self.filter_value}_{self.measure}"

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Clean up unused columns, filter for the state, and rename the value column."""
        dataset = marginalize(dataset, self.unused_columns)
        if self.filter_value == "total":
            dataset = marginalize(dataset, [*self.filters])
        else:
            dataset = filter_data(dataset, self.filters)
        dataset = dataset.rename(columns={"value": self.new_value_column_name})
        return dataset


class TransitionCounts(SimDataFormatter):
    """Formatter for simulation data that contains transition counts."""

    def __init__(self, entity: str, start_state: str, end_state: str) -> None:
        super().__init__(
            measure="transition_count",
            entity=entity,
            filter_value=f"{start_state}_to_{end_state}",
        )


class StatePersonTime(SimDataFormatter):
    """Formatter for simulation data that contains person time."""

    def __init__(self, entity: str | None = None, filter_value: str | None = None) -> None:
        super().__init__(
            measure="person_time",
            entity=entity or "total",
            filter_value=filter_value or "total",
        )


class Deaths(SimDataFormatter):
    """Formatter for simulation data that contains death counts."""

    def __init__(self, cause: str) -> None:
        """
        Initialize the Deaths formatter with cause-specific or all-cause settings.

        Parameters
        ----------
        cause : str, optional
            The specific cause of death to filter for. If None, all deaths are included.
        """

        self.measure = self.data_key = "deaths"
        self.unused_columns = ["measure", "entity_type"]
        self.filter_value = "total" if cause == "all_causes" else cause
        self.filters = {"entity": [self.filter_value], "sub_entity": [self.filter_value]}
        self.new_value_column_name = f"{self.filter_value}_{self.measure}"
