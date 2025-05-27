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

    def __init__(self, type: str, cause: str, filter_value: str) -> None:
        self.type = type
        self.cause = cause
        self.data_key = f"{self.type}_{self.cause}"
        self.redundant_columns = {
            "measure": self.type,
            "entity_type": "cause",
            "entity": self.cause,
        }
        self.filter_column = "sub_entity"
        self.filter_value = filter_value
        self.new_value_column_name = f"{self.filter_value}_{self.type}"

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
        super().__init__("transition_count", cause, f"{start_state}_to_{end_state}")


class PersonTime(SimDataFormatter):
    """Formatter for simulation data that contains person time."""

    def __init__(self, cause: str, state: str | None = None) -> None:
        super().__init__("person_time", cause, state or "total")


class TotalPersonTime(SimDataFormatter):
    """
    Formatter for retrieving total person time across all states in the population.

    This formatter dynamically selects a person time dataset from the available simulation
    outputs, aggregates across all states, and returns the total person time. It's useful
    for calculations that need a population-wide denominator, such as mortality rates.
    """

    def __init__(self) -> None:
        """Initialize the TotalPersonTime formatter with population-level settings."""
        super().__init__("person_time", "population", "total")
        self._dynamic_data_key = None  # Will be determined at runtime

    def format_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up redundant columns, filter for the state, and rename the value column.

        This overrides the parent method to handle datasets that may have different
        entity values than the one specified in the formatter.
        """
        # Instead of checking for entity match, get the actual entity from the data
        entity = dataset.index.get_level_values("entity").unique()[0]

        # Remove redundant columns that should be constant
        for column, value in {"measure": self.type, "entity_type": "cause"}.items():
            dataset = _drop_redundant_index(
                dataset,
                column,
                value,
            )

        # Drop the entity column (we don't need it anymore)
        dataset = dataset.droplevel(["entity"])

        # Continue with the original aggregation logic
        dataset = marginalize(dataset, [self.filter_column])
        dataset = dataset.rename(columns={"value": self.new_value_column_name})
        return dataset

    @property
    def data_key(self) -> str:
        """
        Return the data key, getting it dynamically if not already set.

        Returns
        -------
        str
            The person time dataset key to use

        Raises
        ------
        ValueError
            If no data key has been set yet via get_data_key
        """
        if self._dynamic_data_key is None:
            raise ValueError(
                "No data key has been set. Call get_data_key with a DataLoader instance first."
            )
        return self._dynamic_data_key

    @data_key.setter
    def data_key(self, value: str) -> None:
        self._dynamic_data_key = value

    def get_data_key(self, data_loader) -> str:
        """
        Dynamically determine which person time dataset to use based on available outputs.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader instance with access to simulation outputs

        Returns
        -------
        str
            The selected person time dataset key

        Raises
        ------
        ValueError
            If no person time datasets are available
        """
        all_outputs = data_loader.get_sim_outputs()
        person_time_datasets = [d for d in all_outputs if d.startswith("person_time_")]

        if not person_time_datasets:
            raise ValueError("No person time datasets available")

        self._dynamic_data_key = person_time_datasets[0]
        return self._dynamic_data_key

    @staticmethod
    def validate_person_time_consistency(data_loader, tolerance: float = 0.01) -> bool:
        """
        Validate that all person time datasets sum to approximately the same total.

        Parameters
        ----------
        data_loader : DataLoader
            The data loader instance
        tolerance : float, optional
            The fractional tolerance for differences between totals

        Returns
        -------
        bool
            True if all totals are consistent, False otherwise
        """
        from vivarium_testing_utils.automated_validation.data_loader import DataSource

        all_outputs = data_loader.get_sim_outputs()
        person_time_datasets = [d for d in all_outputs if d.startswith("person_time_")]

        if len(person_time_datasets) < 2:
            return True  # Not enough datasets to compare

        totals = []
        for dataset in person_time_datasets:
            data = data_loader.get_dataset(dataset, DataSource.SIM)
            # Marginalize across sub_entity to get total
            marginalized = marginalize(data, ["sub_entity"])
            # Sum across all remaining stratifications
            total = marginalized["value"].sum()
            totals.append(total)

        # Check if all totals are within tolerance of each other
        reference = totals[0]
        for total in totals[1:]:
            if abs(total - reference) / reference > tolerance:
                return False

        return True


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
