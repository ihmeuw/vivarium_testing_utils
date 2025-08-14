from abc import ABC, abstractmethod
from typing import Any, Collection, Literal

import numpy as np
import pandas as pd

from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation import calculations
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.visualization import dataframe_utils


class Comparison(ABC):
    """A Comparison is the basic testing unit to compare two datasets, a "test" dataset and a
    "reference" dataset. The test dataset is the one that is being validated, while the reference
    dataset is the one that is used as a benchmark. The comparison operates on a *measure* of the two datasets,
    typically a derived quantity of the test data such as incidence rate or prevalence."""

    measure: Measure
    test_source: DataSource
    test_datasets: dict[str, pd.DataFrame]
    reference_source: DataSource
    reference_data: pd.DataFrame
    reference_weights: pd.DataFrame
    test_scenarios: dict[str, str] | None
    reference_scenarios: dict[str, str] | None
    allowed_stratifications: Collection[str]

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        """A summary of the test data and reference data, including:
        - the measure key
        - source
        - index columns
        - size
        - number of draws
        - a sample of the input draws.
        """
        pass

    @abstractmethod
    def get_frame(
        self,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "",
        ascending: bool = False,
        aggregate_draws: bool = False,
    ) -> pd.DataFrame:
        """Get a DataFrame of the comparison data, with naive comparison of the test and reference.

        Parameters:
        -----------
        stratifications
            The stratifications to use for the comparison
        num_rows
            The number of rows to return. If "all", return all rows.
        sort_by
            The column to sort by. Default is "percent_error".
        ascending
            Whether to sort in ascending order. Default is False.
        aggregate_draws
            If True, aggregate over draws and seeds to show means and 95% uncertainty intervals.
        Returns:
        --------
        A DataFrame of the comparison data.
        """
        pass

    @abstractmethod
    def verify(self, stratifications: Collection[str] = ()):  # type: ignore[no-untyped-def]
        pass


class FuzzyComparison(Comparison):
    """A FuzzyComparison is a comparison that requires statistical hypothesis testing
    to determine if the distributions of the datasets are the same. We require both the numerator and
    denominator for the test data, to be able to calculate the statistical power."""

    def __init__(
        self,
        measure: RatioMeasure,
        test_source: DataSource,
        test_datasets: dict[str, pd.DataFrame],
        reference_source: DataSource,
        reference_data: pd.DataFrame,
        reference_weights: pd.DataFrame,
        test_scenarios: dict[str, str] | None = None,
        reference_scenarios: dict[str, str] | None = None,
        stratifications: Collection[str] = (),
    ):
        self.measure: RatioMeasure = measure

        self.test_source = test_source
        self.test_scenarios: dict[str, str] = test_scenarios if test_scenarios else {}
        self.test_datasets = {
            key: calculations.filter_data(dataset, self.test_scenarios, drop_singles=False)
            for key, dataset in test_datasets.items()
        }
        self.reference_source = reference_source
        self.reference_scenarios: dict[str, str] = (
            reference_scenarios if reference_scenarios else {}
        )
        self.reference_data = calculations.filter_data(
            reference_data, self.reference_scenarios, drop_singles=False
        )
        self.reference_weights = reference_weights

        if stratifications:
            # TODO: MIC-6075
            raise NotImplementedError(
                "Non-default stratifications require rate aggregations, which are not currently supported."
            )
        self.allowed_stratifications = stratifications

    @property
    def metadata(self) -> pd.DataFrame:
        """A summary of the test data and reference data, including:
        - the measure key
        - source
        - index columns
        - size
        - number of draws
        - a sample of the input draws.
        """
        measure_key = self.measure.measure_key
        test_info = self._get_metadata_from_datasets("test")
        reference_info = self._get_metadata_from_datasets("reference")
        return dataframe_utils.format_metadata(measure_key, test_info, reference_info)

    def get_frame(
        self,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "",
        ascending: bool = False,
        aggregate_draws: bool = False,
    ) -> pd.DataFrame:
        """Get a DataFrame of the comparison data, with naive comparison of the test and reference.

        Parameters:
        -----------
        stratifications
            The stratifications to use for the comparison
        num_rows
            The number of rows to return. If "all", return all rows.
        sort_by
            The column to sort by. Default for non-aggregated data is "percent_error", for aggregation default is to not sort.
        ascending
            Whether to sort in ascending order. Default is False.
        aggregate_draws
            If True, aggregate over draws to show means and 95% uncertainty intervals.
            Changes the output columns to show mean, 2.5%, and 97.5 for each dataset.
        Returns:
        --------
        A DataFrame of the comparison data.
        """

        test_proportion_data, reference_data = self._align_datasets(stratifications)
        test_proportion_data = test_proportion_data.rename(columns={"value": "rate"}).dropna()
        reference_data = reference_data.rename(columns={"value": "rate"}).dropna()

        if aggregate_draws:
            test_proportion_data = self._aggregate_over_draws(test_proportion_data)
            reference_data = self._aggregate_over_draws(reference_data)

        test_proportion_data = test_proportion_data.add_prefix("test_")
        reference_data = reference_data.add_prefix("reference_")

        merged_data = pd.merge(
            test_proportion_data, reference_data, left_index=True, right_index=True
        )

        if not aggregate_draws:
            merged_data["percent_error"] = (
                (merged_data["test_rate"] - merged_data["reference_rate"])
                / merged_data["reference_rate"]
            ) * 100

        if sort_by:
            sort_key = abs if sort_by == "percent_error" else None
            merged_data = merged_data.sort_values(
                by=sort_by,
                key=sort_key,
                ascending=ascending,
            )

        return merged_data if num_rows == "all" else merged_data.head(n=num_rows)

    def _aggregate_over_draws(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data over draws and seeds, computing mean and 95% uncertainty intervals."""
        # Get the levels to group by (everything except draws and seeds)
        group_levels = [level for level in data.index.names if level != DRAW_INDEX]
        # Group by the remaining levels and aggregate
        aggregated_data = data.groupby(group_levels, sort=False, observed=True)[
            "rate"
        ].describe(percentiles=[0.025, 0.975])

        return aggregated_data[["mean", "2.5%", "97.5%"]]

    def verify(self, stratifications: Collection[str] = ()):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def _get_metadata_from_datasets(
        self, dataset_key: Literal["test", "reference"]
    ) -> dict[str, Any]:
        """Organize the data information into a dictionary for display by a styled pandas DataFrame.
        Apply formatting to values that need special handling.

        Parameters:
        -----------
        dataset
            The dataset to get the metadata from. Either "test" or "reference".
        Returns:
        --------
        A dictionary containing the formatted data information.

        """
        if dataset_key == "test":
            source = self.test_source
            dataframe = self.measure.get_measure_data_from_ratio(**self.test_datasets)
        elif dataset_key == "reference":
            source = self.reference_source
            dataframe = self.reference_data
        else:
            raise ValueError("dataset must be either 'test' or 'reference'")

        data_info: dict[str, Any] = {}

        # Source as string
        data_info["source"] = source.value

        # Index columns as comma-separated string
        index_cols = dataframe.index.names
        data_info["index_columns"] = ", ".join(str(col) for col in index_cols)

        # Size as formatted string
        size = dataframe.shape
        data_info["size"] = f"{size[0]:,} rows × {size[1]:,} columns"

        # Draw information
        if DRAW_INDEX in dataframe.index.names:
            num_draws = dataframe.index.get_level_values(DRAW_INDEX).nunique()
            data_info["num_draws"] = f"{num_draws:,}"
            draw_values = list(dataframe.index.get_level_values(DRAW_INDEX).unique())
            data_info[DRAW_INDEX + "s"] = dataframe_utils.format_draws_sample(draw_values)

        # Seeds information
        if SEED_INDEX in dataframe.index.names:
            num_seeds = dataframe.index.get_level_values(SEED_INDEX).nunique()
            data_info["num_seeds"] = f"{num_seeds:,}"

        return data_info

    def _align_datasets(
        self, stratifications: Collection[str] = ()
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Resolve any index mismatches between the test and reference datasets."""
        # Get union of test data index names

        combined_test_index_names = {
            index_name
            for key in self.test_datasets
            for index_name in self.test_datasets[key].index.names
        }
        reference_index_names = set(self.reference_data.index.names)

        # Get index levels that are only in the test data.
        test_only_indexes = combined_test_index_names - reference_index_names
        reference_only_indexes = reference_index_names - combined_test_index_names
        # Don't aggregate over the scenarios, yet, because we may need them to join the datasets.
        test_indexes_to_marginalize = test_only_indexes.difference(
            tuple(self.test_scenarios.keys()), [DRAW_INDEX]
        )
        reference_indexes_to_drop = reference_only_indexes.difference(
            tuple(self.reference_scenarios.keys()), [DRAW_INDEX]
        )

        # If the test data has any index levels that are not in the reference data, marginalize
        # over those index levels.
        test_datasets = {
            key: calculations.marginalize(
                self.test_datasets[key], test_indexes_to_marginalize
            )
            for key in self.test_datasets
        }

        # Aggregate over indices
        # FIXME: Ref data and ref weights break because of not matching index levels.
        reference_data = self.aggregate_strata_reference(
            strata=[
                x
                for x in self.reference_data.index.names
                if x not in reference_indexes_to_drop
            ]
        )

        converted_test_data = self.measure.get_measure_data_from_ratio(**test_datasets)
        # Reference data can be a float or dataframe
        if not isinstance(reference_data, pd.DataFrame):
            reference_data = pd.DataFrame(
                {"value": reference_data * len(converted_test_data.index)},
                index=converted_test_data.index,
            )

        ## At this point, the only non-common index levels should be scenarios and draws.
        return converted_test_data, reference_data

    def aggregate_strata_reference(
        self, strata: Collection[str] = ()
    ) -> pd.DataFrame | float:
        for stratum in strata:
            if (
                stratum not in self.reference_data.index.names
                and stratum not in self.reference_weights.index.names
            ):
                raise ValueError(
                    f"Stratum '{stratum}' not found in reference data or weights."
                )
        return calculations.weighted_average(
            self.reference_data, self.reference_weights, strata
        )
