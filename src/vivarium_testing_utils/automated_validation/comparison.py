from abc import ABC, abstractmethod
from typing import Any, Collection, Literal

import numpy as np
import pandas as pd
from loguru import logger

from vivarium_testing_utils.automated_validation.bundle import RatioMeasureDataBundle
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX
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
    test_bundle: RatioMeasureDataBundle
    reference_bundle: RatioMeasureDataBundle

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
        stratifications: Collection[str] | Literal["all"] = "all",
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
        test_bundle: RatioMeasureDataBundle,
        reference_bundle: RatioMeasureDataBundle,
    ):
        self.measure: RatioMeasure = measure
        self.test_bundle = test_bundle
        self.reference_bundle = reference_bundle

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
        test_info = self.test_bundle.get_metadata()
        reference_info = self.reference_bundle.get_metadata()
        return dataframe_utils.format_metadata(measure_key, test_info, reference_info)

    def get_frame(
        self,
        stratifications: Collection[str] | Literal["all"] = "all",
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

        test_proportion_data, reference_data = self._align_datasets(
            stratifications, aggregate_draws
        )
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
        # If data doesn't have draws, return data
        if DRAW_INDEX not in data.index.names:
            logger.warning("Data does not have draws. Returning data without aggregating.")
            return data
        # If data only has draws, aggregate and cast single value to a dataframe
        if DRAW_INDEX in data.index.names and len(data.index.names) == 1:
            data = data.describe(percentiles=[0.025, 0.975])
            aggregated_data = data.T
            aggregated_data.index = pd.Index([0], name="index")
        else:
            # Get the levels to group by (everything except draws and seeds)
            group_levels = [level for level in data.index.names if level != DRAW_INDEX]
            # Group by the remaining levels and aggregate
            aggregated_data = data.groupby(group_levels, sort=False, observed=True)[
                "rate"
            ].describe(percentiles=[0.025, 0.975])

        return aggregated_data[["mean", "2.5%", "97.5%"]]

    def verify(self, stratifications: Collection[str] = ()):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def _align_datasets(
        self,
        stratifications: Collection[str] | Literal["all"] = "all",
        aggregate_draws: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Resolve any index mismatches between the test and reference datasets."""

        # Get index levels that are only in the test data.
        test_only_indexes = self.test_bundle.index_names - self.reference_bundle.index_names
        reference_only_indexes = (
            self.reference_bundle.index_names - self.test_bundle.index_names
        )
        # Don't aggregate over the scenarios, yet, because we may need them to join the datasets.
        test_indexes_to_marginalize = test_only_indexes.difference(
            tuple(self.test_bundle.scenarios.keys()), [DRAW_INDEX]
        )
        reference_indexes_to_drop = reference_only_indexes.difference(
            tuple(self.reference_bundle.scenarios.keys()), [DRAW_INDEX]
        )

        # Aggregate over indices for reference data
        if stratifications == "all":
            stratifications = [
                x
                for x in self.reference_bundle.index_names
                if x not in reference_indexes_to_drop
            ]
        aggregated_reference_data = self.reference_bundle.get_measure_data(
            stratifications=stratifications,
        )

        # If the test data has any index levels that are not in the reference data, marginalize
        # over those index levels.
        test_idx_to_marginalize = set(test_indexes_to_marginalize).union(
            set(
                [
                    idx
                    for idx in self.test_bundle.index_names
                    if idx not in stratifications and idx != DRAW_INDEX
                ]
            )
        )
        test_idx_to_stratify = self.test_bundle.index_names.difference(
            test_idx_to_marginalize
        )
        stratified_test_data = self.test_bundle.get_measure_data(
            stratifications=test_idx_to_stratify
        )

        stratified_test_data = stratified_test_data.rename(columns={"value": "rate"})
        aggregated_reference_data = aggregated_reference_data.rename(
            columns={"value": "rate"}
        )
        if aggregate_draws:
            stratified_test_data = self._aggregate_over_draws(stratified_test_data)
            aggregated_reference_data = self._aggregate_over_draws(aggregated_reference_data)
        stratified_test_data = stratified_test_data.add_prefix("test_")
        aggregated_reference_data = aggregated_reference_data.add_prefix("reference_")

        stratified_test_data, aggregated_reference_data = self._cast_across_indexes(
            stratified_test_data, aggregated_reference_data
        )

        ## At this point, the only non-common index levels should be scenarios and draws.
        return stratified_test_data, aggregated_reference_data

    def _cast_across_indexes(
        self, test_data: pd.DataFrame, reference_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align dataset indexes if stratifications is an empty list
        One dataset might have a single row, so we cast that row to match the other's length
        If both datasets are one row, they will already have the same index."""
        if len(test_data) == 1 and len(reference_data) != 1:
            test_data = pd.concat(
                [test_data] * len(reference_data), ignore_index=True
            ).set_index(reference_data.index)
        elif len(reference_data) == 1 and len(test_data) != 1:
            reference_data = pd.concat(
                [reference_data] * len(test_data), ignore_index=True
            ).set_index(test_data.index)

        return test_data, reference_data
