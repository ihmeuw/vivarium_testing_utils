from abc import ABC, abstractmethod
from typing import Collection, Literal

import pandas as pd

from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_transformation import calculations
from vivarium_testing_utils.automated_validation.data_transformation.measurement_data import (
    MeasureDataBundle,
    RatioMeasureDataBundle,
)
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
    test_data: MeasureDataBundle
    reference_data: MeasureDataBundle
    stratifications: Collection[str]

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
        test_data: RatioMeasureDataBundle,
        reference_data: RatioMeasureDataBundle,
        stratifications: Collection[str] = (),
    ):
        self.measure: RatioMeasure = measure
        self.test_data = test_data
        self.reference_data = reference_data

        if stratifications:
            # TODO: MIC-6075
            raise NotImplementedError(
                "Non-default stratifications require rate aggregations, which are not currently supported."
            )
        self.stratifications = stratifications

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
        test_info = self.test_data.get_metadata()
        reference_info = self.reference_data.get_metadata()
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
        if stratifications:
            # TODO: MIC-6075
            raise NotImplementedError(
                "Non-default stratifications require rate aggregations, which are not currently supported."
            )

        test_proportion_data, reference_data = self._align_datasets()

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

    def _align_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Resolve any index mismatches between the test and reference datasets."""
        # Get union of test data index names
        test_datasets = self.test_data.datasets
        reference_data = self.reference_data.measure_data

        combined_test_index_names = {
            index_name
            for key in test_datasets
            for index_name in test_datasets[key].index.names
        }
        reference_index_names = set(reference_data.index.names)
        # Get index levels that are only in the test data.
        test_only_indexes = combined_test_index_names - reference_index_names
        reference_only_indexes = reference_index_names - combined_test_index_names
        # Don't aggregate over the scenarios, yet, because we may need them to join the datasets.
        test_indexes_to_marginalize = test_only_indexes.difference(
            tuple(self.test_data.scenarios.keys()), [DRAW_INDEX]
        )
        reference_indexes_to_drop = reference_only_indexes.difference(
            tuple(self.reference_data.scenarios.keys()), [DRAW_INDEX]
        )

        # If the test data has any index levels that are not in the reference data, marginalize
        # over those index levels.
        test_datasets = {
            key: calculations.marginalize(test_datasets[key], test_indexes_to_marginalize)
            for key in test_datasets
        }
        ## HACK: We would rather handle this in the RatioMeasurementData class.
        converted_test_data = self.measure.get_measure_data_from_ratio(**test_datasets)

        # Drop any singular index levels from the reference data if they are not in the test data.
        # If any ref-only index level is not singular, raise an error.
        redundant_ref_indexes = set(calculations.get_singular_indices(reference_data).keys())
        if not reference_indexes_to_drop.issubset(redundant_ref_indexes):
            # TODO: MIC-6075
            diff = reference_indexes_to_drop - redundant_ref_indexes
            raise ValueError(
                f"Reference data has non-trivial index levels {diff} that are not in the test data. "
                "We cannot currently marginalize over these index levels."
            )
        reference_data = reference_data.droplevel(list(reference_indexes_to_drop))

        ## At this point, the only non-common index levels should be scenarios and draws.
        return converted_test_data, reference_data
