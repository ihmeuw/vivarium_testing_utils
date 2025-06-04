from abc import ABC, abstractmethod
from typing import Any, Collection, Literal

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    get_singular_indices,
    marginalize,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.visualization.dataframe_utils import (
    format_draws_sample,
    format_metadata,
)


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
    def get_diff(
        self,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "percent_error",
        ascending: bool = False,
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
        stratifications: Collection[str] = (),
    ):
        self.measure: RatioMeasure = measure
        self.test_source = test_source
        self.test_datasets = (
            test_datasets  # Dictionary with 'numerator' and 'denominator' keys
        )
        self.reference_source = reference_source
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
        test_info = self._get_metadata_from_dataset("test")
        reference_info = self._get_metadata_from_dataset("reference")
        return format_metadata(measure_key, test_info, reference_info)

    def get_diff(
        self,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "percent_error",
        ascending: bool = False,
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
        Returns:
        --------
        A DataFrame of the comparison data.
        """
        if stratifications:
            # TODO: MIC-6075
            raise NotImplementedError(
                "Non-default stratifications require rate aggregations, which are not currently supported."
            )

        test_data, reference_data = self._align_datasets()

        test_data = test_data.rename(columns={"value": "test_rate"}).dropna()
        reference_data = reference_data.rename(columns={"value": "reference_rate"}).dropna()

        merged_data = pd.merge(test_data, reference_data, left_index=True, right_index=True)
        merged_data["percent_error"] = (
            (merged_data["test_rate"] - merged_data["reference_rate"])
            / merged_data["reference_rate"]
        ) * 100
        sort_key = abs if sort_by == "percent_error" else None
        sorted_data = merged_data.sort_values(
            by=sort_by,
            key=sort_key,
            ascending=ascending,
        )
        if num_rows == "all":
            return sorted_data
        else:
            return sorted_data.head(n=num_rows)

    def verify(self, stratifications: Collection[str] = ()):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def _get_metadata_from_dataset(
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
            # For test data, use the combined DataFrame for metadata
            if isinstance(self.test_datasets, dict):
                # Combine numerator and denominator for metadata display
                numerator_renamed = self.test_datasets["numerator"].rename(
                    columns={"value": self.measure.numerator.name}
                )
                denominator_renamed = self.test_datasets["denominator"].rename(
                    columns={"value": self.measure.denominator.name}
                )
                dataframe = pd.concat([numerator_renamed, denominator_renamed], axis=1)
            else:
                dataframe = self.test_datasets
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
        if "input_draw" in dataframe.index.names:
            num_draws = dataframe.index.get_level_values("input_draw").nunique()
            data_info["num_draws"] = f"{num_draws:,}"
            draw_values = list(dataframe.index.get_level_values("input_draw").unique())
            data_info["input_draws"] = format_draws_sample(draw_values)

        # Seeds information
        if "random_seed" in dataframe.index.names:
            num_seeds = dataframe.index.get_level_values("random_seed").nunique()
            data_info["num_seeds"] = f"{num_seeds:,}"

        return data_info

    def _align_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Resolve any index mismatches between the test and reference datasets."""
        # Get union of test data index names
        combined_test_index_names = set(self.test_datasets["numerator"].index.names).union(
            set(self.test_datasets["denominator"].index.names)
        )

        reference_data = self.reference_data.copy()

        # Drop any singular index levels from the reference data if they are not in the test data.
        # If any ref-only index level is not singular, raise an error.
        ref_only_indexes = [
            index
            for index in self.reference_data.index.names
            if index not in combined_test_index_names
        ]
        redundant_ref_indexes = get_singular_indices(self.reference_data).keys()
        for index_name in ref_only_indexes:
            if not index_name in redundant_ref_indexes:
                # TODO: MIC-6075
                raise ValueError(
                    f"Reference data has non-trivial index {index_name} that is not in the test data."
                    "We cannot currently marginalize over this index."
                )
            else:
                reference_data = reference_data.droplevel(index_name)

        # Apply marginalization to the converted test data
        test_only_indexes = [
            index
            for index in combined_test_index_names
            if index not in reference_data.index.names
        ]
        stratified_numerator = marginalize(self.test_datasets["numerator"], test_only_indexes)
        stratified_denominator = marginalize(
            self.test_datasets["denominator"], test_only_indexes
        )

        converted_test_data = self.measure.get_measure_data_from_ratio(
            stratified_numerator, stratified_denominator
        )
        return converted_test_data, reference_data
