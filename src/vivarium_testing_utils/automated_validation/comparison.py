from abc import ABC, abstractmethod
from typing import Collection, Literal

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    stratify,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.visualization.dataframe_utils import (
    data_info,
    format_metadata_pandas,
)


class Comparison(ABC):
    """A Comparison is the basic testing unit to compare two datasets, a "test" dataset and a
    "reference" dataset. The test dataset is the one that is being validated, while the reference
    dataset is the one that is used as a benchmark. The comparison operates on a *measure* of the two datasets,
    typically a derived quantity of the test data such as incidence rate or prevalence."""

    measure: Measure
    test_source: DataSource
    test_data: pd.DataFrame
    reference_source: DataSource
    reference_data: pd.DataFrame
    stratifications: list[str]

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def verify(self, stratifications: Collection[str] = ()):
        pass

    @abstractmethod
    def get_diff(
        self,
        stratifications: Collection[str] = (),
        num_rows: int | str = 10,
        sort_by: str = "percent_error",
        ascending: bool = False,
    ) -> pd.DataFrame:
        pass


class FuzzyComparison(Comparison):
    """A FuzzyComparison is a comparison that requires statistical hypothesis testing
    to determine if the distributions of the datasets are the same. We require both the numerator and
    denominator for the test data, to be able to calculate the statistical power."""

    def __init__(
        self,
        measure: RatioMeasure,
        test_source: DataSource,
        test_data: pd.DataFrame,
        reference_source: DataSource,
        reference_data: pd.DataFrame,
        stratifications: Collection[str] = (),
    ):
        self.measure = measure
        self.test_source = test_source
        self.test_data = test_data
        self.reference_source = reference_source
        self.reference_data = reference_data.rename(columns={"value": "reference_rate"})
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
        test_info = data_info(self.test_source, self.test_data)
        reference_info = data_info(self.reference_source, self.reference_data)
        return format_metadata_pandas(measure_key, test_info, reference_info)

    def get_diff(
        self,
        stratifications: list[str],
        num_rows: int | str = 10,
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
        converted_test_data = self.measure.get_measure_data_from_ratio(self.test_data).rename(
            columns={"value": "test_rate"}
        )
        stratified_test_data = stratify(converted_test_data, stratifications, agg="mean")
        stratified_reference_data = stratify(self.reference_data, stratifications, agg="mean")
        stratified_test_data, stratified_reference_data = align_indexes(
            [stratified_test_data, stratified_reference_data]
        )
        merged_data = pd.concat([stratified_test_data, stratified_reference_data], axis=1)
        merged_data["percent_error"] = (
            (merged_data["test_rate"] - merged_data["reference_rate"])
            / merged_data["reference_rate"]
        ) * 100
        sorted_data = merged_data.sort_values(
            by=sort_by,
            ascending=ascending,
        )
        if num_rows == "all":
            return sorted_data
        else:
            return sorted_data.head(n=num_rows)

    def verify(self, stratifications: Collection[str] = ()):
        raise NotImplementedError
