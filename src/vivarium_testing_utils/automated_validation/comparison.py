from abc import ABC, abstractmethod
from typing import Collection

import pandas as pd
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    stratify,
    align_indexes,
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

    @abstractmethod
    def verify(self, stratifications: Collection[str] = ()):
        pass

    @abstractmethod
    def summarize(self, stratifications: Collection[str] = ()):
        pass

    @abstractmethod
    def heads(self, stratifications: Collection[str] = ()):
        pass


class FuzzyComparison(Comparison):
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
        self.reference_data = reference_data.rename(columns={"value": "Reference Rate"})
        self.stratifications = stratifications

    def verify(self, stratifications: Collection[str] = ()):
        raise NotImplementedError

    def summarize(self):
        measure_key = self.measure.measure_key
        test_info = self._data_info(self.test_source, self.test_data)
        reference_info = self._data_info(self.reference_source, self.reference_data)
        return {
            "measure_key": measure_key,
            "test_source": test_info,
            "reference_source": reference_info,
        }

    def heads(
        self,
        stratifications: list[str],
        num_rows: int = 10,
        sort_by: str = "Percent Error",
        ascending: bool = False,
    ):
        aligned_ratio_data, aligned_reference_data = align_indexes(
            [self.test_data, self.reference_data],
        )
        converted_test_data = self.measure.get_measure_data_from_ratio(
            aligned_ratio_data
        ).rename(columns={"value": "Test Rate"})
        converted_test_data = stratify(
            converted_test_data,
            stratifications,
        )
        converted_reference_data = stratify(aligned_reference_data, stratifications)
        merged_data = pd.concat([converted_test_data, converted_reference_data], axis=1)
        merged_data["Percent Error"] = (
            (merged_data["Test Rate"] - merged_data["Reference Rate"])
            / merged_data["Reference Rate"]
        ) * 100
        return merged_data.sort_values(
            by=sort_by,
            ascending=ascending,
        ).head(num_rows)

    def _data_info(self, source: DataSource, dataframe: pd.DataFrame) -> dict[str, str]:
        """Return a dictionary of the data source and the dataframe."""
        data_info: dict[str, str] = {}
        data_info["source"] = source.value
        data_info["index_columns"] = dataframe.index.names
        data_info["size"] = dataframe.shape
        if "input_draw" in dataframe.index.names:
            data_info["num_draws"] = dataframe.index.get_level_values("input_draw").nunique()
            data_info["input_draw"] = dataframe.index.get_level_values("input_draw").unique()
        else:
            data_info["num_draws"] = 0
        if source == DataSource.SIM:
            data_info["num_seeds"] = dataframe.index.get_level_values("random_seed").nunique()
        return data_info
