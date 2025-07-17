import pandas as pd
from abc import ABC
from vivarium_testing_utils.automated_validation.constants import DataSource
from vivarium_testing_utils.automated_validation.data_loader import DataLoader
from vivarium_testing_utils.automated_validation.data_transformation import (
    calculations,
    age_groups,
    utils,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)


class MeasureDataBundle(ABC):
    measure: Measure
    source: DataSource
    datasets: dict[str, pd.DataFrame]
    scenarios: dict[str, str] | None


class RatioMeasureDataBundle:

    def __init__(
        self,
        measure: RatioMeasure,
        source: DataSource,
        data_loader: DataLoader,
        age_group_df: pd.DataFrame,
        scenarios: dict[str, str] | None = None,
    ) -> None:
        self.measure = measure
        self.source = source
        self.scenarios = scenarios or {}
        datasets = data_loader.get_bulk_data(source, self.dataset_names)
        datasets = self._transform_data(datasets)
        datasets = age_groups.format_bulk_from_df(datasets, age_group_df)
        self.datasets = {
            key: calculations.filter_data(dataset, scenarios, drop_singles=False)
            for key, dataset in datasets.items()
        }

    @property
    def dataset_names(self) -> dict[str, str]:
        """Return a dictionary of required datasets for the specified source."""
        if self.source == DataSource.SIM:
            return self.measure.sim_datasets
        elif self.source == DataSource.ARTIFACT:
            return self.measure.artifact_datasets
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    def _get_raw_data_from_source(
        self, source: DataSource, data_loader: DataLoader
    ) -> dict[str, pd.DataFrame]:
        """Get the raw datasets from the given source."""
        return {
            dataset_name: data_loader.get_data(data_key, source)
            for dataset_name, data_key in self.dataset_names.items()
        }

    def _transform_data(
        self, datasets: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame] | pd.DataFrame:
        """Apply a transformation function to the data."""
        if self.source == DataSource.SIM:
            return self.measure.get_ratio_datasets_from_sim(**datasets)
        elif self.source == DataSource.ARTIFACT:
            return {"data": self.measure.get_measure_data_from_artifact(**datasets)}
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    @property
    @utils.check_io(out=SingleNumericColumn)
    def measure_data(self) -> pd.DataFrame:
        """Process data from the specified source into a format suitable for calculations."""
        if self.source == DataSource.SIM:
            return self.measure.get_measure_data_from_ratio(**self.datasets)
        elif self.source == DataSource.ARTIFACT:
            return self.datasets["data"]
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

    @staticmethod
    def _format_age_groups(
        datasets: dict[str, pd.DataFrame], age_groups_df: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Format the age groups in the datasets."""
        return {
            dataset_name: age_groups.format_dataframe_from_age_bin_df(dataset, age_groups_df)
            for dataset_name, dataset in datasets.items()
        }
