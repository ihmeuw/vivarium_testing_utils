import pandas as pd
from abc import ABC
from vivarium_testing_utils.automated_validation.constants import DataSource
from vivarium_testing_utils.automated_validation.data_loader import DataLoader
from vivarium_testing_utils.automated_validation.data_transformation import (
    calculations,
    age_groups,
    utils,
)
from typing import Any
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.visualization import dataframe_utils
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
            key: calculations.filter_data(dataset, self.scenarios, drop_singles=False)
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

    def get_metadata(self) -> dict[str, Any]:
        """Organize the data information into a dictionary for display by a styled pandas DataFrame.
        Apply formatting to values that need special handling.

        Returns:
        --------
        A dictionary containing the formatted data information.

        """
        dataframe = self.measure_data
        data_info: dict[str, Any] = {}

        # Source as string
        data_info["source"] = self.source.value

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
