from abc import ABC
from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.constants import (
    DRAW_INDEX,
    SEED_INDEX,
    DataSource,
)
from vivarium_testing_utils.automated_validation.data_loader import DataLoader
from vivarium_testing_utils.automated_validation.data_transformation import (
    age_groups,
    calculations,
    utils,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.visualization import dataframe_utils


class MeasureDataBundle(ABC):
    measure: Measure
    source: DataSource
    data_loader: DataLoader
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
        self.data_loader = data_loader
        self.scenarios = scenarios if scenarios is not None else {}
        self.datasets = self._get_formatted_datasets(age_group_df)

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

    def _get_formatted_datasets(
        self, age_group_data: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        """Formats measure datasets depending on the source."""
        raw_datasets = self.data_loader._get_raw_data_from_source(
            self.measure.get_required_datasets(self.source), self.source
        )
        if self.source == DataSource.SIM:
            datasets = self.measure.get_ratio_datasets_from_sim(
                **raw_datasets,
            )
        elif self.source == DataSource.ARTIFACT:
            data = self.measure.get_measure_data(self.source, **raw_datasets)
            raw_weights = self.data_loader._get_raw_data_from_source(
                self.measure.rate_aggregation_weights.weight_keys, self.source
            )
            weights = self.measure.rate_aggregation_weights.get_weights(**raw_weights)
            datasets = {"data": data, "weights": weights}
        elif self.source == DataSource.GBD:
            raise NotImplementedError
        elif self.source == DataSource.CUSTOM:
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported data source: {self.source}")

        datasets = {
            dataset_name: age_groups.format_dataframe_from_age_bin_df(dataset, age_group_data)
            for dataset_name, dataset in datasets.items()
        }

        return datasets
