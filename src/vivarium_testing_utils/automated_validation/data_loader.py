from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
import yaml
from vivarium import Artifact

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    clean_artifact_data,
    marginalize,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.utils import check_io


class DataSource(Enum):
    SIM = "sim"
    GBD = "gbd"
    ARTIFACT = "artifact"
    CUSTOM = "custom"

    @classmethod
    def from_str(cls, source: str) -> DataSource:
        try:
            return cls(source)
        except ValueError:
            raise ValueError(f"Source {source} not recognized. Must be one of {DataSource}")


NONSTANDARD_ARTIFACT_KEYS = {"population.age_bins"}


class DataLoader:
    def __init__(self, sim_output_dir: Path, cache_size_mb: int = 1000):
        self._sim_output_dir = sim_output_dir
        self._cache_size_mb = cache_size_mb

        self._results_dir = self._sim_output_dir / "results"
        self._raw_datasets: dict[DataSource, dict[str, pd.DataFrame]] = {
            data_source: {} for data_source in DataSource
        }
        self._loader_mapping = {
            DataSource.SIM: self._load_from_sim,
            DataSource.GBD: self._load_from_gbd,
            DataSource.ARTIFACT: self._load_from_artifact,
        }
        self._artifact = self._load_artifact(self._sim_output_dir)

        # Initialize derived dataset person time total
        person_time_total = self._create_person_time_total_dataset()
        if person_time_total is not None:
            self._add_to_cache(
                dataset_key="person_time_total", data=person_time_total, source=DataSource.SIM
            )

    def _create_person_time_total_dataset(self) -> pd.DataFrame | None:
        """
        Create a derived dataset that aggregates total person time across all causes.

        This dataset can be used as a denominator for population-level measures like
        mortality rates.
        """
        all_outputs = self.get_sim_outputs()
        person_time_keys = [d for d in all_outputs if d.startswith("person_time_")]

        if not person_time_keys:
            return None  # No person time datasets to aggregate

        totals = []
        person_time_datasets = []
        for dataset_key in person_time_keys:
            data = self.get_dataset(dataset_key, DataSource.SIM)
            data = _convert_to_total_person_time(data)
            # Sum across all remaining stratifications
            total = data["value"].sum()
            totals.append(total)
            person_time_datasets.append(data)

        # get dataset with largest total
        largest_total_dataset = person_time_datasets[totals.index(max(totals))]

        return largest_total_dataset

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets in the given simulation output directory.
        Only return the filename, not the extension."""
        return list(
            set(str(f.stem) for f in self._results_dir.glob("*.parquet"))
            | set(self._raw_datasets[DataSource.SIM].keys())
        )

    def get_artifact_keys(self) -> list[str]:
        return self._artifact.keys

    def get_dataset(self, dataset_key: str, source: DataSource) -> pd.DataFrame:
        """Return the dataset from the cache if it exists, otherwise load it from the source."""
        try:
            return self._raw_datasets[source][dataset_key].copy()
        except KeyError:
            if source == DataSource.CUSTOM:
                raise ValueError(
                    f"No custom dataset found for {dataset_key}."
                    "Please upload a dataset using ValidationContext.upload_custom_data."
                )
            dataset = self._load_from_source(dataset_key, source)
            self._add_to_cache(dataset_key, source, dataset)
            return dataset

    def upload_custom_data(self, dataset_key: str, data: pd.DataFrame) -> None:
        self._add_to_cache(dataset_key, DataSource.CUSTOM, data)

    def _load_from_source(self, dataset_key: str, source: DataSource) -> pd.DataFrame:
        """Load the data from the given source via the loader mapping."""
        if source == DataSource.ARTIFACT and (
            dataset_key in NONSTANDARD_ARTIFACT_KEYS or dataset_key.endswith(".categories")
        ):
            # Load nonstandard artifact keys from the artifact
            return self._load_nonstandard_artifact(dataset_key)
        return self._loader_mapping[source](dataset_key)

    def _add_to_cache(self, dataset_key: str, source: DataSource, data: pd.DataFrame) -> None:
        """Update the raw_datasets cache with the given data."""
        if dataset_key in self._raw_datasets.get(source, {}):
            raise ValueError(f"Dataset {dataset_key} already exists in the cache.")
        self._raw_datasets[source].update({dataset_key: data.copy()})

    @check_io(out=SimOutputData)
    def _load_from_sim(self, dataset_key: str) -> pd.DataFrame:
        """Load the data from the simulation output directory and set the non-value columns as indices."""
        sim_data = pd.read_parquet(self._results_dir / f"{dataset_key}.parquet")
        if "value" not in sim_data.columns:
            raise ValueError(f"{dataset_key}.parquet requires a column labeled 'value'.")
        multi_index_df = sim_data.set_index(sim_data.columns.drop("value").tolist())
        # ensure index levels are in order ["measure", "entity_type", "entity", "sub_entity"]
        # and then whatever else is in the index
        REQUIRED_INDEX_LEVELS = [
            "measure",
            "entity_type",
            "entity",
            "sub_entity",
        ]
        multi_index_df = multi_index_df.reorder_levels(
            [level for level in REQUIRED_INDEX_LEVELS]
            + [
                level
                for level in multi_index_df.index.names
                if level not in REQUIRED_INDEX_LEVELS
            ]
        )
        return multi_index_df

    @staticmethod
    def _load_artifact(results_dir: Path) -> Artifact:
        model_spec_path = results_dir / "model_specification.yaml"
        artifact_path = yaml.safe_load(model_spec_path.open("r"))["configuration"][
            "input_data"
        ]["artifact_path"]
        return Artifact(artifact_path)

    def _load_nonstandard_artifact(self, dataset_key: str) -> pd.DataFrame:
        """Load artifact data for nonstandard (e.g. not draw or single numeric) keys."""
        data: pd.DataFrame = self._artifact.load(dataset_key)
        self._artifact.clear_cache()
        return data

    @check_io(out=SingleNumericColumn)
    def _load_from_artifact(self, dataset_key: str) -> pd.DataFrame:
        """Load data directly from artifact, assuming correctly formatted data."""
        data: pd.DataFrame = self._artifact.load(dataset_key)
        self._artifact.clear_cache()
        return clean_artifact_data(dataset_key, data)

    def _load_from_gbd(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError


#################
# Helper Methods#
#################


def _convert_to_total_person_time(data: pd.DataFrame) -> pd.DataFrame:
    old_index_names = data.index.names
    data = marginalize(data, ["entity_type", "entity", "sub_entity"])
    data["entity_type"] = "none"
    data["entity"] = "total"
    data["sub_entity"] = "total"
    # Reconstruct the index with the same column order as before
    data = data.reset_index().set_index(old_index_names)
    return data
