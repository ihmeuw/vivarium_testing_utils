from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
import yaml
from layered_config_tree import ConfigurationKeyError, LayeredConfigTree
from vivarium import Artifact


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


class DataLoader:
    def __init__(self, sim_output_dir: str, cache_size_mb: int = 1000):
        self._sim_output_dir = Path(sim_output_dir)
        self._results_dir = self._sim_output_dir / "results"
        self._cache_size_mb = cache_size_mb
        self._raw_datasets = LayeredConfigTree(
            {data_source: {} for data_source in DataSource}
        )
        self._loader_mapping = {
            DataSource.SIM: self._load_from_sim,
            DataSource.GBD: self._load_from_gbd,
            DataSource.ARTIFACT: self._load_from_artifact,
            DataSource.CUSTOM: self._raise_custom_data_error,
        }
        self._metadata = LayeredConfigTree()
        self._artifact = self._load_artifact(self._sim_output_dir)

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets in the given simulation output directory.
        Only return the filename, not the extension."""
        return [str(f.stem) for f in self._results_dir.glob("*.parquet")]

    def get_artifact_keys(self) -> list[str]:
        return self._artifact.keys

    def get_dataset(self, dataset_key: str, source: DataSource) -> pd.DataFrame:
        """Return the dataset from the cache if it exists, otherwise load it from the source."""
        try:
            return self._raw_datasets[source][dataset_key].copy()
        except ConfigurationKeyError:
            dataset = self._load_from_source(dataset_key, source)
            self._add_to_cache(dataset_key, source, dataset)
            return dataset

    def _load_from_source(self, dataset_key: str, source: DataSource) -> None:
        """Load the data from the given source via the loader mapping."""
        return self._loader_mapping[source](dataset_key)

    def _add_to_cache(self, dataset_key: str, source: DataSource, data: pd.DataFrame) -> None:
        """Update the raw_datasets cache with the given data."""
        if dataset_key in self._raw_datasets.get(source, {}):
            raise ValueError(f"Dataset {dataset_key} already exists in the cache.")
        self._raw_datasets.update({source: {dataset_key: data.copy()}})

    def _load_from_sim(self, dataset_key: str) -> pd.DataFrame:
        """Load the data from the simulation output directory and set the non-value columns as indices."""
        sim_data = pd.read_parquet(self._results_dir / f"{dataset_key}.parquet")
        if "value" not in sim_data.columns:
            raise ValueError(f"{dataset_key}.parquet requires a column labeled 'value'.")
        sim_data = sim_data.set_index(sim_data.columns.drop("value").tolist())
        return sim_data

    @staticmethod
    def _load_artifact(results_dir: str) -> Artifact:
        model_spec_path = Path(results_dir) / "model_specification.yaml"
        artifact_path = yaml.safe_load(model_spec_path.open("r"))["configuration"][
            "input_data"
        ]["artifact_path"]
        return Artifact(artifact_path)

    def _load_from_artifact(self, dataset_key: str) -> pd.DataFrame:
        data = self._artifact.load(dataset_key)
        self._artifact.clear_cache()
        return data

    def _load_from_gbd(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError

    def _raise_custom_data_error(self, dataset_key: str) -> pd.DataFrame:
        raise ValueError(
            f"No custom dataset found for {dataset_key}."
            "Please upload a dataset using ValidationContext.upload_custom_data."
        )
