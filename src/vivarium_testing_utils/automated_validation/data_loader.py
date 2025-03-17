from pathlib import Path

import pandas as pd
from layered_config_tree import ConfigurationKeyError, LayeredConfigTree


class DataLoader:
    def __init__(self, results_dir: str, cache_size_mb: int = 1000):
        self.results_dir = Path(results_dir)
        self.sim_output_dir = self.results_dir / "results"
        self.cache_size_mb = cache_size_mb
        self.raw_datasets = LayeredConfigTree(
            {"sim": {}, "gbd": {}, "artifact": {}, "custom": {}}
        )
        self.loader_mapping = {
            "sim": self.load_from_sim,
            "gbd": self.load_from_gbd,
            "artifact": self.load_from_artifact,
            "custom": self.load_custom,
        }
        self.metadata = LayeredConfigTree()
        self.artifact = None  # Just stubbing this out for now

    def load_from_source(self, dataset_key: str, source: str) -> None:
        """Load the data from the given source via the loader mapping."""
        return self.loader_mapping[source](dataset_key)

    def get_dataset(self, dataset_key: str, source: str) -> pd.DataFrame:
        """Return the dataset from the cache if it exists, otherwise load it from the source."""
        try:
            return self.raw_datasets[source][dataset_key]
        except ConfigurationKeyError:
            dataset = self.load_from_source(dataset_key, source)
            self.add_to_datasets(dataset_key, source, dataset)
            return dataset

    def add_to_datasets(self, dataset_key: str, source: str, data: pd.DataFrame) -> None:
        """Update the raw_datasets cache with the given data."""
        self.raw_datasets.update({source: {dataset_key: data}})

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets in the given simulation output directory.
        Only return the filename, not the extension."""
        return [str(f.stem) for f in self.sim_output_dir.glob("*.parquet")]

    def get_artifact_keys(self) -> list[str]:
        raise NotImplementedError

    def load_from_sim(self, dataset_key: str) -> pd.DataFrame:
        """Load the data from the simulation output directory and set the non-value columns as indices."""
        sim_data = pd.read_parquet(self.sim_output_dir / f"{dataset_key}.parquet")
        if "value" not in sim_data.columns:
            raise ValueError(f"Value column not found in {dataset_key}.parquet")
        sim_data = sim_data.set_index(sim_data.columns.drop("value").tolist())
        return sim_data

    def load_from_artifact(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError

    def load_from_gbd(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError

    def load_custom(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError
