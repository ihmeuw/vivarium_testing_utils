import pandas as pd
from layered_config_tree import LayeredConfigTree, ConfigurationKeyError
from pathlib import Path


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
        return self.loader_mapping[source](dataset_key)

    def get_dataset(self, dataset_key: str, source: str) -> pd.DataFrame:
        try:
            return self.raw_datasets[source][dataset_key]
        except ConfigurationKeyError:
            dataset = self.load_from_source(dataset_key, source)
            self.add_to_datasets(dataset_key, source, dataset)
            return dataset

    def add_to_datasets(self, dataset_key: str, source: str, data: pd.DataFrame) -> None:
        self.raw_datasets.update({source: {dataset_key: data}})

    def sim_outputs(self) -> list[str]:
        # return only filenames
        return [str(f.stem) for f in self.sim_output_dir.glob("*.parquet")]

    def artifact_keys(self) -> list[str]:
        pass

    def load_from_sim(self, dataset_key: str) -> pd.DataFrame:
        return pd.read_parquet(self.sim_output_dir / f"{dataset_key}.parquet")

    def load_from_artifact(self, dataset_key: str) -> pd.DataFrame:
        pass

    def load_from_gbd(self, dataset_key: str) -> pd.DataFrame:
        pass

    def load_custom(self, dataset_key: str) -> pd.DataFrame:
        pass
