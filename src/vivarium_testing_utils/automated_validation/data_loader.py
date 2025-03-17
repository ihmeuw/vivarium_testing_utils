import pandas as pd
from layered_config_tree import LayeredConfigTree


class DataLoader:
    def __init__(self, results_dir: str, cache_size_mb: int = 1000):
        self.results_dir = results_dir
        self.cache_size_mb = cache_size_mb
        self.raw_datasets = LayeredConfigTree()
        self.metadata = LayeredConfigTree()
        self.artifact = None  # Just stubbing this out for now

    def load_data(self, dataset_key: str, data_type: str) -> None:
        raise NotImplementedError

    def get_dataset(self, dataset_key: str, data_type: str) -> pd.DataFrame:
        raise NotImplementedError

    def sim_outputs(self) -> list[str]:
        raise NotImplementedError

    def artifact_keys(self) -> list[str]:
        raise NotImplementedError

    def load_from_sim(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError

    def load_from_artifact(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError

    def load_from_gbd(self, dataset_key: str) -> pd.DataFrame:
        raise NotImplementedError
