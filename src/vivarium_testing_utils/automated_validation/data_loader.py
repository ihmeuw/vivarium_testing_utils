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
        pass

    def get_dataset(self, dataset_key: str, data_type: str) -> pd.DataFrame:
        pass

    def sim_outputs(self) -> list[str]:
        pass

    def artifact_keys(self) -> list[str]:
        pass

    def load_from_sim(self, dataset_key: str) -> pd.DataFrame:
        pass

    def load_from_artifact(self, dataset_key: str) -> pd.DataFrame:
        pass

    def load_from_gbd(self, dataset_key: str) -> pd.DataFrame:
        pass
