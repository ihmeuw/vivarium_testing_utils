import pandas as pd
from layered_config_tree import LayeredConfigTree


class DataLoader:

    def __init__(self, results_dir: str, cache_size_mb: int = 1000):
        self.results_dir = results_dir
        self.cache_size_mb = cache_size_mb
        self.raw_datasets = LayeredConfigTree()
        self.metadata = LayeredConfigTree()
        self.artifact = None  # Just stubbing this out for now

    def load_data(dataset_key: str) -> None:
        pass

    def get_dataset(dataset_key: str) -> pd.DataFrame:
        pass

    def sim_outputs():
        pass

    def artifact_keys():
        pass

    def load_from_sim():
        pass

    def load_from_artifact():
        pass

    def load_from_gbd():
        pass
