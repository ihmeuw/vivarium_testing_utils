from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource


def test_get_sim_outputs(sim_result_dir: Path) -> None:
    """Test we have the correctly truncated sim data keys"""
    data_loader = DataLoader(sim_result_dir)
    assert set(data_loader.get_sim_outputs()) == {
        "deaths",
        "person_time_cause",
        "transition_count_cause",
    }


def test_get_dataset_bad_source(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the source is not recognized"""
    data_loader = DataLoader(sim_result_dir)
    with pytest.raises(ValueError):
        data_loader.get_dataset("deaths", "foo")


def test_get_dataset(sim_result_dir: Path) -> None:
    """Ensure that we load data from disk if needed, and don't if not."""
    data_loader = DataLoader(sim_result_dir)
    # check that we call load_from_source the first time we call get_dataset
    data_loader.load_from_source = MagicMock()
    data_loader.get_dataset("deaths", "sim"), pd.DataFrame
    data_loader.load_from_source.assert_called_once_with("deaths", DataSource.SIM)
    # check that we don't call load_from_source the second time we call get_dataset
    data_loader.load_from_source = MagicMock()
    data_loader.get_dataset("deaths", "sim"), pd.DataFrame
    data_loader.load_from_source.assert_not_called()


@pytest.mark.parametrize(
    "dataset_key, source",
    [
        ("deaths", DataSource.SIM),
    ],
)
def load_from_source(dataset_key: str, source: DataSource, sim_result_dir: Path) -> None:
    """Ensure we can sensibly load using key / source combinations"""
    data_loader = DataLoader(sim_result_dir)
    assert not data_loader.raw_datasets.get(source).get(dataset_key)
    data_loader.load_from_source(dataset_key, source)
    assert data_loader.raw_datasets.get(source).get(dataset_key)


def test_add_to_datasets(sim_result_dir: Path) -> None:
    """Ensure that we can add data to the cache"""
    df = pd.DataFrame({"baz": [1, 2, 3]})
    data_loader = DataLoader(sim_result_dir)
    data_loader.add_to_datasets("foo", "bar", df)
    assert data_loader.raw_datasets.get("bar").get("foo").equals(df)


def test_load_from_sim(sim_result_dir: Path) -> None:
    """Ensure that we can load data from the simulation output directory"""
    data_loader = DataLoader(sim_result_dir)
    person_time_cause = data_loader.load_from_sim("deaths")
    assert person_time_cause.shape == (8, 1)
    # check that value is column and rest are indices
    assert person_time_cause.index.names == [
        "measure",
        "entity_type",
        "entity",
        "sub_entity",
        "age_group",
        "sex",
        "input_draw",
        "random_seed",
    ]
    assert person_time_cause.columns == ["value"]
