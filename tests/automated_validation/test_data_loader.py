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


def test_get_dataset(sim_result_dir: Path) -> None:
    """Ensure that we load data from disk if needed, and don't if not."""
    data_loader = DataLoader(sim_result_dir)
    # check that we call load_from_source the first time we call get_dataset
    data_loader._load_from_source = MagicMock()
    data_loader.get_dataset("deaths", DataSource.SIM), pd.DataFrame
    data_loader._load_from_source.assert_called_once_with("deaths", DataSource.SIM)
    # check that we don't call load_from_source the second time we call get_dataset
    data_loader._load_from_source = MagicMock()
    data_loader.get_dataset("deaths", DataSource.SIM), pd.DataFrame
    data_loader._load_from_source.assert_not_called()
    
def test_get_dataset_custom(sim_result_dir: Path) -> None:
    """Ensure that we can load custom data"""
    data_loader = DataLoader(sim_result_dir)
    custom_data = pd.DataFrame({"foo": [1, 2, 3]})
    
    with pytest.raises(ValueError):
        data_loader.get_dataset("foo", DataSource.CUSTOM)
    data_loader._add_to_cache("foo", DataSource.CUSTOM, custom_data)

    assert data_loader.get_dataset("foo", DataSource.CUSTOM).equals(custom_data)


@pytest.mark.parametrize(
    "dataset_key, source",
    [
        ("deaths", DataSource.SIM),
        ("cause.cause.incidence_rate", DataSource.ARTIFACT),
        # Add more sources here later
    ],
)
def test__load_from_source(
    dataset_key: str, source: DataSource, sim_result_dir: Path
) -> None:
    """Ensure we can sensibly load using key / source combinations"""
    data_loader = DataLoader(sim_result_dir)
    assert not data_loader._raw_datasets.get(source).get(dataset_key)
    dataset = data_loader._load_from_source(dataset_key, source)
    assert dataset is not None


def test__add_to_cache(sim_result_dir: Path) -> None:
    """Ensure that we can add data to the cache, but not the same key twice"""
    df = pd.DataFrame({"baz": [1, 2, 3]})
    data_loader = DataLoader(sim_result_dir)
    data_loader._add_to_cache("foo", DataSource.SIM, df)
    assert data_loader._raw_datasets.get(DataSource.SIM).get("foo").equals(df)
    with pytest.raises(ValueError):
        data_loader._add_to_cache("foo", DataSource.SIM, df)


def test__load_from_sim(sim_result_dir: Path) -> None:
    """Ensure that we can load data from the simulation output directory"""
    data_loader = DataLoader(sim_result_dir)
    person_time_cause = data_loader._load_from_sim("deaths")
    assert person_time_cause.shape == (8, 1)
    # check that value is column and rest are indices
    assert set(person_time_cause.index.names) == {
        "measure",
        "entity_type",
        "entity",
        "sub_entity",
        "age_group",
        "sex",
        "input_draw",
        "random_seed",
    }
    assert set(person_time_cause.columns) == {"value"}


def test__load_artifact(sim_result_dir: Path) -> None:
    """Ensure that we can load the artifact itself"""
    artifact = DataLoader._load_artifact(sim_result_dir)
    assert set(artifact.keys) == {"metadata.keyspace", "cause.cause.incidence_rate"}


def test__load_from_artifact(sim_result_dir: Path) -> None:
    """Ensure that we can load data from the artifact directory"""
    data_loader = DataLoader(sim_result_dir)
    art_dataset = data_loader._load_from_artifact("cause.cause.incidence_rate")
    assert art_dataset.shape == (12, 5)
    # check that value is column and rest are indices
    assert set(art_dataset.index.names) == {
        "sex",
        "age_start",
        "age_end",
        "year_start",
        "year_end",
    }
    assert set(art_dataset.columns) == {"draw_0", "draw_1", "draw_2", "draw_3", "draw_4"}
