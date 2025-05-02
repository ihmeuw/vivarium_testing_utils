from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource


def test_get_sim_outputs(sim_result_dir: Path) -> None:
    """Test we have the correctly truncated sim data keys"""
    data_loader = DataLoader(sim_result_dir)
    assert set(data_loader.get_sim_outputs()) == {
        "deaths",
        "person_time_disease",
        "transition_count_disease",
    }


def test_get_dataset(sim_result_dir: Path) -> None:
    """Ensure that we load data from disk if needed, and don't if not."""
    data_loader = DataLoader(sim_result_dir)
    # check that we call load_from_source the first time we call get_dataset
    with patch.object(data_loader, "_load_from_source") as mock_load:
        mock_load.return_value = pd.DataFrame()  # Set appropriate return value
        result = data_loader.get_dataset("deaths", DataSource.SIM)
        assert isinstance(result, pd.DataFrame)
        mock_load.assert_called_once_with("deaths", DataSource.SIM)

    # Second call should use cached data
    with patch.object(data_loader, "_load_from_source") as mock_load:
        result = data_loader.get_dataset("deaths", DataSource.SIM)
        assert isinstance(result, pd.DataFrame)
        mock_load.assert_not_called()


def test_get_dataset_custom(sim_result_dir: Path) -> None:
    """Ensure that we can load custom data"""
    data_loader = DataLoader(sim_result_dir)
    custom_data = pd.DataFrame({"foo": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="No custom dataset found for foo."
        "Please upload a dataset using ValidationContext.upload_custom_data.",
    ):
        data_loader.get_dataset("foo", DataSource.CUSTOM)
    data_loader._add_to_cache("foo", DataSource.CUSTOM, custom_data)

    assert data_loader.get_dataset("foo", DataSource.CUSTOM).equals(custom_data)


@pytest.mark.parametrize(
    "dataset_key, source",
    [
        ("deaths", DataSource.SIM),
        ("cause.disease.incidence_rate", DataSource.ARTIFACT),
        # Add more sources here later
    ],
)
def test__load_from_source(
    dataset_key: str, source: DataSource, sim_result_dir: Path
) -> None:
    """Ensure we can sensibly load using key / source combinations"""
    data_loader = DataLoader(sim_result_dir)
    assert not data_loader._raw_datasets[source].get(dataset_key)
    dataset = data_loader._load_from_source(dataset_key, source)
    assert dataset is not None


def test__add_to_cache(sim_result_dir: Path) -> None:
    """Ensure that we can add data to the cache, but not the same key twice"""
    df = pd.DataFrame({"baz": [1, 2, 3]})
    data_loader = DataLoader(sim_result_dir)
    data_loader._add_to_cache("foo", DataSource.SIM, df)
    assert data_loader._raw_datasets[DataSource.SIM]["foo"].equals(df)
    with pytest.raises(ValueError, match="Dataset foo already exists in the cache."):
        data_loader._add_to_cache("foo", DataSource.SIM, df)


def test_cache_immutable(sim_result_dir: Path) -> None:
    """Ensure that we can't mutate the cached data."""
    data_loader = DataLoader(sim_result_dir)
    source_data = pd.DataFrame({"foo": [1, 2, 3]})
    data_loader._add_to_cache("foo", DataSource.CUSTOM, source_data)
    # Mutate source data
    source_data["foo"] = 0
    cached_data = data_loader.get_dataset("foo", DataSource.CUSTOM)
    assert not cached_data.equals(source_data)

    # Mutate returned cache data
    cached_data["foo"] = [4, 5, 6]
    assert not data_loader.get_dataset("foo", DataSource.CUSTOM).equals(cached_data)


def test__load_from_sim(
    sim_result_dir: Path,
) -> None:
    """Ensure that we can load data from the simulation output directory"""
    data_loader = DataLoader(sim_result_dir)
    deaths = data_loader._load_from_sim("deaths")
    assert deaths.shape == (8, 1)
    # check that value is column and rest are indices
    assert set(deaths.index.names) == {
        "measure",
        "entity_type",
        "entity",
        "sub_entity",
        "age_group",
        "sex",
        "input_draw",
        "random_seed",
    }
    assert set(deaths.columns) == {"value"}


def test__load_artifact(sim_result_dir: Path) -> None:
    """Ensure that we can load the artifact itself"""
    artifact = DataLoader._load_artifact(sim_result_dir)
    assert set(artifact.keys) == {
        "metadata.keyspace",
        "cause.disease.incidence_rate",
    }


def test__load_from_artifact(
    sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can load data from the artifact directory"""
    data_loader = DataLoader(sim_result_dir)
    art_dataset = data_loader._load_from_artifact("cause.disease.incidence_rate")
    assert art_dataset.equals(artifact_disease_incidence)
    # check that value is column and rest are indices
    assert set(art_dataset.index.names) == {
        "stratify_column",
        "input_draw",
    }
    assert set(art_dataset.columns) == {"value"}
