from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX
from vivarium_testing_utils.automated_validation.data_loader import (
    DataLoader,
    DataSource,
    _convert_to_total_person_time,
)
from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeSchema,
)


def test_get_sim_outputs(sim_result_dir: Path) -> None:
    """Test we have the correctly truncated sim data keys"""
    data_loader = DataLoader(sim_result_dir)
    assert set(data_loader.get_sim_outputs()) == {
        "deaths",
        "person_time_disease",
        "transition_count_disease",
        "person_time_child_stunting",
        "person_time_total",
    }


def test_get_data(sim_result_dir: Path) -> None:
    """Ensure that we load data from disk if needed, and don't if not."""
    data_loader = DataLoader(sim_result_dir)
    # check that we call load_from_source the first time we call get_data
    with patch.object(data_loader, "_load_from_source") as mock_load:
        mock_load.return_value = pd.DataFrame()  # Set appropriate return value
        result = data_loader.get_data("deaths", DataSource.SIM)
        assert isinstance(result, pd.DataFrame)
        mock_load.assert_called_once_with("deaths", DataSource.SIM)

    # Second call should use cached data
    with patch.object(data_loader, "_load_from_source") as mock_load:
        result = data_loader.get_data("deaths", DataSource.SIM)
        assert isinstance(result, pd.DataFrame)
        mock_load.assert_not_called()


def test_get_data_custom(sim_result_dir: Path) -> None:
    """Ensure that we can load custom data"""
    data_loader = DataLoader(sim_result_dir)
    custom_data = pd.DataFrame({"foo": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="No custom data found for foo."
        "Please upload data using ValidationContext.upload_custom_data.",
    ):
        data_loader.get_data("foo", DataSource.CUSTOM)
    data_loader._add_to_cache("foo", DataSource.CUSTOM, custom_data)

    assert data_loader.get_data("foo", DataSource.CUSTOM).equals(custom_data)


@pytest.mark.parametrize(
    "data_key, source",
    [
        ("deaths", DataSource.SIM),
        ("cause.disease.incidence_rate", DataSource.ARTIFACT),
        # Add more sources here later
    ],
)
def test__load_from_source(data_key: str, source: DataSource, sim_result_dir: Path) -> None:
    """Ensure we can sensibly load using key / source combinations"""
    data_loader = DataLoader(sim_result_dir)
    assert not data_loader._raw_data_cache[source].get(data_key)
    data = data_loader._load_from_source(data_key, source)
    assert data is not None


def test__add_to_cache(sim_result_dir: Path) -> None:
    """Ensure that we can add data to the cache, but not the same key twice"""
    df = pd.DataFrame({"baz": [1, 2, 3]})
    data_loader = DataLoader(sim_result_dir)
    data_loader._add_to_cache("foo", DataSource.SIM, df)
    assert data_loader._raw_data_cache[DataSource.SIM]["foo"].equals(df)
    with pytest.raises(ValueError, match="Data for foo already exist in the cache."):
        data_loader._add_to_cache("foo", DataSource.SIM, df)


def test_cache_immutable(sim_result_dir: Path) -> None:
    """Ensure that we can't mutate the cached data."""
    data_loader = DataLoader(sim_result_dir)
    source_data = pd.DataFrame({"foo": [1, 2, 3]})
    data_loader._add_to_cache("foo", DataSource.CUSTOM, source_data)
    # Mutate source data
    source_data["foo"] = 0
    cached_data = data_loader.get_data("foo", DataSource.CUSTOM)
    assert not cached_data.equals(source_data)

    # Mutate returned cache data
    cached_data["foo"] = [4, 5, 6]
    assert not data_loader.get_data("foo", DataSource.CUSTOM).equals(cached_data)


def test__load_from_sim(sim_result_dir: Path, deaths_data: pd.DataFrame) -> None:
    """Ensure that we can load data from the simulation output directory"""
    data_loader = DataLoader(sim_result_dir)
    deaths = data_loader._load_from_sim("deaths")
    assert deaths.equals(deaths_data)


def test__load_artifact(
    sim_result_dir: Path, _artifact_keys_mapper: dict[str, pd.DataFrame | str]
) -> None:
    """Ensure that we can load the artifact itself"""
    artifact = DataLoader._load_artifact(sim_result_dir)
    assert set(artifact.keys) == set(
        list(_artifact_keys_mapper.keys()) + ["metadata.keyspace"]
    )


def test__load_from_artifact(
    sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can load data from the artifact directory"""
    data_loader = DataLoader(sim_result_dir)
    art_data = data_loader._load_from_artifact("cause.disease.incidence_rate")
    assert art_data.equals(artifact_disease_incidence)
    # check that value is column and rest are indices
    assert set(art_data.index.names) == {
        "common_stratify_column",
        "other_stratify_column",
        DRAW_INDEX,
    }
    assert set(art_data.columns) == {"value"}


def test__load_nonstandard_artifact(
    sim_result_dir: Path, sample_age_schema: AgeSchema, risk_categories: dict[str, str]
) -> None:
    """Ensure that we can load non-standard data types from the artifact directory"""
    data_loader = DataLoader(sim_result_dir)
    age_bins = data_loader._load_from_artifact("population.age_bins")
    pd.testing.assert_frame_equal(
        age_bins,
        sample_age_schema.to_dataframe(),
    )
    loaded_risk_categories = data_loader._load_from_artifact(
        "risk_factor.risky_risk.categories"
    )
    assert loaded_risk_categories == risk_categories


def test__create_person_time_total(
    sim_result_dir: Path, total_person_time_data: pd.DataFrame
) -> None:
    """Test _create_person_time_total_dataset when one person time dataset exists."""
    data_loader = DataLoader(sim_result_dir)
    person_time_total = data_loader.get_data("person_time_total", DataSource.SIM)
    pd.testing.assert_frame_equal(person_time_total, total_person_time_data)


def test__create_person_time_total_dataset_no_datasets(sim_result_dir: Path) -> None:
    """Test _create_person_time_total_dataset when no person time datasets exist."""
    data_loader = DataLoader(sim_result_dir)

    with patch.object(data_loader, "get_sim_outputs") as mock_get_outputs, patch.object(
        data_loader, "upload_custom_data"
    ) as mock_upload:
        # no person time datasets
        mock_get_outputs.return_value = ["deaths", "transition_count_disease"]
        result = data_loader._create_person_time_total_dataset()
        assert result is None
        mock_upload.assert_not_called()


def test__convert_to_total_pt(person_time_data: pd.DataFrame) -> None:
    """Test _convert_to_total_pt function converts entity and sub_entity to 'total'."""
    result = _convert_to_total_person_time(person_time_data)

    # Check that entity and sub_entity columns are all 'total'
    assert all(result.reset_index()["entity"] == "total")
    assert all(result.reset_index()["sub_entity"] == "total")

    # Check that the index/ column structure is preserved
    assert result.index.names == person_time_data.index.names
    assert list(result.columns) == ["value"]

    # Check that values are preserved (marginalized and aggregated)
    expected_total_value = person_time_data["value"].sum()
    actual_total_value = result["value"].sum()
    assert actual_total_value == expected_total_value
