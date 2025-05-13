from typing import Any

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.visualization.dataframe_utils import (
    get_metadata_from_dataset,
    format_metadata_pandas,
)

MEASURE_KEY = "test_measure"


@pytest.fixture
def test_info() -> dict[str, Any]:
    return {
        "source": "sim",
        "index_columns": ["year", "sex", "age"],
        "size": (100, 5),
        "num_draws": 10,
        "input_draw": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }


@pytest.fixture
def reference_info() -> dict[str, Any]:
    return {
        "source": "artifact",
        "index_columns": ["year", "sex", "age"],
        "size": (50, 3),
        "num_draws": 0,
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a multi-index dataframe with input_draw and random_seed."""
    index = pd.MultiIndex.from_product(
        [
            [2019, 2020],
            ["male", "female"],
            [0, 1, 2],  # input_draw
            [0, 1],  # random_seed
        ],
        names=["year", "sex", "input_draw", "random_seed"],
    )
    return pd.DataFrame({"value": range(len(index))}, index=index)


@pytest.fixture
def sample_dataframe_no_draws() -> pd.DataFrame:
    """Create a sample dataframe without input_draw and random_seed."""
    # Create a multi-index dataframe without input_draw
    index = pd.MultiIndex.from_product(
        [
            [2019, 2020],
            ["male", "female"],
            [30, 40, 50],  # age
        ],
        names=["year", "sex", "age"],
    )
    return pd.DataFrame({"value": range(len(index))}, index=index)


def test_data_info_sim_with_draws(sample_dataframe: pd.DataFrame) -> None:
    """Test the data_info function for SIM data with draws."""

    result = get_metadata_from_dataset(DataSource.SIM, sample_dataframe)

    assert result["source"] == "sim"
    assert result["index_columns"] == ["year", "sex", "input_draw", "random_seed"]
    assert result["size"] == (
        24,
        1,
    )  # 2 years * 2 sexes * 3 draws * 2 seeds = 24 rows, 1 column
    assert result["num_draws"] == 3
    assert list(result["input_draw"]) == [0, 1, 2]
    assert result["num_seeds"] == 2


def test_data_info_artifact_with_draws(sample_dataframe: pd.DataFrame) -> None:
    """Test the data_info function for ARTIFACT data with draws."""

    result = get_metadata_from_dataset(DataSource.ARTIFACT, sample_dataframe)

    assert result["source"] == "artifact"
    assert result["index_columns"] == ["year", "sex", "input_draw", "random_seed"]
    assert result["size"] == (24, 1)
    assert result["num_draws"] == 3
    assert list(result["input_draw"]) == [0, 1, 2]
    # Should not have num_seeds since source is not SIM
    assert "num_seeds" not in result


def test_data_info_no_draws(sample_dataframe_no_draws: pd.DataFrame) -> None:
    """Test the data_info function for data without draws."""

    result = get_metadata_from_dataset(DataSource.GBD, sample_dataframe_no_draws)

    assert result["source"] == "gbd"
    assert result["index_columns"] == ["year", "sex", "age"]
    assert result["size"] == (12, 1)  # 2 years * 2 sexes * 3 ages = 12 rows, 1 column
    assert result["num_draws"] == 0
    assert "input_draw" not in result
    assert "num_seeds" not in result


def test_format_metadata_pandas_basic(
    test_info: dict[str, Any], reference_info: dict[str, Any]
) -> None:
    """Test the format_metadata_pandas function with basic data."""
    styled_df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    # Check the data in the underlying DataFrame
    data = styled_df.data  # type: ignore[attr-defined]
    assert data["Property"][0] == "Measure Key"
    assert data["Test Data"][0] == "test_measure"
    assert data["Reference Data"][0] == "test_measure"

    # Check sources
    assert data["Test Data"][1] == "sim"
    assert data["Reference Data"][1] == "artifact"

    # Check index columns
    assert data["Test Data"][2] == "year, sex, age"
    assert data["Reference Data"][2] == "year, sex, age"

    # Check size
    assert data["Test Data"][3] == "100 rows × 5 columns"
    assert data["Reference Data"][3] == "50 rows × 3 columns"

    # Check num_draws
    assert data["Test Data"][4] == "10"
    assert data["Reference Data"][4] == "0"

    # Check draw sample
    assert data["Test Data"][5] == "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"


def test_format_metadata_pandas_missing_fields() -> None:

    test_info = {"source": "sim"}
    reference_info = {"source": "artifact"}

    styled_df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    # Check that default values are used for missing fields
    data = styled_df.data  # type: ignore[attr-defined]

    # Default index columns
    assert data["Test Data"][2] == ""
    assert data["Reference Data"][2] == ""

    # Default size
    assert data["Test Data"][3] == "0 rows × 0 columns"
    assert data["Reference Data"][3] == "0 rows × 0 columns"

    # Default num_draws
    assert data["Test Data"][4] == "0"
    assert data["Reference Data"][4] == "0"

    # Default draw sample
    assert data["Test Data"][5] == "[]"
    assert data["Reference Data"][5] == "[]"


def test_format_metadata_pandas_many_draws() -> None:
    test_info = {
        "source": "sim",
        "index_columns": ["year", "sex", "age"],
        "size": (1000, 5),
        "num_draws": 100,
        "input_draw": list(range(100)),
    }
    reference_info = {
        "source": "gbd",
        "index_columns": ["year", "sex", "age"],
        "size": (50, 3),
        "num_draws": 0,
    }

    styled_df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)
    data = styled_df.data  # type: ignore[attr-defined]

    # Check the formatted draw sample for many draws
    draw_sample = data["Test Data"][5]
    assert "[0, 1, 2, 3, 4]" in draw_sample
    assert "..." in draw_sample
    assert "[95, 96, 97, 98, 99]" in draw_sample


def test_format_metadata_pandas_styling(
    test_info: dict[str, Any], reference_info: dict[str, Any]
) -> None:

    styled_df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    table_styles = styled_df.table_styles  # type: ignore[attr-defined]

    # Check header styling
    header_styles = [style for style in table_styles if style.get("selector") == "th"]
    assert len(header_styles) > 0


def test_format_metadata_pandas_with_empty_draws(
    test_info: dict[str, Any], reference_info: dict[str, Any]
) -> None:
    # Set num_draws to 0 in test_info
    test_info["num_draws"] = 0
    test_info["input_draw"] = []  # No input_draws
    reference_info["num_draws"] = 0
    reference_info["input_draw"] = []  # No input_draws
    styled_df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)
    data = styled_df.data  # type: ignore[attr-defined]

    # Check draw sample with empty draws
    assert data["Test Data"][5] == "[]"
    assert data["Reference Data"][5] == "[]"
