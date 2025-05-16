from typing import Any

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.visualization.dataframe_utils import (
    format_metadata_pandas,
    get_metadata_from_dataset,
)

MEASURE_KEY = "test_measure"


@pytest.fixture
def test_info() -> dict[str, Any]:
    """Info dictionary with draws."""
    return {
        "source": "sim",
        "index_columns": "year, sex, age, input_draw",
        "size": "100 rows × 5 columns",
        "num_draws": "10",
        "input_draws": "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
    }


@pytest.fixture
def reference_info() -> dict[str, Any]:
    """Info dictionary without draws."""
    return {
        "source": "artifact",
        "index_columns": "year, sex, age",
        "size": "50 rows × 3 columns",
        "num_draws": "0",
        "input_draws": "[]",
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """A multi-index dataframe with input_draw and random_seed."""
    index = pd.MultiIndex.from_product(
        [
            [2019, 2020],
            ["male", "female"],
            [0, 1, 2],
            [0, 1],
        ],
        names=["year", "sex", "input_draw", "random_seed"],
    )
    return pd.DataFrame({"value": range(len(index))}, index=index)


@pytest.fixture
def sample_dataframe_no_draws() -> pd.DataFrame:
    """A sample dataframe without input_draw and random_seed."""
    index = pd.MultiIndex.from_product(
        [
            [2019, 2020],
            ["male", "female"],
            [30, 40, 50],  # age
        ],
        names=["year", "sex", "age"],
    )
    return pd.DataFrame({"value": range(len(index))}, index=index)


@pytest.mark.parametrize(
    "source, expected_source",
    [
        (DataSource.SIM, "sim"),
        (DataSource.GBD, "gbd"),
        (DataSource.ARTIFACT, "artifact"),
    ],
)
def test_get_metadata_from_dataset(
    source: DataSource, expected_source: str, sample_dataframe: pd.DataFrame
) -> None:
    """Test we can extract metadata from a dataframe with draws."""

    result = get_metadata_from_dataset(source, sample_dataframe)

    assert result["source"] == expected_source
    assert result["index_columns"] == "year, sex, input_draw, random_seed"
    assert (
        result["size"] == "24 rows × 1 columns"
    )  # 2 years * 2 sexes * 3 draws * 2 seeds = 24 rows, 1 column
    assert result["num_draws"] == "3"
    assert result["input_draws"] == "[0, 1, 2]"
    assert result["num_seeds"] == "2"


def test_get_metadata_from_dataset_no_draws(sample_dataframe_no_draws: pd.DataFrame) -> None:
    """Test we can extract metadata from a dataframe with draws."""

    result = get_metadata_from_dataset(DataSource.GBD, sample_dataframe_no_draws)

    assert result["source"] == "gbd"
    assert result["index_columns"] == "year, sex, age"
    assert (
        result["size"] == "12 rows × 1 columns"
    )  # 2 years * 2 sexes * 3 ages = 12 rows, 1 column
    assert result["num_draws"] == "0"
    assert result["input_draws"] == "[]"  # Now includes empty input_draws


def test_format_metadata_pandas_basic(
    test_info: dict[str, Any], reference_info: dict[str, Any]
) -> None:
    """Test we can format metadata into a pandas DataFrame."""
    df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    expected_metadata = [
        ("Measure Key", "test_measure", "test_measure"),
        ("Source", "sim", "artifact"),
        ("Index Columns", "year, sex, age, input_draw", "year, sex, age"),
        ("Size", "100 rows × 5 columns", "50 rows × 3 columns"),
        ("Num Draws", "10", "0"),
        ("Input Draws", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", "[]"),
    ]

    assert df.index.name == "Property"
    assert df.shape == (6, 2)
    assert df.columns.tolist() == ["Test Data", "Reference Data"]
    for property_name, test_value, reference_value in expected_metadata:
        assert df.loc[property_name]["Test Data"] == test_value
        assert df.loc[property_name]["Reference Data"] == reference_value


def test_format_metadata_pandas_missing_fields() -> None:
    """Test we can format metadata into a pandas DataFrame wtih missing fields."""
    test_info = {"source": "sim"}
    reference_info = {"source": "artifact"}

    df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    for i in range(2, 6):
        assert df["Test Data"][i] == "N/A"
        assert df["Reference Data"][i] == "N/A"


def test_format_metadata_pandas_many_draws() -> None:
    """Test the formatted draw sample for many draws."""
    test_info = {
        "source": "sim",
        "index_columns": "year, sex, age",
        "size": "1,000 rows × 5 columns",
        "num_draws": "100",
        "input_draws": "[0, 1, 2, 3, 4] ... [95, 96, 97, 98, 99]",  # Pre-formatted draws
    }
    reference_info = {
        "source": "gbd",
        "index_columns": "year, sex, age",
        "size": "50 rows × 3 columns",
        "num_draws": "0",
        "input_draws": "[]",
    }

    df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    # Check the formatted draw sample for many draws
    draw_sample = df["Test Data"][5]
    assert draw_sample == "[0, 1, 2, 3, 4] ... [95, 96, 97, 98, 99]"


def test_format_metadata_pandas_with_empty_draws(
    test_info: dict[str, Any], reference_info: dict[str, Any]
) -> None:
    """Test we can format metadata into a pandas DataFrame with no draws."""
    # Set num_draws to 0 in test_info
    test_info["num_draws"] = "0"
    test_info["input_draws"] = "[]"  # Empty input_draws already formatted
    reference_info["num_draws"] = "0"
    reference_info["input_draws"] = "[]"  # Empty input_draws already formatted
    df = format_metadata_pandas(MEASURE_KEY, test_info, reference_info)

    # Check draw sample with empty draws
    assert df["Test Data"][5] == "[]"
    assert df["Reference Data"][5] == "[]"
