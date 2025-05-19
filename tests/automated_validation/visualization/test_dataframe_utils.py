from typing import Any

import pytest

from vivarium_testing_utils.automated_validation.visualization.dataframe_utils import (
    format_draws_sample,
    format_metadata,
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
        "input_draws": "[0, 1, 2, 3]",
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


def test_format_metadata_basic(
    test_info: dict[str, Any], reference_info: dict[str, Any]
) -> None:
    """Test we can format metadata into a pandas DataFrame."""
    df = format_metadata(MEASURE_KEY, test_info, reference_info)

    expected_metadata = [
        ("Measure Key", "test_measure", "test_measure"),
        ("Source", "sim", "artifact"),
        ("Index Columns", "year, sex, age, input_draw", "year, sex, age"),
        ("Size", "100 rows × 5 columns", "50 rows × 3 columns"),
        ("Num Draws", "10", "0"),
        ("Input Draws", "[0, 1, 2, 3]", "[]"),
    ]

    assert df.index.name == "Property"
    assert df.shape == (6, 2)
    assert df.columns.tolist() == ["Test Data", "Reference Data"]
    for property_name, test_value, reference_value in expected_metadata:
        assert df.loc[property_name]["Test Data"] == test_value
        assert df.loc[property_name]["Reference Data"] == reference_value


def test_format_metadata_missing_fields() -> None:
    """Test we can format metadata into a pandas DataFrame wtih missing fields."""
    test_info = {"source": "sim"}
    reference_info = {"source": "artifact"}

    df = format_metadata(MEASURE_KEY, test_info, reference_info)

    for i in range(2, 6):
        assert df["Test Data"][i] == "N/A"
        assert df["Reference Data"][i] == "N/A"


@pytest.mark.parametrize(
    "draws",
    [
        [0, 1, 2, 3],
        [0, 1, 2],
        [0],
        [],
    ],
)
def test_format_draws(draws: list[int]) -> None:
    """Test formatting a small number of draws."""
    # Test with a small list of draws (less than 2 * max_display)
    assert format_draws_sample(draws) == str(draws)


def test_format_draws_sample_large() -> None:
    """Test formatting a large number of draws."""
    # Test with a large list of draws (more than 2 * max_display)
    draws = list(range(20))
    result = format_draws_sample(draws)
    assert result == "[0, 1, 2] ... [17, 18, 19]"

    result = format_draws_sample(draws, max_display=5)
    assert result == "[0, 1, 2, 3, 4] ... [15, 16, 17, 18, 19]"
