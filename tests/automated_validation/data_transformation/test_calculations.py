import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    aggregate_sum,
    filter_data,
    linear_combination,
    ratio,
    weighted_average,
)


@pytest.fixture
def intermediate_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [4, 5, 6, 7],
            "c": [1, 1, 0, 1],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("x", 0),
                ("x", 1),
                ("y", 0),
                ("y", 1),
            ],
            names=["group", "time"],
        ),
    )


@pytest.fixture
def filter_test_data() -> pd.DataFrame:
    """Create a DataFrame with multiple index levels for testing filter_data."""
    return pd.DataFrame(
        {
            "value": [10, 20, 30, 40, 50, 60, 70, 80],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("location_1", "sex_1", "age_1"),
                ("location_1", "sex_1", "age_2"),
                ("location_1", "sex_2", "age_1"),
                ("location_1", "sex_2", "age_2"),
                ("location_2", "sex_1", "age_1"),
                ("location_2", "sex_1", "age_2"),
                ("location_2", "sex_2", "age_1"),
                ("location_2", "sex_2", "age_2"),
            ],
            names=["location", "sex", "age"],
        ),
    )


@pytest.mark.parametrize(
    "filter_cols,drop_singles,expected_index_names,expected_values",
    [
        # Test filtering to single value with drop_singles=True (default)
        (
            {"location": "location_1"},
            True,
            ["sex", "age"],
            [10, 20, 30, 40],
        ),
        # Test filtering to single value with drop_singles=False
        (
            {"location": "location_1"},
            False,
            ["location", "sex", "age"],
            [10, 20, 30, 40],
        ),
        # Test filtering to multiple values (drop_singles should not affect this)
        (
            {"sex": ["sex_1", "sex_2"], "age": "age_1"},
            True,
            ["location", "sex"],
            [10, 30, 50, 70],
        ),
        (
            {"sex": ["sex_1", "sex_2"], "age": "age_1"},
            False,
            ["location", "sex", "age"],
            [10, 30, 50, 70],
        ),
        # Test filtering with multiple single values
        (
            {"location": "location_1", "sex": "sex_1"},
            True,
            ["age"],
            [10, 20],
        ),
        (
            {"location": "location_1", "sex": "sex_1"},
            False,
            ["location", "sex", "age"],
            [10, 20],
        ),
    ],
)
def test_filter_data(
    filter_test_data: pd.DataFrame,
    filter_cols: dict[str, str],
    drop_singles: bool,
    expected_index_names: list[str],
    expected_values: list[int | float],
) -> None:
    """Test filtering DataFrame with different drop_singles settings."""
    result = filter_data(filter_test_data, filter_cols, drop_singles=drop_singles)
    assert list(result.index.names) == expected_index_names
    assert list(result["value"]) == expected_values


def test_filter_data_empty_result(filter_test_data: pd.DataFrame) -> None:
    """Test that filter_data raises ValueError when result is empty."""
    with pytest.raises(ValueError, match="DataFrame is empty after filtering"):
        filter_data(filter_test_data, {"location": "nonexistent_location"})


def test_ratio(intermediate_data: pd.DataFrame) -> None:
    """Test taking ratio of two DataFrames with 'value' columns"""
    # Create separate numerator and denominator DataFrames
    numerator_a = pd.DataFrame(
        {"value": intermediate_data["a"]}, index=intermediate_data.index
    )
    denominator_b = pd.DataFrame(
        {"value": intermediate_data["b"]}, index=intermediate_data.index
    )
    denominator_c = pd.DataFrame(
        {"value": intermediate_data["c"]}, index=intermediate_data.index
    )

    # Test normal ratio calculation
    assert ratio(numerator_a, denominator_b).equals(
        pd.DataFrame({"value": [1 / 4, 2 / 5, 3 / 6, 4 / 7]}, index=intermediate_data.index)
    )

    # Test ratio with zero denominator
    pd.testing.assert_frame_equal(
        ratio(numerator_a, denominator_c),
        pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0]}, index=intermediate_data.index),
    )


def test_aggregate_sum(intermediate_data: pd.DataFrame) -> None:
    """Test aggregating over different combinations of value columns."""
    assert aggregate_sum(intermediate_data, ["group"]).equals(
        pd.DataFrame(
            {
                "a": [3, 7],
                "b": [9, 13],
                "c": [2, 1],
            },
            index=pd.Index(["x", "y"], name="group"),
        )
    )
    assert aggregate_sum(intermediate_data, ["time"]).equals(
        pd.DataFrame(
            {
                "a": [4, 6],
                "b": [10, 12],
                "c": [1, 2],
            },
            index=pd.Index([0, 1], name="time"),
        )
    )
    assert aggregate_sum(intermediate_data, ["group", "time"]).equals(intermediate_data)
    # test non-existent index column
    with pytest.raises(KeyError):
        aggregate_sum(intermediate_data, ["foo"])


def test_linear_combination(intermediate_data: pd.DataFrame) -> None:
    """Test linear combination of two columns in a multi-indexed DataFrame"""
    assert linear_combination(intermediate_data, 1, "a", 1, "b").equals(
        pd.DataFrame({"value": [5, 7, 9, 11]}, index=intermediate_data.index)
    )
    assert linear_combination(intermediate_data, 2, "a", -1, "b").equals(
        pd.DataFrame({"value": [-2, -1, 0, 1]}, index=intermediate_data.index)
    )
    # test non-existent column
    with pytest.raises(KeyError):
        linear_combination(intermediate_data, 1, "a", 1, "foo")


def test_aggregate_sum_preserves_string_order() -> None:
    """Test that aggregate_sum preserves the order of string index levels."""
    # Create a dataframe with string index that has a non-alphabetical order
    df = pd.DataFrame(
        {"value": [1, 2, 3, 4]},
        index=pd.Index(["c", "a", "d", "b"], name="category"),
    )

    # The result should maintain the original order
    result = aggregate_sum(df, ["category"])
    expected_order = pd.Index(["c", "a", "d", "b"], name="category")
    assert list(result.index) == list(expected_order)


def test_weighted_average(filter_test_data: pd.DataFrame) -> None:
    """Test weighted average with different stratification scenarios."""
    weights = filter_test_data.copy() - 1

    # Test with no stratifications (overall weighted average)
    result_no_strat = weighted_average(filter_test_data, weights, [])
    # Total data sum: 360, Total weights sum: 352, Weighted average: 360/1 = 360 (single value)
    expected_no_strat = pd.Series([360.0], index=pd.Index(["value"]))
    result_no_strat.equals(expected_no_strat)

    # Test with location stratification
    result_location = weighted_average(filter_test_data, weights, ["location"])
    # Data by location: location_1: 100, location_2: 260
    # Weights by location: location_1: 96, location_2: 256
    # Weighted sum: (100 * 96) + (260 * 256) = 76160, Total weights: 352
    # Result: 76160 / 352 = 216.36363636363637
    expected_location = pd.Series([216.36363636363637], index=pd.Index(["value"]))
    result_location.equals(expected_location)

    # Test with location and sex stratification
    result_location_sex = weighted_average(filter_test_data, weights, ["location", "sex"])
    # More granular calculation - each location/sex combination weighted
    # This should return the same as no stratification since we're not actually stratifying at this level
    expected_location_sex = pd.Series([360.0], index=pd.Index(["value"]))
    result_location_sex.equals(expected_location_sex)


def test_weighted_average_different_index(filter_test_data: pd.DataFrame) -> None:
    """Test weighted average with different index levels."""
    weights = filter_test_data.copy() - 1
    # Create a new DataFrame with a different index level
    weights.index = weights.index.droplevel("sex")

    # Test the weighted average calculation
    result = weighted_average(filter_test_data, weights, ["location"])

    # Manually calculate expected result
    # Data aggregated by location: location_1: 100, location_2: 260
    # Weights aggregated by location: location_1: 96, location_2: 256
    # Weighted sum: (100 * 96) + (260 * 256) = 9600 + 66560 = 76160
    # Total weights: 96 + 256 = 352
    # Expected result: 76160 / 352 = 216.36363636363637
    expected = pd.Series([216.36363636363637], index=pd.Index(["value"]))

    pd.testing.assert_series_equal(result, expected)
