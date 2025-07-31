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


@pytest.fixture
def weights():
    return pd.DataFrame(
        {
            "value": [1, 2, 3, 4],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("location_1", "sex_1", "age_1"),
                ("location_1", "sex_1", "age_2"),
                ("location_2", "sex_1", "age_1"),
                ("location_2", "sex_1", "age_2"),
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
    # No aggregation: both data and weights keep original shape
    # aggregate_data * aggregate_weights = elementwise multiplication of full DataFrames
    # Then .sum() sums ALL values in the resulting DataFrame
    # Numerator: (10*9 + 20*19 + 30*29 + 40*39 + 50*49 + 60*59 + 70*69 + 80*79) = 20040
    # Denominator: (9+19+29+39+49+59+69+79) = 352
    # Result: 20040 / 352 = 56.931818
    expected_no_strat = pd.Series([20040 / 352], index=pd.Index(["value"]))
    assert result_no_strat.equals(expected_no_strat)

    # Test with location stratification
    result_location = weighted_average(filter_test_data, weights, ["location"])
    # Data aggregated by location: location_1: 100, location_2: 260
    # Weights aggregated by location: location_1: 96, location_2: 256
    # aggregate_data * aggregate_weights = [100*96, 260*256] = [9600, 66560]
    # Numerator: (9600 + 66560) = 76160
    # Denominator: (96 + 256) = 352
    # Result: 76160 / 352 = 216.36363636363637
    expected_location = pd.Series([216.36363636363637], index=pd.Index(["value"]))
    assert result_location.equals(expected_location)

    # Test with location and sex stratification
    result_location_sex = weighted_average(filter_test_data, weights, ["location", "sex"])
    # Data by location/sex: (location_1,sex_1):30, (location_1,sex_2):70, (location_2,sex_1):110, (location_2,sex_2):150
    # Weights by location/sex: (location_1,sex_1):28, (location_1,sex_2):68, (location_2,sex_1):108, (location_2,sex_2):148
    # aggregate_data * aggregate_weights = [30*28, 70*68, 110*108, 150*148] = [840, 4760, 11880, 22200]
    # Numerator: (840 + 4760 + 11880 + 22200) = 39680
    # Denominator: (28 + 68 + 108 + 148) = 352
    # Result: 39680 / 352 = 112.72727272727273
    expected_location_sex = pd.Series([112.72727272727273], index=pd.Index(["value"]))
    assert result_location_sex.equals(expected_location_sex)
    # Denominator: 28 + 68 + 108 + 148 = 352
    # Result: 39680 / 352 = 112.72727272727273
    expected_location_sex = pd.Series([112.72727272727273], index=pd.Index(["value"]))
    assert result_location_sex.equals(expected_location_sex)


def test_weighted_average_subset_index_levels(filter_test_data: pd.DataFrame) -> None:
    """Test weighted average with different index levels."""
    weights = filter_test_data.copy() - 1
    # Create a new DataFrame with a different index level
    weights.index = weights.index.droplevel("sex")

    # Test the weighted average calculation
    result = weighted_average(filter_test_data, weights, ["location"])

    # Manually calculate expected result
    # Data aggregated by location: location_1: 100, location_2: 260
    # Weights aggregated by location: location_1: 96, location_2: 256 (same as before since weights were duplicated when sex level dropped)
    # Numerator: (100 * 96) + (260 * 256) = 9600 + 66560 = 76160
    # Denominator: 96 + 256 = 352
    # Result: 76160 / 352 = 216.36363636363637
    expected = pd.Series([216.36363636363637], index=pd.Index(["value"]))

    assert result.equals(expected)
