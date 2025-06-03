import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeSchema,
    format_dataframe,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    aggregate_sum,
    linear_combination,
    ratio,
    resolve_age_groups,
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

    # Test mismatched indexes
    mismatched_denominator = pd.DataFrame(
        {"value": [1, 2]}, index=pd.Index(["a", "b"], name="different")
    )
    with pytest.raises(
        ValueError, match="Numerator and denominator DataFrames must have identical indexes"
    ):
        ratio(numerator_a, mismatched_denominator)

    # Test missing value column in numerator
    bad_numerator = pd.DataFrame(
        {"wrong_col": intermediate_data["a"]}, index=intermediate_data.index
    )
    with pytest.raises(ValueError, match="Numerator DataFrame must have a 'value' column"):
        ratio(bad_numerator, denominator_b)

    # Test missing value column in denominator
    bad_denominator = pd.DataFrame(
        {"wrong_col": intermediate_data["b"]}, index=intermediate_data.index
    )
    with pytest.raises(ValueError, match="Denominator DataFrame must have a 'value' column"):
        ratio(numerator_a, bad_denominator)


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


def test_resolve_age_groups(
    sample_df_with_ages: pd.DataFrame,
    sample_age_group_df: pd.DataFrame,
    person_time_data: pd.DataFrame,
) -> None:
    """Test we can reconcile age groups with the data."""
    # Ensure that if the age groups are in the data, we can format the data
    formatted_df = resolve_age_groups(sample_df_with_ages, sample_age_group_df)
    context_age_schema = AgeSchema.from_dataframe(sample_age_group_df)
    pd.testing.assert_frame_equal(
        formatted_df,
        format_dataframe(context_age_schema, sample_df_with_ages),
    )

    formatted_df = resolve_age_groups(person_time_data, sample_age_group_df)
    pd.testing.assert_frame_equal(
        formatted_df,
        person_time_data,
    )


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
