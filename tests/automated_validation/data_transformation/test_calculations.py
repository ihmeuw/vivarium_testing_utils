import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeSchema,
    format_dataframe,
    AGE_GROUP_COLUMN,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    aggregate_sum,
    linear_combination,
    ratio,
    resolve_age_groups,
    custom_sort_dataframe_by_level,
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
    """Test taking ratio of two columns in a multi-indexed DataFrame"""
    assert ratio(intermediate_data, "a", "b").equals(
        pd.DataFrame({"value": [1 / 4, 2 / 5, 3 / 6, 4 / 7]}, index=intermediate_data.index)
    )
    pd.testing.assert_frame_equal(
        ratio(intermediate_data, "a", "c"),
        pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0]}, index=intermediate_data.index),
    )
    # test non-existent column
    with pytest.raises(KeyError):
        ratio(intermediate_data, "a", "foo")


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


def test_custom_sort_dataframe_by_level(sample_age_schema: AgeSchema) -> None:
    """Test sorting a DataFrame by age according to a schema."""
    # Create a DataFrame with age groups in random order
    df = pd.DataFrame(
        {
            "foo": [3.0, 1.0, 2.0],
            "bar": [6.0, 4.0, 5.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "10_to_15"),
                ("cause", "disease", "0_to_5"),
                ("cause", "disease", "5_to_10"),
            ],
            names=["cause", "disease", AGE_GROUP_COLUMN],
        ),
    )

    # Sort the DataFrame by age according to the schema
    sorted_df = custom_sort_dataframe_by_level(
        level=AGE_GROUP_COLUMN,
        order=[group.name for group in sample_age_schema.age_groups],
        df=df,
    )

    # Create the expected DataFrame with age groups in the order defined by the schema
    expected_df = pd.DataFrame(
        {
            "foo": [1.0, 2.0, 3.0],
            "bar": [4.0, 5.0, 6.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_5"),
                ("cause", "disease", "5_to_10"),
                ("cause", "disease", "10_to_15"),
            ],
            names=["cause", "disease", AGE_GROUP_COLUMN],
        ),
    )

    pd.testing.assert_frame_equal(sorted_df, expected_df)


def test_custom_sort_dataframe_by_level_invalid() -> None:
    """Test sorting a DataFrame by age with invalid age groups."""
    # Create a DataFrame with age groups not in the schema
    df = pd.DataFrame(
        {
            "foo": [3.0, 1.0, 2.0],
            "bar": [6.0, 4.0, 5.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "10_to_15"),
                ("cause", "disease", "0_to_5"),
                ("cause", "disease", "invalid_age_group"),
            ],
            names=["cause", "disease", AGE_GROUP_COLUMN],
        ),
    )

    # Create an AgeSchema without the invalid age group
    schema = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])

    # Check that sorting raises a ValueError
    with pytest.raises(ValueError, match="DataFrame age_group values do not match target"):
        custom_sort_dataframe_by_level(
            level=AGE_GROUP_COLUMN, order=[group.name for group in schema.age_groups], df=df
        )
