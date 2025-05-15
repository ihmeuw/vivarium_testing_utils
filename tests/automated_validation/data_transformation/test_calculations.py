import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeSchema,
    format_dataframe,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    aggregate,
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
    """Test taking ratio of two columns in a multi-indexed DataFrame"""
    assert ratio(intermediate_data, "a", "b").equals(
        pd.DataFrame({"value": [1 / 4, 2 / 5, 3 / 6, 4 / 7]}, index=intermediate_data.index)
    )
    assert ratio(intermediate_data, "a", "c").equals(
        pd.DataFrame({"value": [1, 2, float("inf"), 4]}, index=intermediate_data.index)
    )
    # test non-existent column
    with pytest.raises(KeyError):
        ratio(intermediate_data, "a", "foo")


def test_aggregate(intermediate_data: pd.DataFrame) -> None:
    """Test aggregating over different combinations of value columns."""
    assert aggregate(intermediate_data, ["group"], agg="sum").equals(
        pd.DataFrame(
            {
                "a": [3, 7],
                "b": [9, 13],
                "c": [2, 1],
            },
            index=pd.Index(["x", "y"], name="group"),
        )
    )
    assert aggregate(intermediate_data, ["time"], agg="sum").equals(
        pd.DataFrame(
            {
                "a": [4, 6],
                "b": [10, 12],
                "c": [1, 2],
            },
            index=pd.Index([0, 1], name="time"),
        )
    )
    assert aggregate(intermediate_data, ["group", "time"], agg="sum").equals(
        intermediate_data
    )
    # test non-existent index column
    with pytest.raises(KeyError):
        aggregate(intermediate_data, ["foo"], agg="sum")


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
