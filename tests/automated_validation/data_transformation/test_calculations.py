import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    aggregate_sum,
    linear_combination,
    ratio,
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
        pd.Series([5, 7, 9, 11], index=intermediate_data.index)
    )
    assert linear_combination(intermediate_data, 2, "a", -1, "b").equals(
        pd.Series([-2, -1, 0, 1], index=intermediate_data.index)
    )
    # test non-existent column
    with pytest.raises(KeyError):
        linear_combination(intermediate_data, 1, "a", 1, "foo")
