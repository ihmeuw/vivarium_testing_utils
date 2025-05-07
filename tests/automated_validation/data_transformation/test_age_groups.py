from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeGroup,
    AgeSchema,
)


def test_age_group() -> None:
    """Test the AgeGroup class instantiation."""
    group = AgeGroup("0_to_5_years", 0, 5)
    assert group.name == "0_to_5_years"
    assert group.start == 0
    assert group.end == 5
    assert group.span == 5


@pytest.mark.parametrize(
    "name, start, end, match",
    [
        ("0_to_5_years", -1, 1, "Negative start age"),
        ("0_to_5_years", 1, -1, "Negative end age"),
        ("0_to_5_years", 5, 4, "End age must be greater than start age."),
    ],
)
def test_age_group_invalid(name: str, start: int, end: int, match: str) -> None:
    """Test the AgeGroup class instantiation with invalid parameters."""
    with pytest.raises(ValueError, match=match):
        AgeGroup(name, start, end)


def test_age_group_eq() -> None:
    """Test the equality operator for AgeGroup."""
    group1 = AgeGroup("0_to_5_years", 0, 5)
    group2 = AgeGroup("0_to_5", 0, 5)
    group3 = AgeGroup("5_to_10_years", 5, 10)
    assert group1 == group2
    assert group1 != group3


@pytest.mark.parametrize(
    "string, ages",
    [
        ("0_to_5_years", (0, 5)),
        ("0_to_6_months", (0, 0.5)),
        ("0_to_8_days", (0, 0.02191780821917808)),
        ("14_to_17", (14, 17)),
    ],
)
def test_age_group_from_string(string: str, ages: tuple[int | float, int | float]) -> None:
    """Test AgeGroup instantiation from string."""
    group = AgeGroup.from_string(string)
    assert group.name == string
    assert group.start == ages[0]
    assert np.isclose(group.end, ages[1])
    assert np.isclose(group.span, ages[1] - ages[0])


@pytest.mark.parametrize(
    "string, match",
    [
        ("invalid_format", "Invalid age group name format:"),
        (
            "0_to_5_invalid_unit",
            "Invalid unit: invalid_unit. Must be 'days', 'months', or 'years'.",
        ),
    ],
)
def test_age_group_invalid_string(string: str, match: str) -> None:
    """Test AgeGroup instantiation from invalid string."""
    with pytest.raises(ValueError, match=match):
        AgeGroup.from_string(string)


def test_age_group_from_range() -> None:
    """Test AgeGroup instantiation from range."""
    group = AgeGroup.from_range(0, 5)
    assert group.name == "0_to_5"
    assert group.start == 0
    assert group.end == 5
    assert group.span == 5


@pytest.mark.parametrize(
    "group_name, group_ages, fraction",
    [
        ("0_to_5_years", (0, 5), 1.0),
        ("0_to_10_years", (0, 10), 1.0),
        ("3_to_8_years", (3, 8), 2 / 5),
        ("6_to_10_years", (6, 10), 0.0),
    ],
)
def test_age_group_fraction_contained_by(
    group_name: str, group_ages: tuple[float | int, float | int], fraction: float
) -> None:
    """Test that we get the correct amount of overlap between two age groups."""
    group = AgeGroup("0_to_5_years", 0, 5)

    other_group = AgeGroup(group_name, *group_ages)
    assert group.fraction_contained_by(other_group) == fraction


def test_age_schema() -> None:
    """Test the AgeSchema class instantiation."""
    schema = AgeSchema([AgeGroup("0_to_5", 0, 5), AgeGroup("5_to_10", 5, 10)])
    assert len(schema) == 2
    assert schema[0] == AgeGroup("0_to_5", 0, 5)
    assert schema[1] == AgeGroup("5_to_10", 5, 10)
    assert schema.range == (0, 10)
    assert schema.span == 10


@pytest.mark.parametrize(
    "age_groups, err_match",
    [
        ([("0_to_5", 0, 5), ("4_to_10", 4, 10)], "Overlapping age groups"),
        ([("0_to_5", 0, 5), ("6_to_10", 6, 10)], "Gap between consecutive age groups"),
        ([], "No age groups provided"),
    ],
)
def test_age_schema_validation(
    age_groups: list[tuple[str, float | int, float | int]], err_match: str
) -> None:
    """Test we get errors for invalid combinations of age groups."""
    with pytest.raises(ValueError, match=err_match):
        AgeSchema.from_tuples(age_groups)


def test_age_schema_from_tuples() -> None:
    """Test instantion of AgeSchema from tuples."""
    schema = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])
    assert len(schema) == 2
    assert schema[0] == AgeGroup("0_to_5", 0, 5)
    assert schema[1] == AgeGroup("5_to_10", 5, 10)
    assert schema.range == (0, 10)
    assert schema.span == 10


def test_age_schema_from_ranges() -> None:
    """Test instantion of AgeSchema from ranges."""
    schema = AgeSchema.from_ranges([(0, 5), (5, 10)])
    assert len(schema) == 2
    assert schema[0] == AgeGroup("0_to_5", 0, 5)
    assert schema[1] == AgeGroup("5_to_10", 5, 10)
    assert schema.range == (0, 10)
    assert schema.span == 10


def test_age_schema_from_strings() -> None:
    """Test instantion of AgeSchema from strings."""
    schema = AgeSchema.from_strings(["0_to_5", "5_to_10"])
    assert len(schema) == 2
    assert schema[0] == AgeGroup("0_to_5", 0, 5)
    assert schema[1] == AgeGroup("5_to_10", 5, 10)
    assert schema.range == (0, 10)
    assert schema.span == 10


def test_age_schema_from_dataframe() -> None:
    """Test instantion of AgeSchema from a DataFrame."""
    df = pd.DataFrame(
        {
            "foo": [1.0, 2.0, 3.0, 4.0],
            "bar": [5.0, 6.0, 7.0, 8.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_5"),
                ("cause", "disease", "5_to_10"),
                ("cause", "disease", "10_to_15"),
                ("cause", "disease", "15_to_20"),
            ],
            names=["cause", "disease", "age_group"],
        ),
    )

    target_age_schema = AgeSchema.from_dataframe(df)
    assert len(target_age_schema.age_groups) == 4
    assert target_age_schema.range == (0, 20)
    assert target_age_schema.span == 20


def test_age_schema_eq() -> None:
    """Test the equality operator for AgeSchema."""
    schema1 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])
    schema2 = AgeSchema.from_tuples([("foo", 0, 5), ("bar", 5, 10)])
    schema3 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_15", 5, 15)])
    assert schema1 == schema2
    assert schema1 != schema3


def test_age_schema_contains() -> None:
    """Test the contains method for AgeSchema."""
    schema = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])

    assert AgeGroup("different_name", 0, 5) in schema
    assert AgeGroup("0_to_5", 0, 5) in schema
    assert not AgeGroup("10_to_15", 10, 15) in schema


def test_age_schema_is_subset() -> None:
    """Test we can see whether one schema is a subset of another."""
    schema1 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])
    schema2 = AgeSchema.from_tuples(
        [("0_to_5", 0, 5), ("5_to_10", 5, 10), ("10_to_15", 10, 15)]
    )
    schema3 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_15", 5, 15)])
    assert schema1.is_subset(schema2)
    assert not schema1.is_subset(schema3)


def test_age_schema_validate_compatible() -> None:
    """Test whether one schema can be transformed to another."""
    schema1 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])
    schema2 = AgeSchema.from_tuples([("0_to_4", 0, 4), ("4_to_10", 4, 10)])
    schema3 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_15", 5, 15)])
    schema1.validate_compatible(schema2)
    with pytest.raises(
        ValueError,
        match="Age schemas have different ranges",
    ):
        schema1.validate_compatible(schema3)


def test_age_schema_get_transform_matrix() -> None:
    """Test we can get a transform matrix between two schemas."""
    schema1 = AgeSchema.from_tuples([("0_to_2", 0, 2), ("2_to_5", 2, 5)])
    schema2 = AgeSchema.from_tuples([("0_to_1", 0, 1), ("1_to_3", 1, 3), ("3_to_5", 3, 5)])
    transform_matrix = schema1.get_transform_matrix(schema2)
    expected_matrix = pd.DataFrame(
        {
            "0_to_1": [1.0, 0.0],
            "1_to_3": [0.5, 0.5],
            "3_to_5": [0.0, 1.0],
        },
        index=["0_to_2", "2_to_5"],
    )
    pd.testing.assert_frame_equal(transform_matrix, expected_matrix)


def test_rebin_dataframe() -> None:
    """Test we can transform a DataFrame to a new age schema."""
    df = pd.DataFrame(
        {
            "foo": [1.0, 2.0, 3.0, 4.0],
            "bar": [5.0, 6.0, 7.0, 8.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_5"),
                ("cause", "disease", "5_to_10"),
                ("cause", "disease", "10_to_15"),
                ("cause", "disease", "15_to_20"),
            ],
            names=["cause", "disease", "age_group"],
        ),
    )

    target_age_schema = AgeSchema.from_tuples(
        [
            ("0_to_10", 0, 10),
            ("10_to_20", 10, 20),
        ]
    )

    rebinned_df = target_age_schema.rebin_dataframe(df)
    expected_df = pd.DataFrame(
        {
            "foo": [3.0, 7.0],
            "bar": [11.0, 15.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_10"),
                ("cause", "disease", "10_to_20"),
            ],
            names=["cause", "disease", "age_group"],
        ),
    )
    assert rebinned_df.equals(expected_df)


def test_rebin_dataframe_uneven() -> None:
    """Test we can transform a DataFrame to a new age schema with uneven groups."""
    df = pd.DataFrame(
        {
            "foo": [1.0, 2.0, 3.0, 4.0],
            "bar": [5.0, 6.0, 7.0, 8.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_5"),
                ("cause", "disease", "5_to_10"),
                ("cause", "disease", "10_to_15"),
                ("cause", "disease", "15_to_20"),
            ],
            names=["cause", "disease", "age_group"],
        ),
    )

    target_age_schema = AgeSchema.from_tuples(
        [
            ("0_to_3", 0, 3),
            ("3_to_4", 3, 4),
            ("4_to_7", 4, 7),
            ("7_to_20", 7, 20),
        ]
    )
    expected_foo = {
        "0_to_3": 1.0 * 3 / 5,
        "3_to_4": 1.0 * 1 / 5,
        "4_to_7": 1.0 * 1 / 5 + 2.0 * 2 / 5,
        "7_to_20": 2.0 * 3 / 5 + 3.0 + 4.0,
    }
    expected_bar = {
        "0_to_3": 5.0 * 3 / 5,
        "3_to_4": 5.0 * 1 / 5,
        "4_to_7": 5.0 * 1 / 5 + 6.0 * 2 / 5,
        "7_to_20": 6.0 * 3 / 5 + 7.0 + 8.0,
    }

    rebinned_df = target_age_schema.rebin_dataframe(df)
    expected_df = pd.DataFrame(
        {
            "foo": expected_foo.values(),
            "bar": expected_bar.values(),
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_3"),
                ("cause", "disease", "3_to_4"),
                ("cause", "disease", "4_to_7"),
                ("cause", "disease", "7_to_20"),
            ],
            names=["cause", "disease", "age_group"],
        ),
    )
    pd.testing.assert_frame_equal(rebinned_df, expected_df)
