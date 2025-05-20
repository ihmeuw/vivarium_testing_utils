from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AGE_END_COLUMN,
    AGE_GROUP_COLUMN,
    AGE_START_COLUMN,
    AgeGroup,
    AgeRange,
    AgeSchema,
    AgeTuple,
    format_dataframe,
    get_transform_matrix,
    rebin_dataframe,
)


def test_age_group() -> None:
    """Test the AgeGroup class instantiation."""
    group = AgeGroup("foo", 5, 13)
    assert group.name == "foo"
    assert group.start == 5
    assert group.end == 13
    assert group.span == 8


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
def test_age_group_from_string(string: str, ages: AgeRange) -> None:
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
    group_name: str, group_ages: AgeRange, fraction: float
) -> None:
    """Test that we get the correct amount of overlap between two age groups."""
    group = AgeGroup("0_to_5_years", 0, 5)

    other_group = AgeGroup(group_name, *group_ages)
    assert group.fraction_contained_by(other_group) == fraction


def check_example_age_schema(age_schema: AgeSchema) -> None:
    """Check that the example age schema was instantiated correctly, regardless of method."""
    assert len(age_schema) == 3
    assert age_schema[0] == AgeGroup("0_to_5", 0, 5)
    assert age_schema[1] == AgeGroup("5_to_10", 5, 10)
    assert age_schema[2] == AgeGroup("10_to_15", 10, 15)
    assert age_schema.range == (0, 15)
    assert age_schema.span == 15


def test_age_schema_instantiation(
    sample_age_tuples: list[AgeTuple],
    sample_df_with_ages: pd.DataFrame,
) -> None:
    """Test the AgeSchema class instantiation."""
    for age_schema in [
        AgeSchema.from_tuples(sample_age_tuples),
        AgeSchema.from_ranges([(tuple[1], tuple[2]) for tuple in sample_age_tuples]),
        AgeSchema.from_strings([tuple[0] for tuple in sample_age_tuples]),
        AgeSchema.from_dataframe(sample_df_with_ages),
    ]:
        check_example_age_schema(age_schema)


@pytest.mark.parametrize(
    "age_groups, err_match",
    [
        ([("0_to_5", 0, 5), ("4_to_10", 4, 10)], "Overlapping age groups"),
        ([("0_to_5", 0, 5), ("6_to_10", 6, 10)], "Gap between consecutive age groups"),
        ([], "No age groups provided"),
    ],
)
def test_age_schema_validation(age_groups: list[AgeTuple], err_match: str) -> None:
    """Test we get errors for invalid combinations of age groups."""
    with pytest.raises(ValueError, match=err_match):
        AgeSchema.from_tuples(age_groups)


def test_age_schema_to_dataframe(
    sample_age_schema: AgeSchema, sample_age_group_df: pd.DataFrame
) -> None:
    """Test we can convert an AgeSchema to a DataFrame."""
    pd.testing.assert_frame_equal(sample_age_schema.to_dataframe(), sample_age_group_df)


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


def test_age_schema_is_subset(sample_age_schema: AgeSchema) -> None:
    """Test we can see whether one schema is a subset of another."""
    subset_schema = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])
    not_subset_schema = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_15", 5, 15)])

    assert subset_schema.is_subset(sample_age_schema)
    assert not not_subset_schema.is_subset(sample_age_schema)


def test_age_schema_can_coerce_to() -> None:
    """Test whether one schema can be transformed to another."""
    schema1 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_10", 5, 10)])
    schema2 = AgeSchema.from_tuples([("0_to_4", 0, 4), ("4_to_10", 4, 10)])
    schema3 = AgeSchema.from_tuples([("0_to_5", 0, 5), ("5_to_15", 5, 15)])

    assert schema1.can_coerce_to(schema2)
    assert schema2.can_coerce_to(schema1)
    assert schema1.can_coerce_to(schema3)
    assert not schema3.can_coerce_to(schema1)


def test_age_schema_get_transform_matrix(sample_age_schema: AgeSchema) -> None:
    """Test we can get a transform matrix between two schemas."""
    new_schema = AgeSchema.from_tuples([("0_to_7.5", 0, 7.5), ("7.5_to_15", 7.5, 15)])
    transform_matrix = get_transform_matrix(sample_age_schema, new_schema)
    expected_matrix = pd.DataFrame(
        {
            "0_to_5": [1.0, 0.0],
            "5_to_10": [0.5, 0.5],
            "10_to_15": [0.0, 1.0],
        },
        index=["0_to_7.5", "7.5_to_15"],
    )

    pd.testing.assert_frame_equal(transform_matrix, expected_matrix)


def test_age_schema_format_cols(
    sample_age_schema: AgeSchema, sample_df_with_ages: pd.DataFrame
) -> None:
    """Test we can format a DataFrame with only age groups."""
    for dataframe in [
        sample_df_with_ages,
        sample_df_with_ages.droplevel([AGE_START_COLUMN, AGE_END_COLUMN]),
        sample_df_with_ages.droplevel([AGE_START_COLUMN, AGE_END_COLUMN]),
    ]:
        pd.testing.assert_frame_equal(
            format_dataframe(sample_age_schema, dataframe),
            sample_df_with_ages,
        )


def test_age_schema_format_dataframe_invalid(sample_age_schema: AgeSchema) -> None:
    """Test we get an error if we try to format a DataFrame with invalid age groups."""
    df = pd.DataFrame(
        {
            "foo": [1.0, 2.0],
            "bar": [5.0, 6.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "25_to_30"),
                ("cause", "disease", "30_to_40"),
            ],
            names=["cause", "disease", AGE_GROUP_COLUMN],
        ),
    )
    with pytest.raises(ValueError, match="Cannot coerce"):
        format_dataframe(sample_age_schema, df)


def test_age_schema_format_dataframe_rebin(sample_df_with_ages: pd.DataFrame) -> None:
    """Test we that format_dataframe rebins the dataframe when necessary."""
    target_age_schema = AgeSchema.from_tuples(
        [
            ("0_to_3", 0, 3),
            ("3_to_4", 3, 4),
            ("4_to_7", 4, 7),
            ("7_to_15", 7, 15),
        ]
    )
    formatted_df = format_dataframe(target_age_schema, sample_df_with_ages)
    pd.testing.assert_frame_equal(
        formatted_df, rebin_dataframe(target_age_schema, sample_df_with_ages)
    )


def test_rebin_dataframe(sample_df_with_ages: pd.DataFrame) -> None:
    """Test we can transform a DataFrame to a new age schema with uneven groups."""
    df = sample_df_with_ages.droplevel([AGE_START_COLUMN, AGE_END_COLUMN])

    target_age_schema = AgeSchema.from_tuples(
        [
            ("0_to_3", 0, 3),
            ("3_to_4", 3, 4),
            ("4_to_7", 4, 7),
            ("7_to_15", 7, 15),
        ]
    )
    expected_foo = {
        "0_to_3": 1.0 * 3 / 5,
        "3_to_4": 1.0 * 1 / 5,
        "4_to_7": 1.0 * 1 / 5 + 2.0 * 2 / 5,
        "7_to_15": 2.0 * 3 / 5 + 3.0,
    }
    expected_bar = {
        "0_to_3": 4.0 * 3 / 5,
        "3_to_4": 4.0 * 1 / 5,
        "4_to_7": 4.0 * 1 / 5 + 5.0 * 2 / 5,
        "7_to_15": 5.0 * 3 / 5 + 6.0,
    }

    rebinned_df = rebin_dataframe(target_age_schema, df)
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
                ("cause", "disease", "7_to_15"),
            ],
            names=["cause", "disease", AGE_GROUP_COLUMN],
        ),
    )
    pd.testing.assert_frame_equal(rebinned_df, expected_df)
