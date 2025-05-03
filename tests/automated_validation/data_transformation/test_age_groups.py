from __future__ import annotations

import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeGroup,
    AgeSchema,
    rebin_dataframe,
    reformat_artifact_dataframe,
)


def test_age_group():
    # Test the AgeGroup class
    group = AgeGroup("0_to_5_years", 0, 5)
    assert group.name == "0_to_5_years"
    assert group.start == 0
    assert group.end == 5
    assert group.span == 5


@pytest.mark.parametrize(
    "string, ages",
    [
        ("0_to_5_years", (0, 5)),
        ("0_to_6_months", (0, 0.5)),
        ("0_to_8_days", (0, 0.02191780821917808)),
        ("14_to_17", (14, 17)),
    ],
)
def test_age_group_from_string(string, ages):
    group = AgeGroup.from_string(string)
    assert group.name == string
    assert group.start == ages[0]
    assert np.isclose(group.end, ages[1])
    assert np.isclose(group.span, ages[1] - ages[0])


@pytest.mark.parametrize(
    "string, match",
    [
        ("invalid_format", "Invalid age group name format:"),
        ("5_to_0_years", "End age must be greater than or equal to start age."),
        (
            "0_to_5_invalid_unit",
            "Invalid unit: invalid_unit. Must be 'days', 'months', or 'years'.",
        ),
    ],
)
def test_age_group_invalid_string(string, match):
    # Test invalid string format
    with pytest.raises(ValueError, match=match):
        AgeGroup.from_string(string)


@pytest.mark.parametrize(
    "group_name, group_ages, fraction",
    [
        ("0_to_5_years", (0, 5), 1.0),
        ("0_to_10_years", (0, 10), 1.0),
        ("3_to_8_years", (3, 8), 2 / 5),
        ("6_to_10_years", (6, 10), 0.0),
    ],
)
def test_age_group_fraction_contained_by(group_name, group_ages, fraction):
    # Test fraction_contained_by method
    group = AgeGroup("0_to_5_years", 0, 5)

    other_group = AgeGroup(group_name, *group_ages)
    assert group.fraction_contained_by(other_group) == fraction


def test_reformat_artifact_dataframe():
    artifact_df = pd.DataFrame(
        {"value": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0", "5"),
                ("cause", "disease", "5", "10"),
                ("cause", "disease", "10", "15"),
                ("cause", "disease", "15", "20"),
            ],
            names=["cause", "disease", "age_start", "age_end"],
        ),
    )
    reformatted_df = reformat_artifact_dataframe(artifact_df)
    expected_df = pd.DataFrame(
        {"value": [1, 2, 3, 4]},
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
    assert reformatted_df.equals(expected_df)


def test_rebin_dataframe():
    # Create a sample DataFrame with age groups
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
    # Create a target age schema
    target_age_schema = AgeSchema.from_dict(
        {
            "0_to_10": (0, 10),
            "10_to_20": (10, 20),
        }
    )
    # Rebin the DataFrame
    rebinned_df = rebin_dataframe(df, target_age_schema)
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


def test_rebin_dataframe_uneven():
    # Create a sample DataFrame with age groups
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
    # Create a target age schema
    target_age_schema = AgeSchema.from_dict(
        {
            "0_to_3": (0, 3),
            "3_to_4": (3, 4),
            "4_to_7": (4, 7),
            "7_to_20": (7, 20),
        }
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
    # Rebin the DataFrame
    rebinned_df = rebin_dataframe(df, target_age_schema)
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
