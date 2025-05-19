from pathlib import Path

import pandas as pd
import pytest
from pytest_mock import MockFixture
from vivarium.framework.artifact.artifact import ArtifactException

from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.measures import Incidence
from vivarium_testing_utils.automated_validation.interface import ValidationContext


@pytest.mark.skip("Not implemented")
def test_add_comparison_bad_source(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the source is not recognized"""
    context = ValidationContext(sim_result_dir)
    with pytest.raises(ValueError, match="Source bad_source not recognized"):
        context.add_comparison("cause.disease.incidence", "bad_source", "gbd")


def test_upload_custom_data(sim_result_dir: Path) -> None:
    """Ensure that we can upload custom data and retrieve it"""
    context = ValidationContext(sim_result_dir)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)
    assert context.get_raw_dataset("foo", "custom").equals(df)


def test_show_raw_dataset(
    sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can show the raw dataset"""
    context = ValidationContext(sim_result_dir)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)

    # Ensure loading with a string instead of a DataSource enum works
    assert context.get_raw_dataset("foo", "custom").equals(df)
    assert context.get_raw_dataset("deaths", "sim").shape == (8, 1)
    assert context.get_raw_dataset("cause.disease.incidence_rate", "artifact").equals(
        artifact_disease_incidence
    )


def test__get_age_groups_art(sim_result_dir: Path, mocker: MockFixture) -> None:
    """Ensure that we grab age groups 'from the artifact' when available"""
    age_groups = pd.DataFrame(
        {
            "foo": ["bar"],
        },
    )

    # mock dataloader to return age groups
    mocker.patch(
        "vivarium_testing_utils.automated_validation.interface.DataLoader._load_from_source",
        return_value=age_groups,
    )
    context = ValidationContext(sim_result_dir)
    assert context.age_groups.equals(age_groups)


def test__get_age_groups_gbd(sim_result_dir: Path, mocker: MockFixture) -> None:
    """Test that if age groups are not available from the artifact, we get them from vivarium_inputs"""
    age_groups = pd.DataFrame(
        {
            "foo": ["bar"],
        },
    )
    mocker.patch(
        "vivarium_testing_utils.automated_validation.interface.DataLoader._load_from_source",
        side_effect=ArtifactException(),
    )

    mocker.patch(
        "vivarium_inputs.get_age_bins",
        return_value=age_groups,
    )
    context = ValidationContext(sim_result_dir)
    assert context.age_groups.equals(age_groups)


def test___get_raw_datasets_from_source(
    sim_result_dir: Path,
    transition_count_data: pd.DataFrame,
    person_time_data: pd.DataFrame,
    artifact_disease_incidence: pd.DataFrame,
) -> None:
    """Ensure that we can get raw datasets from a source"""
    context = ValidationContext(sim_result_dir)
    measure = Incidence("disease")
    test_raw_datasets = context._get_raw_datasets_from_source(measure, DataSource.SIM)
    ref_raw_datasets = context._get_raw_datasets_from_source(measure, DataSource.ARTIFACT)

    assert test_raw_datasets["numerator_data"].equals(transition_count_data)
    assert test_raw_datasets["denominator_data"].equals(person_time_data)
    assert ref_raw_datasets["artifact_data"].equals(artifact_disease_incidence)


def test_add_comparison(
    sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can add a comparison"""
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir)
    context.add_comparison(measure_key, "sim", "artifact", [])
    assert measure_key in context.comparisons
    comparison = context.comparisons[measure_key]

    assert comparison.measure.measure_key == measure_key
    assert comparison.stratifications == []
    expected_ratio_data = pd.DataFrame(
        {
            "susceptible_to_disease_to_disease_transition_count": [3.0, 5.0],
            "susceptible_to_disease_person_time": [17.0, 29.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert comparison.test_data.equals(expected_ratio_data)
    assert comparison.reference_data.equals(
        artifact_disease_incidence.rename(columns={"value": "reference_rate"})
    )


######################################
# Tests for NotImplementedError cases#
######################################


def test_not_implemented(sim_result_dir: Path) -> None:
    """Test that ValidationContext.add_comparison raises NotImplementedError when test_source is not 'sim'."""
    context = ValidationContext(sim_result_dir)

    with pytest.raises(
        NotImplementedError,
        match="Comparison for artifact source not implemented. Must be SIM.",
    ):
        context.add_comparison("cause.disease.incidence_rate", "artifact", "gbd")

    with pytest.raises(
        NotImplementedError, match="Non-default stratifications require rate aggregations"
    ):
        context.add_comparison("cause.disease.incidence_rate", "sim", "artifact")
        context.get_frame("cause.disease.incidence_rate", stratifications=["foo", "bar"])


@pytest.mark.skip("Not implemented")
def test_metadata() -> None:
    """Ensure that we can summarize a comparison"""
    pass


@pytest.mark.skip("Not implemented")
def test_get_frame() -> None:
    """Ensure that we can verify a comparison"""
    pass
