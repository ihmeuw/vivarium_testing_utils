from pathlib import Path

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.interface import ValidationContext
from vivarium_testing_utils.automated_validation.data_transformation.measures import Incidence
from vivarium_testing_utils.automated_validation.data_loader import DataSource


@pytest.mark.skip("Not implemented")
def test_add_comparison_bad_source(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the source is not recognized"""
    context = ValidationContext(sim_result_dir, None)
    with pytest.raises(ValueError, match="Source bad_source not recognized"):
        context.add_comparison("cause.disease.incidence", "bad_source", "gbd")


def test_upload_custom_data(sim_result_dir: Path) -> None:
    """Ensure that we can upload custom data and retrieve it"""
    context = ValidationContext(sim_result_dir, None)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)
    assert context.get_raw_dataset("foo", "custom").equals(df)


def test_show_raw_dataset(sim_result_dir: Path, artifact_disease_incidence) -> None:
    """Ensure that we can show the raw dataset"""
    context = ValidationContext(sim_result_dir, None)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)

    # Ensure loading with a string instead of a DataSource enum works
    assert context.get_raw_dataset("foo", "custom").equals(df)
    assert context.get_raw_dataset("deaths", "sim").shape == (8, 1)
    assert context.get_raw_dataset("cause.disease.incidence_rate", "artifact").equals(
        artifact_disease_incidence
    )


def test___get_raw_datasets_from_source(
    sim_result_dir: Path,
    transition_count_data: pd.DataFrame,
    person_time_data: pd.DataFrame,
    artifact_disease_incidence: pd.DataFrame,
) -> None:
    """Ensure that we can get raw datasets from a source"""
    context = ValidationContext(sim_result_dir, None)
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
    context = ValidationContext(sim_result_dir, None)
    context.add_comparison(measure_key, DataSource.SIM, DataSource.ARTIFACT, [])
    assert measure_key in context.comparisons
    comparison = context.comparisons[measure_key]

    assert comparison.measure.measure_key == measure_key
    assert comparison.stratifications == []
    expected_ratio_data = pd.DataFrame(
        {
            "susceptible_to_disease_to_disease_transition_count": [3, 5],
            "susceptible_to_disease_person_time": [17, 29],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert comparison.test_data.equals(expected_ratio_data)
    assert comparison.reference_data.equals(artifact_disease_incidence)
