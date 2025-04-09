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
        context.add_comparison("cause.malaria.incidence", "bad_source", "gbd")


def test_upload_custom_data(sim_result_dir: Path) -> None:
    """Ensure that we can upload custom data and retrieve it"""
    context = ValidationContext(sim_result_dir, None)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)
    assert context.get_raw_dataset("foo", "custom").equals(df)


def test_show_raw_dataset(sim_result_dir: Path) -> None:
    """Ensure that we can show the raw dataset"""
    context = ValidationContext(sim_result_dir, None)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)

    # Ensure loading with a string instead of a DataSource enum works
    assert context.get_raw_dataset("foo", "custom").equals(df)
    assert context.get_raw_dataset("deaths", "sim").shape == (8, 1)
    assert context.get_raw_dataset("cause.malaria.incidence_rate", "artifact").shape == (
        12,
        5,
    )


def test___get_raw_datasets_from_source(sim_result_dir: Path) -> None:
    """Ensure that we can get raw datasets from a source"""
    context = ValidationContext(sim_result_dir, None)
    measure = Incidence("malaria")
    test_raw_datasets = context._get_raw_datasets_from_source(measure, DataSource.SIM)
    ref_raw_datasets = context._get_raw_datasets_from_source(measure, DataSource.ARTIFACT)


def test_add_comparison(sim_result_dir: Path) -> None:
    """Ensure that we can add a comparison"""
    measure_key = "cause.malaria.incidence_rate"
    context = ValidationContext(sim_result_dir, None)
    context.add_comparison(measure_key, DataSource.SIM, DataSource.ARTIFACT, [])
    assert measure_key in context.comparisons
    comparison = context.comparisons[measure_key]
