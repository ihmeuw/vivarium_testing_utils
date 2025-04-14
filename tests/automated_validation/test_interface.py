from pathlib import Path

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.interface import ValidationContext


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


def test_show_raw_dataset(
    sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
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
