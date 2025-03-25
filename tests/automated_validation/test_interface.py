from pathlib import Path

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.interface import ValidationContext


@pytest.mark.skip("Not implemented")
def test_add_comparison_bad_source(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the source is not recognized"""
    context = ValidationContext(sim_result_dir, None)
    with pytest.raises(ValueError):
        context.add_comparison("cause.cause.incidence", "bad_source", "gbd")


def test_upload_custom_data(sim_result_dir: Path) -> None:
    """Ensure that we can upload custom data and retrieve it"""
    context = ValidationContext(sim_result_dir, None)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data(df, "foo")
    assert context.show_raw_dataset("foo", "custom").equals(df)
