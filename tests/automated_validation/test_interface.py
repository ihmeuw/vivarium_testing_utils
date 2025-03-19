from pathlib import Path

import pytest

from vivarium_testing_utils.automated_validation.interface import ValidationContext


@pytest.mark.skip("Not implemented")
def test_add_comparison_bad_source(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the source is not recognized"""
    context = ValidationContext(sim_result_dir, None)
    with pytest.raises(ValueError):
        context.add_comparison("cause.cause.incidence", "bad_source", "gbd")
