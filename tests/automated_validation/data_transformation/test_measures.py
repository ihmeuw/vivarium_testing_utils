from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Incidence,
    Prevalence,
    Remission,
)


def test_incidence() -> None:
    """Test the Incidence measure."""
    cause = "disease"
    measure = Incidence(cause)
    assert measure.measure_key == f"incidence_{cause}"
    assert measure.sim_datasets == {
        "numerator": f"transition_count_{cause}",
        "denominator": f"person_time_{cause}",
    }
    assert measure.artifact_datasets == {"measure_data": measure.measure_key}
