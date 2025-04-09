from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Incidence,
    Prevalence,
    Remission,
)


def test_incidence() -> None:
    """Test the Incidence measure."""
    cause = "cause"
    measure = Incidence(cause)
    assert measure.measure_key == f"cause.{cause}.incidence_rate"
    assert measure.sim_datasets == {
        "numerator_data": f"transition_count_{cause}",
        "denominator_data": f"person_time_{cause}",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}
