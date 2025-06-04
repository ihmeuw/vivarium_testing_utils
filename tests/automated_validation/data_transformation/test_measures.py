import pandas as pd
from pandas.testing import assert_frame_equal

from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    CauseSpecificMortalityRate,
    ExcessMortalityRate,
    Incidence,
    Prevalence,
    SIRemission,
)


def test_incidence(
    transition_count_data: pd.DataFrame, person_time_data: pd.DataFrame
) -> None:
    """Test the Incidence measure."""
    cause = "disease"
    measure = Incidence(cause)
    assert measure.measure_key == f"cause.{cause}.incidence_rate"
    assert measure.sim_datasets == {
        "numerator_data": f"transition_count_{cause}",
        "denominator_data": f"person_time_{cause}",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    numerator_data, denominator_data = measure.get_ratio_data_from_sim(
        numerator_data=transition_count_data,
        denominator_data=person_time_data,
    )
    expected_numerator_data = pd.DataFrame(
        {
            "value": [3.0, 5.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [17.0, 29.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert numerator_data.equals(expected_numerator_data)
    assert denominator_data.equals(expected_denominator_data)

    measure_data_from_ratio = measure.get_measure_data_from_ratio(
        numerator_data=numerator_data, denominator_data=denominator_data
    )

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=transition_count_data, denominator_data=person_time_data
    )
    expected_measure_data = pd.DataFrame(
        {"value": [3 / 17.0, 5 / 29.0]},
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert measure_data.equals(expected_measure_data)
    assert measure_data_from_ratio.equals(expected_measure_data)


def test_prevalence(person_time_data: pd.DataFrame) -> None:
    """Test the Prevalence measure."""
    cause = "disease"
    measure = Prevalence(cause)
    assert measure.measure_key == f"cause.{cause}.prevalence"
    assert measure.sim_datasets == {
        "numerator_data": f"person_time_{cause}",
        "denominator_data": f"person_time_{cause}",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    numerator_data, denominator_data = measure.get_ratio_data_from_sim(
        numerator_data=person_time_data,
        denominator_data=person_time_data,
    )
    expected_numerator_data = pd.DataFrame(
        {
            "value": [23.0, 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [17.0 + 23.0, 29.0 + 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert numerator_data.equals(expected_numerator_data)
    assert denominator_data.equals(expected_denominator_data)
    measure_data_from_ratio = measure.get_measure_data_from_ratio(
        numerator_data=numerator_data, denominator_data=denominator_data
    )
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=person_time_data, denominator_data=person_time_data
    )
    expected_measure_data = pd.DataFrame(
        {"value": [23.0 / (17.0 + 23.0), 37.0 / (29.0 + 37.0)]},
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert measure_data.equals(expected_measure_data)
    assert measure_data_from_ratio.equals(expected_measure_data)


def test_si_remission(
    transition_count_data: pd.DataFrame, person_time_data: pd.DataFrame
) -> None:
    """Test the SIRemission measure."""
    cause = "disease"
    measure = SIRemission(cause)
    assert measure.measure_key == f"cause.{cause}.remission_rate"
    assert measure.sim_datasets == {
        "numerator_data": f"transition_count_{cause}",
        "denominator_data": f"person_time_{cause}",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    numerator_data, denominator_data = measure.get_ratio_data_from_sim(
        numerator_data=transition_count_data,
        denominator_data=person_time_data,
    )
    expected_numerator_data = pd.DataFrame(
        {
            "value": [7.0, 13.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [23.0, 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert numerator_data.equals(expected_numerator_data)
    assert denominator_data.equals(expected_denominator_data)
    measure_data_from_ratio = measure.get_measure_data_from_ratio(
        numerator_data, denominator_data
    )
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=transition_count_data, denominator_data=person_time_data
    )
    expected_measure_data = pd.DataFrame(
        {"value": [7 / 23.0, 13 / 37.0]},
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert measure_data.equals(expected_measure_data)
    assert measure_data_from_ratio.equals(expected_measure_data)


def test_all_cause_mortality_rate(
    deaths_data: pd.DataFrame, total_person_time_data: pd.DataFrame
) -> None:
    """Test the CauseMortalityRate measurefor all causes."""
    measure = CauseSpecificMortalityRate("all_causes")
    assert measure.measure_key == "cause.all_causes.cause_specific_mortality_rate"
    assert measure.sim_datasets == {
        "numerator_data": "deaths",
        "denominator_data": "person_time_total",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    numerator_data, denominator_data = measure.get_ratio_data_from_sim(
        numerator_data=deaths_data,
        denominator_data=total_person_time_data,
    )

    # Expected dataframe for the numerator and denominator data
    # The Deaths formatter with no cause will marginalize over entity and sub_entity
    # to get total deaths by stratify_column
    expected_numerator_data = pd.DataFrame(
        {
            "value": [5.0, 9.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [
                40.0,
                66.0,
            ],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(numerator_data, expected_numerator_data)
    assert_frame_equal(denominator_data, expected_denominator_data)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=deaths_data, denominator_data=total_person_time_data
    )

    expected_measure_data = pd.DataFrame(
        {"value": [5.0 / 40.0, 9.0 / 66.0]},
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(measure_data, expected_measure_data)


def test_cause_specific_mortality_rate(
    deaths_data: pd.DataFrame, total_person_time_data: pd.DataFrame
) -> None:
    """Test the CauseSpecificMortalityRate measure."""
    cause = "disease"
    measure = CauseSpecificMortalityRate(cause)
    assert measure.measure_key == f"cause.{cause}.cause_specific_mortality_rate"
    assert measure.sim_datasets == {
        "numerator_data": f"deaths",
        "denominator_data": "person_time_total",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    numerator_data, denominator_data = measure.get_ratio_data_from_sim(
        numerator_data=deaths_data,
        denominator_data=total_person_time_data,
    )

    # Expected dataframe for the numerator and denominator data
    # The Deaths formatter with a specific cause will filter for that cause
    # The TotalPersonTime formatter will marginalize person_time over all states
    expected_numerator_data = pd.DataFrame(
        {
            "value": [2.0, 4.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [
                40.0,
                66.0,
            ],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(numerator_data, expected_numerator_data)
    assert_frame_equal(denominator_data, expected_denominator_data)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=deaths_data, denominator_data=total_person_time_data
    )

    expected_measure_data = pd.DataFrame(
        {"value": [2.0 / 40.0, 4.0 / 66.0]},
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(measure_data, expected_measure_data)


def test_excess_mortality_rate(
    deaths_data: pd.DataFrame, person_time_data: pd.DataFrame
) -> None:
    """Test the ExcessMortalityRate measure."""
    cause = "disease"
    measure = ExcessMortalityRate(cause)
    assert measure.measure_key == f"cause.{cause}.excess_mortality_rate"
    assert measure.sim_datasets == {
        "numerator_data": f"deaths",
        "denominator_data": f"person_time_{cause}",
    }

    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    numerator_data, denominator_data = measure.get_ratio_data_from_sim(
        numerator_data=deaths_data,
        denominator_data=person_time_data,
    )

    # Expected dataframe for the numerator and denominator data
    # The Deaths formatter with a specific cause will filter for that cause
    # The PersonTime formatter with a specific state will filter for that state
    expected_numerator_data = pd.DataFrame(
        {
            "value": [2.0, 4.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [
                23.0,
                37.0,
            ],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(numerator_data, expected_numerator_data)
    assert_frame_equal(denominator_data, expected_denominator_data)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=deaths_data, denominator_data=person_time_data
    )

    expected_measure_data = pd.DataFrame(
        {"value": [2.0 / 23.0, 4.0 / 37.0]},
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(measure_data, expected_measure_data)
