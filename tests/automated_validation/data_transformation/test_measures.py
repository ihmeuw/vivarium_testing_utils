import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Incidence,
    Prevalence,
    SIRemission,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    RatioData,
    SimOutputData,
    SingleNumericColumn,
)
from pandera.typing import DataFrame


def test_incidence(
    transition_count_data: DataFrame[SimOutputData],
    person_time_data: DataFrame[SimOutputData],
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

    ratio_data = measure.get_ratio_data_from_sim(
        numerator_data=transition_count_data,
        denominator_data=person_time_data,
    )
    expected_ratio_data = DataFrame[RatioData](
        {
            "susceptible_to_disease_to_disease_transition_count": [3.0, 5.0],
            "susceptible_to_disease_person_time": [17.0, 29.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert ratio_data.equals(expected_ratio_data)

    measure_data_from_ratio = measure.get_measure_data_from_ratio(ratio_data=ratio_data)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=transition_count_data, denominator_data=person_time_data
    )
    expected_measure_data = pd.Series(
        [3 / 17.0, 5 / 29.0],
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert measure_data.equals(expected_measure_data)
    assert measure_data_from_ratio.equals(expected_measure_data)


def test_prevalence(person_time_data: DataFrame[SimOutputData]) -> None:
    """Test the Prevalence measure."""
    cause = "disease"
    measure = Prevalence(cause)
    assert measure.measure_key == f"cause.{cause}.prevalence"
    assert measure.sim_datasets == {
        "numerator_data": f"person_time_{cause}",
        "denominator_data": f"person_time_{cause}",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    ratio_data = measure.get_ratio_data_from_sim(
        numerator_data=person_time_data,
        denominator_data=person_time_data,
    )
    expected_ratio_data = DataFrame[RatioData](
        {
            "disease_person_time": [23.0, 37.0],
            "total_person_time": [17.0 + 23.0, 29.0 + 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert ratio_data.equals(expected_ratio_data)
    measure_data_from_ratio = measure.get_measure_data_from_ratio(ratio_data=ratio_data)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=person_time_data, denominator_data=person_time_data
    )
    expected_measure_data = pd.Series(
        [23.0 / (17.0 + 23.0), 37.0 / (29.0 + 37.0)],
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert measure_data.equals(expected_measure_data)
    assert measure_data_from_ratio.equals(expected_measure_data)


def test_si_remission(
    transition_count_data: DataFrame[SimOutputData],
    person_time_data: DataFrame[SimOutputData],
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

    ratio_data = measure.get_ratio_data_from_sim(
        numerator_data=transition_count_data,
        denominator_data=person_time_data,
    )
    expected_ratio_data = DataFrame[RatioData](
        {
            "disease_to_susceptible_to_disease_transition_count": [7.0, 13.0],
            "disease_person_time": [23.0, 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert ratio_data.equals(expected_ratio_data)
    measure_data_from_ratio = measure.get_measure_data_from_ratio(ratio_data=ratio_data)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=transition_count_data, denominator_data=person_time_data
    )
    expected_measure_data = pd.Series(
        [7 / 23.0, 13 / 37.0],
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert measure_data.equals(expected_measure_data)
    assert measure_data_from_ratio.equals(expected_measure_data)
