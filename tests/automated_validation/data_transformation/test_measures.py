import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    TotalPopulationPersonTime,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    CauseSpecificMortalityRate,
    ExcessMortalityRate,
    Incidence,
    PopulationStructure,
    Prevalence,
    RatioMeasure,
    RiskExposure,
    SIRemission,
    get_measure_from_key,
)


def get_expected_dataframe(value_1: float, value_2: float) -> pd.DataFrame:
    """Create the expected dataframe by passing in two values to a reliable index."""
    return pd.DataFrame(
        {
            "value": [value_1, value_2],
        },
        index=pd.MultiIndex.from_tuples(
            [("A", "baseline"), ("B", "baseline")],
            names=["stratify_column", "scenario"],
        ),
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

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=transition_count_data,
        denominator_data=person_time_data,
    )
    assert ratio_datasets["numerator_data"].equals(get_expected_dataframe(3.0, 5.0))
    assert ratio_datasets["denominator_data"].equals(get_expected_dataframe(17.0, 29.0))

    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=transition_count_data, denominator_data=person_time_data
    )
    expected_measure_data = get_expected_dataframe(3 / 17.0, 5 / 29.0)

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

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=person_time_data,
        denominator_data=person_time_data,
    )

    assert ratio_datasets["numerator_data"].equals(get_expected_dataframe(23.0, 37.0))
    assert ratio_datasets["denominator_data"].equals(
        get_expected_dataframe(17.0 + 23.0, 29.0 + 37.0)
    )
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=person_time_data, denominator_data=person_time_data
    )
    expected_measure_data = get_expected_dataframe(23.0 / (17.0 + 23.0), 37.0 / (29.0 + 37.0))
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

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=transition_count_data,
        denominator_data=person_time_data,
    )

    assert ratio_datasets["numerator_data"].equals(get_expected_dataframe(7.0, 13.0))
    assert ratio_datasets["denominator_data"].equals(get_expected_dataframe(23.0, 37.0))
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=transition_count_data, denominator_data=person_time_data
    )
    expected_measure_data = get_expected_dataframe(7.0 / 23.0, 13.0 / 37.0)
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

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=deaths_data,
        denominator_data=total_person_time_data,
    )

    # Expected dataframe for the numerator and denominator data
    # The Deaths formatter with no cause will marginalize over entity and sub_entity
    # to get total deaths by stratify_column
    assert_frame_equal(ratio_datasets["numerator_data"], get_expected_dataframe(5.0, 9.0))
    assert_frame_equal(ratio_datasets["denominator_data"], get_expected_dataframe(40.0, 66.0))
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=deaths_data, denominator_data=total_person_time_data
    )

    expected_measure_data = get_expected_dataframe(5.0 / 40.0, 9.0 / 66.0)
    assert_frame_equal(measure_data, expected_measure_data)
    assert_frame_equal(measure_data_from_ratio, expected_measure_data)


def test_cause_specific_mortality_rate(
    deaths_data: pd.DataFrame,
    total_person_time_data: pd.DataFrame,
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

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=deaths_data,
        denominator_data=total_person_time_data,
    )

    # Expected dataframe for the numerator and denominator data
    # The Deaths formatter with a specific cause will filter for that cause
    # The TotalPersonTime formatter will marginalize person_time over all states
    assert_frame_equal(ratio_datasets["numerator_data"], get_expected_dataframe(2.0, 4.0))
    assert_frame_equal(ratio_datasets["denominator_data"], get_expected_dataframe(40.0, 66.0))
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)
    measure_data = measure.get_measure_data_from_sim(
        numerator_data=deaths_data, denominator_data=total_person_time_data
    )

    expected_measure_data = get_expected_dataframe(2.0 / 40.0, 4.0 / 66.0)
    assert_frame_equal(measure_data, expected_measure_data)
    assert_frame_equal(measure_data_from_ratio, expected_measure_data)


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

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=deaths_data,
        denominator_data=person_time_data,
    )

    # Expected dataframe for the numerator and denominator data
    # The Deaths formatter with a specific cause will filter for that cause
    # The PersonTime formatter with a specific state will filter for that state
    assert_frame_equal(ratio_datasets["numerator_data"], get_expected_dataframe(2.0, 4.0))
    assert_frame_equal(ratio_datasets["denominator_data"], get_expected_dataframe(23.0, 37.0))
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=deaths_data, denominator_data=person_time_data
    )

    expected_measure_data = get_expected_dataframe(2.0 / 23.0, 4.0 / 37.0)
    assert_frame_equal(measure_data, expected_measure_data)
    assert_frame_equal(measure_data_from_ratio, expected_measure_data)


def test_risk_exposure(risk_state_person_time_data: pd.DataFrame) -> None:
    """Test the RiskExposure measure."""
    risk_factor = "child_stunting"
    measure = RiskExposure(risk_factor)
    assert measure.measure_key == f"risk_factor.{risk_factor}.exposure"
    assert measure.sim_datasets == {
        "numerator_data": f"person_time_{risk_factor}",
        "denominator_data": f"person_time_{risk_factor}",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=risk_state_person_time_data,
        denominator_data=risk_state_person_time_data,
    )

    # Expected ratio data:
    # Numerator: person time in each specific risk state (cat1, cat2, cat3)
    # Denominator: total person time across all risk states for each stratification
    # Total person time per stratification: A = 8+12+15 = 35, B = 20+6+10 = 36
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("cat1", "A"),
            ("cat2", "A"),
            ("cat3", "A"),
            ("cat1", "B"),
            ("cat2", "B"),
            ("cat3", "B"),
        ],
        names=["parameter", "stratify_column"],
    )
    expected_numerator_data = pd.DataFrame(
        {
            "value": [8.0, 12.0, 15.0, 20.0, 6.0, 10.0],
        },
        index=expected_index,
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [35.0, 35.0, 35.0, 36.0, 36.0, 36.0],
        },
        index=expected_index,
    )

    assert_frame_equal(ratio_datasets["numerator_data"], expected_numerator_data)
    assert_frame_equal(ratio_datasets["denominator_data"], expected_denominator_data)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=risk_state_person_time_data,
        denominator_data=risk_state_person_time_data,
    )

    expected_measure_data = pd.DataFrame(
        {
            "value": [
                8.0 / 35.0,
                12.0 / 35.0,
                15.0 / 35.0,
                20.0 / 36.0,
                6.0 / 36.0,
                10.0 / 36.0,
            ]
        },
        index=expected_index,
    )
    assert_frame_equal(measure_data, expected_measure_data)


def test_population_structure(person_time_data: pd.DataFrame) -> None:
    """Test the PopulationStructure measure."""
    scenario_columns = ["scenario"]
    measure = PopulationStructure(scenario_columns)

    assert measure.measure_key == "population.structure"
    assert measure.sim_datasets == {
        "numerator_data": "person_time_total",
        "denominator_data": "person_time_total",
    }
    assert measure.artifact_datasets == {"artifact_data": measure.measure_key}

    ratio_datasets = measure.get_ratio_datasets_from_sim(
        numerator_data=person_time_data,
        denominator_data=person_time_data,
    )

    expected_denominator_data = pd.DataFrame(
        {
            "value": [17.0 + 23.0 + 29.0 + 37.0],
        },
        index=pd.Index(
            ["baseline"],
            name="scenario",
        ),
    )

    assert_frame_equal(
        ratio_datasets["numerator_data"], get_expected_dataframe(17.0 + 23.0, 29.0 + 37.0)
    )
    assert_frame_equal(ratio_datasets["denominator_data"], expected_denominator_data)
    measure_data_from_ratio = measure.get_measure_data_from_ratio(**ratio_datasets)

    measure_data = measure.get_measure_data_from_sim(
        numerator_data=person_time_data, denominator_data=person_time_data
    )

    expected_measure_data = get_expected_dataframe(
        (17.0 + 23.0) / (17.0 + 23.0 + 29.0 + 37.0),
        (29.0 + 37.0) / (17.0 + 23.0 + 29.0 + 37.0),
    )
    assert_frame_equal(measure_data, expected_measure_data)
    assert_frame_equal(measure_data_from_ratio, expected_measure_data)


@pytest.mark.parametrize(
    "measure_key,expected_class",
    [
        ("cause.heart_disease.incidence_rate", Incidence),
        ("cause.diabetes.prevalence", Prevalence),
        ("cause.tuberculosis.remission_rate", SIRemission),
        ("cause.cancer.cause_specific_mortality_rate", CauseSpecificMortalityRate),
        ("cause.stroke.excess_mortality_rate", ExcessMortalityRate),
        ("risk_factor.child_wasting.exposure", RiskExposure),
        ("population.structure", PopulationStructure),
    ],
)
def test_get_measure_from_key(measure_key: str, expected_class: type[RatioMeasure]) -> None:
    """Test get_measure_from_key for 3-part measure keys."""
    scenario_columns = ["scenario"]

    measure = get_measure_from_key(measure_key, scenario_columns)
    assert isinstance(measure, expected_class)
    assert measure.measure_key == measure_key
    if measure_key == "population.structure":
        assert isinstance(measure.denominator, TotalPopulationPersonTime)
        assert measure.denominator.scenario_columns == scenario_columns


@pytest.mark.parametrize(
    "invalid_key,expected_error",
    [
        ("invalid", ValueError),
        ("too.many.parts.here", ValueError),
        ("", ValueError),
        ("invalid_entity.something.measure", KeyError),
        ("cause.heart_disease.invalid_measure", KeyError),
        ("risk_factor.child_wasting.invalid_measure", KeyError),
        ("population.invalid_measure", KeyError),
    ],
)
def test_get_measure_from_key_invalid_inputs(
    invalid_key: str, expected_error: type[Exception]
) -> None:
    """Test get_measure_from_key with invalid inputs."""
    scenario_columns = ["scenario"]

    with pytest.raises(expected_error):
        get_measure_from_key(invalid_key, scenario_columns)
