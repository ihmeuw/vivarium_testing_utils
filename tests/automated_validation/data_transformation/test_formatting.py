import pandas as pd
from pandas.testing import assert_frame_equal

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    Deaths,
    RiskStatePersonTime,
    StatePersonTime,
    TransitionCounts,
)


def test_transition_counts(transition_count_data: pd.DataFrame) -> None:
    """Test TransitionCounts formatting."""
    formatter = TransitionCounts("disease", "susceptible_to_disease", "disease")
    # assert formatter has right number of attrs
    assert len(formatter.__dict__) == 7
    assert formatter.measure == "transition_count"
    assert formatter.entity == "disease"
    assert formatter.data_key == "transition_count_disease"
    assert formatter.filter_value == "susceptible_to_disease_to_disease"
    assert formatter.filters == {"sub_entity": ["susceptible_to_disease_to_disease"]}
    assert (
        formatter.new_value_column_name
        == "susceptible_to_disease_to_disease_transition_count"
    )
    assert formatter.unused_columns == ["measure", "entity_type", "entity"]

    expected_dataframe = pd.DataFrame(
        {
            formatter.new_value_column_name: [3.0, 5.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert_frame_equal(formatter.format_dataset(transition_count_data), expected_dataframe)


def test_person_time(person_time_data: pd.DataFrame) -> None:
    """Test PersonTime formatting."""
    # Create a mock dataset
    formatter = StatePersonTime("disease", "disease")
    assert len(formatter.__dict__) == 7
    assert formatter.measure == "person_time"
    assert formatter.entity == "disease"
    assert formatter.data_key == "person_time_disease"
    assert formatter.filters == {"sub_entity": ["disease"]}
    assert formatter.new_value_column_name == "disease_person_time"
    assert formatter.unused_columns == ["measure", "entity_type", "entity"]

    expected_dataframe = pd.DataFrame(
        {
            "disease_person_time": [23.0, 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert_frame_equal(formatter.format_dataset(person_time_data), expected_dataframe)


def test_person_time_state_total(person_time_data: pd.DataFrame) -> None:
    """Test PersonTime formatting with total state."""
    formatter = StatePersonTime("disease")
    assert len(formatter.__dict__) == 7
    assert formatter.measure == "person_time"
    assert formatter.entity == "disease"
    assert formatter.data_key == "person_time_disease"
    assert formatter.filters == {"sub_entity": ["total"]}
    assert formatter.new_value_column_name == "total_person_time"
    assert formatter.unused_columns == ["measure", "entity_type", "entity"]

    expected_dataframe = pd.DataFrame(
        {
            "total_person_time": [17.0 + 23.0, 29.0 + 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert_frame_equal(formatter.format_dataset(person_time_data), expected_dataframe)


def test_total_person_time(total_person_time_data: pd.DataFrame) -> None:
    """Test StatePersonTime formatter initialization with total."""
    formatter = StatePersonTime()

    assert formatter.measure == "person_time"
    assert formatter.entity == "total"
    assert formatter.data_key == "person_time_total"
    assert formatter.new_value_column_name == "total_person_time"
    assert formatter.unused_columns == ["measure", "entity_type", "entity"]
    assert formatter.filters == {"sub_entity": ["total"]}

    expected_dataframe = pd.DataFrame(
        {
            "total_person_time": [17.0 + 23.0, 29.0 + 37.0],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert_frame_equal(formatter.format_dataset(total_person_time_data), expected_dataframe)


def test_deaths_cause_specific(deaths_data: pd.DataFrame) -> None:
    """Test Deaths formatter with a specific cause."""
    formatter = Deaths("disease")

    assert formatter.measure == "deaths"
    assert formatter.data_key == "deaths"
    assert formatter.filters == {"entity": ["disease"], "sub_entity": ["disease"]}
    assert formatter.new_value_column_name == "disease_deaths"
    assert formatter.unused_columns == ["measure", "entity_type"]

    # Filter out only data related to the disease itself, since we want
    # deaths directly attributed to the disease
    expected_dataframe = pd.DataFrame(
        {
            "disease_deaths": [2.0, 4.0],  # Deaths data for the disease itself
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert_frame_equal(formatter.format_dataset(deaths_data), expected_dataframe)


def test_deaths_all_causes(deaths_data: pd.DataFrame) -> None:
    """Test Deaths formatter for all causes."""
    formatter = Deaths("all_causes")

    assert formatter.measure == "deaths"
    assert formatter.data_key == "deaths"
    assert formatter.filters == {"entity": ["total"], "sub_entity": ["total"]}
    assert formatter.new_value_column_name == "total_deaths"
    assert formatter.unused_columns == ["measure", "entity_type"]

    expected_dataframe = pd.DataFrame(
        {
            "total_deaths": [5.0, 9.0],  # All deaths, regardless of cause
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )
    assert_frame_equal(formatter.format_dataset(deaths_data), expected_dataframe)


def test_risk_state_person_time(risk_state_person_time_data: pd.DataFrame) -> None:
    """Test RiskStatePersonTime formatting without sum_all."""
    formatter = RiskStatePersonTime("child_stunting")

    assert formatter.entity == "child_stunting"
    assert formatter.data_key == "person_time_child_stunting"
    assert formatter.sum_all == False
    assert formatter.new_value_column_name == "person_time"
    assert formatter.unused_columns == ["measure", "entity_type", "entity"]

    expected_dataframe = pd.DataFrame(
        {
            "person_time": [100.0, 150.0, 200.0, 250.0, 75.0, 125.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cat1", "A"),
                ("cat2", "A"),
                ("cat3", "A"),
                ("cat1", "B"),
                ("cat2", "B"),
                ("cat3", "B"),
            ],
            names=["parameter", "stratify_column"],
        ),
    )

    assert_frame_equal(
        formatter.format_dataset(risk_state_person_time_data), expected_dataframe
    )


def test_risk_state_person_time_sum_all(risk_state_person_time_data: pd.DataFrame) -> None:
    """Test RiskStatePersonTime formatting with sum_all=True."""
    formatter = RiskStatePersonTime("child_stunting", sum_all=True)

    assert formatter.entity == "child_stunting"
    assert formatter.data_key == "person_time_child_stunting"
    assert formatter.sum_all == True
    assert formatter.new_value_column_name == "person_time_total"
    assert formatter.unused_columns == ["measure", "entity_type", "entity"]

    # With sum_all=True, each risk state gets the total person time for its stratification
    # Total for A = 100+150+200 = 450, Total for B = 250+75+125 = 450
    expected_dataframe = pd.DataFrame(
        {
            "person_time_total": [450.0, 450.0, 450.0, 450.0, 450.0, 450.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cat1", "A"),
                ("cat2", "A"),
                ("cat3", "A"),
                ("cat1", "B"),
                ("cat2", "B"),
                ("cat3", "B"),
            ],
            names=["parameter", "stratify_column"],
        ),
    )

    assert_frame_equal(
        formatter.format_dataset(risk_state_person_time_data), expected_dataframe
    )
