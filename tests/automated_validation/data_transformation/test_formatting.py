import pandas as pd
from pandas.testing import assert_frame_equal

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
    TotalPersonTime,
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
    formatter = PersonTime("disease", "disease")
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
    formatter = PersonTime("disease")
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


def test_total_person_time_init(total_person_time_data: pd.DataFrame) -> None:
    """Test TotalPersonTime formatter initialization."""
    formatter = TotalPersonTime()

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
