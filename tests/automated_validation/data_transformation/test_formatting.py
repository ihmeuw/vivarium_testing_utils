import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    Deaths,
    PersonTime,
    TotalPersonTime,
    TransitionCounts,
    _drop_redundant_index,
)


def test__drop_redundant_index() -> None:
    """Test drop_redundant_index function."""
    # Create a mock dataset
    dataset = pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("redundant_column", "heterogeneous"),
                ("redundant_column", "values"),
                ("redundant_column", "in_this"),
                ("redundant_column", "column"),
            ],
            names=["redundant_column", "interesting_column"],
        ),
    )
    expected_dataframe = pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
        },
        index=pd.Index(
            ["heterogeneous", "values", "in_this", "column"],
            name="interesting_column",
        ),
    )
    # Call the function to drop the redundant index
    formatted_dataset = _drop_redundant_index(dataset, "redundant_column", "redundant_column")
    assert_frame_equal(formatted_dataset, expected_dataframe)

    # Test that we raise an error if the column has more than one value
    dataset = dataset.copy()
    dataset.index = pd.MultiIndex.from_tuples(
        [
            ("redundant_column", "heterogeneous"),
            ("redundant_column", "values"),
            ("redundant_column", "in_this"),
            ("not_redundant!", "column"),
        ],
        names=["redundant_column", "interesting_column"],
    )
    with pytest.raises(ValueError):
        _drop_redundant_index(dataset, "redundant_column", "redundant_column")


def test_transition_counts(transition_count_data: pd.DataFrame) -> None:
    """Test TransitionCounts formatting."""
    formatter = TransitionCounts("disease", "susceptible_to_disease", "disease")
    # assert formatter has right number of attrs
    assert len(formatter.__dict__) == 7
    assert formatter.measure == "transition_count"
    assert formatter.entity == "disease"
    assert formatter.data_key == "transition_count_disease"
    assert formatter.filter_value == "susceptible_to_disease_to_disease"
    assert formatter.filter_column == "sub_entity"
    assert (
        formatter.new_value_column_name
        == "susceptible_to_disease_to_disease_transition_count"
    )
    assert formatter.redundant_columns == {
        "measure": "transition_count",
        "entity_type": "cause",
        "entity": "disease",
    }

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
    assert formatter.filter_column == "sub_entity"
    assert formatter.new_value_column_name == "disease_person_time"
    assert formatter.redundant_columns == {
        "measure": "person_time",
        "entity_type": "cause",
        "entity": "disease",
    }

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
    assert formatter.filter_column == "sub_entity"
    assert formatter.new_value_column_name == "total_person_time"
    assert formatter.redundant_columns == {
        "measure": "person_time",
        "entity_type": "cause",
        "entity": "disease",
    }

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
    assert formatter.redundant_columns == {
        "measure": "person_time",
        "entity_type": "cause",
        "entity": "total",
    }

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
    assert formatter.filter_columns == ["entity", "sub_entity"]
    assert formatter.new_value_column_name == "disease_deaths"
    assert formatter.redundant_columns == {
        "measure": "deaths",
        "entity_type": "cause",
    }

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
    formatter = Deaths()

    assert formatter.measure == "deaths"
    assert formatter.data_key == "deaths"
    assert formatter.filter_columns == ["entity", "sub_entity"]
    assert formatter.new_value_column_name == "total_deaths"
    assert formatter.redundant_columns == {
        "measure": "deaths",
        "entity_type": "cause",
    }

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
