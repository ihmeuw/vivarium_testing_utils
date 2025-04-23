import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
    TransitionCounts,
    _drop_redundant_index,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
)
from pandera.typing import DataFrame


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
    assert formatted_dataset.equals(expected_dataframe)

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


def test_transition_counts(transition_count_data: DataFrame[SimOutputData]) -> None:
    """Test TransitionCounts formatting."""
    formatter = TransitionCounts("disease", "susceptible_to_disease", "disease")
    # assert formatter has right number of attrs
    assert len(formatter.__dict__) == 7
    assert formatter.type == "transition_count"
    assert formatter.cause == "disease"
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

    assert formatter.format_dataset(transition_count_data).equals(expected_dataframe)


def test_person_time(person_time_data: DataFrame[SimOutputData]) -> None:
    """Test PersonTime formatting."""
    # Create a mock dataset
    formatter = PersonTime("disease", "disease")
    assert len(formatter.__dict__) == 7
    assert formatter.type == "person_time"
    assert formatter.cause == "disease"
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

    assert formatter.format_dataset(person_time_data).equals(expected_dataframe)


def test_total_pt(person_time_data: DataFrame[SimOutputData]) -> None:
    """Test PersonTime formatting with total state."""
    formatter = PersonTime("disease")
    assert len(formatter.__dict__) == 7
    assert formatter.type == "person_time"
    assert formatter.cause == "disease"
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

    assert formatter.format_dataset(person_time_data).equals(expected_dataframe)
