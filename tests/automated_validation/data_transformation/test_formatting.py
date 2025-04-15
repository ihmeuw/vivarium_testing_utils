import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
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
    assert formatted_dataset.equals(expected_dataframe)


def test_transition_counts(transition_count_data: pd.DataFrame) -> None:
    """Test TransitionCounts formatting."""
    formatter = TransitionCounts("disease", "susceptible_to_disease", "disease")
    assert formatter.type == "transition_count"
    assert formatter.cause == "disease"
    assert formatter.data_key == "transition_count_disease"
    assert formatter.filter_value == "susceptible_to_disease_to_disease"
    assert formatter.filter_column == "sub_entity"
    assert (
        formatter.new_value_column_name
        == "susceptible_to_disease_to_disease_transition_count"
    )

    expected_dataframe = pd.DataFrame(
        {
            formatter.new_value_column_name: [3, 5],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert formatter.format_dataset(transition_count_data).equals(expected_dataframe)


def test_person_time(person_time_data: pd.DataFrame) -> None:
    """Test PersonTime formatting."""
    # Create a mock dataset
    formatter = PersonTime("disease", "disease")
    assert formatter.type == "person_time"
    assert formatter.cause == "disease"
    assert formatter.data_key == "person_time_disease"
    assert formatter.filter_column == "sub_entity"
    assert formatter.new_value_column_name == "disease_person_time"

    expected_dataframe = pd.DataFrame(
        {
            "disease_person_time": [23, 37],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert formatter.format_dataset(person_time_data).equals(expected_dataframe)


def test_total_pt(person_time_data: pd.DataFrame) -> None:
    """Test PersonTime formatting with total state."""
    formatter = PersonTime("disease")
    assert formatter.type == "person_time"
    assert formatter.cause == "disease"
    assert formatter.data_key == "person_time_disease"
    assert formatter.filter_column == "sub_entity"
    assert formatter.new_value_column_name == "total_person_time"

    expected_dataframe = pd.DataFrame(
        {
            "total_person_time": [17 + 23, 29 + 37],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert formatter.format_dataset(person_time_data).equals(expected_dataframe)
