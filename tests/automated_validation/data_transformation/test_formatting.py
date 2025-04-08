import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
    TransitionCounts,
    drop_redundant_index,
)


def test_drop_redundant_index() -> None:
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
    formatted_dataset = drop_redundant_index(dataset, "redundant_column", "redundant_column")
    assert formatted_dataset.equals(expected_dataframe)


def test_transition_counts() -> None:
    """Test TransitionCounts formatting."""
    formatter = TransitionCounts("disease", "A", "B")
    assert formatter.type == "transition_count"
    assert formatter.cause == "disease"
    assert formatter.data_key == "transition_count_disease"
    assert formatter.start_state == "A"
    assert formatter.end_state == "B"
    assert formatter.transition_string == "A_TO_B"
    assert formatter.groupby_column == "disease_transition"
    assert formatter.renamed_column == "A_TO_B_transition_count"

    # Create a mock dataset
    dataset = pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("disease", "True", "A_TO_B"),
                ("disease", "False", "A_TO_B"),
                ("disease", "True", "B_TO_C"),
                ("disease", "False", "B_TO_C"),
            ],
            names=["cause", "stratify_column", "disease_transition"],
        ),
    )
    expected_dataframe = pd.DataFrame(
        {
            "A_TO_B_transition_count": [10, 20],
        },
        index=pd.Index(
            ["True", "False"],
            name="stratify_column",
        ),
    )

    assert formatter.format_dataset(dataset).equals(expected_dataframe)


def test_person_time() -> None:
    """Test PersonTime formatting."""
    # Create a mock dataset
    formatter = PersonTime("disease", "infected")
    assert formatter.type == "person_time"
    assert formatter.cause == "disease"
    assert formatter.data_key == "person_time_disease"
    assert formatter.state == "infected"
    assert formatter.groupby_column == "disease_state"
    assert formatter.renamed_column == "infected_person_time"

    dataset = pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("disease", "susceptible", "A"),
                ("disease", "infected", "A"),
                ("disease", "susceptible", "B"),
                ("disease", "infected", "B"),
            ],
            names=["cause", "disease_state", "stratify_column"],
        ),
    )
    expected_dataframe = pd.DataFrame(
        {
            "infected_person_time": [20, 40],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert formatter.format_dataset(dataset).equals(expected_dataframe)


def test_total_pt():
    """Test PersonTime formatting with total state."""
    formatter = PersonTime("disease")
    assert formatter.type == "person_time"
    assert formatter.cause == "disease"
    assert formatter.data_key == "person_time_disease"
    assert formatter.state == "total"
    assert formatter.groupby_column == "disease_state"
    assert formatter.renamed_column == "total_person_time"

    # Create a mock dataset
    dataset = pd.DataFrame(
        {
            "value": [10, 20, 30, 40],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("disease", "susceptible", "A"),
                ("disease", "infected", "A"),
                ("disease", "susceptible", "B"),
                ("disease", "infected", "B"),
            ],
            names=["cause", "disease_state", "stratify_column"],
        ),
    )
    expected_dataframe = pd.DataFrame(
        {
            "total_person_time": [30, 70],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    assert formatter.format_dataset(dataset).equals(expected_dataframe)
