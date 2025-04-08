import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
    PersonTime,
    SimDataFormatter,
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

    formatter = TransitionCounts("disease", "A", "B")
    formatted_dataset = formatter.format_dataset(dataset)
    assert formatted_dataset.equals(expected_dataframe)


def test_person_time() -> None:
    """Test PersonTime formatting."""
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
            "infected_person_time": [20, 40],
        },
        index=pd.Index(
            ["A", "B"],
            name="stratify_column",
        ),
    )

    formatter = PersonTime("disease", "infected")
    formatted_dataset = formatter.format_dataset(dataset)
    assert formatted_dataset.equals(expected_dataframe)


def test_total_pt():
    """Test PersonTime formatting with total state."""
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

    formatter = PersonTime("disease")
    formatted_dataset = formatter.format_dataset(dataset)
    assert formatted_dataset.equals(expected_dataframe)
