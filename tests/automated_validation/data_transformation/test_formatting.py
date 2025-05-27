import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from unittest.mock import MagicMock, patch

from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
)
from vivarium_testing_utils.automated_validation.data_transformation.formatting import (
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


def test_total_pt(person_time_data: pd.DataFrame) -> None:
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


def test_total_person_time_init() -> None:
    """Test TotalPersonTime formatter initialization."""
    formatter = TotalPersonTime()

    assert formatter.measure == "person_time"
    assert formatter.entity == "population"
    assert formatter.filter_value == "total"
    assert formatter.filter_column == "sub_entity"
    assert formatter.new_value_column_name == "total_person_time"

    # Should raise error if data_key accessed before get_data_key is called
    with pytest.raises(ValueError, match="No data key has been set"):
        _ = formatter.data_key


def test_total_person_time_get_data_key() -> None:
    """Test TotalPersonTime get_data_key method."""
    formatter = TotalPersonTime()

    # Create a mock data loader
    mock_data_loader = MagicMock()
    mock_data_loader.get_sim_outputs.return_value = [
        "person_time_disease",
        "deaths",
        "person_time_other",
    ]

    # Test data key selection
    data_key = formatter.get_data_key(mock_data_loader)
    assert data_key == "person_time_disease"
    assert formatter.data_key == "person_time_disease"  # Now accessible via property

    # Test setting data_key directly
    formatter.data_key = "custom_person_time"
    assert formatter.data_key == "custom_person_time"

    # Test error when no person time datasets
    formatter = TotalPersonTime()  # Fresh instance
    mock_data_loader.get_sim_outputs.return_value = ["deaths", "transition_count_disease"]
    with pytest.raises(ValueError, match="No person time datasets available"):
        formatter.get_data_key(mock_data_loader)


def test_total_person_time_validate_consistency() -> None:
    """Test TotalPersonTime validate_person_time_consistency method."""
    # Create a mock data loader
    mock_data_loader = MagicMock()

    # Test with only one person time dataset
    mock_data_loader.get_sim_outputs.return_value = ["person_time_disease"]
    assert TotalPersonTime.validate_person_time_consistency(mock_data_loader) is True

    # Test with consistent person time datasets
    mock_data_loader.get_sim_outputs.return_value = [
        "person_time_disease",
        "person_time_other",
    ]

    # Create two datasets with the same total
    disease_pt = pd.DataFrame(
        {"value": [10, 20]},
        index=pd.MultiIndex.from_tuples(
            [
                ("person_time", "cause", "disease", "state1", "group1"),
                ("person_time", "cause", "disease", "state2", "group1"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )

    other_pt = pd.DataFrame(
        {"value": [15, 15]},
        index=pd.MultiIndex.from_tuples(
            [
                ("person_time", "cause", "other", "state1", "group1"),
                ("person_time", "cause", "other", "state2", "group1"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )

    # Set up mock to return appropriate data
    def side_effect(dataset_key, source):
        if dataset_key == "person_time_disease":
            return disease_pt
        elif dataset_key == "person_time_other":
            return other_pt

    mock_data_loader.get_dataset.side_effect = side_effect

    # Both datasets sum to 30, should be consistent
    assert TotalPersonTime.validate_person_time_consistency(mock_data_loader) is True

    # Test with inconsistent person time datasets
    inconsistent_pt = pd.DataFrame(
        {"value": [10, 10]},
        index=pd.MultiIndex.from_tuples(
            [
                ("person_time", "cause", "other", "state1", "group1"),
                ("person_time", "cause", "other", "state2", "group1"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )

    def inconsistent_side_effect(dataset_key, source):
        if dataset_key == "person_time_disease":
            return disease_pt
        elif dataset_key == "person_time_other":
            return inconsistent_pt

    mock_data_loader.get_dataset.side_effect = inconsistent_side_effect

    # Dataset 1 sums to 30, dataset 2 sums to 20, should be inconsistent
    assert TotalPersonTime.validate_person_time_consistency(mock_data_loader) is False

    # Test with custom tolerance
    assert (
        TotalPersonTime.validate_person_time_consistency(mock_data_loader, tolerance=0.5)
        is True
    )
