from unittest import mock

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.comparison import (
    DataSource,
    FuzzyComparison,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    get_singular_indices,
)


@pytest.fixture
def test_data() -> pd.DataFrame:
    """A sample test data DataFrame with draws."""
    return pd.DataFrame(
        {"numerator": [10, 20, 30], "denominator": [100, 100, 100]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0, 0), ("2020", "female", 0, 0), ("2025", "male", 0, 0)],
            names=["year", "sex", "age", "input_draw"],
        ),
    )


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """A sample test data DataFrame without draws."""
    return pd.DataFrame(
        {"value": [0.11, 0.21, 0.29]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
            names=["year", "sex", "age"],
        ),
    )


@pytest.fixture
def mock_ratio_measure() -> RatioMeasure:
    """Create generic mock RatioMeasure for testing."""

    def _get_measure_data_from_ratio(test_data: pd.DataFrame) -> pd.DataFrame:
        measure_data = test_data.copy()
        measure_data["value"] = measure_data["numerator"] / measure_data["denominator"]
        measure_data = measure_data.drop(columns=["numerator", "denominator"])
        return measure_data

    measure = mock.Mock(spec=RatioMeasure)
    measure.measure_key = "mock_measure"
    measure.get_measure_data_from_ratio.side_effect = _get_measure_data_from_ratio
    return measure


def test_fuzzy_comparison_init(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    """Test the initialization of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        stratifications=[],
    )

    assert comparison.measure == mock_ratio_measure
    assert comparison.test_source == DataSource.SIM
    assert comparison.test_data.equals(test_data)
    assert comparison.reference_source == DataSource.GBD
    assert "reference_rate" in comparison.reference_data.columns
    assert not "value" in comparison.reference_data.columns
    assert list(comparison.stratifications) == []


def test_fuzzy_comparison_metadata(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    """Test the metadata property of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    metadata = comparison.metadata

    expected_metadata = [
        ("Measure Key", "mock_measure", "mock_measure"),
        ("Source", "sim", "gbd"),
        ("Index Columns", "year, sex, age, input_draw", "year, sex, age"),
        ("Size", "3 rows × 2 columns", "3 rows × 1 columns"),
        ("Num Draws", "1", "0"),
        ("Input Draws", "[0]", "[]"),
    ]
    assert metadata.index.name == "Property"
    assert metadata.shape == (6, 2)
    assert metadata.columns.tolist() == ["Test Data", "Reference Data"]
    for property_name, test_value, reference_value in expected_metadata:
        assert metadata.loc[property_name]["Test Data"] == test_value
        assert metadata.loc[property_name]["Reference Data"] == reference_value


def test_fuzzy_comparison_get_diff(
    mock_ratio_measure: RatioMeasure,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
) -> None:
    """Test the get_diff method of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    diff = comparison.get_diff(stratifications=[], num_rows=1)
    assert len(diff) == 1
    assert "test_rate" in diff.columns
    assert "reference_rate" in diff.columns
    assert "percent_error" in diff.columns

    # Test returning all rows
    all_diff = comparison.get_diff(stratifications=[], num_rows="all")
    assert len(all_diff) == 3

    # Test sorting
    # descending order
    sorted_desc = comparison.get_diff(sort_by="percent_error", ascending=False)
    assert sorted_desc.iloc[0]["percent_error"] >= sorted_desc.iloc[-1]["percent_error"]
    sorted_asc = comparison.get_diff(sort_by="percent_error", ascending=True)
    assert sorted_asc.iloc[0]["percent_error"] <= sorted_asc.iloc[-1]["percent_error"]


def test_fuzzy_comparison_init_with_stratifications(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    """Test that FuzzyComparison raises NotImplementedError when initialized with non-empty stratifications."""
    with pytest.raises(
        NotImplementedError, match="Non-default stratifications require rate aggregations"
    ):
        FuzzyComparison(
            mock_ratio_measure,
            DataSource.SIM,
            test_data,
            DataSource.GBD,
            reference_data,
            stratifications=["year"],
        )


def test_fuzzy_comparison_get_diff_with_stratifications(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    """Test that FuzzyComparison.get_diff raises NotImplementedError when called with non-empty stratifications."""
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    with pytest.raises(
        NotImplementedError, match="Non-default stratifications require rate aggregations"
    ):
        comparison.get_diff(stratifications=["year"])


def test_fuzzy_comparison_verify_not_implemented(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    """ "FuzzyComparison.verify() is not implemented."""
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    with pytest.raises(NotImplementedError):
        comparison.verify()


def test_fuzzy_comparison_align_datasets_with_singular_reference_index(
    mock_ratio_measure: RatioMeasure,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
) -> None:
    """Test that _align_datasets correctly handles singular reference-only indices."""
    reference_data_with_singular_index = reference_data.copy()
    reference_data_with_singular_index["location"] = ["Global", "Global", "Global"]
    reference_data_with_singular_index.set_index("location", append=True, inplace=True)

    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data_with_singular_index,
    )

    # Verify the singular index exists
    assert "location" in comparison.reference_data.index.names
    assert "location" not in comparison.test_data.index.names

    # Verify it's detected as a singular index
    singular_indices = get_singular_indices(comparison.reference_data)
    assert "location" in singular_indices
    assert singular_indices["location"] == "Global"

    # Execute
    test_data, reference_data = comparison._align_datasets()

    # Verify the singular index was dropped
    assert "location" not in reference_data.index.names
    assert test_data.shape[0] == reference_data.shape[0]


def test_fuzzy_comparison_align_datasets_with_non_singular_reference_index(
    mock_ratio_measure: RatioMeasure,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
) -> None:
    """Test that _align_datasets raises ValueError for non-singular reference-only indices."""
    reference_data_with_non_singular_index = reference_data.copy()
    reference_data_with_non_singular_index["location"] = ["Global", "USA", "USA"]
    reference_data_with_non_singular_index.set_index("location", append=True, inplace=True)

    # Setup
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data_with_non_singular_index,
    )

    # Verify the non-singular index exists
    assert "location" in comparison.reference_data.index.names
    assert "location" not in comparison.test_data.index.names

    # Verify it's not detected as a singular index
    singular_indices = get_singular_indices(comparison.reference_data)
    assert "location" not in singular_indices

    # Execute and verify error is raised with correct message
    with pytest.raises(ValueError, match="Reference data has non-trivial index location"):
        comparison._align_datasets()
