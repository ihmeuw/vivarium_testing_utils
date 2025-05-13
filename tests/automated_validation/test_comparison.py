from unittest import mock

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.comparison import (
    DataSource,
    FuzzyComparison,
    RatioMeasure,
)


@pytest.fixture
def mock_ratio_measure() -> RatioMeasure:
    measure = mock.Mock(spec=RatioMeasure)
    measure.measure_key = "mock_measure"
    measure.get_measure_data_from_ratio.return_value = pd.DataFrame(
        {"value": [0.1, 0.2, 0.3]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
            names=["year", "sex", "age"],
        ),
    )
    return measure


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {"numerator": [10, 20, 30], "denominator": [100, 100, 100]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0, 0), ("2020", "female", 0, 0), ("2025", "male", 0, 0)],
            names=["year", "sex", "age", "input_draw"],
        ),
    )


@pytest.fixture
def reference_data() -> pd.DataFrame:
    return pd.DataFrame(
        {"value": [0.11, 0.21, 0.29]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
            names=["year", "sex", "age"],
        ),
    )


def test_fuzzy_comparison_init(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    # Test initialization
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        stratifications=["year", "sex"],
    )

    assert comparison.measure == mock_ratio_measure
    assert comparison.test_source == DataSource.SIM
    assert comparison.test_data.equals(test_data)
    assert comparison.reference_source == DataSource.GBD
    assert "reference_rate" in comparison.reference_data.columns
    assert list(comparison.stratifications) == ["year", "sex"]


def test_fuzzy_comparison_metadata(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    metadata = comparison.metadata
    assert metadata is not None
    # The metadata returns a styled dataframe, check the underlying data
    data = metadata.data
    assert data["Property"][0] == "Measure Key"
    assert data["Test Data"][0] == "mock_measure"
    assert data["Reference Data"][0] == "mock_measure"

    # Check sources
    assert data["Test Data"][1] == "sim"
    assert data["Reference Data"][1] == "gbd"

    # Check num_draws
    assert data["Test Data"][4] == "1"
    assert data["Reference Data"][4] == "0"


@pytest.mark.parametrize(
    "stratifications, expected_rows",
    [
        (["year"], 2),  # Stratify by year only (2020, 2025)
        (["sex"], 2),  # Stratify by sex only (male, female)
        (["year", "sex"], 3),  # Stratify by year and sex (all original combinations)
        ([], 3),  # No stratification, return all rows
    ],
)
def test_fuzzy_comparison_get_diff(
    mock_ratio_measure: RatioMeasure,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    stratifications: list[str],
    expected_rows: int,
) -> None:
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    diff = comparison.get_diff(stratifications=stratifications)
    assert len(diff) == min(expected_rows, 10)  # Default num_rows is 10
    assert "test_rate" in diff.columns
    assert "reference_rate" in diff.columns
    assert "percent_error" in diff.columns

    # Test returning all rows
    all_diff = comparison.get_diff(stratifications=stratifications, num_rows="all")
    assert len(all_diff) == expected_rows

    # Test sorting
    sorted_asc = comparison.get_diff(
        stratifications=stratifications, sort_by="percent_error", ascending=True
    )
    assert sorted_asc.iloc[0]["percent_error"] <= sorted_asc.iloc[-1]["percent_error"]


def test_fuzzy_comparison_verify_not_implemented(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    with pytest.raises(NotImplementedError):
        comparison.verify()
