from unittest import mock

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.comparison import (
    DataSource,
    FuzzyComparison,
    RatioMeasure,
)


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
def mock_ratio_measure(test_data: pd.DataFrame) -> RatioMeasure:
    measure_data = test_data.copy()
    measure_data["value"] = measure_data["numerator"] / measure_data["denominator"]
    measure_data = measure_data.drop(columns=["numerator", "denominator"])

    measure = mock.Mock(spec=RatioMeasure)
    measure.measure_key = "mock_measure"
    measure.get_measure_data_from_ratio.return_value = measure_data
    return measure


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
    assert not "value" in comparison.reference_data.columns
    assert list(comparison.stratifications) == ["year", "sex"]


def test_fuzzy_comparison_metadata(
    mock_ratio_measure: RatioMeasure, test_data: pd.DataFrame, reference_data: pd.DataFrame
) -> None:
    comparison = FuzzyComparison(
        mock_ratio_measure, DataSource.SIM, test_data, DataSource.GBD, reference_data
    )

    metadata = comparison.metadata.data

    assert metadata["Property"][0] == "Measure Key"
    assert metadata["Test Data"][0] == "mock_measure"
    assert metadata["Reference Data"][0] == "mock_measure"

    # Check sources
    assert metadata["Test Data"][1] == "sim"
    assert metadata["Reference Data"][1] == "gbd"

    # Check index columns
    assert metadata["Test Data"][2] == "year, sex, age, input_draw"
    assert metadata["Reference Data"][2] == "year, sex, age"
    # Check size
    assert metadata["Test Data"][3] == "3 rows × 2 columns"
    assert metadata["Reference Data"][3] == "3 rows × 1 columns"

    # Check num_draws
    assert metadata["Test Data"][4] == "1"
    assert metadata["Reference Data"][4] == "0"

    # Check draw sample
    assert metadata["Test Data"][5] == "[0]"
    assert metadata["Reference Data"][5] == "[]"


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
    assert len(diff) == expected_rows
    assert diff.index.names == stratifications
    assert "test_rate" in diff.columns
    assert "reference_rate" in diff.columns
    assert "percent_error" in diff.columns

    # Test returning all rows
    all_diff = comparison.get_diff(stratifications=stratifications, num_rows="all")
    assert len(all_diff) == expected_rows

    # Test sorting
    # descending order
    sorted_desc = comparison.get_diff(
        stratifications=stratifications, sort_by="percent_error", ascending=False
    )
    assert sorted_desc.iloc[0]["percent_error"] >= sorted_desc.iloc[-1]["percent_error"]
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
