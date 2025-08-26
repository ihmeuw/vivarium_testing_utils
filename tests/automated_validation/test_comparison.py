from collections.abc import Collection
from unittest import mock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_check import check

from vivarium_testing_utils.automated_validation.comparison import DataSource, FuzzyComparison
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_transformation import calculations
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
)


@pytest.fixture
def test_data() -> dict[str, pd.DataFrame]:
    """A sample test data dictionary with separate numerator and denominator DataFrames."""
    index = pd.MultiIndex.from_tuples(
        [
            ("2020", "male", 0, 1, 1337, "baseline"),
            ("2020", "female", 0, 5, 1337, "baseline"),
            ("2025", "male", 0, 2, 42, "baseline"),
            ("2025", "male", 0, 2, 50, "baseline"),  # Add a seed to get marginalized over
        ],
        names=["year", "sex", "age", DRAW_INDEX, SEED_INDEX, "scenario"],
    )
    numerator_df = pd.DataFrame({"value": [10, 20, 30, 35]}, index=index)
    denominator_df = pd.DataFrame({"value": [100, 100, 100, 100]}, index=index)
    return {"numerator_data": numerator_df, "denominator_data": denominator_df}


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """A sample test data DataFrame without draws."""
    return pd.DataFrame(
        {"value": [0.12, 0.2, 0.29]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
            names=["year", "sex", "age"],
        ),
    )


@pytest.fixture
def mock_ratio_measure() -> RatioMeasure:
    """Create generic mock RatioMeasure for testing."""
    # Create mock formatters
    mock_numerator = mock.Mock()
    mock_numerator.name = "numerator"

    mock_denominator = mock.Mock()
    mock_denominator.name = "denominator"

    measure = mock.Mock(spec=RatioMeasure)
    measure.measure_key = "mock_measure"
    measure.numerator = mock_numerator
    measure.denominator = mock_denominator
    measure.get_measure_data_from_ratio.side_effect = calculations.ratio
    return measure


@pytest.fixture
def reference_weights() -> pd.DataFrame:
    """A sample weights DataFrame."""
    return pd.DataFrame(
        {"value": [0.15, 0.25, 0.35]},
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
            names=["year", "sex", "age"],
        ),
    )


def test_fuzzy_comparison_init(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test the initialization of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
        test_scenarios={"scenario": "baseline"},
    )
    # Scenario is dropped from test datasets in the FuzzyComparison constructor
    test_data = {key: dataset.droplevel("scenario") for key, dataset in test_data.items()}

    with check:
        assert comparison.measure == mock_ratio_measure
        assert comparison.test_source == DataSource.SIM
        assert comparison.test_datasets.keys() == {"numerator_data", "denominator_data"}
        assert comparison.test_datasets["numerator_data"].equals(test_data["numerator_data"])
        assert comparison.test_datasets["denominator_data"].equals(
            test_data["denominator_data"]
        )
        assert comparison.reference_source == DataSource.GBD
        assert comparison.reference_data.equals(reference_data)
        assert comparison.test_scenarios == {"scenario": "baseline"}
        assert not comparison.reference_scenarios


def test_fuzzy_comparison_metadata(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test the metadata property of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
        test_scenarios={"scenario": "baseline"},
    )

    metadata = comparison.metadata

    expected_metadata = [
        ("Measure Key", "mock_measure", "mock_measure"),
        ("Source", "sim", "gbd"),
        (
            "Index Columns",
            "year, sex, age, input_draw, random_seed",
            "year, sex, age",
        ),
        ("Size", "4 rows × 1 columns", "3 rows × 1 columns"),
        ("Num Draws", "3", "N/A"),
        ("Input Draws", "[1, 2, 5]", "N/A"),
        ("Num Seeds", "3", "N/A"),
    ]
    assert metadata.index.name == "Property"
    assert metadata.shape == (7, 2)
    assert metadata.columns.tolist() == ["Test Data", "Reference Data"]
    for property_name, test_value, reference_value in expected_metadata:
        assert metadata.loc[property_name]["Test Data"] == test_value
        assert metadata.loc[property_name]["Reference Data"] == reference_value


def test_fuzzy_comparison_get_frame(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test the get_frame method of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )

    diff = comparison.get_frame(num_rows=1)

    with check:
        assert len(diff) == 1
        assert "test_rate" in diff.columns
        assert "reference_rate" in diff.columns
        assert "percent_error" in diff.columns
        assert DRAW_INDEX in diff.index.names
        assert SEED_INDEX not in diff.index.names

    # Test returning all rows
    all_diff = comparison.get_frame(num_rows="all")
    assert len(all_diff) == 3

    # Test sorting
    # descending order
    sorted_desc = comparison.get_frame(sort_by="percent_error", ascending=False)
    for i in range(len(sorted_desc) - 1):
        assert abs(sorted_desc.iloc[i]["percent_error"]) >= abs(
            sorted_desc.iloc[i + 1]["percent_error"]
        )
    sorted_asc = comparison.get_frame(sort_by="percent_error", ascending=True)
    for i in range(len(sorted_asc) - 1):
        assert abs(sorted_asc.iloc[i]["percent_error"]) <= abs(
            sorted_asc.iloc[i + 1]["percent_error"]
        )

    # Test sorting by reference rate
    sorted_by_ref = comparison.get_frame(sort_by="reference_rate", ascending=True)
    for i in range(len(sorted_by_ref) - 1):
        assert (
            sorted_by_ref.iloc[i]["reference_rate"]
            <= sorted_by_ref.iloc[i + 1]["reference_rate"]
        )


def test_fuzzy_comparison_get_frame_aggregated_draws(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test the get_frame method of the FuzzyComparison class with aggregated draws."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )
    diff = comparison.get_frame(num_rows="all", aggregate_draws=True)
    expected_df = pd.DataFrame(
        {
            "test_mean": [0.1, 0.2, 0.325],
            "test_2.5%": [0.1, 0.2, 0.325],
            "test_97.5%": [0.1, 0.2, 0.325],
            # Reference data has no draws and we have no stratifications so we just return the reference data
            "reference_rate": list(reference_data["value"]),
        },
        index=pd.MultiIndex.from_tuples(
            [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
            names=["year", "sex", "age"],
        ),
    )
    assert_frame_equal(diff, expected_df)


@pytest.mark.parametrize("stratifications", [None, ["year"], []])
@pytest.mark.parametrize("aggregate", [True, False])
@pytest.mark.parametrize("draws", ["test", "reference", "both", "neither"])
def test_fuzzy_comparison_get_frame_parametrized(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
    stratifications: Collection[str] | None,
    aggregate: bool,
    draws: str,
) -> None:
    """Test that FuzzyComparison.get_frame raises NotImplementedError when called with non-empty stratifications."""
    draw_values = list(
        test_data["numerator_data"].index.get_level_values(DRAW_INDEX).unique()
    )
    if draws == "test":
        # This is how the fixtures are set up
        for key in test_data:
            assert "input_draw" in test_data[key].index.names
        assert "input_draw" not in reference_data.index.names
        assert "input_draw" not in reference_weights.index.names
    if draws in ["reference", "both"]:
        # Remove draws from test data and add draws index level to reference datasets
        reference_data = _add_draws_to_dataframe(reference_data, draw_values)
        reference_weights = _add_draws_to_dataframe(reference_weights, draw_values)
    if draws in ["reference", "neither"]:
        # Remove draws from test dataset
        test_data = {
            dataset_key: test_data[dataset_key]
            .groupby(
                [
                    level
                    for level in test_data[dataset_key].index.names
                    if level != "input_draw"
                ]
            )
            .sum()
            for dataset_key in test_data
        }

    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )

    data = comparison.get_frame(stratifications=stratifications, aggregate_draws=aggregate)
    if stratifications is None:
        expected_index_names = [
            col
            for col in test_data["numerator_data"].index.names
            if col not in ["input_draw", "random_seed", "scenario"]
        ]
        if not aggregate and draws != "neither":
            expected_index_names += ["input_draw"]
        assert set(data.index.names) == set(expected_index_names)
    elif stratifications == ["year"]:
        assert set(data.index.names) == {"year"} if aggregate else {"year", "input_draw"}
    else:
        # stratifications is [] and all index levels are aggregated over
        assert not data.empty
        assert set(data.index.names) == {"index"} if aggregate else {"input_draw"}
    if aggregate:
        schema_mapper = {
            "test": {"test_mean", "test_2.5%", "test_97.5%", "reference_rate"},
            "reference": {"test_rate", "reference_mean", "reference_2.5%", "reference_97.5%"},
            "both": {
                "test_mean",
                "test_2.5%",
                "test_97.5%",
                "reference_mean",
                "reference_2.5%",
                "reference_97.5%",
            },
            "neither": {"test_rate", "reference_rate"},
        }
        expected_columns = schema_mapper[draws]
    else:
        expected_columns = {"test_rate", "reference_rate", "percent_error"}
    assert set(data.columns) == expected_columns


def test_fuzzy_comparison_verify_not_implemented(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """ "FuzzyComparison.verify() is not implemented."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )

    with pytest.raises(NotImplementedError):
        comparison.verify()


def test_get_metadata_from_dataset(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test we can extract metadata from a dataframe with draws."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )
    result = comparison._get_metadata_from_datasets("test")
    with check:
        assert result["source"] == DataSource.SIM.value
        assert result["index_columns"] == "year, sex, age, input_draw, random_seed, scenario"
        assert (
            result["size"] == "4 rows × 1 columns"
        )  # 2 years * 2 sexes * 3 draws * 2 seeds = 24 rows, 1 column
        assert result["num_draws"] == "3"
        assert result["input_draws"] == "[1, 2, 5]"
        assert result["num_seeds"] == "3"


def test_get_metadata_from_dataset_no_draws(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test we can extract metadata from a dataframe with draws."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )
    result = comparison._get_metadata_from_datasets("reference")
    with check:
        assert result["source"] == DataSource.GBD.value
        assert result["index_columns"] == "year, sex, age"
        assert (
            result["size"] == "3 rows × 1 columns"
        )  # 2 years * 2 sexes * 3 ages = 12 rows, 1 column
        assert "num_draws" not in result
        assert "input_draws" not in result
        assert "num_seeds" not in result


def test_fuzzy_comparison_align_datasets_with_non_singular_reference_index(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
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
        reference_weights,
    )

    # Verify the non-singular index exists
    assert "location" in comparison.reference_data.index.names
    assert "location" not in comparison.test_datasets["numerator_data"].index.names

    # Verify it's not detected as a singular index
    singular_indices = calculations.get_singular_indices(comparison.reference_data)
    assert "location" not in singular_indices


def test_fuzzy_comparison_align_datasets_calculation(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test _align_datasets with varying denominators to ensure ratios are calculated correctly."""

    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
        test_scenarios={"scenario": "baseline"},
    )

    aligned_test_data, aligned_reference_data = comparison._align_datasets()
    pd.testing.assert_frame_equal(aligned_reference_data, reference_data)

    expected_values = [10 / 100, 20 / 100, (30 + 35) / (100 + 100)]
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("2020", "male", 0, 1),
            ("2020", "female", 0, 5),
            ("2025", "male", 0, 2),
        ],
        names=["year", "sex", "age", DRAW_INDEX],
    )
    assert_frame_equal(
        aligned_test_data,
        pd.DataFrame(
            {"value": expected_values},
            index=expected_index,
        ),
    )


def test_aggregate_strata(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
) -> None:
    """Test that aggregate_strata correctly aggregates data."""
    comparison = FuzzyComparison(
        mock_ratio_measure,
        DataSource.SIM,
        test_data,
        DataSource.GBD,
        reference_data,
        reference_weights,
    )

    aggregated = comparison.aggregate_strata_reference(reference_data, ["age", "sex"])
    # (0, Male) = (0.12 * 0.15 + 0.29 * 0.35) / (0.15 + 0.35)
    expected = pd.DataFrame(
        {
            "value": [
                (0.12 * 0.15 + 0.29 * 0.35) / (0.15 + 0.35),
                (0.2 * 0.25) / 0.25,
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (0, "male"),
                (0, "female"),
            ],
            names=["age", "sex"],
        ),
    )
    assert isinstance(aggregated, pd.DataFrame)
    pd.testing.assert_frame_equal(aggregated, expected)

    with pytest.raises(ValueError, match="not found in reference data or weights"):
        comparison.aggregate_strata_reference(reference_data, ["dog", "cat"])


def _add_draws_to_dataframe(df: pd.DataFrame, draw_values: list[int]) -> pd.DataFrame:
    """Add a 'input_draw' index level to the DataFrame."""
    df["input_draw"] = draw_values
    return df.set_index("input_draw", append=True).sort_index()
