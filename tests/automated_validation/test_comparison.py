from unittest import mock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_check import check

from vivarium_testing_utils.automated_validation.comparison import (
    FuzzyComparison,
)
from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, SEED_INDEX
from vivarium_testing_utils.automated_validation.data_transformation import calculations
from vivarium_testing_utils.automated_validation.data_transformation.data_bundle import (
    RatioMeasureDataBundle,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.constants import DataSource
from vivarium_testing_utils.automated_validation.data_loader import DataLoader

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
def age_group_df() -> pd.DataFrame:
    """Create a simple age group DataFrame for testing."""
    return pd.DataFrame(
        {"age_start": [0], "age_end": [5]}, index=pd.Index(["0_to_5"], name="age_group")
    ).set_index(["age_start", "age_end"], append=True)


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
    measure.sim_datasets = {
        "numerator_data": "numerator_data",
        "denominator_data": "denominator_data",
    }
    measure.artifact_datasets = {"data": "data"}
    measure.get_measure_data_from_ratio.side_effect = calculations.ratio
    measure.get_ratio_datasets_from_sim.side_effect = lambda **datasets: datasets
    measure.get_measure_data_from_artifact.side_effect = lambda data: data
    return measure


@pytest.fixture
def mock_dataloader(
    test_data: dict[str, pd.DataFrame], reference_data: pd.DataFrame
) -> DataLoader:
    """Create a mock DataLoader that returns test data for SIM source."""
    mock_loader = mock.Mock(spec=DataLoader)

    def mock_get_bulk_data(source, data_keys):
        if source == DataSource.SIM:
            return test_data
        elif source == DataSource.ARTIFACT:
            return {"data": reference_data}
        else:
            return {}

    mock_loader.get_bulk_data.side_effect = mock_get_bulk_data
    return mock_loader


@pytest.fixture
def test_data_bundle(
    mock_ratio_measure: RatioMeasure,
    mock_dataloader: DataLoader,
    age_group_df: pd.DataFrame,
) -> RatioMeasureDataBundle:
    """Create a RatioMeasureDataBundle object for test data."""
    return RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mock_dataloader,
        age_group_df=age_group_df,
        scenarios={"scenario": "baseline"},
    )


@pytest.fixture
def reference_data_bundle(
    mock_ratio_measure: RatioMeasure,
    mock_dataloader: DataLoader,
    age_group_df: pd.DataFrame,
) -> RatioMeasureDataBundle:
    """Create a RatioMeasureDataBundle object for reference data."""
    return RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mock_dataloader,
        age_group_df=age_group_df,
    )


def test_fuzzy_comparison_init(
    mock_ratio_measure: RatioMeasure,
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """Test the initialization of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=test_data_bundle,
        reference_data=reference_data_bundle,
        stratifications=[],
    )

    with check:
        assert comparison.measure == mock_ratio_measure
        assert comparison.test_data.source == DataSource.SIM
        assert comparison.test_data.datasets.keys() == {"numerator_data", "denominator_data"}
        assert comparison.reference_data.source == DataSource.ARTIFACT
        assert comparison.test_data.scenarios == {"scenario": "baseline"}
        assert comparison.reference_data.scenarios == {}
        assert list(comparison.stratifications) == []


def test_fuzzy_comparison_metadata(
    mock_ratio_measure: RatioMeasure,
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """Test the metadata property of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=test_data_bundle,
        reference_data=reference_data_bundle,
    )

    metadata = comparison.metadata

    expected_metadata = [
        ("Measure Key", "mock_measure", "mock_measure"),
        ("Source", "sim", "artifact"),
        (
            "Index Columns",
            "year, sex, age, input_draw, random_seed, scenario",
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
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """Test the get_frame method of the FuzzyComparison class."""
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=test_data_bundle,
        reference_data=reference_data_bundle,
    )

    diff = comparison.get_frame(stratifications=[], num_rows=1)

    with check:
        assert len(diff) == 1
        assert "test_rate" in diff.columns
        assert "reference_rate" in diff.columns
        assert "percent_error" in diff.columns
        assert DRAW_INDEX in diff.index.names
        assert SEED_INDEX not in diff.index.names

    # Test returning all rows
    all_diff = comparison.get_frame(stratifications=[], num_rows="all")
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


def test_fuzzy_comparison_get_frame_aggregated(
    mock_ratio_measure: RatioMeasure,
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """Test the get_frame method of the FuzzyComparison class with aggregated draws."""
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=test_data_bundle,
        reference_data=reference_data_bundle,
    )
    diff = comparison.get_frame(stratifications=[], num_rows="all", aggregate_draws=True)
    expected_df = pd.DataFrame(
        {
            "test_mean": [0.1, 0.2, 0.325],
            "test_2.5%": [0.1, 0.2, 0.325],
            "test_97.5%": [0.1, 0.2, 0.325],
            "reference_mean": [0.12, 0.2, 0.29],
            "reference_2.5%": [0.12, 0.2, 0.29],
            "reference_97.5%": [0.12, 0.2, 0.29],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("2020", "male", 0, "baseline"),
                ("2020", "female", 0, "baseline"),
                ("2025", "male", 0, "baseline"),
            ],
            names=["year", "sex", "age", "scenario"],
        ),
    )
    assert_frame_equal(diff, expected_df)


def test_fuzzy_comparison_init_with_stratifications(
    mock_ratio_measure: RatioMeasure,
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """Test that FuzzyComparison raises NotImplementedError when initialized with non-empty stratifications."""
    with pytest.raises(
        NotImplementedError, match="Non-default stratifications require rate aggregations"
    ):
        FuzzyComparison(
            measure=mock_ratio_measure,
            test_data=test_data_bundle,
            reference_data=reference_data_bundle,
            stratifications=["year"],
        )


def test_fuzzy_comparison_get_frame_with_stratifications(
    mock_ratio_measure: RatioMeasure,
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """Test that FuzzyComparison.get_frame raises NotImplementedError when called with non-empty stratifications."""
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=test_data_bundle,
        reference_data=reference_data_bundle,
    )

    with pytest.raises(
        NotImplementedError, match="Non-default stratifications require rate aggregations"
    ):
        comparison.get_frame(stratifications=["year"])


def test_fuzzy_comparison_verify_not_implemented(
    mock_ratio_measure: RatioMeasure,
    test_data_bundle: RatioMeasureDataBundle,
    reference_data_bundle: RatioMeasureDataBundle,
) -> None:
    """ "FuzzyComparison.verify() is not implemented."""
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=test_data_bundle,
        reference_data=reference_data_bundle,
    )

    with pytest.raises(NotImplementedError):
        comparison.verify()


def test_fuzzy_comparison_align_datasets_with_singular_reference_index(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
) -> None:
    """Test that _align_datasets correctly handles singular reference-only indices."""
    reference_data_with_singular_index = reference_data.copy()
    reference_data_with_singular_index["location"] = ["Global", "Global", "Global"]
    reference_data_with_singular_index.set_index("location", append=True, inplace=True)

    # Create specific mock objects for this test
    mock_test_data = mock.Mock(spec=RatioMeasureDataBundle)
    mock_test_data.datasets = test_data
    mock_test_data.scenarios = {"scenario": "baseline"}

    mock_ref_data = mock.Mock(spec=RatioMeasureDataBundle)
    mock_ref_data.datasets = {"data": reference_data_with_singular_index}
    mock_ref_data.scenarios = {}
    mock_ref_data.measure_data = reference_data_with_singular_index

    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=mock_test_data,
        reference_data=mock_ref_data,
    )

    # Verify the singular index exists
    assert "location" in reference_data_with_singular_index.index.names
    assert "location" not in test_data["numerator_data"].index.names

    # Verify it's detected as a singular index
    singular_indices = calculations.get_singular_indices(reference_data_with_singular_index)
    assert "location" in singular_indices
    assert singular_indices["location"] == "Global"

    # Execute
    aligned_test_data, aligned_reference_data = comparison._align_datasets()

    # Verify the singular index was dropped
    assert "location" not in aligned_reference_data.index.names
    assert aligned_test_data.shape[0] == aligned_reference_data.shape[0]


def test_fuzzy_comparison_align_datasets_with_non_singular_reference_index(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
) -> None:
    """Test that _align_datasets raises ValueError for non-singular reference-only indices."""
    reference_data_with_non_singular_index = reference_data.copy()
    reference_data_with_non_singular_index["location"] = ["Global", "USA", "USA"]
    reference_data_with_non_singular_index.set_index("location", append=True, inplace=True)

    # Create specific mock objects for this test
    mock_test_data = mock.Mock(spec=RatioMeasureDataBundle)
    mock_test_data.datasets = test_data
    mock_test_data.scenarios = {"scenario": "baseline"}

    mock_ref_data = mock.Mock(spec=RatioMeasureDataBundle)
    mock_ref_data.datasets = {"data": reference_data_with_non_singular_index}
    mock_ref_data.scenarios = {}
    mock_ref_data.measure_data = reference_data_with_non_singular_index

    # Setup
    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=mock_test_data,
        reference_data=mock_ref_data,
    )

    # Verify the non-singular index exists
    assert "location" in reference_data_with_non_singular_index.index.names
    assert "location" not in test_data["numerator_data"].index.names

    # Verify it's not detected as a singular index
    singular_indices = calculations.get_singular_indices(
        reference_data_with_non_singular_index
    )
    assert "location" not in singular_indices

    # Execute and verify error is raised with correct message
    with pytest.raises(
        ValueError, match="Reference data has non-trivial index levels {'location'}"
    ):
        comparison._align_datasets()


def test_fuzzy_comparison_align_datasets_calculation(
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    reference_data: pd.DataFrame,
) -> None:
    """Test _align_datasets with varying denominators to ensure ratios are calculated correctly."""

    # Create specific mock objects for this test
    mock_test_data = mock.Mock(spec=RatioMeasureDataBundle)
    mock_test_data.datasets = test_data
    mock_test_data.scenarios = {"scenario": "baseline"}

    mock_ref_data = mock.Mock(spec=RatioMeasureDataBundle)
    mock_ref_data.datasets = {"data": reference_data}
    mock_ref_data.scenarios = {}
    mock_ref_data.measure_data = reference_data

    comparison = FuzzyComparison(
        measure=mock_ratio_measure,
        test_data=mock_test_data,
        reference_data=mock_ref_data,
    )

    aligned_test_data, aligned_reference_data = comparison._align_datasets()

    assert_frame_equal(aligned_reference_data, reference_data)

    expected_values = [10 / 100, 20 / 100, (30 + 35) / (100 + 100)]
    expected_index = pd.MultiIndex.from_tuples(
        [
            ("2020", "male", 0, 1, "baseline"),
            ("2020", "female", 0, 5, "baseline"),
            ("2025", "male", 0, 2, "baseline"),
        ],
        names=["year", "sex", "age", DRAW_INDEX, "scenario"],
    )
    assert_frame_equal(
        aligned_test_data,
        pd.DataFrame(
            {"value": expected_values},
            index=expected_index,
        ),
    )
