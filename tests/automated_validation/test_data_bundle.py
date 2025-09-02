"""Tests for the RatioMeasureDataBundle class."""

from unittest import mock

import pandas as pd
import pytest
from pytest_mock import MockFixture

from vivarium_testing_utils.automated_validation.bundle import RatioMeasureDataBundle
from vivarium_testing_utils.automated_validation.constants import (
    DRAW_INDEX,
    SEED_INDEX,
    DataSource,
)
from vivarium_testing_utils.automated_validation.data_loader import DataLoader
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    RatioMeasure,
)


def test_init_with_sim_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test instantiation with SIM data source."""
    # Mock the data loading process
    mock_datasets = {
        "numerator_data": pd.DataFrame({"value": [1, 2, 3]}),
        "denominator_data": pd.DataFrame({"value": [10, 20, 30]}),
    }
    mocker.patch.object(
        DataLoader,
        "_get_raw_data_from_source",
        return_value=mock_datasets,
    )

    mock_data_loader = mocker.MagicMock(spec=DataLoader)

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mock_data_loader,
        age_group_df=sample_age_group_df,
        scenarios={"scenario": "baseline"},
    )

    assert bundle.measure == mock_ratio_measure
    assert bundle.source == DataSource.SIM
    assert bundle.data_loader == mock_data_loader
    assert bundle.scenarios == {"scenario": "baseline"}
    assert bundle.datasets == mock_datasets


def test_init_with_artifact_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test instantiation with ARTIFACT data source."""
    mock_datasets = {
        "data": pd.DataFrame({"value": [0.1, 0.2, 0.3]}),
        "weights": pd.DataFrame({"value": [1.0, 1.0, 1.0]}),
    }
    mocker.patch.object(
        DataLoader,
        "_get_raw_data_from_source",
        return_value=mock_datasets,
    )

    mock_data_loader = mocker.MagicMock(spec=DataLoader)

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mock_data_loader,
        age_group_df=sample_age_group_df,
    )

    assert bundle.measure == mock_ratio_measure
    assert bundle.source == DataSource.ARTIFACT
    assert bundle.data_loader == mock_data_loader
    assert bundle.scenarios == {}
    assert bundle.datasets == mock_datasets


def test_init_with_default_scenarios(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test instantiation with default scenarios (None)."""
    mock_datasets = {"data": pd.DataFrame({"value": [1, 2, 3]})}
    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value=mock_datasets,
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    assert bundle.scenarios == {}


def test_dataset_names_property_sim_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test dataset_names property for SIM source."""
    expected_datasets = {
        "numerator_data": "test_numerator",
        "denominator_data": "test_denominator",
    }
    mock_ratio_measure.sim_datasets = expected_datasets

    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    assert bundle.dataset_names == expected_datasets


def test_dataset_names_property_artifact_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test dataset_names property for ARTIFACT source."""
    expected_datasets = {"artifact_data": "test.artifact.key"}
    mock_ratio_measure.artifact_datasets = expected_datasets

    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    assert bundle.dataset_names == expected_datasets


def test_dataset_names_property_unsupported_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test dataset_names property raises error for unsupported source."""
    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.GBD,  # Unsupported source
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    with pytest.raises(ValueError, match="Unsupported data source: DataSource.GBD"):
        _ = bundle.dataset_names


def test_transform_data_sim_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _transform_data method for SIM source."""
    input_datasets = {
        "numerator_data": pd.DataFrame({"value": [1, 2]}),
        "denominator_data": pd.DataFrame({"value": [10, 20]}),
    }
    expected_output = {"processed_num": pd.DataFrame(), "processed_den": pd.DataFrame()}

    mock_ratio_measure.get_ratio_datasets_from_sim.return_value = expected_output
    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    result = bundle._transform_data(input_datasets)

    assert result == expected_output
    mock_ratio_measure.get_ratio_datasets_from_sim.assert_called_once_with(**input_datasets)


def test_transform_data_artifact_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _transform_data method for ARTIFACT source."""
    input_datasets = {"artifact_data": pd.DataFrame({"value": [0.1, 0.2]})}
    expected_data = pd.DataFrame({"value": [0.05, 0.1]})

    mock_ratio_measure.get_measure_data_from_artifact.return_value = expected_data
    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    result = bundle._transform_data(input_datasets)

    assert "data" in result
    pd.testing.assert_frame_equal(result["data"], expected_data)
    mock_ratio_measure.get_measure_data_from_artifact.assert_called_once_with(
        **input_datasets
    )


def test_transform_data_unsupported_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _transform_data method raises error for unsupported source."""
    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.CUSTOM,  # Unsupported source
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    with pytest.raises(ValueError, match="Unsupported data source: DataSource.CUSTOM"):
        bundle._transform_data({})


def test_measure_data_property_sim_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test measure_data property for SIM source."""
    mock_datasets = {
        "numerator_data": pd.DataFrame({"value": [1, 2]}),
        "denominator_data": pd.DataFrame({"value": [10, 20]}),
    }
    expected_result = pd.DataFrame({"value": [0.1, 0.1]})

    mock_ratio_measure.get_measure_data_from_ratio.return_value = expected_result
    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value=mock_datasets,
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    result = bundle.measure_data

    pd.testing.assert_frame_equal(result, expected_result)
    mock_ratio_measure.get_measure_data_from_ratio.assert_called_once_with(**mock_datasets)


def test_measure_data_property_artifact_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test measure_data property for ARTIFACT source."""
    expected_data = pd.DataFrame({"value": [0.1, 0.2]})
    mock_datasets = {"data": expected_data, "weights": pd.DataFrame()}

    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value=mock_datasets,
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    result = bundle.measure_data

    pd.testing.assert_frame_equal(result, expected_data)


def test_measure_data_property_unsupported_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test measure_data property raises error for unsupported source."""
    mocker.patch.object(RatioMeasureDataBundle, "_get_formatted_datasets")

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.GBD,  # Unsupported source
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    with pytest.raises(ValueError, match="Unsupported data source: DataSource.GBD"):
        _ = bundle.measure_data


def test_get_metadata_basic_structure(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test get_metadata method returns correct basic structure."""
    test_data = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [("A", "male"), ("A", "female"), ("B", "male")],
            names=["location", "sex"],
        ),
    )

    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value={"data": test_data},
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    metadata = bundle.get_metadata()

    assert metadata["source"] == "artifact"
    assert metadata["index_columns"] == "location, sex"
    assert metadata["size"] == "3 rows × 1 columns"


def test_get_metadata_with_draws(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test get_metadata method with draw information."""
    test_data = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_tuples(
            [("A", 0), ("A", 1), ("B", 0), ("B", 1)],
            names=["location", DRAW_INDEX],
        ),
    )

    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value={"data": test_data},
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    metadata = bundle.get_metadata()

    assert metadata["num_draws"] == "2"
    assert DRAW_INDEX + "s" in metadata


def test_get_metadata_with_seeds(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test get_metadata method with seed information."""
    test_data = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=pd.MultiIndex.from_tuples(
            [
                ("A", 1337),
                ("A", 1338),
                ("A", 1339),
                ("B", 1337),
                ("B", 1338),
                ("B", 1339),
            ],
            names=["location", SEED_INDEX],
        ),
    )

    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value={"data": test_data},
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    metadata = bundle.get_metadata()

    assert metadata["num_seeds"] == "3"


def test_get_formatted_datasets_sim_source_integration(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _get_formatted_datasets method for SIM source integration."""
    # Mock the data loader and measure methods
    mock_data_loader = mocker.MagicMock(spec=DataLoader)
    raw_datasets = {
        "numerator_data": pd.DataFrame({"value": [1, 2]}),
        "denominator_data": pd.DataFrame({"value": [10, 20]}),
    }
    mock_data_loader._get_raw_data_from_source.return_value = raw_datasets

    processed_datasets = {
        "numerator_data": pd.DataFrame({"value": [0.1, 0.2]}),
        "denominator_data": pd.DataFrame({"value": [1.0, 2.0]}),
    }
    mock_ratio_measure.get_ratio_datasets_from_sim.return_value = processed_datasets
    mock_ratio_measure.get_required_datasets.return_value = {
        "numerator_data": "test_num",
        "denominator_data": "test_den",
    }

    # Mock the age_groups formatting function
    mock_format_func = mocker.patch(
        "vivarium_testing_utils.automated_validation.data_transformation.age_groups.format_dataframe_from_age_bin_df"
    )
    mock_format_func.side_effect = lambda df, age_df: df  # Return unchanged for simplicity

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mock_data_loader,
        age_group_df=sample_age_group_df,
    )

    # Verify the methods were called correctly
    mock_ratio_measure.get_required_datasets.assert_called_with(DataSource.SIM)
    mock_data_loader._get_raw_data_from_source.assert_called_with(
        mock_ratio_measure.get_required_datasets.return_value, DataSource.SIM
    )
    mock_ratio_measure.get_ratio_datasets_from_sim.assert_called_with(**raw_datasets)

    # Verify datasets are formatted correctly
    assert len(bundle.datasets) == 2
    assert "numerator_data" in bundle.datasets
    assert "denominator_data" in bundle.datasets


def test_get_formatted_datasets_artifact_source_integration(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _get_formatted_datasets method for ARTIFACT source integration."""
    # Mock the data loader and measure methods
    mock_data_loader = mocker.MagicMock(spec=DataLoader)
    raw_datasets = {"artifact_data": pd.DataFrame({"value": [0.1, 0.2]})}
    mock_data_loader._get_raw_data_from_source.return_value = raw_datasets

    processed_data = pd.DataFrame({"value": [0.05, 0.1]})
    mock_ratio_measure.get_measure_data.return_value = processed_data
    mock_ratio_measure.get_required_datasets.return_value = {"artifact_data": "test.key"}

    # Mock rate aggregation weights
    mock_weights = mock.Mock()
    mock_weights.weight_keys = {"population": "population.structure"}
    mock_weights.get_weights.return_value = pd.DataFrame({"value": [1.0, 1.0]})
    mock_ratio_measure.rate_aggregation_weights = mock_weights

    raw_weights = {"population": pd.DataFrame({"value": [100, 200]})}
    mock_data_loader._get_raw_data_from_source.side_effect = [raw_datasets, raw_weights]

    # Mock the age_groups formatting function
    mock_format_func = mocker.patch(
        "vivarium_testing_utils.automated_validation.data_transformation.age_groups.format_dataframe_from_age_bin_df"
    )
    mock_format_func.side_effect = lambda df, age_df: df  # Return unchanged for simplicity

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mock_data_loader,
        age_group_df=sample_age_group_df,
    )

    # Verify the methods were called correctly
    mock_ratio_measure.get_required_datasets.assert_called_with(DataSource.ARTIFACT)
    assert mock_data_loader._get_raw_data_from_source.call_count == 2
    mock_ratio_measure.get_measure_data.assert_called_with(
        DataSource.ARTIFACT, **raw_datasets
    )
    mock_weights.get_weights.assert_called_with(**raw_weights)

    # Verify datasets contain both data and weights
    assert len(bundle.datasets) == 2
    assert "data" in bundle.datasets
    assert "weights" in bundle.datasets


def test_get_formatted_datasets_gbd_source_raises(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _get_formatted_datasets raises NotImplementedError for GBD source."""
    mock_data_loader = mocker.MagicMock(spec=DataLoader)
    mock_data_loader._get_raw_data_from_source.return_value = {}

    with pytest.raises(NotImplementedError):
        RatioMeasureDataBundle(
            measure=mock_ratio_measure,
            source=DataSource.GBD,
            data_loader=mock_data_loader,
            age_group_df=sample_age_group_df,
        )


def test_get_formatted_datasets_custom_source_raises(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test _get_formatted_datasets raises NotImplementedError for CUSTOM source."""
    mock_data_loader = mocker.MagicMock(spec=DataLoader)
    mock_data_loader._get_raw_data_from_source.return_value = {}

    with pytest.raises(NotImplementedError):
        RatioMeasureDataBundle(
            measure=mock_ratio_measure,
            source=DataSource.CUSTOM,
            data_loader=mock_data_loader,
            age_group_df=sample_age_group_df,
        )
