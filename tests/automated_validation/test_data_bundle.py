"""Tests for the RatioMeasureDataBundle class."""

from typing import cast
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


def test_get_metadata(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
    test_data: dict[str, pd.DataFrame],
) -> None:
    """Test get_metadata method returns correct basic structure."""

    mocker.patch.object(
        RatioMeasureDataBundle,
        "_get_formatted_datasets",
        return_value=test_data,
    )

    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )

    metadata = bundle.get_metadata()

    assert metadata["source"] == "sim"
    assert metadata["index_columns"] == "year, sex, age, input_draw, random_seed, scenario"
    assert metadata["size"] == "4 rows × 1 columns"


@pytest.mark.parametrize("source", [DataSource.GBD, DataSource.CUSTOM])
def test_get_formatted_datasets_not_implemented_source(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
    source: DataSource,
) -> None:
    """Test _get_formatted_datasets raises NotImplementedError for GBD source."""
    mock_data_loader = mocker.MagicMock(spec=DataLoader)
    mock_data_loader._get_raw_data_from_source.return_value = {}

    with pytest.raises(NotImplementedError):
        RatioMeasureDataBundle(
            measure=mock_ratio_measure,
            source=source,
            data_loader=mock_data_loader,
            age_group_df=sample_age_group_df,
        )


def test_get_measure_data(
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
) -> None:
    """Test that aggregate_stratifications correctly aggregates data."""

    aggregated = comparison._aggregate_reference_stratifications(
        reference_bundle.datasets["data"], ["age", "sex"]
    )
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
        comparison._aggregate_reference_stratifications(
            reference_bundle.datasets["data"], ["dog", "cat"]
        )
