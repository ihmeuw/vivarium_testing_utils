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


@pytest.mark.parametrize("stratifications", [[], [DRAW_INDEX, SEED_INDEX]])
def test_aggregate_scenario_stratifications(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    test_data: dict[str, pd.DataFrame],
    sample_age_group_df: pd.DataFrame,
    stratifications: list[str],
) -> None:
    # Scenario is dropped from test datasets in the DataBundle formatting
    test_data = {key: dataset.droplevel("scenario") for key, dataset in test_data.items()}

    # mock loading of datasets
    mocker.patch(
        "vivarium_testing_utils.automated_validation.bundle.RatioMeasureDataBundle._get_formatted_datasets",
        return_value=test_data,
    )
    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.SIM,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
        scenarios={"scenario": "baseline"},
    )
    # This is marginalizing the stratifications out of the test data
    aggregated = bundle._aggregate_scenario_stratifications(test_data, stratifications)

    if not stratifications:
        aggregated.equals(test_data["numerator_data"] / test_data["denominator_data"])
    else:
        assert bundle.index_names.difference(set(stratifications)) == set(
            aggregated.index.names
        )
        expected = pd.DataFrame(
            data={
                "value": [10 / 100, 20 / 100, (30 + 35) / (100 + 100)],
            },
            index=pd.MultiIndex.from_tuples(
                [("2020", "male", 0), ("2020", "female", 0), ("2025", "male", 0)],
                names=["year", "sex", "age"],
            ),
        )
        pd.testing.assert_frame_equal(aggregated, expected)


def test_aggregate_reference_stratifications():
    pass
