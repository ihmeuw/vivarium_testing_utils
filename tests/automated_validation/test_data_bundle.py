"""Tests for the RatioMeasureDataBundle class."""

from typing import Literal, cast

import pandas as pd
import pytest
from pytest_mock import MockFixture

from vivarium_testing_utils.automated_validation.bundle import RatioMeasureDataBundle
from vivarium_testing_utils.automated_validation.constants import DataSource
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
def test_dataset_names_value_error(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    sample_age_group_df: pd.DataFrame,
    source: DataSource,
) -> None:
    """Test _get_formatted_datasets raises NotImplementedError for GBD source."""
    mock_data_loader = mocker.MagicMock(spec=DataLoader)
    mock_data_loader._get_raw_data_from_source.return_value = {}

    with pytest.raises(ValueError):
        RatioMeasureDataBundle(
            measure=mock_ratio_measure,
            source=source,
            data_loader=mock_data_loader,
            age_group_df=sample_age_group_df,
        )


@pytest.mark.parametrize("stratifications", [[], ["year", "sex", "age"]])
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
        assert list(stratifications) == list(aggregated.index.names)
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


@pytest.mark.parametrize("stratifications", ["all", ["age", "sex"]])
def test_aggregate_reference_stratifications(
    mocker: MockFixture,
    mock_ratio_measure: RatioMeasure,
    reference_data: pd.DataFrame,
    reference_weights: pd.DataFrame,
    sample_age_group_df: pd.DataFrame,
    stratifications: list[str] | Literal["all"],
) -> None:
    # mock loading of datasets
    mocker.patch(
        "vivarium_testing_utils.automated_validation.bundle.RatioMeasureDataBundle._get_formatted_datasets",
        return_value={"data": reference_data},
    )
    mocker.patch(
        "vivarium_testing_utils.automated_validation.bundle.RatioMeasureDataBundle._get_aggregated_weights",
        return_value=reference_weights,
    )
    bundle = RatioMeasureDataBundle(
        measure=mock_ratio_measure,
        source=DataSource.ARTIFACT,
        data_loader=mocker.MagicMock(spec=DataLoader),
        age_group_df=sample_age_group_df,
    )
    aggregated = bundle._aggregate_artifact_stratifications(stratifications)

    if stratifications == "all":
        aggregated.equals(reference_data)
    else:
        assert set(stratifications) == set(aggregated.index.names)
        expected = pd.DataFrame(
            data={
                "value": [
                    ((0.15 * 0.12) + (0.35 * 0.29)) / (0.15 + 0.35),
                    (0.20 * 0.25) / 0.25,
                ]
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("male", 0),
                    ("female", 0),
                ],
                names=["sex", "age"],
            ),
        )
        pd.testing.assert_frame_equal(aggregated, expected)
