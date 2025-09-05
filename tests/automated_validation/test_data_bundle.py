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
