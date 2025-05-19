from pathlib import Path

import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AGE_END_COLUMN,
    AGE_GROUP_COLUMN,
    AGE_START_COLUMN,
    AgeSchema,
    AgeTuple,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.utils import check_io


@pytest.fixture
def sim_result_dir() -> Path:
    return Path(__file__).parent / "data/sim_outputs"


@check_io(out=SingleNumericColumn)
@pytest.fixture
def transition_count_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [3.0, 5.0, 7.0, 13.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (
                    "transition_count",
                    "cause",
                    "disease",
                    "susceptible_to_disease_to_disease",
                    "A",
                ),
                (
                    "transition_count",
                    "cause",
                    "disease",
                    "susceptible_to_disease_to_disease",
                    "B",
                ),
                (
                    "transition_count",
                    "cause",
                    "disease",
                    "disease_to_susceptible_to_disease",
                    "A",
                ),
                (
                    "transition_count",
                    "cause",
                    "disease",
                    "disease_to_susceptible_to_disease",
                    "B",
                ),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )


@check_io(out=SingleNumericColumn)
@pytest.fixture
def person_time_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [17.0, 23.0, 29.0, 37.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("person_time", "cause", "disease", "susceptible_to_disease", "A"),
                ("person_time", "cause", "disease", "disease", "A"),
                ("person_time", "cause", "disease", "susceptible_to_disease", "B"),
                ("person_time", "cause", "disease", "disease", "B"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )


@pytest.fixture
def raw_artifact_disease_incidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "draw_0": [0.17, 0.13],
            "draw_1": [0.18, 0.14],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A",),
                ("B",),
            ],
            names=["stratify_column"],
        ),
    )


@check_io(out=SingleNumericColumn)
@pytest.fixture
def artifact_disease_incidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [
                0.17,
                0.18,
                0.13,
                0.14,
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A", 0),
                ("A", 1),
                ("B", 0),
                ("B", 1),
            ],
            names=["stratify_column", "input_draw"],
        ),
    )


@pytest.fixture
def sample_age_tuples() -> list[AgeTuple]:
    return [
        ("0_to_5", 0, 5),
        ("5_to_10", 5, 10),
        ("10_to_15", 10, 15),
    ]


@pytest.fixture
def sample_age_schema(
    sample_age_tuples: list[AgeTuple],
) -> AgeSchema:
    return AgeSchema.from_tuples(sample_age_tuples)


@pytest.fixture
def sample_age_group_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            AGE_GROUP_COLUMN: ["0_to_5", "5_to_10", "10_to_15"],
            AGE_START_COLUMN: [0.0, 5.0, 10.0],
            AGE_END_COLUMN: [5.0, 10.0, 15.0],
        }
    ).set_index([AGE_GROUP_COLUMN, AGE_START_COLUMN, AGE_END_COLUMN])


@pytest.fixture
def sample_df_with_ages() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "foo": [1.0, 2.0, 3.0],
            "bar": [4.0, 5.0, 6.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_5", 0.0, 5.0),
                ("cause", "disease", "5_to_10", 5.0, 10.0),
                ("cause", "disease", "10_to_15", 10.0, 15.0),
            ],
            names=["cause", "disease", AGE_GROUP_COLUMN, AGE_START_COLUMN, AGE_END_COLUMN],
        ),
    )
