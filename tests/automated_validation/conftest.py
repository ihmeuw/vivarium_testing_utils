from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    SimOutputData,
    SingleNumericColumn,
)


@pytest.fixture
def sim_result_dir() -> Path:
    return Path(__file__).parent / "data/sim_outputs"


@pa.check_io(out=SingleNumericColumn.to_schema())
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


@pa.check_io(out=SingleNumericColumn.to_schema())
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
                ("A"),
                ("B"),
            ],
            names=["stratify_column"],
        ),
    )


@pa.check_io(out=SingleNumericColumn.to_schema())
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
            names=["stratify_column", "draw"],
        ),
    )
