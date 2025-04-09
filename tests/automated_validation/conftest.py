from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def sim_result_dir() -> Path:
    return Path(__file__).parent / "data/sim_outputs"


@pytest.fixture
def transition_count_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [3, 5, 7, 13],
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


@pytest.fixture
def person_time_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": [17, 23, 29, 37],
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
