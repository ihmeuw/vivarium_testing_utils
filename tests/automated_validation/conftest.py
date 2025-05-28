from pathlib import Path
import pandas as pd
import pytest
import yaml

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
from vivarium.framework.artifact import Artifact


@pytest.fixture
def sim_result_dir(
    tmp_path: Path,
    transition_count_data: pd.DataFrame,
    person_time_data: pd.DataFrame,
    deaths_data: pd.DataFrame,
    raw_artifact_disease_incidence: pd.DataFrame,
    sample_age_group_df: pd.DataFrame,
):
    """Create a temporary directory for simulation outputs."""
    # Create the directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    # Save Sim DataFrames
    transition_count_data.reset_index().to_parquet(
        results_dir / "transition_count_disease.parquet"
    )
    person_time_data.reset_index().to_parquet(results_dir / "person_time_disease.parquet")
    deaths_data.reset_index().to_parquet(results_dir / "deaths.parquet")

    # Create Artifact
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "artifact.hdf"
    artifact = Artifact(artifact_path)
    artifact.write("cause.disease.incidence_rate", raw_artifact_disease_incidence)
    artifact.write("population.age_bins", sample_age_group_df)
    # Save model specification
    with open(tmp_path / "model_specification.yaml", "w") as f:
        yaml.dump(get_model_spec(artifact_path), f)

    return tmp_path


def get_model_spec(artifact_path: Path) -> dict:
    """Sample model specification for testing."""
    return {
        "configuration": {
            "input_data": {
                "artifact_path": str(artifact_path),
            }
        }
    }


@pytest.fixture
def deaths_data() -> pd.DataFrame:
    """Sample deaths data for testing."""
    # This is sample data - adjust according to your actual data structure
    return pd.DataFrame(
        {
            "value": [2.0, 3.0, 4.0, 5.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("deaths", "cause", "disease", "susceptible_to_disease", "A"),
                ("deaths", "cause", "disease", "disease", "A"),
                ("deaths", "cause", "disease", "susceptible_to_disease", "B"),
                ("deaths", "cause", "disease", "disease", "B"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )


@pytest.fixture
def transition_count_data() -> pd.DataFrame:
    """Raw transition count data to be saved to parquet."""
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


@pytest.fixture
def person_time_data() -> pd.DataFrame:
    """Raw person time data to be saved to parquet."""
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
    """Raw artifact disease incidence data."""
    return pd.DataFrame(
        {
            "draw_0": [0.17, 0.13],
            "draw_1": [0.18, 0.14],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (
                    "A",
                    "C",
                ),
                (
                    "B",
                    "D",
                ),
            ],
            names=["stratify_column", "other_stratify_column"],
        ),
    )


@check_io(out=SingleNumericColumn)
@pytest.fixture
def artifact_disease_incidence() -> pd.DataFrame:
    """Processed artifact disease incidence data."""
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
                ("A", "C", 0),
                ("A", "C", 1),
                ("B", "D", 0),
                ("B", "D", 1),
            ],
            names=["stratify_column", "other_stratify_column", "input_draw"],
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
