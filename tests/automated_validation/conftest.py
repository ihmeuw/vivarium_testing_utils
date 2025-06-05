from pathlib import Path

import pandas as pd
import pytest
import yaml
from pytest import TempPathFactory
from vivarium.framework.artifact import Artifact

from vivarium_testing_utils.automated_validation.data_loader import (
    _convert_to_total_person_time,
)
from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AGE_END_COLUMN,
    AGE_GROUP_COLUMN,
    AGE_START_COLUMN,
    AgeSchema,
    AgeTuple,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.utils import check_io


@check_io(out=SingleNumericColumn)
def _create_transition_count_data() -> pd.DataFrame:
    """Create transition count data for testing."""
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
def _create_person_time_data() -> pd.DataFrame:
    """Create person time data for testing."""
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


@check_io(out=SingleNumericColumn)
def _create_deaths_data() -> pd.DataFrame:
    """Create deaths data for testing."""
    return pd.DataFrame(
        {
            "value": [2.0, 3.0, 4.0, 5.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("deaths", "cause", "disease", "disease", "A"),
                ("deaths", "cause", "other_causes", "other_causes", "A"),
                ("deaths", "cause", "disease", "disease", "B"),
                ("deaths", "cause", "other_causes", "other_causes", "B"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )


@check_io(out=DrawData)
def _create_raw_artifact_disease_incidence() -> pd.DataFrame:
    """Create raw artifact disease incidence data for testing."""
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


def _create_sample_age_group_df() -> pd.DataFrame:
    """Create sample age group data for testing."""
    return pd.DataFrame(
        {
            AGE_GROUP_COLUMN: ["0_to_5", "5_to_10", "10_to_15"],
            AGE_START_COLUMN: [0.0, 5.0, 10.0],
            AGE_END_COLUMN: [5.0, 10.0, 15.0],
        }
    ).set_index([AGE_GROUP_COLUMN, AGE_START_COLUMN, AGE_END_COLUMN])


@check_io(out=SingleNumericColumn)
def _create_risk_state_person_time_data() -> pd.DataFrame:
    """Create risk state person time data for testing."""
    return pd.DataFrame(
        {
            "value": [8.0, 12.0, 15.0, 20.0, 6.0, 10.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("person_time", "rei", "child_stunting", "cat1", "A"),
                ("person_time", "rei", "child_stunting", "cat2", "A"),
                ("person_time", "rei", "child_stunting", "cat3", "A"),
                ("person_time", "rei", "child_stunting", "cat1", "B"),
                ("person_time", "rei", "child_stunting", "cat2", "B"),
                ("person_time", "rei", "child_stunting", "cat3", "B"),
            ],
            names=["measure", "entity_type", "entity", "sub_entity", "stratify_column"],
        ),
    )


@check_io(out=DrawData)
def _create_raw_artifact_risk_exposure() -> pd.DataFrame:
    """Create raw artifact risk exposure data for testing."""
    return pd.DataFrame(
        {
            "draw_0": [0.25, 0.35, 0.40, 0.30, 0.20, 0.50],
            "draw_1": [0.28, 0.32, 0.42, 0.28, 0.22, 0.48],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A", "cat1"),
                ("A", "cat2"),
                ("A", "cat3"),
                ("B", "cat1"),
                ("B", "cat2"),
                ("B", "cat3"),
            ],
            names=["stratify_column", "parameter"],
        ),
    )


@pytest.fixture(scope="session")
def sim_result_dir(tmp_path_factory: TempPathFactory) -> Path:
    """Create a temporary directory for simulation outputs."""
    # Create the temporary directory at session scope
    tmp_path = tmp_path_factory.mktemp("sim_data")

    # Create the directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)

    # Create data directly within this session-scoped fixture
    # so we don't depend on function-scoped fixtures
    _transition_count_data = _create_transition_count_data()
    _person_time_data = _create_person_time_data()
    _deaths_data = _create_deaths_data()
    _raw_artifact_disease_incidence = _create_raw_artifact_disease_incidence()
    _sample_age_group_df = _create_sample_age_group_df()
    _risk_state_person_time_data = _create_risk_state_person_time_data()
    _raw_artifact_risk_exposure = _create_raw_artifact_risk_exposure()

    # Save Sim DataFrames
    _transition_count_data.reset_index().to_parquet(
        results_dir / "transition_count_disease.parquet"
    )
    _person_time_data.reset_index().to_parquet(results_dir / "person_time_disease.parquet")
    _deaths_data.reset_index().to_parquet(results_dir / "deaths.parquet")
    _risk_state_person_time_data.reset_index().to_parquet(
        results_dir / "person_time_child_stunting.parquet"
    )

    # Create Artifact
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    artifact_path = artifact_dir / "artifact.hdf"
    artifact = Artifact(artifact_path)
    artifact.write("cause.disease.incidence_rate", _raw_artifact_disease_incidence)
    artifact.write("risk_factor.child_stunting.exposure", _raw_artifact_risk_exposure)
    artifact.write("population.age_bins", _sample_age_group_df)
    # Save model specification
    with open(tmp_path / "model_specification.yaml", "w") as f:
        yaml.dump(get_model_spec(artifact_path), f)

    return tmp_path


def get_model_spec(artifact_path: Path) -> dict[str, dict[str, dict[str, str]]]:
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
    return _create_deaths_data()


@pytest.fixture
def transition_count_data() -> pd.DataFrame:
    """Raw transition count data to be saved to parquet."""
    return _create_transition_count_data()


@pytest.fixture
def person_time_data() -> pd.DataFrame:
    return _create_person_time_data()


@pytest.fixture
def total_person_time_data(
    person_time_data: pd.DataFrame,
) -> pd.DataFrame:
    """Total person time data."""
    return _convert_to_total_person_time(person_time_data)


@pytest.fixture
def raw_artifact_disease_incidence() -> pd.DataFrame:
    """Raw artifact disease incidence data."""
    return _create_raw_artifact_disease_incidence()


@pytest.fixture
@check_io(out=SingleNumericColumn)
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
    return _create_sample_age_group_df()


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


@pytest.fixture
def risk_state_person_time_data() -> pd.DataFrame:
    """Risk state person time data for testing."""
    return _create_risk_state_person_time_data()


@pytest.fixture
def raw_artifact_risk_exposure() -> pd.DataFrame:
    """Raw artifact risk exposure data."""
    return _create_raw_artifact_risk_exposure()


@pytest.fixture
@check_io(out=SingleNumericColumn)
def artifact_risk_exposure() -> pd.DataFrame:
    """Processed artifact risk exposure data."""
    return pd.DataFrame(
        {
            "value": [
                0.25,
                0.28,  # A, cat1, draws 0 and 1
                0.35,
                0.32,  # A, cat2, draws 0 and 1
                0.40,
                0.42,  # A, cat3, draws 0 and 1
                0.30,
                0.28,  # B, cat1, draws 0 and 1
                0.20,
                0.22,  # B, cat2, draws 0 and 1
                0.50,
                0.48,  # B, cat3, draws 0 and 1
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A", "cat1", 0),
                ("A", "cat1", 1),
                ("A", "cat2", 0),
                ("A", "cat2", 1),
                ("A", "cat3", 0),
                ("A", "cat3", 1),
                ("B", "cat1", 0),
                ("B", "cat1", 1),
                ("B", "cat2", 0),
                ("B", "cat2", 1),
                ("B", "cat3", 0),
                ("B", "cat3", 1),
            ],
            names=["stratify_column", "parameter", "input_draw"],
        ),
    )
