from pathlib import Path

import pandas as pd
import pytest
import yaml
from pytest import TempPathFactory
from vivarium.framework.artifact import Artifact

from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX
from vivarium_testing_utils.automated_validation.data_loader import (
    _convert_to_total_person_time,
)
from vivarium_testing_utils.automated_validation.data_transformation import utils
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


@utils.check_io(out=SingleNumericColumn)
def _create_transition_count_data() -> pd.DataFrame:
    """Create transition count data for testing."""
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 8.0],
        },
        index=pd.MultiIndex.from_product(
            [
                ["transition_count"],
                ["cause"],
                ["disease"],
                ["susceptible_to_disease_to_disease", "disease_to_susceptible_to_disease"],
                ["A", "B"],
                ["baseline"],
                ["foo", "bar"],
            ],
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "common_stratify_column",
                "scenario",
                "tc_unique_stratification",
            ],
        ),
    )


@utils.check_io(out=SingleNumericColumn)
def _create_person_time_data() -> pd.DataFrame:
    """Create person time data for testing."""
    return pd.DataFrame(
        {
            "value": [7.0, 10.0, 12.0, 17.0, 9.0, 14.0, 15.0, 22.0],
        },
        index=pd.MultiIndex.from_product(
            [
                ["person_time"],
                ["cause"],
                ["disease"],
                ["susceptible_to_disease", "disease"],
                ["A", "B"],
                ["baseline"],
                ["foo", "bar"],
            ],
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "common_stratify_column",
                "scenario",
                "pt_unique_stratification",
            ],
        ),
    )


@utils.check_io(out=SingleNumericColumn)
def _create_deaths_data() -> pd.DataFrame:
    """Create deaths data for testing."""
    return pd.DataFrame(
        {
            "value": [2.0, 3.0, 4.0, 5.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("deaths", "cause", "disease", "disease", "A", "baseline"),
                ("deaths", "cause", "other_causes", "other_causes", "A", "baseline"),
                ("deaths", "cause", "disease", "disease", "B", "baseline"),
                ("deaths", "cause", "other_causes", "other_causes", "B", "baseline"),
            ],
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "common_stratify_column",
                "scenario",
            ],
        ),
    )


@utils.check_io(out=DrawData)
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
            names=["common_stratify_column", "other_stratify_column"],
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


@utils.check_io(out=SingleNumericColumn)
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
            names=[
                "measure",
                "entity_type",
                "entity",
                "sub_entity",
                "common_stratify_column",
            ],
        ),
    )


@utils.check_io(out=DrawData)
def _create_raw_artifact_risk_exposure() -> pd.DataFrame:
    """Create raw artifact risk exposure data for testing."""
    return pd.DataFrame(
        {
            "draw_0": [0.25, 0.35, 0.40, 0.30, 0.20, 0.50],
            "draw_1": [0.28, 0.32, 0.42, 0.28, 0.22, 0.48],
        },
        index=pd.MultiIndex.from_product(
            [
                ["A", "B"],
                ["cat1", "cat2", "cat3"],
            ],
            names=["common_stratify_column", "parameter"],
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
@utils.check_io(out=SingleNumericColumn)
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
            names=["common_stratify_column", "other_stratify_column", DRAW_INDEX],
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
            "value": [1.0, 2.0, 3.0],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("cause", "disease", "0_to_5", 0.0, 5.0),
                ("cause", "disease", "5_to_10", 5.0, 10.0),
                ("cause", "disease", "10_to_15", 10.0, 15.0),
            ],
            names=[
                "cause",
                "disease",
                AGE_GROUP_COLUMN,
                AGE_START_COLUMN,
                AGE_END_COLUMN,
            ],
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
@utils.check_io(out=SingleNumericColumn)
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
        index=pd.MultiIndex.from_product(
            [
                ["A", "B"],
                ["cat1", "cat2", "cat3"],
                [0, 1],
            ],
            names=["common_stratify_column", "parameter", "input_draw"],
        ),
    )


@pytest.fixture
def artifact_relative_risk() -> pd.DataFrame:
    """Sample relative risks artifact data."""
    return pd.DataFrame(
        {
            "value": [1.5, 2.0, 1.8, 1.2],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("disease", "excess_mortality_rate", "cat1", "B", 0),
                ("disease", "excess_mortality_rate", "cat1", "B", 1),
                ("disease", "excess_mortality_rate", "cat2", "D", 0),
                ("disease", "excess_mortality_rate", "cat2", "D", 1),
            ],
            names=[
                "affected_entity",
                "affected_measure",
                "parameter",
                "other_stratify_column",
                DRAW_INDEX,
            ],
        ),
    )


@pytest.fixture
def artifact_excess_mortality_rate() -> pd.DataFrame:
    """Sample excess mortality rate artifact data."""
    return pd.DataFrame(
        {
            "value": [0.02, 0.03, 0.01, 0.04],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("B", 0),
                ("B", 1),
                ("D", 0),
                ("D", 1),
            ],
            names=["other_stratify_column", DRAW_INDEX],
        ),
    )


@pytest.fixture
def risk_categories() -> dict[str, str]:
    """Sample risk categories mapping."""
    return {"cat1": "high", "cat2": "medium", "cat3": "low", "cat4": "unexposed"}
