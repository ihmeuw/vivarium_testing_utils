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
            "draw_0": [0.17, 0.13, 0.18, 0.14, 0.25, 0.26, 0.35, 0.36],
            "draw_1": [0.18, 0.14, 0.19, 0.15, 0.23, 0.20, 0.30, 0.32],
        },
        index=_get_artifact_index(),
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


def _create_risk_categories() -> dict[str, str]:
    """Create sample risk categories mapping."""
    return {
        "cat1": "high",
        "cat2": "medium",
        "cat3": "low",
        "cat4": "unexposed",
    }


@pytest.fixture(scope="session")
def sim_result_dir(
    tmp_path_factory: TempPathFactory, _artifact_keys_mapper: dict[str, pd.DataFrame]
) -> Path:
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
    _risk_state_person_time_data = _create_risk_state_person_time_data()

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
    for key, data in _artifact_keys_mapper.items():
        artifact.write(key, data)

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
def artifact_disease_incidence() -> pd.DataFrame:
    """Processed artifact disease incidence data."""
    return pd.DataFrame(
        {
            "value": [
                0.17,
                0.18,
                0.13,
                0.14,
                0.18,
                0.19,
                0.14,
                0.15,
                0.25,
                0.23,
                0.26,
                0.20,
                0.35,
                0.30,
                0.36,
                0.32,
            ],
        },
        index=pd.MultiIndex.from_tuples(
            [
                ("A", "C", 0.0, 5.0, 0),
                ("A", "C", 0.0, 5.0, 1),
                ("A", "C", 5.0, 10.0, 0),
                ("A", "C", 5.0, 10.0, 1),
                ("A", "D", 0.0, 5.0, 0),
                ("A", "D", 0.0, 5.0, 1),
                ("A", "D", 5.0, 10.0, 0),
                ("A", "D", 5.0, 10.0, 1),
                ("B", "C", 0.0, 5.0, 0),
                ("B", "C", 0.0, 5.0, 1),
                ("B", "C", 5.0, 10.0, 0),
                ("B", "C", 5.0, 10.0, 1),
                ("B", "D", 0.0, 5.0, 0),
                ("B", "D", 0.0, 5.0, 1),
                ("B", "D", 5.0, 10.0, 0),
                ("B", "D", 5.0, 10.0, 1),
            ],
            names=[
                "common_stratify_column",
                "other_stratify_column",
                "age_start",
                "age_end",
                "input_draw",
            ],
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
    return _create_risk_categories()


def _artifact_population_structure() -> pd.DataFrame:
    """Sample population structure artifact data."""
    pop = pd.DataFrame(
        {
            "value": [1000, 2000, 1500, 2500, 3000, 4000, 3500, 4500],
        },
        index=_get_artifact_index(),
    )
    pop["location"] = "Global"
    pop.set_index("location", append=True, inplace=True)
    index_order = [
        "location",
        "common_stratify_column",
        "other_stratify_column",
        AGE_START_COLUMN,
        AGE_END_COLUMN,
    ]
    pop = pop.reset_index().set_index(index_order)

    return pop


def _make_artifact_prevalence() -> pd.DataFrame:
    """Sample prevalence artifact data."""
    return pd.DataFrame(
        {
            "draw_0": [0.1, 0.2, 0.15, 0.25, 0.3, 0.4, 0.35, 0.45],
            "draw_1": [0.08, 0.12, 0.09, 0.11, 0.25, 0.26, 0.35, 0.36],
        },
        index=_get_artifact_index(),
    )


@pytest.fixture(scope="session")
def _artifact_keys_mapper() -> dict[str, pd.DataFrame | dict[str, str]]:
    _raw_artifact_disease_incidence = _create_raw_artifact_disease_incidence()
    _raw_artifact_risk_exposure = _create_raw_artifact_risk_exposure()
    _sample_age_group_df = _create_sample_age_group_df()
    _risk_categories = _create_risk_categories()
    _population_structure = _artifact_population_structure()
    _artifact_prevalence = _make_artifact_prevalence()
    return {
        "cause.disease.incidence_rate": _raw_artifact_disease_incidence,
        "risk_factor.child_stunting.exposure": _raw_artifact_risk_exposure,
        "population.age_bins": _sample_age_group_df,
        "risk_factor.risky_risk.categories": _risk_categories,
        "population.structure": _population_structure,
        "cause.disease.prevalence": _artifact_prevalence,
    }


def _get_artifact_index() -> pd.MultiIndex:
    """Create an expected dataframe for testing."""
    return pd.MultiIndex.from_tuples(
        [
            ("A", "C", 0.0, 5.0),
            ("A", "C", 5.0, 10.0),
            ("A", "D", 0.0, 5.0),
            ("A", "D", 5.0, 10.0),
            ("B", "C", 0.0, 5.0),
            ("B", "C", 5.0, 10.0),
            ("B", "D", 0.0, 5.0),
            ("B", "D", 5.0, 10.0),
        ],
        names=[
            "common_stratify_column",
            "other_stratify_column",
            AGE_START_COLUMN,
            AGE_END_COLUMN,
        ],
    )
