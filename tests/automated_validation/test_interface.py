from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_mock import MockFixture
from vivarium.framework.artifact.artifact import ArtifactException

from tests.automated_validation.conftest import IS_ON_SLURM
from vivarium_testing_utils.automated_validation.constants import (
    DRAW_INDEX,
    INPUT_DATA_INDEX_NAMES,
)
from vivarium_testing_utils.automated_validation.data_loader import DataLoader
from vivarium_testing_utils.automated_validation.data_transformation import age_groups
from vivarium_testing_utils.automated_validation.interface import ValidationContext


def test_context_initialization(
    sim_result_dir: Path, sample_age_group_df: pd.DataFrame
) -> None:
    """Ensure that we can initialize a ValidationContext with a simulation result directory"""
    context = ValidationContext(sim_result_dir, scenario_columns=["foo"])
    assert isinstance(context, ValidationContext)
    assert isinstance(context.data_loader, DataLoader)
    assert_frame_equal(context.age_groups, sample_age_group_df)
    assert context.comparisons == {}
    assert context.scenario_columns == ["foo"]


@pytest.mark.skip("Not implemented")
def test_add_comparison_bad_source(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the source is not recognized"""
    context = ValidationContext(sim_result_dir)
    with pytest.raises(ValueError, match="Source bad_source not recognized"):
        context.add_comparison("cause.disease.incidence", "bad_source", "gbd")


def test_upload_custom_data(sim_result_dir: Path) -> None:
    """Ensure that we can upload custom data and retrieve it"""
    context = ValidationContext(sim_result_dir)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)
    assert context.get_raw_data("foo", "custom").equals(df)


def test_get_raw_data(
    sim_result_dir: Path, deaths_data: pd.DataFrame, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can show the raw data"""
    context = ValidationContext(sim_result_dir)
    df = pd.DataFrame({"baz": [1, 2, 3]})
    context.upload_custom_data("foo", df)

    # Ensure loading with a string instead of a DataSource enum works
    assert context.get_raw_data("foo", "custom").equals(df)
    assert context.get_raw_data("deaths", "sim").equals(deaths_data)
    assert context.get_raw_data("cause.disease.incidence_rate", "artifact").equals(
        artifact_disease_incidence
    )


def test__get_age_groups_art(sim_result_dir: Path, mocker: MockFixture) -> None:
    """Ensure that we grab age groups 'from the artifact' when available"""
    age_groups = pd.DataFrame(
        {
            "foo": ["bar"],
        },
    )

    # mock dataloader to return age groups
    mocker.patch(
        "vivarium_testing_utils.automated_validation.data_loader.Artifact.load",
        return_value=age_groups,
    )
    context = ValidationContext(sim_result_dir)
    assert context.age_groups.equals(age_groups)


def test__get_age_groups_gbd(sim_result_dir: Path, mocker: MockFixture) -> None:
    """Test that if age groups are not available from the artifact, we get them from vivarium_inputs"""
    age_groups = pd.DataFrame(
        {
            "foo": ["bar"],
        },
    )

    def selective_load_side_effect(data_key: str) -> pd.DataFrame:
        if data_key == "population.age_bins":
            raise ArtifactException()
        # For other keys like "population.location", return a mock value
        return pd.DataFrame({"mock_data": [1, 2, 3]})

    mocker.patch(
        "vivarium_testing_utils.automated_validation.data_loader.Artifact.load",
        side_effect=selective_load_side_effect,
    )

    mocker.patch(
        "vivarium_inputs.get_age_bins",
        return_value=age_groups,
    )
    context = ValidationContext(sim_result_dir)
    assert context.age_groups.equals(age_groups)


def test_add_comparison_bad_scenarios(sim_result_dir: Path) -> None:
    """Ensure that we raise an error if the scenarios are not provided correctly"""
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir, scenario_columns=["scenario_column"])

    # Test with missing scenarios
    with pytest.raises(ValueError, match="missing scenarios for: {'scenario_column'}"):
        context.add_comparison(measure_key, "sim", "artifact")


def test_add_comparison(
    sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can add a comparison"""
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir)
    context.add_comparison(measure_key, "sim", "artifact")
    assert measure_key in context.comparisons
    comparison = context.comparisons[measure_key]

    assert comparison.measure.measure_key == measure_key  # type: ignore [attr-defined]

    # Test that test_data is now a dictionary with numerator and denominator
    assert isinstance(comparison.test_bundle.datasets, dict)
    assert "numerator_data" in comparison.test_bundle.datasets
    assert "denominator_data" in comparison.test_bundle.datasets

    expected_index = pd.MultiIndex.from_tuples(
        [("A", "baseline"), ("B", "baseline")],
        names=["stratify_column", "scenario"],
    )

    expected_numerator_data = pd.DataFrame(
        {
            "value": [3.0, 5.0],
        },
        index=expected_index,
    )
    expected_denominator_data = pd.DataFrame(
        {
            "value": [17.0, 29.0],
        },
        index=expected_index,
    )

    assert comparison.test_bundle.datasets["numerator_data"].equals(expected_numerator_data)
    assert comparison.test_bundle.datasets["denominator_data"].equals(
        expected_denominator_data
    )
    # Update artifact reference data to match simulation format
    artifact_disease_incidence = age_groups.format_dataframe_from_age_bin_df(
        artifact_disease_incidence, context.age_groups
    )
    assert comparison.reference_bundle.datasets["data"].equals(artifact_disease_incidence)


def test_get_frame(sim_result_dir: Path) -> None:
    """Ensure that we can verify a comparison"""
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir)
    context.add_comparison(measure_key, "sim", "artifact")
    data = context.get_frame(measure_key)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert set(data.index.names) == {"common_stratify_column", "input_draw"}
    assert set(data.columns) == {"test_rate", "reference_rate", "percent_error"}

    # Test stratification works - there are only two columns and we do not remove input draw
    # so this will return the same dataframe
    data2 = context.get_frame(measure_key, stratifications=["common_stratify_column"])
    assert isinstance(data2, pd.DataFrame)
    assert not data2.empty
    assert set(data2.index.names) == {"common_stratify_column", "input_draw"}
    assert set(data2.columns) == {"test_rate", "reference_rate", "percent_error"}


def test_metadata(sim_result_dir: Path, mocker: MockFixture) -> None:
    """Ensure that we can summarize a comparison"""
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir)
    context.add_comparison(measure_key, "sim", "artifact")

    mocker.patch(
        "vivarium_testing_utils.automated_validation.interface.Path.name",
        "2025_01_01_00_00_00",
    )
    mocker.patch(
        "vivarium_testing_utils.automated_validation.interface.os.path.getmtime",
        return_value=1735718340,  # Represents Dec 31 23:59
    )
    metadata = context.metadata(measure_key)

    assert set(metadata.index) == {
        "Measure Key",
        "Source",
        "Shared Indices",
        "Source Specific Indices",
        "Size",
        "Num Draws",
        "Input Draws",
        "Run Time",
    }
    # Metadata is already tesed with comparison and bundle. Run time is the only metadata from interface
    assert metadata["Test Data"].loc["Run Time"] == "Jan 01 00:00 2025"
    assert metadata["Reference Data"]["Run Time"] == "Dec 31 23:59 2024"


######################################
# Tests for NotImplementedError cases#
######################################


def test_plot_comparison(sim_result_dir: Path, mocker: MockFixture) -> None:
    """Test that ValidationContext.plot_comparison correctly calls plot_utils.plot_comparison"""
    # Setup
    mock_figure = mocker.Mock(spec=plt.Figure)
    mock_plot_comparison = mocker.patch(
        "vivarium_testing_utils.automated_validation.visualization.plot_utils.plot_comparison",
        return_value=mock_figure,
    )

    # Create a context and add a comparison
    context = ValidationContext(sim_result_dir)
    measure_key = "cause.disease.incidence_rate"
    context.add_comparison(measure_key, "sim", "artifact")

    # Call plot_comparison with various parameters
    plot_type = "line"
    condition = {"sex": "male"}
    x_axis = "age_group"
    result = context.plot_comparison(
        comparison_key=measure_key, type=plot_type, condition=condition, x_axis=x_axis
    )

    # Assert plot_utils.plot_comparison was called with correct arguments
    mock_plot_comparison.assert_called_once()
    args, kwargs = mock_plot_comparison.call_args

    # Check arguments
    assert args[0] == context.comparisons[measure_key]  # comparison object
    assert args[1] == plot_type  # type
    assert args[2] == condition  # condition
    assert kwargs["x_axis"] == x_axis  # additional kwargs

    # Check return value
    assert result == mock_figure


@pytest.mark.parametrize("test_source", ["sim", "artifact"])
def test_add_comparison_different_test_source(
    test_source: str, sim_result_dir: Path, artifact_disease_incidence: pd.DataFrame
) -> None:
    """Ensure that we can add a comparison"""
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir)
    context.add_comparison(measure_key, test_source, "artifact")
    assert measure_key in context.comparisons
    comparison = context.comparisons[measure_key]

    assert comparison.measure.measure_key == measure_key  # type: ignore [attr-defined]

    # Test that test_data is now a dictionary with numerator and denominator
    assert isinstance(comparison.test_bundle.datasets, dict)
    if test_source == "sim":
        assert "numerator_data" in comparison.test_bundle.datasets
        assert "denominator_data" in comparison.test_bundle.datasets
    else:
        assert "data" in comparison.test_bundle.datasets


@pytest.mark.parametrize("test_source", ["sim", "artifact"])
def test_get_frame_different_test_source(test_source: str, sim_result_dir: Path) -> None:
    measure_key = "cause.disease.incidence_rate"
    context = ValidationContext(sim_result_dir)
    context.add_comparison(measure_key, test_source, "artifact")
    data = context.get_frame(measure_key)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert set(data.columns) == {"test_rate", "reference_rate", "percent_error"}


@pytest.mark.parametrize(
    "data_key",
    [
        "risk_factor.child_wasting.exposure",
        "risk_factor.child_wasting.relative_risk",
        "population.structure",
        "cause.diarrheal_diseases.remission_rate",
        "cause.diarrheal_diseases.cause_specific_mortality_rate",
        "cause.diarrheal_diseases.incidence_rate",
        "cause.diarrheal_diseases.prevalence",
        "cause.diarrheal_diseases.excess_mortality_rate",
    ],
)
def test_cache_gbd_data(sim_result_dir: Path, data_key: str) -> None:
    """Tests that we can cache custom GBD and retreive it. More importantly, tests that
    GBD data is properly mapped from id columns to value columns upon caching."""
    if not IS_ON_SLURM:
        pytest.skip("No access to slurm shared filesystem available for testing.")

    context = ValidationContext(sim_result_dir)
    # NOTE: Some of these CSVs are reused but have the same schema. Users will be expected to
    # make the correct get draws calls. For example, prevalence and incidence can be pull with
    # one call and then filtered down or pulled separately but they have the same schema.
    measure_data_mapper = {
        "risk_factor.child_wasting.exposure": "exposure",
        "risk_factor.child_wasting.relative_risk": "relative_risks",
        "population.structure": "population_structure",
        "cause.diarrheal_diseases.remission_rate": "remission_rate",
        "cause.diarrheal_diseases.cause_specific_mortality_rate": "remission_rate",
        "cause.diarrheal_diseases.incidence_rate": "incidence",
        "cause.diarrheal_diseases.prevalence": "incidence",
        "cause.diarrheal_diseases.excess_mortality_rate": "remission_rate",
    }

    file_name = measure_data_mapper[data_key] + ".csv"
    file_path = Path(__file__).parent / "gbd_data" / file_name
    gbd_data = pd.read_csv(file_path)
    context.cache_gbd_data(data_key, gbd_data)

    cached_data = context.get_raw_data(data_key, "gbd")
    assert set(cached_data.columns) == {"value"}
    index_cols = [
        "location",
        "sex",
        "age_start",
        "age_end",
        "year_start",
        "year_end",
    ]
    if data_key in [
        "risk_factor.child_wasting.exposure",
        "risk_factor.child_wasting.relative_risk",
    ]:
        index_cols.append("parameter")
        if data_key == "risk_factor.child_wasting.relative_risk":
            index_cols.append("affected_entity")
    if data_key != "population.structure":
        index_cols.append(DRAW_INDEX)

    assert set(cached_data.index.names) == (set(index_cols))
    assert set(cached_data.columns) == {"value"}


@pytest.mark.parametrize(
    "comparison_key",
    [
        "risk_factor.child_wasting.exposure",
        "risk_factor.child_wasting.relative_risk",
        "cause.diarrheal_diseases.remission_rate",
        "cause.diarrheal_diseases.cause_specific_mortality_rate",
        "cause.diarrheal_diseases.incidence_rate",
        "cause.diarrheal_diseases.prevalence",
        "cause.diarrheal_diseases.excess_mortality_rate",
    ],
)
def test_get_frame_column_order(comparison_key: str, sim_result_dir: Path) -> None:
    """Tests that get_frame returns data with the correct index column order."""

    idx_tuples = [
        ("Persephone", "Male", 0, 5, 2023, 2024),
        ("Persephone", "Male", 5, 10, 2023, 2024),
        ("Persephone", "Male", 10, 15, 2023, 2024),
        ("Persephone", "Female", 0, 5, 2023, 2024),
        ("Persephone", "Female", 5, 10, 2023, 2024),
        ("Persephone", "Female", 10, 15, 2023, 2024),
    ]
    idx_names = [
        INPUT_DATA_INDEX_NAMES.LOCATION,
        INPUT_DATA_INDEX_NAMES.SEX,
        INPUT_DATA_INDEX_NAMES.AGE_START,
        INPUT_DATA_INDEX_NAMES.AGE_END,
        INPUT_DATA_INDEX_NAMES.YEAR_START,
        INPUT_DATA_INDEX_NAMES.YEAR_END,
    ]
    data = pd.DataFrame(
        {
            "test_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "reference_rate": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            "percent_error": [33.3, 20.0, 14.3, 11.1, 9.1, 7.7],
        },
        index=pd.MultiIndex.from_tuples(
            idx_tuples,
            names=idx_names,
        ),
    )
    if comparison_key in [
        "risk_factor.child_wasting.exposure",
        "risk_factor.child_wasting.relative_risk",
    ]:
        # Add parameter level with cat1 and cat2 for each existing index group
        new_tuples = [(*tup, param) for tup in idx_tuples for param in ["cat1", "cat2"]]

        data = pd.DataFrame(
            {
                "test_rate": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6],
                "reference_rate": [
                    0.15,
                    0.15,
                    0.25,
                    0.25,
                    0.35,
                    0.35,
                    0.45,
                    0.45,
                    0.55,
                    0.55,
                    0.65,
                    0.65,
                ],
                "percent_error": [
                    33.3,
                    33.3,
                    20.0,
                    20.0,
                    14.3,
                    14.3,
                    11.1,
                    11.1,
                    9.1,
                    9.1,
                    7.7,
                    7.7,
                ],
            },
            index=pd.MultiIndex.from_tuples(
                new_tuples,
                names=idx_names + [INPUT_DATA_INDEX_NAMES.PARAMETER],
            ),
        )
        if comparison_key == "risk_factor.child_wasting.relative_risk":
            # Add "affected_entity" level with value "lost_in_space"
            data = data.assign(affected_entity="lost_in_space")
            data = data.set_index(INPUT_DATA_INDEX_NAMES.AFFECTED_ENTITY, append=True)

    # NOTE: The index levels have been added in the order they are expected to be returned to users
    expected_order = list(data.index.names)
    wrong_order = [
        INPUT_DATA_INDEX_NAMES.YEAR_END,
        INPUT_DATA_INDEX_NAMES.YEAR_START,
        INPUT_DATA_INDEX_NAMES.AGE_END,
        INPUT_DATA_INDEX_NAMES.AGE_START,
        INPUT_DATA_INDEX_NAMES.SEX,
        INPUT_DATA_INDEX_NAMES.LOCATION,
    ]
    for level in [INPUT_DATA_INDEX_NAMES.AFFECTED_ENTITY, INPUT_DATA_INDEX_NAMES.PARAMETER]:
        if level in data.index.names:
            wrong_order.append(level)
    unsorted = data.reorder_levels(wrong_order)
    assert list(unsorted.index.names) == wrong_order

    context = ValidationContext(sim_result_dir)
    # Add entity and measure to expect levels at front
    expected_order = [
        INPUT_DATA_INDEX_NAMES.ENTITY,
        INPUT_DATA_INDEX_NAMES.MEASURE,
    ] + expected_order
    sorted = context.format_ui_data_index(unsorted, comparison_key)
    assert list(sorted.index.names) == expected_order
