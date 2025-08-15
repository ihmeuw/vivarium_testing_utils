from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_mock import MockFixture
from vivarium.framework.artifact.artifact import ArtifactException

from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation import age_groups
from vivarium_testing_utils.automated_validation.data_transformation.measures import Incidence
from vivarium_testing_utils.automated_validation.interface import ValidationContext


def test_context_initialization(
    sim_result_dir: Path, sample_age_group_df: pd.DataFrame
) -> None:
    """Ensure that we can initialize a ValidationContext with a simulation result directory"""
    context = ValidationContext(sim_result_dir, scenario_columns=["foo"])
    assert isinstance(context, ValidationContext)
    assert isinstance(context._data_loader, DataLoader)
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
    mocker.patch(
        "vivarium_testing_utils.automated_validation.data_loader.Artifact.load",
        side_effect=ArtifactException(),
    )

    mocker.patch(
        "vivarium_inputs.get_age_bins",
        return_value=age_groups,
    )
    context = ValidationContext(sim_result_dir)
    assert context.age_groups.equals(age_groups)


def test___get_raw_data_from_source(
    sim_result_dir: Path,
    transition_count_data: pd.DataFrame,
    person_time_data: pd.DataFrame,
    artifact_disease_incidence: pd.DataFrame,
) -> None:
    """Ensure that we can get raw data from a source"""
    context = ValidationContext(sim_result_dir)
    measure = Incidence("disease")
    test_raw_data = context._get_raw_data_from_source(
        measure.get_required_datasets(DataSource.SIM), DataSource.SIM
    )
    ref_raw_data = context._get_raw_data_from_source(
        measure.get_required_datasets(DataSource.ARTIFACT), DataSource.ARTIFACT
    )

    assert test_raw_data["numerator_data"].equals(transition_count_data)
    assert test_raw_data["denominator_data"].equals(person_time_data)
    assert ref_raw_data["artifact_data"].equals(artifact_disease_incidence)


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

    assert comparison.measure.measure_key == measure_key

    # Test that test_data is now a dictionary with numerator and denominator
    assert isinstance(comparison.test_datasets, dict)
    assert "numerator_data" in comparison.test_datasets
    assert "denominator_data" in comparison.test_datasets

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

    assert comparison.test_datasets["numerator_data"].equals(expected_numerator_data)
    assert comparison.test_datasets["denominator_data"].equals(expected_denominator_data)
    # Update artifact reference data to match simulation format
    artifact_disease_incidence = age_groups.format_dataframe_from_age_bin_df(
        artifact_disease_incidence, context.age_groups
    )
    assert comparison.reference_data.equals(artifact_disease_incidence)


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


######################################
# Tests for NotImplementedError cases#
######################################


def test_not_implemented(sim_result_dir: Path) -> None:
    """Test that ValidationContext.add_comparison raises NotImplementedError when test_source is not 'sim'."""
    context = ValidationContext(sim_result_dir)

    with pytest.raises(
        NotImplementedError,
        match="Comparison for artifact source not implemented. Must be SIM.",
    ):
        context.add_comparison("cause.disease.incidence_rate", "artifact", "gbd")


@pytest.mark.skip("Not implemented")
def test_metadata() -> None:
    """Ensure that we can summarize a comparison"""
    pass


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
