from typing import Any
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from vivarium_testing_utils.automated_validation.comparison import Comparison
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.visualization.plot_utils import (
    _append_condition_to_title,
    _conditionalize,
    _format_title,
    _get_combined_data,
    _get_unconditioned_index_names,
    _line_plot,
    _rel_plot,
    plot_comparison,
)


# ==================
# Shared Fixtures
# ==================
@pytest.fixture
def test_title() -> str:
    """Create a test title for the plot."""
    return "Test Title"


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create test data with two unconditioned variables where the second has more unique values."""
    # First variable (sex) has 2 values, second (age_group) has 3
    index = pd.MultiIndex.from_product(
        [
            ["male", "female"],  # sex - 2 values
            ["A", "B", "C"],  # age_group - 3 values
            ["Test", "Reference"],  # source - 2 values
            ["North", "South"],  # region 2 values
            ["susceptible", "infected", "recovered"],  # disease_state - 2 values
            [0],  # input_draw
        ],
        names=["sex", "age_group", "source", "region", "disease_state", "input_draw"],
    )
    data = pd.DataFrame(
        [i / 10 for i in range(2 * 3 * 2 * 2 * 3)],
        index=index,
        columns=["value"],
    )
    return data


@pytest.fixture
def sample_comparison(sample_data: pd.DataFrame, mocker: MockerFixture) -> Mock:
    # Create test and reference data
    test_data = sample_data.xs("Test", level="source")
    reference_data = sample_data.xs("Reference", level="source")

    # Mock Comparison object with the _align_datasets method
    mock_comparison = mocker.Mock(spec=Comparison)
    mock_comparison._align_datasets = mocker.Mock(return_value=(test_data, reference_data))

    # Set up sources
    mock_comparison.test_source = mocker.Mock(spec=DataSource)
    mock_comparison.test_source.name = "test"
    mock_comparison.test_scenarios = {}
    mock_comparison.reference_source = mocker.Mock(spec=DataSource)
    mock_comparison.reference_source.name = "reference"
    mock_comparison.reference_scenarios = {}

    # Set up measure
    mock_comparison.measure = mocker.Mock()
    mock_comparison.measure.measure_key = "measure.test_measure"

    # Type narrow for mypy
    assert isinstance(mock_comparison, Mock)

    return mock_comparison


@pytest.fixture
def mock_relplot_setup(mocker: MockerFixture) -> tuple[Mock, Mock]:
    """Set up mocked seaborn relplot with trackable figure attributes."""
    mock_relplot = mocker.patch("seaborn.relplot")

    mock_figure = mocker.Mock()
    mock_figure._test_attributes = {}

    # Mock the methods we want to test
    mock_figure.suptitle = mocker.Mock(
        side_effect=lambda title, y=None, fontsize=None, **kwargs: mock_figure._test_attributes.update(
            {"title": title, "title_y": y, "title_fontsize": fontsize}
        )
    )

    mock_figure.legend = mocker.Mock(
        side_effect=lambda **kwargs: mock_figure._test_attributes.update(
            {"legend_params": kwargs}
        )
    )

    # Set up the return value
    mock_relplot.return_value = mocker.Mock()
    mock_relplot.return_value.figure = mock_figure
    mock_relplot.return_value._legend = mocker.Mock()
    mock_relplot.return_value.set_axis_labels = mocker.Mock()
    mock_relplot.return_value.set_xticklabels = mocker.Mock()
    mock_relplot.return_value.map = mocker.Mock()
    mock_relplot.return_value.tight_layout = mocker.Mock()

    return mock_relplot, mock_figure


# ==================
# Helper Functions
# ==================


def assert_figure_attributes(
    mock_figure: Mock,
    title: str,
    y: float = 1.02,
    fontsize: int = 16,
    legend_loc: str = "upper right",
) -> None:
    """Assert that figure attributes were set correctly."""
    assert mock_figure.suptitle.called
    assert mock_figure._test_attributes.get("title") == title
    assert mock_figure._test_attributes.get("title_y") == y
    assert mock_figure._test_attributes.get("title_fontsize") == fontsize

    assert mock_figure.legend.called
    assert mock_figure._test_attributes.get("legend_params", {}).get("loc") == legend_loc


def assert_relplot_settings(mock_relplot: Mock, x_axis: str = "age_group") -> None:
    """Assert that relplot settings were applied correctly."""
    mock_relplot.return_value.set_axis_labels.assert_called_once_with(x_axis, "Proportion")
    mock_relplot.return_value.set_xticklabels.assert_called_once_with(rotation=30)
    mock_relplot.return_value.map.assert_called_once()
    assert mock_relplot.return_value.map.call_args[0][0] == plt.grid
    mock_relplot.return_value.tight_layout.assert_called_once()


# ===================================
# Tests for plot_comparison
# ===================================


class TestPlotComparison:
    def test_valid_type(self, sample_comparison: Comparison, mocker: MockerFixture) -> None:
        # Setup
        mock_line_plot = mocker.patch(
            "vivarium_testing_utils.automated_validation.visualization.plot_utils._line_plot"
        )
        mock_line_plot.return_value = plt.figure()

        # Call the function
        fig = plot_comparison(
            comparison=sample_comparison,
            type="line",
            condition={"sex": "male"},
            x_axis="age_group",
        )

        # Assert
        mock_line_plot.assert_called_once()
        assert "title" in mock_line_plot.call_args[1]
        assert "combined_data" in mock_line_plot.call_args[1]
        assert mock_line_plot.call_args[1]["x_axis"] == "age_group"

    def test_invalid_type(self, sample_comparison: Comparison) -> None:
        with pytest.raises(NotImplementedError, match="Unsupported plot type"):
            plot_comparison(comparison=sample_comparison, type="invalid")


# ===================================
# Tests for line_plot
# ===================================


class TestLinePlot:
    def test_subplots_true(
        self, test_title: str, sample_data: pd.DataFrame, mocker: MockerFixture
    ) -> None:

        mock_rel_plot = mocker.patch(
            "vivarium_testing_utils.automated_validation.visualization.plot_utils._rel_plot"
        )
        mock_rel_plot.return_value = plt.figure()
        fig = _line_plot(
            title=test_title, combined_data=sample_data, x_axis="age_group", subplots=True
        )

        # Assert rel_plot was called with correct arguments
        mock_rel_plot.assert_called_once()
        args, kwargs = mock_rel_plot.call_args
        assert kwargs["title"] == test_title
        assert kwargs["x_axis"] == "age_group"
        assert isinstance(kwargs["plot_args"], dict)

        # Verify lineplot specific arguments
        assert kwargs["plot_args"]["marker"] == "o"
        assert kwargs["plot_args"]["markers"] is True
        assert kwargs["plot_args"]["hue"] == "source"
        assert kwargs["plot_args"]["y"] == "value"
        assert kwargs["plot_args"]["errorbar"] == "pi"

    def test_subplots_false(
        self, test_title: str, sample_data: pd.DataFrame, mocker: MockerFixture
    ) -> None:

        mock_fig = mocker.Mock()
        mock_ax = mocker.Mock()
        mock_fig.add_subplot.return_value = mock_ax

        mocker.patch("matplotlib.pyplot.figure", return_value=mock_fig)
        mock_lineplot = mocker.patch("seaborn.lineplot")
        mocker.patch("matplotlib.pyplot.tight_layout")

        figures = _line_plot(
            title=test_title, combined_data=sample_data, x_axis="age_group", subplots=False
        )

        # Check we get a list of figures as expected
        assert isinstance(figures, list)
        assert len(figures) > 0

        # Test that the axis was properly configured
        assert mock_ax.set_title.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.grid.called

        # Assert lineplot was called with expected args
        assert mock_lineplot.called
        kwargs = mock_lineplot.call_args[1]
        assert kwargs["marker"] == "o"
        assert kwargs["markers"] is True
        assert kwargs["hue"] == "source"
        assert kwargs["y"] == "value"
        assert kwargs["errorbar"] == "pi"


# ===================================
# Tests for rel_plot
# ===================================


class TestRelPlot:
    def test_too_many_stratifications(self, sample_data: pd.DataFrame) -> None:
        # Setup data with 3 stratification levels (excluding input_draw and source)
        # This should fail because we allow max 2 stratification levels
        with pytest.raises(ValueError, match="Maximum of.*stratification levels supported"):
            _rel_plot(
                title="Test Title",
                combined_data=sample_data,
                x_axis="age_group",
            )

    def test_two_unconditioned(
        self,
        test_title: str,
        sample_data: pd.DataFrame,
        mock_relplot_setup: tuple[Mock, Mock],
    ) -> None:
        """Test rel_plot with two unconditioned variables where the first has more unique values."""
        filtered_data = sample_data.xs(
            "male", level="sex"
        )  # 2 unconditioned variables remain
        assert isinstance(filtered_data, pd.DataFrame)

        mock_relplot, mock_figure = mock_relplot_setup

        fig = _rel_plot(
            title=test_title,
            combined_data=filtered_data,
            x_axis="age_group",
        )

        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # Disease State has more unique values, so should be row
        assert kwargs["row"] == "disease_state"
        assert kwargs["col"] == "region"

        assert_figure_attributes(mock_figure, test_title)
        assert_relplot_settings(mock_relplot)

    def test_one_unconditioned(
        self,
        test_title: str,
        sample_data: pd.DataFrame,
        mock_relplot_setup: tuple[Mock, Mock],
    ) -> None:
        """Test rel_plot with a single unconditioned variable."""
        filtered_data = sample_data.xs(
            ("male", "North"), level=("sex", "region")
        )  # 1 unconditioned variable remains
        assert isinstance(filtered_data, pd.DataFrame)
        mock_relplot, mock_figure = mock_relplot_setup

        fig = _rel_plot(
            title=test_title,
            combined_data=filtered_data,
            x_axis="age_group",
        )

        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # With one unconditioned variable, it should be the row
        assert kwargs["row"] == "disease_state"
        assert "col" not in kwargs

        assert_figure_attributes(mock_figure, test_title)
        assert_relplot_settings(mock_relplot)

    def test_basic(
        self,
        test_title: str,
        sample_data: pd.DataFrame,
        mock_relplot_setup: tuple[Mock, Mock],
    ) -> None:
        """Test rel_plot with all variables conditioned."""
        filtered_data = sample_data.xs(
            ("male", "North", "susceptible"), level=("sex", "region", "disease_state")
        )
        assert isinstance(filtered_data, pd.DataFrame)

        mock_relplot, mock_figure = mock_relplot_setup

        fig = _rel_plot(title=test_title, combined_data=filtered_data, x_axis="age_group")

        mock_relplot.assert_called_once()

        assert_figure_attributes(mock_figure, test_title)
        assert_relplot_settings(mock_relplot)


# ===================================
# Tests for Helper Functions
# ===================================


class TestHelperFunctions:
    def test_format_title(self) -> None:
        assert _format_title("measure.test_measure") == "Test Measure"
        assert _format_title("measure.compound_name_example") == "Compound Name Example"

    def test_get_unconditioned_index_names(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [("male", "A", "Test", 0, 42)],
            names=["sex", "age_group", "source", "input_draw", "random_seed"],
        )

        # Test with different x_axis values
        assert set(_get_unconditioned_index_names(index, "age_group")) == {"sex"}
        assert set(_get_unconditioned_index_names(index, "sex")) == {"age_group"}

    def test_conditionalize(self, sample_data: pd.DataFrame) -> None:

        filtered_data = _conditionalize({"sex": "male"}, sample_data)

        assert "sex" not in filtered_data.index.names

    @pytest.mark.parametrize(
        "test_index_names,ref_index_names,expected_index_names",
        [
            # Same index structure
            (
                ["sex", "age_group", "region"],
                ["sex", "age_group", "region"],
                ["sex", "age_group", "region", "source"],
            ),
            # Test has extra index level
            (
                ["sex", "age_group", "region", "extra_level"],
                ["sex", "age_group", "region"],
                ["sex", "age_group", "region", "extra_level", "source"],
            ),
            # Reference has extra index level
            (
                ["sex", "age_group", "region"],
                ["sex", "age_group", "region", "extra_level"],
                ["sex", "age_group", "region", "extra_level", "source"],
            ),
            # Both have different extra levels
            (
                ["sex", "age_group", "test_only"],
                ["sex", "age_group", "ref_only"],
                ["sex", "age_group", "ref_only", "test_only", "source"],
            ),
        ],
    )
    def test_get_combined_data(
        self,
        test_index_names: list[str],
        ref_index_names: list[str],
        expected_index_names: list[str],
        mocker: MockerFixture,
    ) -> None:
        """Test _get_combined_data with different index structures."""
        # Create test data with specified index
        test_data = pd.DataFrame(
            {"value": [1, 2, 3, 4]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("male", "A", "North"),
                    ("male", "B", "North"),
                    ("female", "A", "South"),
                    ("female", "B", "South"),
                ],
                names=test_index_names[:3],  # Use first 3 levels
            ),
        )

        # Add extra levels if specified
        if len(test_index_names) > 3:
            for extra_level in test_index_names[3:]:
                test_data[extra_level] = "test_value"
                test_data = test_data.set_index(extra_level, append=True)

        # Create reference data with specified index
        ref_data = pd.DataFrame(
            {"value": [5, 6, 7, 8]},
            index=pd.MultiIndex.from_tuples(
                [
                    ("male", "A", "North"),
                    ("male", "B", "North"),
                    ("female", "A", "South"),
                    ("female", "B", "South"),
                ],
                names=ref_index_names[:3],  # Use first 3 levels
            ),
        )

        # Add extra levels if specified
        if len(ref_index_names) > 3:
            for extra_level in ref_index_names[3:]:
                ref_data[extra_level] = "ref_value"
                ref_data = ref_data.set_index(extra_level, append=True)

        # Mock Comparison object
        mock_comparison = mocker.Mock(spec=Comparison)
        mock_comparison._align_datasets = mocker.Mock(return_value=(test_data, ref_data))
        mock_comparison.test_source = mocker.Mock(spec=DataSource)
        mock_comparison.test_source.name = "test"
        mock_comparison.reference_source = mocker.Mock(spec=DataSource)
        mock_comparison.reference_source.name = "reference"

        # Test the function
        result = _get_combined_data(mock_comparison)

        # Verify the combined data structure
        assert "source" in result.index.names
        assert set(result.index.names) == set(expected_index_names)
        assert set(result.index.get_level_values("source").unique()) == {"Test", "Reference"}

        # Verify that both test and reference data are present
        test_rows = result.xs("Test", level="source")
        ref_rows = result.xs("Reference", level="source")
        assert len(test_rows) == len(test_data)
        assert len(ref_rows) == len(ref_data)

    @pytest.mark.parametrize(
        "condition_dict, expected",
        [
            ({}, "Original Title"),
            ({"sex": "Male"}, "Original Title\nsex = Male"),
            ({"foo": "30"}, "Original Title\nfoo = 30"),
            ({"sex": "Male", "age_group": "A"}, "Original Title\nsex = Male | age_group = A"),
        ],
    )
    def test__append_condition_to_title(
        self, condition_dict: dict[str, Any], expected: str
    ) -> None:
        """Test that empty condition dict returns original title unchanged."""
        assert _append_condition_to_title(condition_dict, "Original Title") == expected
