from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vivarium_testing_utils.automated_validation.comparison import Comparison
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.visualization.plot_utils import (
    _append_source,
    conditionalize,
    format_title,
    get_combined_data,
    get_unconditioned_index_names,
    line_plot,
    plot_comparison,
    rel_plot,
)


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
def sample_comparison(sample_data: pd.DataFrame) -> Comparison:
    # Create test and reference data
    test_data = sample_data.xs("Test", level="source")
    reference_data = sample_data.xs("Reference", level="source")

    # Mock Comparison object with the _align_datasets method
    mock_comparison = Mock()
    mock_comparison._align_datasets = Mock(return_value=(test_data, reference_data))

    # Set up sources
    mock_comparison.test_source = Mock(spec=DataSource)
    mock_comparison.test_source.name = "test"
    mock_comparison.reference_source = Mock(spec=DataSource)
    mock_comparison.reference_source.name = "reference"

    # Set up measure
    mock_comparison.measure = Mock()
    mock_comparison.measure.measure_key = "measure.test_measure"

    return mock_comparison


def test_plot_comparison(sample_comparison: Comparison) -> None:
    # Setup
    with patch(
        "vivarium_testing_utils.automated_validation.visualization.plot_utils.line_plot"
    ) as mock_line_plot:
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


def test_plot_comparison_invalid_type(sample_comparison: Comparison) -> None:
    with pytest.raises(ValueError, match="Unsupported plot type"):
        plot_comparison(comparison=sample_comparison, type="invalid")


def test_line_plot_subplots_true(sample_data) -> None:
    # Setup
    plt.close("all")
    title = "Test Title"

    # Call the function with subplots=True
    with patch(
        "vivarium_testing_utils.automated_validation.visualization.plot_utils.rel_plot"
    ) as mock_rel_plot:
        mock_rel_plot.return_value = plt.figure()
        fig = line_plot(
            title=title, combined_data=sample_data, x_axis="age_group", subplots=True
        )

        # Assert rel_plot was called with correct arguments
        mock_rel_plot.assert_called_once()
        args, kwargs = mock_rel_plot.call_args
        assert kwargs["title"] == title
        assert kwargs["x_axis"] == "age_group"
        assert isinstance(kwargs["plot_args"], dict)

        # Verify lineplot specific arguments
        assert kwargs["plot_args"]["marker"] == "o"
        assert kwargs["plot_args"]["markers"] is True
        assert kwargs["plot_args"]["hue"] == "source"
        assert kwargs["plot_args"]["y"] == "value"
        assert kwargs["plot_args"]["errorbar"] == "pi"


def test_line_plot_subplots_false(sample_data) -> None:
    # Setup
    plt.close("all")
    title = "Test Title"

    # Create mocks for matplotlib components
    mock_fig = Mock()
    mock_ax = Mock()
    mock_fig.add_subplot.return_value = mock_ax

    # Call the function with subplots=False
    with patch("matplotlib.pyplot.figure", return_value=mock_fig), patch(
        "seaborn.lineplot"
    ) as mock_lineplot, patch("matplotlib.pyplot.tight_layout"):
        figures = line_plot(
            title=title, combined_data=sample_data, x_axis="age_group", subplots=False
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


def test_rel_plot_too_many_stratifications(sample_data: pd.DataFrame) -> None:
    # Setup data with 3 stratification levels (excluding input_draw and source)
    # This should fail because we allow max 2 stratification levels
    with pytest.raises(ValueError, match="Maximum of.*stratification levels supported"):
        rel_plot(
            title="Test Title",
            combined_data=sample_data,
            x_axis="age_group",
        )


def test_rel_plot_two_unconditioned(
    sample_data: pd.DataFrame,
) -> None:
    """Test rel_plot with two unconditioned variables where the first has more unique values."""
    plt.close("all")
    title = "Test Title"
    filtered_data = sample_data.xs("male", level="sex")  # 2 unconditioned variables remain
    with patch("seaborn.relplot") as mock_relplot:
        # Create a mock with trackable attributes for testing
        mock_figure = Mock()
        mock_figure._test_attributes = {}

        # Mock the methods we want to test
        mock_figure.suptitle = Mock(
            side_effect=lambda title, y=None, fontsize=None, **kwargs: mock_figure._test_attributes.update(
                {"title": title, "title_y": y, "title_fontsize": fontsize}
            )
        )

        mock_figure.legend = Mock(
            side_effect=lambda **kwargs: mock_figure._test_attributes.update(
                {"legend_params": kwargs}
            )
        )

        # Set up the return value
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = mock_figure
        mock_relplot.return_value._legend = Mock()
        mock_relplot.return_value.set_axis_labels = Mock()
        mock_relplot.return_value.set_xticklabels = Mock()

        fig = rel_plot(
            title=title,
            combined_data=filtered_data,
            x_axis="age_group",
        )

        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # Disease State has more unique values, so should be row
        assert kwargs["row"] == "disease_state"
        assert kwargs["col"] == "region"

        # Test that title was set correctly
        assert mock_figure.suptitle.called
        assert mock_figure._test_attributes.get("title") == title
        assert mock_figure._test_attributes.get("title_y") == 1.02
        assert mock_figure._test_attributes.get("title_fontsize") == 16

        # Test legend configuration
        assert mock_figure.legend.called
        assert (
            mock_figure._test_attributes.get("legend_params", {}).get("loc") == "upper right"
        )

        # Test axis labels were set
        mock_relplot.return_value.set_axis_labels.assert_called_once_with(
            "age_group", "Proportion"
        )

        # Test that xtick labels were rotated
        mock_relplot.return_value.set_xticklabels.assert_called_once_with(rotation=30)


def test_rel_plot_one_unconditioned(sample_data: pd.DataFrame) -> None:
    """Test rel_plot with a single unconditioned variable."""
    plt.close("all")
    title = "Test Title"
    filtered_data = sample_data.xs(
        ("male", "North"), level=["sex", "region"]
    )  # 1 unconditioned variable remains
    with patch("seaborn.relplot") as mock_relplot:
        # Create a mock with trackable attributes for testing
        mock_figure = Mock()
        mock_figure._test_attributes = {}

        # Mock the methods we want to test
        mock_figure.suptitle = Mock(
            side_effect=lambda title, y=None, fontsize=None, **kwargs: mock_figure._test_attributes.update(
                {"title": title, "title_y": y, "title_fontsize": fontsize}
            )
        )

        mock_figure.legend = Mock(
            side_effect=lambda **kwargs: mock_figure._test_attributes.update(
                {"legend_params": kwargs}
            )
        )

        # Set up the return value
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = mock_figure
        mock_relplot.return_value._legend = Mock()
        mock_relplot.return_value.set_axis_labels = Mock()
        mock_relplot.return_value.set_xticklabels = Mock()

        fig = rel_plot(
            title=title,
            combined_data=filtered_data,
            x_axis="age_group",  # This makes sex unconditioned
        )

        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # With one unconditioned variable, it should be the row
        assert kwargs["row"] == "disease_state"
        assert "col" not in kwargs

        # Test that title was set correctly
        assert mock_figure.suptitle.called
        assert mock_figure._test_attributes.get("title") == title
        assert mock_figure._test_attributes.get("title_y") == 1.02
        assert mock_figure._test_attributes.get("title_fontsize") == 16

        # Test legend configuration
        assert mock_figure.legend.called
        assert (
            mock_figure._test_attributes.get("legend_params", {}).get("loc") == "upper right"
        )

        # Test axis labels were set
        mock_relplot.return_value.set_axis_labels.assert_called_once_with(
            "age_group", "Proportion"
        )

        # Test that xtick labels were rotated
        mock_relplot.return_value.set_xticklabels.assert_called_once_with(rotation=30)


def test_rel_plot_basic(
    sample_data: pd.DataFrame,
) -> None:
    # Setup
    plt.close("all")
    filtered_data = sample_data.xs(
        ("male", "North", "susceptible"), level=["sex", "region", "disease_state"]
    )
    title = "Test Title"
    # Call the function
    with patch("seaborn.relplot") as mock_relplot:
        # Create a mock with trackable attributes for testing
        mock_figure = Mock()
        mock_figure._test_attributes = {}

        # Mock the methods we want to test
        mock_figure.suptitle = Mock(
            side_effect=lambda title, y=None, fontsize=None, **kwargs: mock_figure._test_attributes.update(
                {"title": title, "title_y": y, "title_fontsize": fontsize}
            )
        )

        mock_figure.legend = Mock(
            side_effect=lambda **kwargs: mock_figure._test_attributes.update(
                {"legend_params": kwargs}
            )
        )

        # Set up the return value
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = mock_figure
        mock_relplot.return_value._legend = Mock()
        mock_relplot.return_value.set_axis_labels = Mock()
        mock_relplot.return_value.set_xticklabels = Mock()
        mock_relplot.return_value.map = Mock()
        mock_relplot.return_value.tight_layout = Mock()

        fig = rel_plot(title=title, combined_data=filtered_data, x_axis="age_group")

        # Assert relplot was called correctly
        mock_relplot.assert_called_once()

        # Test that title was set correctly
        assert mock_figure.suptitle.called
        assert mock_figure._test_attributes.get("title") == title
        assert mock_figure._test_attributes.get("title_y") == 1.02
        assert mock_figure._test_attributes.get("title_fontsize") == 16

        # Test legend configuration
        assert mock_figure.legend.called
        assert (
            mock_figure._test_attributes.get("legend_params", {}).get("loc") == "upper right"
        )

        # Test axis labels were set
        mock_relplot.return_value.set_axis_labels.assert_called_once_with(
            "age_group", "Proportion"
        )

        # Test that xtick labels were rotated
        mock_relplot.return_value.set_xticklabels.assert_called_once_with(rotation=30)

        # Test that grid was applied
        mock_relplot.return_value.map.assert_called_once()
        assert mock_relplot.return_value.map.call_args[0][0] == plt.grid

        # Test that tight_layout was called
        mock_relplot.return_value.tight_layout.assert_called_once()


def test_format_title() -> None:
    assert format_title("measure.test_measure") == "Test Measure"
    assert format_title("measure.compound_name_example") == "Compound Name Example"


def test_get_unconditioned_index_names() -> None:
    # Setup
    index = pd.MultiIndex.from_tuples(
        [("male", "A", "Test", 0, 42)],
        names=["sex", "age_group", "source", "input_draw", "random_seed"],
    )

    # Test with different x_axis values
    assert set(get_unconditioned_index_names(index, "age_group")) == {"sex"}
    assert set(get_unconditioned_index_names(index, "sex")) == {"age_group"}


def test_append_source() -> None:
    # Setup
    data = pd.DataFrame(
        {"value": [0.1, 0.2]},
        index=pd.MultiIndex.from_tuples(
            [("male", 0), ("female", 0)], names=["sex", "input_draw"]
        ),
    )
    source = Mock(spec=DataSource)
    source.name = "test_source"

    # Call function
    result = _append_source(data, source)

    # Assert
    assert "source" in result.index.names
    assert "Test_source" in result.index.get_level_values("source").unique()


def test_conditionalize(sample_data) -> None:
    title = "Original Title"

    # Call function
    new_title, filtered_data = conditionalize({"sex": "male"}, title, sample_data)

    # Assert
    assert "sex = male" in new_title
    assert "sex" not in filtered_data.index.names


def test_get_combined_data(sample_comparison: Comparison) -> None:
    # Call the function
    result = get_combined_data(sample_comparison)

    # Assert
    assert "source" in result.index.names
    assert set(result.index.get_level_values("source").unique()) == {"Test", "Reference"}
