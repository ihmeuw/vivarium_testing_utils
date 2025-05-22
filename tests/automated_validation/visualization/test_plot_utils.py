import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.comparison import Comparison

from vivarium_testing_utils.automated_validation.visualization.plot_utils import (
    rel_plot,
    line_plot,
    plot_comparison,
    titleify,
    get_unconditioned_index_names,
    _append_source,
    conditionalize,
    get_combined_data,
)


@pytest.fixture
def sample_multi_index_data() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [
            ["male", "female"],  # sex
            ["cat1", "cat2"],  # category
            ["2020", "2021"],  # year
            [0, 1],  # input_draw
        ],
        names=["sex", "category", "year", "input_draw"],
    )
    data = pd.DataFrame(
        np.random.random(len(index)),
        index=index,
        columns=["value"],
    )
    return data


@pytest.fixture
def sample_comparison() -> Comparison:
    # Create test and reference data
    test_data = pd.DataFrame(
        {"value": [0.1, 0.2, 0.3, 0.4]},
        index=pd.MultiIndex.from_tuples(
            [("male", "A", 0), ("male", "B", 0), ("female", "A", 0), ("female", "B", 0)],
            names=["sex", "age_group", "input_draw"],
        ),
    )

    reference_data = pd.DataFrame(
        {"value": [0.15, 0.25, 0.35, 0.45]},
        index=pd.MultiIndex.from_tuples(
            [("male", "A", 0), ("male", "B", 0), ("female", "A", 0), ("female", "B", 0)],
            names=["sex", "age_group", "input_draw"],
        ),
    )

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


def test_line_plot_subplots_true() -> None:
    # Setup
    plt.close("all")
    title = "Test Title"
    data = pd.DataFrame(
        {"value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        index=pd.MultiIndex.from_tuples(
            [
                ("male", "A", "Test", 0),
                ("male", "B", "Test", 0),
                ("female", "A", "Test", 0),
                ("male", "A", "Reference", 0),
                ("male", "B", "Reference", 0),
                ("female", "A", "Reference", 0),
            ],
            names=["sex", "age_group", "source", "input_draw"],
        ),
    )

    # Call the function with subplots=True
    with patch(
        "vivarium_testing_utils.automated_validation.visualization.plot_utils.rel_plot"
    ) as mock_rel_plot:
        mock_rel_plot.return_value = plt.figure()
        fig = line_plot(title=title, combined_data=data, x_axis="age_group", subplots=True)

        # Assert rel_plot was called with correct arguments
        mock_rel_plot.assert_called_once()
        args, kwargs = mock_rel_plot.call_args
        assert kwargs["title"] == title
        assert kwargs["x_axis"] == "age_group"
        assert isinstance(kwargs["plot_args"], dict)


def test_line_plot_subplots_false() -> None:
    # Setup
    plt.close("all")
    title = "Test Title"
    data = pd.DataFrame(
        {"value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        index=pd.MultiIndex.from_tuples(
            [
                ("male", "A", "Test", 0),
                ("male", "B", "Test", 0),
                ("female", "A", "Test", 0),
                ("male", "A", "Reference", 0),
                ("male", "B", "Reference", 0),
                ("female", "A", "Reference", 0),
            ],
            names=["sex", "age_group", "source", "input_draw"],
        ),
    )

    # Call the function with subplots=False
    with patch("matplotlib.figure.Figure.add_subplot"), patch("seaborn.lineplot"), patch(
        "matplotlib.pyplot.tight_layout"
    ):
        figures = line_plot(
            title=title, combined_data=data, x_axis="age_group", subplots=False
        )

        # Check we get a list of figures as expected
        assert isinstance(figures, list)
        assert len(figures) > 0
        assert all(isinstance(fig, plt.Figure) for fig in figures)


def test_rel_plot_basic() -> None:
    # Setup
    plt.close("all")
    title = "Test Title"
    data = pd.DataFrame(
        {"value": [0.1, 0.2, 0.3, 0.4]},
        index=pd.MultiIndex.from_tuples(
            [
                ("male", "A", "Test", 0),
                ("male", "B", "Test", 0),
                ("male", "A", "Reference", 0),
                ("male", "B", "Reference", 0),
            ],
            names=["sex", "age_group", "source", "input_draw"],
        ),
    )

    # Call the function
    with patch("seaborn.relplot") as mock_relplot:
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = Mock()
        mock_relplot.return_value._legend = Mock()

        fig = rel_plot(title=title, combined_data=data, x_axis="age_group")

        # Assert relplot was called correctly
        mock_relplot.assert_called_once()


def test_rel_plot_too_many_stratifications(sample_multi_index_data: pd.DataFrame) -> None:
    # Setup data with 3 stratification levels (excluding input_draw and source)
    # This should fail because we allow max 2 stratification levels
    with pytest.raises(ValueError, match="Maximum of.*stratification levels supported"):
        rel_plot(
            title="Test Title",
            combined_data=sample_multi_index_data,
            x_axis="invalid",  # This will make all columns unconditioned
        )


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


def test_titleify() -> None:
    assert titleify("measure.test_measure") == "Test Measure"
    assert titleify("measure.compound_name_example") == "Compound Name Example"


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


def test_conditionalize() -> None:
    # Setup
    data = pd.DataFrame(
        {"value": [0.1, 0.2, 0.3, 0.4]},
        index=pd.MultiIndex.from_tuples(
            [("male", "A", 0), ("male", "B", 0), ("female", "A", 0), ("female", "B", 0)],
            names=["sex", "age_group", "input_draw"],
        ),
    )
    title = "Original Title"

    # Call function
    new_title, filtered_data = conditionalize({"sex": "male"}, title, data)

    # Assert
    assert "sex = male" in new_title
    assert "sex" not in filtered_data.index.names
    assert len(filtered_data) == 2


def test_get_combined_data(sample_comparison: Comparison) -> None:
    # Call the function
    result = get_combined_data(sample_comparison)

    # Assert
    assert "source" in result.index.names
    assert len(result) == 8  # 4 rows from test + 4 rows from reference
    assert set(result.index.get_level_values("source").unique()) == {"Test", "Reference"}


@pytest.fixture
def data_with_two_unconditioned_first_larger() -> pd.DataFrame:
    """Create test data with two unconditioned variables where the first has more unique values."""
    # First variable (sex) has 3 values, second (age_group) has 2
    index = pd.MultiIndex.from_product(
        [
            ["male", "female", "other"],  # sex - 3 values
            ["A", "B"],  # age_group - 2 values
            ["Test", "Reference"],  # source
            [0],  # input_draw
        ],
        names=["sex", "age_group", "source", "input_draw"],
    )
    data = pd.DataFrame(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        index=index,
        columns=["value"],
    )
    return data


@pytest.fixture
def data_with_two_unconditioned_second_larger() -> pd.DataFrame:
    """Create test data with two unconditioned variables where the second has more unique values."""
    # First variable (sex) has 2 values, second (age_group) has 3
    index = pd.MultiIndex.from_product(
        [
            ["male", "female"],  # sex - 2 values
            ["A", "B", "C"],  # age_group - 3 values
            ["Test", "Reference"],  # source
            [0],  # input_draw
        ],
        names=["sex", "age_group", "source", "input_draw"],
    )
    data = pd.DataFrame(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        index=index,
        columns=["value"],
    )
    return data


@pytest.fixture
def data_with_one_unconditioned() -> pd.DataFrame:
    """Create test data with one unconditioned variable."""
    index = pd.MultiIndex.from_product(
        [
            ["male", "female"],  # sex
            ["Test", "Reference"],  # source
            [0],  # input_draw
        ],
        names=["sex", "source", "input_draw"],
    )
    data = pd.DataFrame(
        [0.1, 0.2, 0.3, 0.4],
        index=index,
        columns=["value"],
    )
    return data


def test_rel_plot_two_unconditioned_first_larger(
    data_with_two_unconditioned_first_larger: pd.DataFrame,
) -> None:
    """Test rel_plot with two unconditioned variables where the first has more unique values."""
    plt.close("all")
    title = "Test Title"

    with patch("seaborn.relplot") as mock_relplot:
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = Mock()
        mock_relplot.return_value._legend = Mock()

        fig = rel_plot(
            title=title,
            combined_data=data_with_two_unconditioned_first_larger,
            x_axis="source",  # This makes both sex and age_group unconditioned
        )

        # Check that relplot was called with the correct arguments
        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # First variable (sex) has more unique values, so should be row
        assert kwargs["row"] == "sex"
        assert kwargs["col"] == "age_group"


def test_rel_plot_two_unconditioned_second_larger(
    data_with_two_unconditioned_second_larger: pd.DataFrame,
) -> None:
    """Test rel_plot with two unconditioned variables where the second has more unique values."""
    plt.close("all")
    title = "Test Title"

    with patch("seaborn.relplot") as mock_relplot:
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = Mock()
        mock_relplot.return_value._legend = Mock()

        fig = rel_plot(
            title=title,
            combined_data=data_with_two_unconditioned_second_larger,
            x_axis="source",  # This makes both sex and age_group unconditioned
        )

        # Check that relplot was called with the correct arguments
        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # Second variable (age_group) has more unique values, so should be row
        assert kwargs["row"] == "age_group"
        assert kwargs["col"] == "sex"


def test_rel_plot_one_unconditioned(data_with_one_unconditioned: pd.DataFrame) -> None:
    """Test rel_plot with a single unconditioned variable."""
    plt.close("all")
    title = "Test Title"

    with patch("seaborn.relplot") as mock_relplot:
        mock_relplot.return_value = Mock()
        mock_relplot.return_value.figure = Mock()
        mock_relplot.return_value._legend = Mock()

        fig = rel_plot(
            title=title,
            combined_data=data_with_one_unconditioned,
            x_axis="source",  # This makes sex unconditioned
        )

        # Check that relplot was called with the correct arguments
        mock_relplot.assert_called_once()
        kwargs = mock_relplot.call_args[1]

        # With one unconditioned variable, it should be the row
        assert kwargs["row"] == "sex"
        assert "col" not in kwargs
