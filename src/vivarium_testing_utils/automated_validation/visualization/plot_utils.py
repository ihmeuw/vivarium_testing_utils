import pandas as pd
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from typing import Any
from vivarium_testing_utils.automated_validation.comparison import Comparison
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

RELPLOT_KWARGS = {
    "height": 5,
    "aspect": 1.3,
    "marker": "o",
    "markers": True,
    "facet_kws": {"sharex": False, "sharey": True},
}


def plot_comparison(comparison: Comparison, type: str, kwargs) -> Figure:
    """Create a plot for the given comparison.

    Args:
        comparison (Comparison): The comparison object to plot.
        type (str): Type of plot to create.
        kwargs: Additional keyword arguments for specific plot types.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    PLOT_TYPE_MAPPING = {
        "line": line_plot,
        "bar": bar_plot,
        "box": box_plot,
        "heatmap": heatmap,
    }
    if type not in PLOT_TYPE_MAPPING:
        raise ValueError(
            f"Unsupported plot type: {type}. Supported types are: {list(PLOT_TYPE_MAPPING.keys())}"
        )
    title = titleify(comparison.measure.measure_key)
    test_data, reference_data = comparison._align_datasets()
    test_data = _append_source(test_data, comparison.test_source)
    reference_data = _append_source(reference_data, comparison.reference_source)

    default_kwargs = {
        "title": title,
        "test_data": test_data,
        "reference_data": reference_data,
    }
    default_kwargs.update(kwargs)

    return PLOT_TYPE_MAPPING[type](**default_kwargs)


def plot_data(dataset: pd.DataFrame, type: str, kwargs):
    raise NotImplementedError


def line_plot(
    title: str,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    x_axis: str = "age_group",
    condition: dict[str, Any] = {},
) -> Figure:
    """Create a stratified line plot using Seaborn's relplot.

    Args:
        title (str): Main title for the plot
        test_data (pd.DataFrame): Test dataset
        test_source (DataSource): Source of test data
        reference_data (pd.DataFrame): Reference dataset
        reference_source (DataSource): Source of reference data
        x_axis (str): Column to use for x-axis
        stratifications (list[str]): List of columns to stratify by (max 2)

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    ALLOWED_STRATIFICATIONS = 2
    all_indexes = test_data.index.names
    stat_cols = ["input_draw", "random_seed"]
    plotted_cols = [x_axis, "source"]
    condition_cols = condition.keys()
    unconditioned = list(
        set(all_indexes) - set(stat_cols) - set(condition_cols) - set(plotted_cols)
    )

    if len(unconditioned) > ALLOWED_STRATIFICATIONS:
        raise ValueError(
            "Maximum of {ALLOWED_STRATIFICATIONS} stratification levels supported with ."
            f"Please conditionalize {len(unconditioned) - ALLOWED_STRATIFICATIONS} of levels {unconditioned}."
        )

    test_data = test_data.reorder_levels(reference_data.index.names)

    # Combine datasets
    combined_data = pd.concat([test_data, reference_data]).reset_index()
    for condition_col, condition_value in condition.items():
        combined_data = combined_data[combined_data[condition_col] == condition_value]

    # Set up relplot parameters based on stratifications
    relplot_kwargs = RELPLOT_KWARGS.copy()
    relplot_kwargs["data"] = combined_data
    relplot_kwargs["hue"] = "source"
    relplot_kwargs["x"] = x_axis
    relplot_kwargs["y"] = "value"  # Assuming 'value' is the y-axis variable
    relplot_kwargs["kind"] = "line"
    relplot_kwargs["errorbar"] = "pi"  # Nonparametric 95% CI

    if unconditioned:
        if len(unconditioned) == 2:
            first, second = unconditioned
            if (
                test_data.index.get_level_values(first).nunique()
                > test_data.index.get_level_values(second).nunique()
            ):
                relplot_kwargs["row"] = first
                relplot_kwargs["col"] = second
            else:
                relplot_kwargs["row"] = second
                relplot_kwargs["col"] = first
        else:
            relplot_kwargs["row"] = unconditioned[0]

    # Create the plot
    g = sns.relplot(**relplot_kwargs)

    # Customize
    g.set_axis_labels(x_axis, "Proportion")
    # Add overall title
    # Add subtitle for conditions
    if condition:
        # Add to title for each condition
        condition_text = f"{' | '.join([f'{k} = {v}' for k, v in condition.items()])}"
        title += f"\n given {condition_text}"

    g.figure.suptitle(title, y=1.02, fontsize=16)
    g.map(plt.grid, alpha=0.5, color="gray")
    g.tight_layout()
    return g


def bar_plot(
    title: str,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    x_axis: str,
    stratifications: list[str],
):
    raise NotImplementedError


def box_plot(
    title: str,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    cat: str,
    stratifications: list[str],
):
    raise NotImplementedError


def heatmap(
    title: str,
    test_data: pd.DataFrame,
    reference_data: pd.DataFrame,
    row: str,
    col: str,
):
    raise NotImplementedError


def save_plot(fig, name, format):
    raise NotImplementedError

##################
# Helper Methods #
##################


def _append_source(
    data: pd.DataFrame,
    source: DataSource,
) -> pd.DataFrame:
    """Append a source column to the DataFrame."""
    data_with_source = data.copy()
    data_with_source["source"] = source.name.lower().capitalize()
    data_with_source.set_index("source", append=True, inplace=True)
    return data_with_source


def titleify(measure_key: str) -> str:
    """Convert a measure key to a more readable format."""
    title = " ".join(measure_key.split(".")[1:])
    title = title.replace("_", " ")
    title = " ".join([word.capitalize() for word in title.split()])
    return title
