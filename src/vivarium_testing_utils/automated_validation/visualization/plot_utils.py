import pandas as pd
from vivarium_testing_utils.automated_validation.data_loader import DataSource

from vivarium_testing_utils.automated_validation.comparison import Comparison
from matplotlib import pyplot as plt
import seaborn as sns

RELPLOT_KWARGS = {
    "height": 5,
    "aspect": 1.3,
    "marker": "o",
    "markers": True,
    "errorbar": ("ci", 97.5),
    "facet_kws": {"sharex": False, "sharey": True},
}

def plot_comparison(comparison: Comparison, type: str, kwargs):
    raise NotImplementedError


def plot_data(dataset: pd.DataFrame, type: str, kwargs):
    raise NotImplementedError


def line_plot(
    title: str,
    test_data: pd.DataFrame,
    test_source: DataSource,
    reference_data: pd.DataFrame,
    reference_source: DataSource,
    x_axis: str,
    stratifications: list[str],
):
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
    if len(stratifications) > 2:
        raise ValueError("Maximum of 2 stratifications supported")

    # Prepare data
    test_data_with_source = test_data.copy()
    test_data_with_source["source"] = test_source.name.lower().capitalize()

    reference_data_with_source = reference_data.copy()
    reference_data_with_source["source"] = reference_source.name.lower().capitalize()

    # Combine datasets
    combined_data = pd.concat(
        [test_data_with_source, reference_data_with_source]
    ).reset_index()

    # Set up relplot parameters based on stratifications
    relplot_kwargs = RELPLOT_KWARGS.copy()
    relplot_kwargs["data"] = combined_data
    relplot_kwargs["hue"] = "source"
    relplot_kwargs["x"] = x_axis
    relplot_kwargs["y"] = "value"  # Assuming 'value' is the y-axis variable
    relplot_kwargs["kind"] = "line"

    # Add stratifications
    if stratifications:
        if len(stratifications) == 1:
            relplot_kwargs["col"] = stratifications[0]
            relplot_kwargs["col_wrap"] = min(3, combined_data[stratifications[0]].nunique())
        else:
            relplot_kwargs["row"] = stratifications[0]
            relplot_kwargs["col"] = stratifications[1]

    # Create the plot
    g = sns.relplot(**relplot_kwargs)

    # Customize
    g.set_axis_labels(x_axis, "Proportion")
    # Add overall title
    g.figure.suptitle(title, y=1.02, fontsize=16)
    g.map(plt.grid, alpha=0.5, color="gray")
    g.tight_layout()
    return g


def bar_plot(comparison: Comparison, x_axis: str, stratifications: list[str]):
    raise NotImplementedError


def box_plot(comparison: Comparison, cat: str, stratifications: list[str]):
    raise NotImplementedError


def heatmap(comparison: Comparison, row: str, col: str):
    raise NotImplementedError


def save_plot(fig, name, format):
    raise NotImplementedError
