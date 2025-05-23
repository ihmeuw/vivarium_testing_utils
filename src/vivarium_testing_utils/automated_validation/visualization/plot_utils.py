# mypy: ignore-errors
import pandas as pd
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from typing import Any
from vivarium_testing_utils.automated_validation.comparison import Comparison
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def plot_comparison(
    comparison: Comparison, type: str, condition: dict[str, Any] = {}, **kwargs: Any
) -> Figure | list[Figure]:
    """Create a plot for the given comparison.

    Args:
        comparison (Comparison): The comparison object to plot.
        type (str): Type of plot to create.
        kwargs: Additional keyword arguments for specific plot types.

    Returns:
        matplotlib.figure.Figure | list[Figure]: The generated figure or list of figures.
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
    title = format_title(comparison.measure.measure_key)

    combined_data = get_combined_data(comparison)

    title, combined_data = conditionalize(condition, title, combined_data)

    default_kwargs = {
        "title": title,
        "combined_data": combined_data,
    }
    default_kwargs.update(kwargs)

    return PLOT_TYPE_MAPPING[type](**default_kwargs)


def plot_data(dataset: pd.DataFrame, type: str, kwargs):
    raise NotImplementedError


def line_plot(
    title: str,
    combined_data: pd.DataFrame,
    x_axis: str,
    subplots: bool = True,
) -> Figure | list[Figure]:
    """
    Create a line plot for the given data.
    Args:
        title: Intended plot title.
        combined_data: Test and Reference data to plot.
        x_axis: Column to use for the x-axis.
        subplots: Whether to create subplots for each condition.
    Returns:
         The generated figure or list of figures."""

    LINEPLOT_KWARGS = {
        "marker": "o",
        "markers": True,
        "hue": "source",
        "y": "value",  # Assuming 'value' is the y-axis variable
        "errorbar": "pi",  # Nonparametric 95% CI
    }

    if subplots:
        return rel_plot(
            title=title,
            combined_data=combined_data,
            x_axis=x_axis,
            plot_args=LINEPLOT_KWARGS,
        )
    else:
        unconditioned = get_unconditioned_index_names(combined_data.index, x_axis)

        # Close any existing figures to avoid conflicts
        plt.close("all")

        # List to store individual figures
        figures = []

        # Create individual figures for each condition
        for grouped_idx, grouped_df in combined_data.groupby(level=unconditioned):
            if not isinstance(grouped_idx, tuple):
                grouped_idx = (grouped_idx,)
            # Create a new figure for each condition
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            condition_text = f"{' | '.join([f'{col} = {val}' for col, val in zip(unconditioned, grouped_idx)])}"
            condition_title = f"{title}\n {condition_text}"

            sns.lineplot(
                data=grouped_df.reset_index(),
                x=x_axis,
                ax=ax,
                **LINEPLOT_KWARGS,
            )

            ax.set_title(condition_title)
            ax.set_xlabel(x_axis)
            ax.set_ylabel("Proportion")
            ax.grid(alpha=0.5, color="gray")

            # Finalize the figure
            plt.tight_layout()

            # Add to our list of figures
            figures.append(fig)

        return figures


def rel_plot(
    title: str,
    combined_data: pd.DataFrame,
    x_axis: str = "age_group",
    plot_args: dict[str, Any] = {},
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
    unconditioned = get_unconditioned_index_names(combined_data.index, x_axis)

    if len(unconditioned) > ALLOWED_STRATIFICATIONS:
        raise ValueError(
            "Maximum of {ALLOWED_STRATIFICATIONS} stratification levels supported with ."
            f"Please conditionalize {len(unconditioned) - ALLOWED_STRATIFICATIONS} of levels {unconditioned}."
        )

    # Set up relplot parameters based on stratifications
    relplot_kwargs = plot_args.copy()
    relplot_kwargs["facet_kws"] = {"sharex": False, "sharey": True}
    relplot_kwargs["kind"] = "line"

    if unconditioned:
        if len(unconditioned) == 2:
            first, second = unconditioned
            if (
                combined_data.index.get_level_values(first).nunique()
                > combined_data.index.get_level_values(second).nunique()
            ):
                relplot_kwargs["row"] = first
                relplot_kwargs["col"] = second
            else:
                relplot_kwargs["row"] = second
                relplot_kwargs["col"] = first
        else:
            relplot_kwargs["row"] = unconditioned[0]

    # Create the plot
    g = sns.relplot(data=combined_data.reset_index(), x=x_axis, **relplot_kwargs)

    # Customize
    g.set_axis_labels(x_axis, "Proportion")
    g.set_xticklabels(rotation=30)

    # Custom Legend
    g._legend.remove()  # Remove the default legend
    g.figure.legend(loc="upper right")

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


def format_title(measure_key: str) -> str:
    """Convert a measure key to a more readable format."""
    title = " ".join(measure_key.split(".")[1:])
    title = title.replace("_", " ")
    title = " ".join([word.capitalize() for word in title.split()])
    return title


def get_unconditioned_index_names(
    index: pd.Index,
    x_axis: str,
) -> list[str]:
    """Get the columns that are not conditioned on."""
    all_index_names = index.names
    stat_cols = ["input_draw", "random_seed"]
    plotted_cols = ["source", x_axis]
    unconditioned = list(set(all_index_names) - set(stat_cols) - set(plotted_cols))
    return unconditioned


def get_combined_data(comparison: Comparison) -> pd.DataFrame:
    """Get the combined data from the test and reference datasets."""
    test_data, reference_data = comparison._align_datasets()
    test_data = _append_source(test_data, comparison.test_source)
    reference_data = _append_source(reference_data, comparison.reference_source)
    test_data = test_data.reorder_levels(reference_data.index.names)
    combined_data = pd.concat([test_data, reference_data])
    return combined_data


def conditionalize(
    condition_dict: dict[str, Any], title: str, data: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    """Filter the data based on the condition dictionary."""
    for condition_level, condition_value in condition_dict.items():
        data = data.query(f"{condition_level} == '{condition_value}'")
        data = data.droplevel(condition_level)

    if condition_dict:
        title += f"\n{' | '.join([f'{k} = {v}' for k, v in condition_dict.items()])}"
    return title, data
