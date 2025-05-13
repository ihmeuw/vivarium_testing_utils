from typing import Any

import pandas as pd
from pandas.io.formats.style import Styler as DfStyler

from vivarium_testing_utils.automated_validation.data_loader import DataSource


def get_metadata_from_dataset(source: DataSource, dataframe: pd.DataFrame) -> dict[str, Any]:
    """Organize the data information into a dictionary for display by a styled pandas DataFrame.

    Parameters:
    -----------
    source
        The source of the data (i.e. sim, artifact, or gbd)
    dataframe
        The DataFrame containing the data to be displayed
    Returns:
    --------
    A dictionary containing the data information.

    """
    data_info: dict[str, Any] = {}
    data_info["source"] = source.value
    data_info["index_columns"] = dataframe.index.names
    data_info["size"] = dataframe.shape
    if "input_draw" in dataframe.index.names:
        data_info["num_draws"] = dataframe.index.get_level_values("input_draw").nunique()
        data_info["input_draw"] = dataframe.index.get_level_values("input_draw").unique()
    else:
        data_info["num_draws"] = 0
    if "random_seed" in dataframe.index.names:
        data_info["num_seeds"] = dataframe.index.get_level_values("random_seed").nunique()
    return data_info


def format_metadata_pandas(
    measure_key: str, test_info: dict[str, Any], reference_info: dict[str, Any]
) -> DfStyler:
    """
    Format the comparison data as a styled pandas DataFrame

    Parameters:
    -----------
    measure_key
        The key of the measure being compared
    test_info
        Information about the test data to be displayed
    reference_info
        Information about the reference data to be displayed
    Returns:
    --------
        Styled DataFrame for display
    """
    # Extract necessary data

    # Create data for summary table
    data = {
        "Property": [
            "Measure Key",
            "Source",
            "Index Columns",
            "Size",
            "Number of Draws",
            "Draw Sample",
        ],
        "Test Data": _get_display_formatting(measure_key, test_info),
        "Reference Data": _get_display_formatting(measure_key, reference_info),
    }

    # Create and style DataFrame
    df = pd.DataFrame(data)

    # Apply styling
    styled_df = df.style.set_properties(
        **{"text-align": "left", "padding": "10px", "border": "1px solid #dddddd"}  # type: ignore[arg-type]
    )

    # Add title as caption
    styled_df = styled_df.set_caption("Comparison Summary").set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "16px"),
                    ("font-weight", "bold"),
                ],
            }
        ]
    )

    # Color headers
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1E1E1E"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("padding", "10px"),
                ],
            }
        ]
    )

    # Alternate row colors
    styled_df = styled_df.apply(
        lambda x: ["background-color: #E6F0FF" if i % 2 == 0 else "" for i in range(len(x))],
        axis=0,
    )

    return styled_df


def _get_display_formatting(measure_key: str, data_info: dict[str, Any]) -> list[str]:
    source = data_info.get("source", "Unknown")
    size = data_info.get("size", (0, 0))
    num_draws = data_info.get("num_draws", 0)
    index_cols: list[str] = data_info.get("index_columns", [])

    return [
        measure_key,
        source,
        ", ".join(str(col) for col in index_cols),
        f"{size[0]:,} rows × {size[1]:,} columns",
        f"{num_draws:,}",
        _format_draws_sample(data_info.get("input_draw", [])),
    ]


def _format_draws_sample(draw_index: Any, max_display: int = 5) -> str:
    """Helper function to format draw samples for display.

    Parameters:
    -----------
    draw_index
        The index of the draws to be formatted
    max_display
        The maximum number of draws to display. If the number of draws exceeds this, the
        function will display the first and last max_display draws, separated by ellipses.
    Returns:
    --------
        A string representation of the draws sample.
    """
    if hasattr(draw_index, "__iter__"):
        # Convert to list if it's any iterable
        draw_list = list(draw_index)

        if len(draw_list) <= max_display * 2:
            return str(draw_list)
        else:
            first = draw_list[:max_display]
            last = draw_list[-max_display:]
            return f"{first} ... {last}"
    else:
        return str(draw_index)
