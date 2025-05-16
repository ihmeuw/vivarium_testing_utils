from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource

REQUIRED_KEYS = ["measure_key", "source", "index_columns", "size", "num_draws", "input_draws"]


def get_metadata_from_dataset(source: DataSource, dataframe: pd.DataFrame) -> dict[str, Any]:
    """Organize the data information into a dictionary for display by a styled pandas DataFrame.
    Apply formatting to values that need special handling.

    Parameters:
    -----------
    source
        The source of the data (i.e. sim, artifact, or gbd)
    dataframe
        The DataFrame containing the data to be displayed
    Returns:
    --------
    A dictionary containing the formatted data information.

    """
    data_info: dict[str, Any] = {}

    # Source as string
    data_info["source"] = source.value

    # Index columns as comma-separated string
    index_cols = dataframe.index.names
    data_info["index_columns"] = ", ".join(str(col) for col in index_cols)

    # Size as formatted string
    size = dataframe.shape
    data_info["size"] = f"{size[0]:,} rows × {size[1]:,} columns"

    # Draw information
    if "input_draw" in dataframe.index.names:
        num_draws = dataframe.index.get_level_values("input_draw").nunique()
        data_info["num_draws"] = f"{num_draws:,}"
        draw_values = dataframe.index.get_level_values("input_draw").unique()
        data_info["input_draws"] = _format_draws_sample(draw_values)
    else:
        data_info["num_draws"] = "0"
        data_info["input_draws"] = "[]"

    # Seeds information
    if "random_seed" in dataframe.index.names:
        num_seeds = dataframe.index.get_level_values("random_seed").nunique()
        data_info["num_seeds"] = f"{num_seeds:,}"

    return data_info


def format_metadata_pandas(
    measure_key: str, test_info: dict[str, Any], reference_info: dict[str, Any]
) -> pd.DataFrame:
    """
    Format the comparison data as a pandas DataFrame

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
        DataFrame for display
    """
    # Start with the required keys in the specified order
    display_keys = list(REQUIRED_KEYS)

    # Get all unique keys from both dictionaries that aren't in REQUIRED_KEYS
    non_standard_keys = sorted(
        (set(test_info.keys()) | set(reference_info.keys())).difference(set(REQUIRED_KEYS))
    )

    # Add non-standard keys to the required keys list
    display_keys.extend(non_standard_keys)

    # Format the keys by replacing underscores with spaces and capitalizing each word
    properties = []
    for key in display_keys:
        formatted_key = " ".join(word.capitalize() for word in key.split("_"))
        properties.append(formatted_key)

    # Add measure key to both test and reference data dictionaries
    test_data = test_info.copy()
    test_data["measure_key"] = measure_key

    reference_data = reference_info.copy()
    reference_data["measure_key"] = measure_key

    # Create the rows for the DataFrame
    test_values = [test_data.get(key, "N/A") for key in display_keys]
    reference_values = [reference_data.get(key, "N/A") for key in display_keys]

    # Create the DataFrame
    return pd.DataFrame(
        {
            "Property": properties,
            "Test Data": test_values,
            "Reference Data": reference_values,
        }
    )


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
