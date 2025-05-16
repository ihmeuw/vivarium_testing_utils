from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource

REQUIRED_KEYS = ["measure_key", "source", "index_columns", "size", "num_draws", "input_draws"]


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
        data_info["input_draws"] = dataframe.index.get_level_values("input_draw").unique()
    else:
        data_info["num_draws"] = 0
    if "random_seed" in dataframe.index.names:
        data_info["num_seeds"] = dataframe.index.get_level_values("random_seed").nunique()
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
    display_keys = list(REQUIRED_KEYS)
    # Get all unique keys from both dictionaries
    non_standard_keys = sorted(
        (set(test_info.keys()) | set(reference_info.keys())).difference(set(REQUIRED_KEYS))
    )

    # Add non-standard keys to the required keys list
    display_keys.extend(non_standard_keys)

    # Format the keys by replacing underscores with spaces and capitalizing each word
    properties = []
    for key in display_keys:
        # Replace underscores with spaces and capitalize each word
        formatted_key = " ".join(word.capitalize() for word in key.split("_"))
        properties.append(formatted_key)

    # Create the DataFrame
    return pd.DataFrame(
        {
            "Property": properties,
            "Test Data": _get_display_formatting(measure_key, test_info, display_keys),
            "Reference Data": _get_display_formatting(
                measure_key, reference_info, display_keys
            ),
        }
    )


def _get_display_formatting(
    measure_key: str, data_info: dict[str, Any], display_keys: list[str]
) -> list[str]:
    """Helper function to format the data information for display dynamically based on available keys.

    Parameters:
    -----------
    measure_key
        The key of the measure being compared
    data_info
        Information about the data to be displayed
    display_keys
        List of keys to display in the specified order

    Returns:
    --------
        A list of strings containing the formatted data information
    """
    result = []

    for key in display_keys:
        if key == "measure_key":
            # Special case for measure_key
            result.append(measure_key)
        elif key == "input_draws":
            # Format draw sample
            result.append(_format_draws_sample(data_info.get("input_draws", [])))
        elif key == "size":
            # Format size as "rows × columns"
            size = data_info.get("size", (0, 0))
            result.append(f"{size[0]:,} rows × {size[1]:,} columns")
        elif key == "index_columns":
            # Format index columns as comma-separated string
            index_cols = data_info.get("index_columns", [])
            result.append(", ".join(str(col) for col in index_cols))
        elif key == "num_draws":
            # Format number with comma separators
            num = data_info.get(key, 0)
            result.append(f"{num:,}")
        else:
            # Default formatting for other keys
            result.append(str(data_info.get(key, "N/A")))

    return result


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
