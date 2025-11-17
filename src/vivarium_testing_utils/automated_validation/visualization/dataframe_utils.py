from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.data_loader import DataSource

REQUIRED_KEYS = ("measure_key", "source", "index_columns", "size", "num_draws", "input_draws")


def format_metadata(
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
    data = pd.DataFrame(
        {
            "Property": properties,
            "Test Data": test_values,
            "Reference Data": reference_values,
        }
    ).set_index(["Property"])
    # Wrap strings to multiple lines to display all information
    data["Test Data"] = data["Test Data"].str.wrap(25)
    data["Reference Data"] = data["Reference Data"].str.wrap(25)
    return data


def format_draws_sample(draw_list: list[int], source: DataSource) -> str:
    """Helper function to format draw samples for display.

    Parameters:
    -----------
    draw_list
        The list of draws to be formatted
    source
        The data source of the draws. One of the DataSource enum values.
    max_display
        The maximum number of draws to display. If the number of draws exceeds this, the
        function will display the first and last max_display draws, separated by ellipses.

    Returns:
    --------
        A string representation of the draws sample.
    """
    draw_list = sorted(draw_list)
    if source == DataSource.SIM:
        # Display all draws run in the simulation
        return str(draw_list)
    elif source in [DataSource.GBD, DataSource.ARTIFACT]:
        if not draw_list:
            return "range()"
        # Display range of draws for GBD and Artifact data
        first = draw_list[0]
        last = draw_list[-1]
        return f"range({first}-{last})"
    else:
        raise ValueError(f"Source {source} not recognized. Must be a valid DataSource")
