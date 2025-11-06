from __future__ import annotations

from typing import Any, Callable, TypeVar

import pandas as pd
import pandera as pa

F = TypeVar("F", bound=Callable[..., Any])


def check_io(**model_dict: type) -> Callable[[F], F]:
    """A wrapper for pa.check_io that automatically converts SchemaModels to schemas.

    Parameters
    ----------
    **model_dict
        Keyword arguments where keys are parameter names and values
        are SchemaModel classes or schema objects.

    Returns
    -------
        A decorator function that wraps the target function with pa.check_io.
    """
    # Convert any SchemaModel classes to schemas
    schema_dict = {}
    for key, value in model_dict.items():
        # Check if it's a SchemaModel class (not instance) and has to_schema method
        if hasattr(value, "to_schema") and callable(value.to_schema):
            schema_dict[key] = value.to_schema()
        else:
            # If it's already a schema or something else, use it as is
            schema_dict[key] = value

    # Return the decorator using pa.check_io with the converted schemas
    return pa.check_io(**schema_dict)


# TODO: Remove this function and references when we can support Series schemas
# more easily
def series_to_dataframe(series: pd.Series[float]) -> pd.DataFrame:
    """Convert a Series to a DataFrame with the Series values as a single column."""
    return series.to_frame(name="value")


def format_custom_gbd_data(raw_gbd: pd.DataFrame) -> pd.DataFrame:
    """Format the output of a get_draws call to have expect index and value columns."""

    sort_order = ["location_id", "sex_id", "age_group_id", "year_id"]
    sorted_data_index = [n for n in sort_order if n in raw_gbd.index.names]
    sorted_data_index.extend([n for n in raw_gbd.index.names if n not in sorted_data_index])

    if isinstance(raw_gbd.index, pd.MultiIndex):
        raw_gbd = raw_gbd.reorder_levels(sorted_data_index)
    raw_gbd = raw_gbd.sort_index()
    return raw_gbd
