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


def drop_extra_columns(raw_gbd: pd.DataFrame, data_key: str) -> pd.DataFrame:
    """Format the output of a get_draws call to have expect index and value columns."""

    value_cols = [col for col in raw_gbd.columns if "draw" in col]
    # Population structure only has "population"
    # Data should only have a "value" column when cached. Draws get converted when cached
    if data_key == "population.structure":
        # Population structure has only "population" column. Rename it
        raw_gbd = raw_gbd.rename(columns={"population": "value"})
        if "value" in raw_gbd.columns:
            value_cols = ["value"]
    if not value_cols:
        raise ValueError(
            f"No value columns found in the data. Columns found: {raw_gbd.columns.tolist()}"
        )

    gbd_cols = ["location_id", "sex_id", "age_group_id", "year_id", "cause_id"]
    measure = data_key.split(".")[-1]
    if measure in ["exposure", "relative_risk"]:
        gbd_cols.append("parameter")
    columns_to_keep = [col for col in raw_gbd.columns if col in gbd_cols + value_cols]
    return raw_gbd[columns_to_keep]


def set_gbd_index(data: pd.DataFrame, data_key: str) -> pd.DataFrame:
    """Set the index of a GBD DataFrame based on the data key."""
    measure = data_key.split(".")[-1]
    gbd_cols = ["location_id", "sex_id", "age_group_id", "year_id"]
    if measure in ["exposure", "relative_risk"]:
        gbd_cols.append("parameter")
    if measure != "relative_risk" and "cause_id" in data.columns:
        data = data.drop(columns=["cause_id"])

    index_cols = [col for col in gbd_cols if col in data.columns]

    formatted = data.set_index(index_cols)
    return formatted


def set_validation_index(data: pd.DataFrame) -> pd.DataFrame:
    """Set the index of cached validation data to expected columns."""
    # Data should only have a "value" column when cached. Draws get converted when cached
    extra_columns = [col for col in data.columns if "draw" not in col]
    if not extra_columns:
        extra_columns = [col for col in data.columns if col != "value"]

    # Preserve existing index order and add extra columns
    sorted_data_index = [n for n in data.index.names]
    sorted_data_index.extend([col for col in extra_columns if col not in sorted_data_index])

    # Reset index to convert existing index to columns, then set new index
    data = data.reset_index()
    data = data.set_index(sorted_data_index)

    return data
