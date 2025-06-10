from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeSchema,
    format_dataframe,
)
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    SingleNumericColumn,
)
from vivarium_testing_utils.automated_validation.data_transformation.utils import (
    check_io,
    series_to_dataframe,
)

DRAW_PREFIX = "draw_"


def filter_data(data: pd.DataFrame, filter_cols: dict[str, list[str]]) -> pd.DataFrame:
    """Filter a DataFrame by the given index columns and values.

    The filter_cols argument
    should be a dictionary where the keys are column names and the values are lists of
    values to keep. If we filter to a single value, drop the column. If the dataframe is empty
    after filtering, raise an error."""
    for col, values in filter_cols.items():
        if len(values) == 1:
            data = data[data.index.get_level_values(col) == values[0]]
            data = data.droplevel([col])
        else:
            data = data[data.index.get_level_values(col).isin(values)]
    if data.empty:
        # TODO: Make sure we handle this case appropriately when we
        # want to automatically add many comparisons
        raise ValueError(
            f"DataFrame is empty after filtering by {filter_cols}. "
            f"Check that the filter values are valid."
        )

    return data


@check_io(
    numerator_data=SingleNumericColumn,
    denominator_data=SingleNumericColumn,
    out=SingleNumericColumn,
)
def ratio(numerator_data: pd.DataFrame, denominator_data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the ratio of two SingleNumericColumn DataFrames.

    Their indexes do not need to match, but must be interoperable.

    Parameters
    ----------
    numerator_data : pd.DataFrame
        SingleNumericColumn DataFrame to use as the numerator
    denominator_data : pd.DataFrame
        SingleNumericColumn DataFrame  to use as the denominator

    Returns
    -------
    pd.DataFrame
        SingleNumericColumn DataFrame containing the ratio values
    """
    zero_denominator = denominator_data["value"] == 0
    if zero_denominator.any():
        logger.warning(
            "Denominator has zero values. "
            "These will be put into the ratio dataframe as NaN."
        )
    ratio_values = numerator_data["value"] / denominator_data["value"]
    ratio_values[zero_denominator] = np.nan
    return series_to_dataframe(ratio_values)


def aggregate_sum(data: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Aggregate the dataframe over the specified index columns by summing."""
    if not groupby_cols:
        return data
    # Use observed=True to avoid sorting categorical levels
    # This is a hack, because we're not technically using pd.Categorical here.
    # TODO: MIC-6090  Use the right abstractions for categorical index columns.
    # You might need to keep this observed=True even after doing that.
    return data.groupby(groupby_cols, sort=False, observed=True).sum()


def stratify(data: pd.DataFrame, stratification_cols: list[str]) -> pd.DataFrame:
    """Stratify the data by the index columns, summing over everything else. Syntactic sugar for aggregate."""
    return aggregate_sum(data, stratification_cols)


def marginalize(data: pd.DataFrame, marginalize_cols: list[str]) -> pd.DataFrame:
    """Sum over marginalize columns, keeping the rest. Syntactic sugar for aggregate."""
    return aggregate_sum(data, [x for x in data.index.names if x not in marginalize_cols])


def linear_combination(
    data: pd.DataFrame, coeff_a: float, col_a: str, coeff_b: float, col_b: str
) -> pd.DataFrame:
    """Return a series that is the linear combination of two columns in a DataFrame."""
    return series_to_dataframe((data[col_a] * coeff_a) + (data[col_b] * coeff_b))


@check_io(out=SingleNumericColumn)
def clean_artifact_data(
    dataset_key: str,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the artifact data by dropping unnecessary columns and renaming the value column."""
    if data.columns.str.startswith(DRAW_PREFIX).all():
        data = _clean_artifact_draws(data)
    elif "value" not in data.columns:
        raise ValueError(f"Artifact {dataset_key} must have draw columns or a value column.")
    return data


@check_io(data=DrawData, out=SingleNumericColumn)
def _clean_artifact_draws(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the artifact data by dropping unnecessary columns and renaming the value column."""
    # Drop unnecessary columns
    # if data has value columns of format draw_1, draw_2, etc., drop the draw_ prefix
    # and melt the data into long format
    data = data.melt(
        var_name="input_draw",
        value_name="value",
        ignore_index=False,
    )
    data["input_draw"] = data["input_draw"].str.replace(DRAW_PREFIX, "", regex=False)
    data["input_draw"] = data["input_draw"].astype(int)
    data = data.set_index("input_draw", append=True).sort_index()
    return data


def resolve_age_groups(data: pd.DataFrame, age_groups: pd.DataFrame) -> pd.DataFrame:
    """Try to merge the age groups with the data. If it fails, just return the data."""
    context_age_schema = AgeSchema.from_dataframe(age_groups)
    try:
        return format_dataframe(context_age_schema, data)
    except ValueError:
        logger.info(
            "Could not resolve age groups. The DataFrame likely has no age data. Returning dataframe as-is."
        )
        return data


def get_singular_indices(data: pd.DataFrame) -> dict[str, Any]:
    """Get index levels and their values that are singular (i.e. have only one unique value)."""
    singular_metadata: dict[str, Any] = {}
    for index_level in data.index.names:
        if data.index.get_level_values(index_level).nunique() == 1:
            singular_metadata[index_level] = data.index.get_level_values(index_level)[0]
    return singular_metadata
