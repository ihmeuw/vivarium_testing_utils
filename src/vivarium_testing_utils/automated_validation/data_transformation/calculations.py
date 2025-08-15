from __future__ import annotations

from typing import Any, Collection, Iterable, Mapping

import numpy as np
import pandas as pd
from loguru import logger

from vivarium_testing_utils.automated_validation.constants import DRAW_INDEX, DRAW_PREFIX
from vivarium_testing_utils.automated_validation.data_transformation import utils
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    SingleNumericColumn,
)


def filter_data(
    data: pd.DataFrame, filter_cols: Mapping[str, str | list[str]], drop_singles: bool = True
) -> pd.DataFrame:
    """Filter a DataFrame by the given index columns and values.

    The filter_cols argument
    should be a dictionary where the keys are column names and the values are lists of
    values to keep. If we filter to a single value, drop the column. If the dataframe is empty
    after filtering, raise an error."""
    for col, values in filter_cols.items():
        if isinstance(values, str):
            values = [values]
        if len(values) == 1:
            data = data[data.index.get_level_values(col) == values[0]]
            if drop_singles:
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


@utils.check_io(
    numerator_data=SingleNumericColumn,
    denominator_data=SingleNumericColumn,
    out=SingleNumericColumn,
)
def ratio(numerator_data: pd.DataFrame, denominator_data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with the ratio of two SingleNumericColumn DataFrames.

    Their indexes do not need to match, but must be interoperable.

    Parameters
    ----------
    numerator_data
        SingleNumericColumn DataFrame to use as the numerator
    denominator_data
        SingleNumericColumn DataFrame  to use as the denominator

    Returns
    -------
        SingleNumericColumn DataFrame containing the ratio values
    """
    zero_denominator = denominator_data["value"] == 0
    if zero_denominator.any():
        logger.warning(
            "Denominator has zero values. "
            "These will be put into the ratio dataframe as NaN."
        )
    denominator_data[zero_denominator] = np.nan
    return numerator_data / denominator_data


def aggregate_sum(data: pd.DataFrame, groupby_cols: Collection[str] = []) -> pd.DataFrame:
    """Aggregate the dataframe over the specified index columns by summing."""
    if not isinstance(groupby_cols, list):
        groupby_cols = list(groupby_cols)
    if not groupby_cols:
        return data
    # Use observed=True to avoid sorting categorical levels
    # This is a hack, because we're not technically using pd.Categorical here.
    # TODO: MIC-6090  Use the right abstractions for categorical index columns.
    # You might need to keep this observed=True even after doing that.
    return data.groupby(groupby_cols, sort=False, observed=True).sum()


def stratify(data: pd.DataFrame, stratification_cols: Collection[str]) -> pd.DataFrame:
    """Stratify the data by the index columns, summing over everything else. Syntactic sugar for aggregate."""
    return aggregate_sum(data, stratification_cols)


def marginalize(data: pd.DataFrame, marginalize_cols: Collection[str]) -> pd.DataFrame:
    """Sum over marginalize columns, keeping the rest. Syntactic sugar for aggregate."""
    return aggregate_sum(data, [x for x in data.index.names if x not in marginalize_cols])


def linear_combination(
    data: pd.DataFrame, coeff_a: float, col_a: str, coeff_b: float, col_b: str
) -> pd.DataFrame:
    """Return a series that is the linear combination of two columns in a DataFrame."""
    return utils.series_to_dataframe((data[col_a] * coeff_a) + (data[col_b] * coeff_b))


@utils.check_io(data=DrawData, out=SingleNumericColumn)
def clean_artifact_draws(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the artifact data by dropping unnecessary columns and renaming the value column."""
    # Drop unnecessary columns
    # if data has value columns of format draw_1, draw_2, etc., drop the draw_ prefix
    # and melt the data into long format
    data = data.melt(
        var_name=DRAW_INDEX,
        value_name="value",
        ignore_index=False,
    )
    data[DRAW_INDEX] = data[DRAW_INDEX].str.replace(DRAW_PREFIX, "", regex=False)
    data[DRAW_INDEX] = data[DRAW_INDEX].astype(int)
    data = data.set_index(DRAW_INDEX, append=True).sort_index()
    return data


def get_singular_indices(data: pd.DataFrame) -> dict[str, Any]:
    """Get index levels and their values that are singular (i.e. have only one unique value)."""
    singular_metadata: dict[str, Any] = {}
    for index_level in data.index.names:
        if data.index.get_level_values(index_level).nunique() == 1:
            singular_metadata[index_level] = data.index.get_level_values(index_level)[0]
    return singular_metadata


@utils.check_io(
    data=SingleNumericColumn,
    weights=SingleNumericColumn,
)
def weighted_average(
    data: pd.DataFrame,
    weights: pd.DataFrame,
    stratifications: list[str] = [],
) -> pd.DataFrame | float:
    """Calculate a weighted average of the data using the provided weights.

    Parameters
    ----------
    data
        DataFrame with the values to average. Must have a 'value' column.
    weights
        DataFrame with the weights to apply to the values in data. Must have a 'value' column.
    stratifications
        List of index level names to use for stratification/grouping.
    Raises
        ------
        ValueError
            If data index levels is not a subset of weights index levels.

    Returns
    -------
        Pandas DataFrame with the weighted average values for each stratification group.

    Examples
    --------

    >>> fish_data = pd.DataFrame(
    ...     {
    ...         "weights": [20, 100, 2, 50],
    ...         "value": [2, 3, 5, 7],
    ...     },
    ...     index=pd.MultiIndex.from_tuples([
    ...         ("Male", "Red"),
    ...         ("Male", "Blue"),
    ...         ("Female", "Red"),
    ...         ("Female", "Blue"),
    ...     ], names=["sex", "color"])
    ... )
    >>> data = pd.DataFrame({"value": fish_data["value"]}, index=fish_data.index)
    >>> weights = pd.DataFrame({"value": fish_data["weights"]}, index=fish_data.index)

    # Weighted average by sex:
    >>> weighted_average(data, weights, ["sex"])
    # Returns:
    #         value
    # sex
    # Male     2.83  # (20*2 + 100*3)/(20+100) = 340/120 ≈ 2.83
    # Female   6.92  # (2*5 + 50*7)/(2+50) = 360/52 ≈ 6.92

    # Weighted average by color:
    >>> weighted_average(data, weights, ["color"])
    # Returns:
    #        value
    # color
    # Red     2.27  # (20*2 + 2*5)/(20+2) = 50/22 ≈ 2.27
    # Blue    4.33  # (100*3 + 50*7)/(100+50) = 650/150 ≈ 4.33

    # Overall weighted average (no stratification):
    >>> weighted_average(data, weights, [])
    # Returns: 3.55  # (20*2 + 100*3 + 2*5 + 50*7)/(20+100+2+50) = 700/172 ≈ 4.07

    """

    # Check if weights has extra index levels compared to data
    data_index_names = set(data.index.names)
    weights_index_names = set(weights.index.names)

    if not data_index_names.issubset(weights_index_names):
        raise ValueError(
            f"Data index levels {data_index_names - weights_index_names} "
            f"are not present in weights index levels {weights_index_names}"
        )

    # If weights has extra index levels, aggregate by summing
    extra_levels = weights_index_names - data_index_names
    if extra_levels:
        # Group by the levels that match data's index and sum over the extra levels
        weights = weights.groupby(level=data.index.names, sort=False, observed=True).sum()

    # Check that index levels are compatible (at least subsets of each other)
    if not data.index.equals(weights.index):
        raise ValueError(
            "Data and weights must have the same index levels. "
            f"Data index: {data.index.names}, Weights index: {weights.index.names}"
        )

    if not stratifications:
        # Return a single float value instead of a one row pandas series
        return float(((data.mul(weights).sum()) / weights.sum()).item())

    numerator = (
        data.mul(weights).groupby(level=stratifications, sort=False, observed=True).sum()
    )
    denominator = weights.groupby(level=stratifications, sort=False, observed=True).sum()
    weighted_avg = numerator / denominator

    return weighted_avg
