from __future__ import annotations

from typing import TypeVar

import pandas as pd
import pandera as pa

from vivarium_testing_utils.automated_validation.data_transformation.age_groups import (
    AgeGroup,
    AgeSchema,
    rebin_dataframe,
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


def align_indexes(datasets: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Put each dataframe on a common index by choosing the intersection of index columns
    and marginalizing over the rest."""
    # Get the common index columns
    common_index = list(set.intersection(*(set(data.index.names) for data in datasets)))

    # Marginalize over the rest
    return [stratify(data, common_index) for data in datasets]


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


def ratio(data: pd.DataFrame, numerator: str, denominator: str) -> pd.DataFrame:
    """Return a series of the ratio of two columns in a DataFrame,
    where the columns are specified by their names."""
    return series_to_dataframe(data[numerator] / data[denominator])


def aggregate_sum(data: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Aggregate the dataframe over the specified index columns by summing."""
    if not groupby_cols:
        return data
    return data.groupby(groupby_cols).sum()


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
        var_name="draw",
        value_name="value",
        ignore_index=False,
    )
    data["draw"] = data["draw"].str.replace(DRAW_PREFIX, "", regex=False)
    data["draw"] = data["draw"].astype(int)
    data = data.set_index("draw", append=True).sort_index()
    return data


def resolve_age_groups(data: pd.DataFrame, age_bins: pd.DataFrame) -> pd.DataFrame:
    """Try to merge the age groups with the data. If it fails, just return the data."""
    try:
        data_age_schema = AgeSchema.from_dataframe(data)
    except ValueError:
        # if the data doesn't have any age information, just return it
        return data
    context_age_schema = AgeSchema.from_dataframe(age_bins)
    if data_age_schema.is_subset(context_age_schema):
        if "age_group" in data.index.names:
            data = data.droplevel("age_group")
        return pd.merge(data, age_bins, left_index=True, right_index=True)

    else:
        return rebin_dataframe(data, context_age_schema)


def align_datasets(
    test_data: pd.DataFrame, ref_data: pd.DataFrame, age_bins: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align the test and reference datasets on the same index."""
    test_data = resolve_age_groups(test_data, age_bins)
    ref_data = resolve_age_groups(ref_data, age_bins)
    test_data, ref_data = align_indexes([test_data, ref_data])
    return test_data, ref_data
