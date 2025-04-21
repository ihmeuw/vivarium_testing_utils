from typing import TypeVar, Union

import pandas as pd
from pandera.typing import DataFrame

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    SingleNumericColumn,
)

DataSet = TypeVar("DataSet", pd.DataFrame, pd.Series)

DRAW_PREFIX = "draw_"


def align_indexes(datasets: list[DataSet]) -> list[DataSet]:
    """Put each dataframe on a common index by choosing the intersection of index columns
    and marginalizing over the rest."""
    # Get the common index columns
    common_index = set.intersection(*(set(data.index.names) for data in datasets))

    # Marginalize over the rest
    return [marginalize(data, common_index) for data in datasets]


def filter_data(data: DataSet, filter_cols: dict[str, list]) -> DataSet:
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


def ratio(data: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    """Return a series of the ratio of two columns in a DataFrame,
    where the columns are specified by their names."""
    return data[numerator] / data[denominator]


def aggregate_sum(data: DataSet, groupby_cols: list[str]) -> DataSet:
    """Aggregate the dataframe over the specified index columns by summing."""
    if not groupby_cols:
        return data
    return data.groupby(groupby_cols).sum()


def stratify(data: DataSet, stratification_cols: list[str]) -> DataSet:
    """Stratify the data by the index columns, summing over everything else. Syntactic sugar for aggregate."""
    return aggregate_sum(data, stratification_cols)


def marginalize(data: DataSet, marginalize_cols: list[str]) -> DataSet:
    """Sum over marginalize columns, keeping the rest. Syntactic sugar for aggregate."""
    return aggregate_sum(data, data.index.names.difference(marginalize_cols))


def linear_combination(
    data: pd.DataFrame, coeff_a: float, col_a: str, coeff_b: float, col_b: str
) -> pd.Series:
    """Return a series that is the linear combination of two columns in a DataFrame."""
    return (data[col_a] * coeff_a) + (data[col_b] * coeff_b)


@pa.check_types
def clean_artifact_data(
    dataset_key: str,
    data: Union[DataFrame[SingleNumericColumn], DataFrame[DrawData]],
) -> DataFrame[SingleNumericColumn]:
    """Clean the artifact data by dropping unnecessary columns and renaming the value column."""
    # Drop unnecessary columns
    # if data has value columns of format draw_1, draw_2, etc., drop the draw_ prefix
    # and melt the data into long format
    if data.columns.str.startswith(DRAW_PREFIX).all():
        data = data.melt(
            var_name="draw",
            value_name="value",
            ignore_index=False,
        )
        data["draw"] = data["draw"].str.replace(DRAW_PREFIX, "", regex=False)
        data["draw"] = data["draw"].astype(int)
        data = data.set_index("draw", append=True).sort_index()
    elif "value" not in data.columns:
        raise ValueError(f"Artifact {dataset_key} must have draw columns or a value column.")
    return data
