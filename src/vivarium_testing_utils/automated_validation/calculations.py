from typing import TypeVar

import pandas as pd

DataSet = TypeVar("DataSet", pd.DataFrame, pd.Series)


def process_raw_data(
    input_data_type: str, raw_data: pd.DataFrame, measure: str
) -> pd.DataFrame:
    raise NotImplementedError


def compute_metric(
    input_data_type: str, intermediate_data: pd.DataFrame, measure: str
) -> pd.Series:
    raise NotImplementedError


def validate_intermediate_data(intermediate_data: DataSet) -> DataSet:
    """Ensure the intermediate data has an appropriate format for further calculations."""
    # raise if any of the columns are not numeric
    if not intermediate_data.applymap(lambda x: isinstance(x, (int, float))).all().all():
        raise ValueError("All value columns must be numeric")


def ratio(data: pd.DataFrame, numerator, denominator) -> pd.Series:
    """Return a series of the ratio of two columns in a DataFrame,
    where the columns are specified by their names."""
    return data[numerator] / data[denominator]


def aggregate_sum(data: DataSet, groupby_cols: list[str]) -> DataSet:
    """Aggregate the dataframe over the specified index columns by summing."""
    return data.groupby(groupby_cols).sum()


def stratify(data: DataSet, stratification_cols: list[str]) -> DataSet:
    """Stratify the data by the index columns, summing over everything else. Syntactic sugar for aggregate."""
    return aggregate_sum(data, stratification_cols)


def marginalize(data: DataSet, marginalize_cols: list[str]) -> DataSet:
    """Sum over marginalize columns, keeping the rest. Syntactic sugar for aggregate."""
    return data.groupby(data.index.names.difference(marginalize_cols)).sum()


def linear_combination(
    data: pd.DataFrame, coeff_a: float, col_a: str, coeff_b: float, col_b: str
) -> pd.Series:
    """Return a series that is the linear combination of two columns in a DataFrame."""
    return (data[col_a] * coeff_a) + (data[col_b] * coeff_b)
