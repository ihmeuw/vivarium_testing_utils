import pandas as pd


def process_raw_data(
    input_data_type: str, raw_data: pd.DataFrame, measure: str
) -> pd.DataFrame:
    raise NotImplementedError


def compute_metric(
    input_data_type: str, intermediate_data: pd.DataFrame, measure: str
) -> pd.Series:
    raise NotImplementedError


def ratio(numerator, denominator):
    raise NotImplementedError


def aggregate(data: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    raise NotImplementedError


def linear_combination(coefficients: list[float], data: pd.DataFrame) -> pd.Series:
    raise NotImplementedError
