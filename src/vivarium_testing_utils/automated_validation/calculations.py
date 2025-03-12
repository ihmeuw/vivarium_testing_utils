import pandas as pd


def process_raw_data(
    input_data_type: str, raw_data: pd.DataFrame, measure: str
) -> pd.DataFrame:
    pass


def compute_metric(
    input_data_type: str, intermediate_data: pd.DataFrame, measure: str
) -> pd.Series:
    pass


def ratio(numerator, denominator):
    pass


def aggregate(data: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    pass


def linear_combination(coefficients: list[float], data: pd.DataFrame) -> pd.Series:
    pass
