from typing import Literal, Optional, Union

import pandas as pd
import pandera as pa
from pandas.api.types import is_any_real_numeric_dtype
from pandera.typing import DataFrame, Index, Series


class SingleNumericValue(pa.DataFrameModel):
    """We restrict many intermediate dataframes to a single numeric column.

    This is a primitive DataFrameModel that checks for this criterion. It is inherited elsewhere.
    """

    # Columns
    value: Series

    @pa.check("value")
    def check_value(cls, series: Series) -> bool:
        return is_any_real_numeric_dtype(series)

    class Config:
        strict = True


class SimOutputData(SingleNumericValue):
    """The output of a simulation is a dataframe with a single numeric column and a multi-index."""

    # Required index levels
    measure: Index[str]
    entity_type: Index[str]
    entity: Index[str]
    sub_entity: Index[str]


class DrawData(pa.DataFrameModel):
    """Draw Data from the Artifact has a large number of 'draw_' columns which must be pivoted."""

    # Columns
    draws: Series = pa.Field(regex=True, alias=r"^draw_\d+")

    @pa.dataframe_check
    def check_draw_columns(cls, df: DataFrame) -> bool:
        # Check that all columns are numeric
        numeric_columns = df.select_dtypes(include=["number"]).columns
        # Check that all columns start with "draw_"
        draw_columns = df.columns[df.columns.str.startswith("draw_")]
        # Check that the number of draw columns is equal to the number of numeric columns
        return len(numeric_columns) == len(draw_columns) == len(df.columns)

    class Config:
        strict = True


RawArtifactData = Union[DataFrame[SingleNumericValue], DataFrame[DrawData]]


class RatioData(pa.DataFrameModel):
    """Ratio data is simulation data that has undergone one processing step to yield a dataframe with two numeric columns.

    The columns will be numerator and denominator for a calculation into a final measure."""

    # Custom Checks
    @pa.dataframe_check
    def check_two_numeric_columns(cls, df: DataFrame) -> bool:
        # Check that there are exactly two numeric columns in the DataFrame
        numeric_columns = df.select_dtypes(include=["number"]).columns
        return len(numeric_columns) == 2
