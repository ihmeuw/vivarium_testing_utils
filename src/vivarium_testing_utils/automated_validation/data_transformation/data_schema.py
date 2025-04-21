from typing import Union

import pandera as pa
from pandera.typing import DataFrame, Index, Series


class SingleNumericColumn(pa.DataFrameModel):
    """We restrict many intermediate dataframes to a single numeric column.

    This is a primitive DataFrameModel that checks for this criterion. We coerce all numeric types to float.
    It is inherited elsewhere.
    """

    # Columns
    value: Series[float] = pa.Field(coerce=True)

    class Config:
        strict = True


class SimOutputData(SingleNumericColumn):
    """The output of a simulation is a dataframe with a single numeric column and a multi-index."""

    # Required index levels
    measure: Index[str]
    entity_type: Index[str]
    entity: Index[str]
    sub_entity: Index[str]


class DrawData(pa.DataFrameModel):
    """Draw Data from the Artifact has a large number of 'draw_' columns which must be pivoted."""

    # Columns
    draws: Series[float] = pa.Field(regex=True, alias=r"^draw_\d+", coerce=True)

    class Config:
        strict = True


RawArtifactData = Union[DataFrame[SingleNumericColumn], DataFrame[DrawData]]


class RatioData(pa.DataFrameModel):
    """Ratio data is a dataframe with two numeric columns.

    The columns will be numerator and denominator for a calculation into a final measure."""

    # Custom Checks
    @pa.dataframe_check
    def check_two_numeric_columns(cls, df: DataFrame) -> bool:
        # Check that there are exactly two numeric columns in the DataFrame
        numeric_columns = df.select_dtypes(include=["number"]).columns
        return len(numeric_columns) == 2
