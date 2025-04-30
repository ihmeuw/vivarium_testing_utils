import pandas as pd
import pandera as pa
from pandera.typing import Index


class SingleNumericColumn(pa.DataFrameModel):
    """We restrict many intermediate dataframes to a single numeric column.

    This is a primitive DataFrameModel that checks for this criterion. We coerce all numeric types to float.
    It is inherited elsewhere.
    """

    # Columns
    value: float = pa.Field(coerce=True)

    # Config is inherited in child classes.
    # I'm not sure entirely how it works, but
    # it seems that the parent config settings will be
    # applied *before* the child config class is applied,
    # so overwriting Config won't make it to non-strict, for example.
    # However, the Config *class* of the child object won't have strict in it.
    class Config:
        strict = True


class SimOutputData(SingleNumericColumn):
    """The output of a simulation is a dataframe with a single numeric column and a multi-index."""

    # Required index levels
    # Index levels have to be in this order, but extra levels are allowed and can be between them.
    measure: Index[str]
    entity_type: Index[str]
    entity: Index[str]
    sub_entity: Index[str]


class DrawData(pa.DataFrameModel):
    """Draw Data from the Artifact has a large number of 'draw_' columns which must be pivoted."""

    # Columns
    draws: float = pa.Field(regex=True, alias=r"^draw_\d+", coerce=True)

    class Config:
        strict = True


class RatioData(pa.DataFrameModel):
    """Ratio data is a dataframe with two numeric columns.

    The columns will be numerator and denominator for a calculation into a final measure."""

    # Custom Checks
    @pa.dataframe_check
    def check_two_numeric_columns(cls, df: pd.DataFrame) -> bool:
        # Check that there are exactly two columns in the DataFrame,
        # and both are numeric
        numeric_columns = df.select_dtypes(include=["number"]).columns
        return len(numeric_columns) == df.shape[1] == 2
