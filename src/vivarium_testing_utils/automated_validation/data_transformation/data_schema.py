from __future__ import annotations
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


def check_io(**model_dict):
    """
    A wrapper for pa.check_io that automatically converts SchemaModels to schemas.

    Args:
        **model_dict: Keyword arguments where keys are parameter names and values
                     are SchemaModel classes or schema objects.

    Returns:
        A decorator function that wraps the target function with pa.check_io.
    """
    # Convert any SchemaModel classes to schemas
    schema_dict = {}
    for key, value in model_dict.items():
        # Check if it's a SchemaModel class (not instance) and has to_schema method
        if hasattr(value, "to_schema") and callable(value.to_schema):
            schema_dict[key] = value.to_schema()
        else:
            # If it's already a schema or something else, use it as is
            schema_dict[key] = value

    # Return the decorator using pa.check_io with the converted schemas
    return pa.check_io(**schema_dict)


# TODO: Remove this function and references when we can support Series schemas
# more easily
def series_to_dataframe(series: pd.Series[float]) -> pd.DataFrame:
    """Convert a Series to a DataFrame with the Series values as a single column."""
    return series.to_frame(name="value")
