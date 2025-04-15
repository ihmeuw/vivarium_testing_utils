import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series
from pandas.api.types import is_any_real_numeric_dtype


class SimOutputData(pa.DataFrameModel):
    # Required index levels
    measure: Index[str]
    entity_type: Index[str]
    entity: Index[str]
    sub_entity: Index[str]

    # Required and only data column
    value: Series

    @pa.check("value")
    def check_value(self, series: Series) -> Series:
        return is_any_real_numeric_dtype(series)

    class Config:
        strict = True  # Prevents extra columns beyond those defined here


class ArtifactData(pa.DataFrameModel):
    # A data schema for artifact data that enters the data loader
    pass


class CustomData(pa.DataFrameModel):
    # A data schema for custom data that enters the data loader
    pass


class RatioData(pa.DataFrameModel):
    # Data schema for ratio data, which can be assembled into a measure.
    pass


class MeasureData(pa.DataFrameModel):
    # Measure data is the final, single-column output correponding to a measure.
    pass
