import pandera as pa
import pandas as pd
from pandera.typing import Index, DataFrame, Series


class SimOutputData(pa.DataFrameModel):
    # A data schema for raw simulation data that enters the data loader
    value: Series[float] = pa.Field(
        pa.Float,
        coerce=True,
        nullable=True,
    )
    pass


class ArtifactData(pa.DataFrameModel):
    # A data schema for artifact data that enters the data loader
    pass


class CustomData(pa.DataFrameModel):
    # A data schema for custom data that enters the data loader
    pass


class IntermediateData(pa.DataFrameModel):
    # Data schema for intermediate data that serves as basis of calculations
    pass


class ProcessedData(pa.DataFrameModel):
    # Schema of processed data that is the output of calculations
    pass
