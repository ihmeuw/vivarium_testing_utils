import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series


class SimOutputData(pa.DataFrameModel):
    # A data schema for raw simulation data that enters the data loader
    pass


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
