from pandera.typing import DataFrame, Series
import pandas as pd
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    RatioData,
    SimOutputData,
    SingleNumericColumn,
)
from typing import Union

PDDataSet = pd.DataFrame | pd.Series[float]
SimDataSet = Union[DataFrame[SimOutputData], Series[float]]
RawArtifactDataSet = DataFrame[SingleNumericColumn] | Series[float] | DataFrame[DrawData]
RawDataSet = SimDataSet | RawArtifactDataSet
