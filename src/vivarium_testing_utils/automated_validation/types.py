from typing import Union

from pandera.typing import DataFrame

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    SimOutputData,
    SingleNumericColumn,
)

SimDataSet = DataFrame[SimOutputData]
RawArtifactDataSet = DataFrame[SingleNumericColumn] | DataFrame[DrawData]
RawDataSet = SimDataSet | RawArtifactDataSet
