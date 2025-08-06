from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation import utils
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SingleNumericColumn,
)


@dataclass
class RateAggregationWeights:
    weight_keys: dict[str, str]
    """Artifact keys for the weights to be aggregated."""
    formula: Callable[..., pd.DataFrame]
    """Function that will compute the aggregated weights"""
    description: str = ""
    """Human-readable description of the formula used to compute the weights."""

    @utils.check_io(out=SingleNumericColumn)
    def get_weights(self, *args, **kwargs) -> pd.DataFrame:
        return self.formula(*args, **kwargs)
