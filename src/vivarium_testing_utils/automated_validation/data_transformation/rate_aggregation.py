from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from vivarium_testing_utils.automated_validation.data_transformation import utils
from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SingleNumericColumn,
)


@dataclass
class RateAggregationWeights:
    weight_keys: dict[str, str]  # Dataset keys needed
    formula: Callable[..., pd.DataFrame]  # Combines weights
    description: str = ""  # Human-readable description of the aggregation

    @utils.check_io(out=SingleNumericColumn)
    def get_weights(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.formula(*args, **kwargs)


POPULATION_WEIGHTED = RateAggregationWeights(
    weight_keys={"population": "population.structure"},
    formula=lambda population: population,
    description="Population-weighted average",
)
