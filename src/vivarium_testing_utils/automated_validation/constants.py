from __future__ import annotations

from enum import Enum

DRAW_PREFIX = "draw_"

DRAW_INDEX = "input_draw"
SEED_INDEX = "random_seed"


class DataSource(Enum):
    SIM = "sim"
    GBD = "gbd"
    ARTIFACT = "artifact"
    CUSTOM = "custom"

    @classmethod
    def from_str(cls, source: str) -> DataSource:
        try:
            return cls(source)
        except ValueError:
            raise ValueError(f"Source {source} not recognized. Must be one of {DataSource}")


LOCATION_ARTIFACT_KEY = "population.location"
POPULATION_STRUCTURE_ARTIFACT_KEY = "population.structure"

GBD_DATA_MAPPER = {
    "exposure": "exposure",
    "relative_risk": "relative_risks",
    "population.structure": "population_structure",
    "remission_rate": "remission_rates",
    "cause_specific_mortality_rate": "cause_specific_mortality_rates",
    "incidence_rate": "incidence",
    "prevalence": "incidence",
}
