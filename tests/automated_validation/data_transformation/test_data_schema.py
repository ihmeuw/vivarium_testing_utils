from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
)
from pandera.errors import SchemaError
import pytest


def test_sim_output(transition_count_data, person_time_data):
    # Test the SimOutputData schema
    sim_output_schema = SimOutputData
    sim_output_schema.validate(transition_count_data)
    sim_output_schema.validate(person_time_data)

    # Test that the schema raises an error for invalid data
    invalid_data = transition_count_data.copy()
    invalid_data["value"] = "invalid"
    with pytest.raises(SchemaError):
        sim_output_schema.validate(invalid_data)

    # Test that the schema raises an error for missing columns
    missing_column_data = transition_count_data.drop(columns=["value"])
    with pytest.raises(SchemaError):
        sim_output_schema.validate(missing_column_data)
