from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    SimOutputData,
)
from pandera.errors import SchemaError
import pytest


def test_sim_output(transition_count_data, person_time_data):
    """
    Test that the SimOutputData schema correctly validates the transition count and person time data.
    We don't want it too permissive or too strict. Note that we are implicitly testing that extra index
    levels are OK, because the sample data has the "stratify_column" index level.
    """
    assert "stratify_column" in transition_count_data.index.names

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

    # Test that the schema raises an error for extra columns
    extra_column_data = transition_count_data.copy()
    extra_column_data["extra_column"] = 0
    with pytest.raises(SchemaError):
        sim_output_schema.validate(extra_column_data)

    # Test that a missing index level raises an error
    missing_index_level_data = transition_count_data.reset_index()
    missing_index_level_data = missing_index_level_data.drop(columns=["measure"])
    with pytest.raises(SchemaError):
        sim_output_schema.validate(missing_index_level_data)
