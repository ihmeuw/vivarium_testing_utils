import pandas as pd
import pytest
from pandera.errors import SchemaError

from vivarium_testing_utils.automated_validation.data_transformation.data_schema import (
    DrawData,
    RatioData,
    SimOutputData,
    SingleNumericColumn,
)


def test_single_numeric_value() -> None:
    """Test that the SingleNumericValue schema correctly validates a DataFrame with a single numeric column."""
    schema = SingleNumericColumn
    data = pd.DataFrame({"value": [1, 2, 3]}, index=pd.Index([0, 1, 2], name="index"))
    schema.validate(data)

    # Test that the schema raises an error for invalid data
    invalid_data = data.copy()
    invalid_data["value"] = "invalid"
    with pytest.raises(SchemaError):
        schema.validate(invalid_data)

    # Test that the schema raises an error for missing columns
    missing_column_data = data.drop(columns=["value"])
    with pytest.raises(SchemaError):
        schema.validate(missing_column_data)

    # Test that the schema raises an error for extra columns
    extra_column_data = data.copy()
    extra_column_data["extra_column"] = 0
    with pytest.raises(SchemaError):
        schema.validate(extra_column_data)


def test_sim_output(
    transition_count_data: pd.DataFrame, person_time_data: pd.DataFrame
) -> None:
    """
    Test that the SimOutputData schema correctly validates the transition count and person time data.
    We don't want it too permissive or too strict. Note that we are implicitly testing that extra index
    levels are OK, because the sample data has the "stratify_column" index level.
    """
    assert "stratify_column" in transition_count_data.index.names
    schema = SimOutputData

    # Test that the SimOutputData schema correctly validates the transition count and person time data
    schema.validate(transition_count_data)
    schema.validate(person_time_data)

    # Test that a missing index level raises an error
    for key in ["measure", "entity_type", "entity", "sub_entity"]:
        missing_index_data = transition_count_data.droplevel(key)
        with pytest.raises(SchemaError):
            schema.validate(missing_index_data)


def test_draw_data(raw_artifact_disease_incidence: pd.DataFrame) -> None:
    """
    Test that the DrawData schema correctly validates the artifact disease incidence data.
    """
    schema = DrawData

    # Test that the DrawData schema correctly validates the artifact disease incidence data
    schema.validate(raw_artifact_disease_incidence)

    # Test that a missing column does not raise an error
    missing_column_data = raw_artifact_disease_incidence.drop(columns=["draw_0"])
    schema.validate(missing_column_data)

    # Test that an extra column raises an error
    extra_column_data = raw_artifact_disease_incidence.copy()
    extra_column_data["extra_column"] = 0
    with pytest.raises(SchemaError):
        schema.validate(extra_column_data)

    # Test that the schema raises an error for invalid data
    invalid_data = raw_artifact_disease_incidence.copy()
    invalid_data["draw_0"] = "invalid"
    with pytest.raises(SchemaError):
        schema.validate(invalid_data)


def test_ratio_data() -> None:
    """
    Test that the RatioData schema correctly validates a DataFrame with two numeric columns.
    """
    schema = RatioData

    # Create a valid DataFrame
    data = pd.DataFrame({"numerator": [1, 2, 3], "denominator": [4, 5, 6]})
    schema.validate(data)

    # Test that the schema raises an error for invalid data
    invalid_data = data.copy()
    invalid_data["numerator"] = "invalid"
    with pytest.raises(SchemaError):
        schema.validate(invalid_data)

    # Test that the schema raises an error for missing columns
    missing_column_data = data.drop(columns=["numerator"])
    with pytest.raises(SchemaError):
        schema.validate(missing_column_data)

    # Test that the schema raises an error for extra columns
    extra_column_data = data.copy()
    extra_column_data["extra_column"] = 0
    with pytest.raises(SchemaError):
        schema.validate(extra_column_data)

    # Test that the schema raises an error for extra columns
    # even if it's not numeric
    extra_column_data = data.copy()
    extra_column_data["extra_column"] = "foo"
    with pytest.raises(SchemaError):
        schema.validate(extra_column_data)
