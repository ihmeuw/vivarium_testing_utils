from __future__ import annotations
import pandas as pd
import re
import numpy as np
from collections import defaultdict
import warnings


class AgeGroup:
    """
    Class to represent an age group with start and end ages.
    """

    def __init__(self, name, start: int, end: int):
        self.name = name
        self.start = start
        self.end = end
        self.span = end - start
        if self.span < 0:
            raise ValueError("End age must be greater than or equal to start age.")

    def __repr__(self):
        return f"AgeGroup(start={self.start}, end={self.end})"

    def fraction_contained_by(self, other: AgeGroup) -> float:
        """
        Return the amount of this group that is contained within another group.
        """
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap <= 0:
            return 0.0
        return overlap / self.span

    @classmethod
    def from_string(cls, name: str):
        """
        Parse age bucket names to extract start and end ages and their units.
        Supports formats like '0_to_6_months', '12_to_15_years', '0_to_8_days', '14_to_17'
        """
        # Extract numbers and unit from the bucket name
        pattern = r"(\d+(?:\.\d+)?)_to_(\d+(?:\.\d+)?)(?:_(\w+))?"
        match = re.match(pattern, name)

        if not match:
            raise ValueError(f"Invalid age group name format: {name}")

        start, end, unit = match.groups()
        start, end = float(start), float(end)

        # Default to years if unit is not specified
        if unit is None:
            unit = "years"

        # Convert all to months for consistent comparison
        if unit == "days":
            start_years = start / 365  # Approximate
            end_years = end / 365
        elif unit == "months":
            start_years = start / 12
            end_years = end / 12
        elif unit == "years":
            start_years = start
            end_years = end
        else:
            raise ValueError(f"Invalid unit: {unit}. Must be 'days', 'months', or 'years'.")

        return cls(name, start_years, end_years)


class AgeSchema:
    """
    Class to represent a schema of age buckets.
    """

    def __init__(self, age_buckets: list[AgeGroup]):
        self.age_buckets = age_buckets
        self.age_buckets.sort(key=lambda x: x.start)
        self._validate()
        self.range = (self.age_buckets[0].start, self.age_buckets[-1].end)
        self.span = self.range[1] - self.range[0]

    def _validate(self):
        """
        Validate the age buckets to ensure they are non-overlapping and ordered.
        """
        for i in range(len(self.age_buckets) - 1):
            if self.age_buckets[i].end > self.age_buckets[i + 1].start:
                raise ValueError(
                    f"Overlapping age buckets: {self.age_buckets[i]} and {self.age_buckets[i + 1]}"
                )
        for bucket in self.age_buckets:
            if bucket.start < 0:
                raise ValueError(f"Negative start age in bucket: {bucket}")
            if bucket.end < 0:
                raise ValueError(f"Negative end age in bucket: {bucket}")
            if bucket.start > bucket.end:
                raise ValueError(f"Start age greater than end age in bucket: {bucket}")
        if len(self.age_buckets) == 0:
            raise ValueError("No age buckets provided.")

    def validate_compatible(self, other: AgeSchema):
        """
        Validate that two age schemas are compatible.
        """
        if abs(self.range[0] - other.range[0]) > 1 / 12:
            raise ValueError(
                f"Age schemas have different ranges: {self.range} and {other.range}"
            )

    def get_converter(self, other: AgeSchema):
        """
        Get a converter mapping between this schema (target) and another schema (source).
        Returns a list of tuples (target_idx, source_idx, fraction) where fraction is
        the proportion of the source bucket that should go into the target bucket.
        """
        converter = []
        for i, target_bucket in enumerate(self.age_buckets):
            for j, source_bucket in enumerate(other.age_buckets):
                # Calculate what fraction of the source bucket should go into the target bucket
                fraction = source_bucket.fraction_contained_by(target_bucket)
                if fraction > 0:
                    converter.append((i, j, fraction))
        return converter

    @classmethod
    def from_dict(cls, age_buckets: dict[str, tuple[int, int]]):
        """
        Create an AgeSchema from a dictionary of age buckets.
        """
        age_groups = []
        for name, (start, end) in age_buckets.items():
            age_groups.append(AgeGroup(name, start, end))
        return cls(age_groups)

    @classmethod
    def from_strings(cls, age_buckets: list[str]):
        """
        Create an AgeSchema from a list of age bucket names.
        """
        age_groups = []
        for name in age_buckets:
            age_groups.append(AgeGroup.from_string(name))
        return cls(age_groups)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """
        Create an AgeSchema from a DataFrame with age bucket names.
        """
        age_groups = []
        for name in df.index.get_level_values("age_group").unique():
            age_groups.append(AgeGroup.from_string(name))
        return cls(age_groups)


def reformat_artifact_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat a DataFrame with age_start and age_end index levels to have a single age_group level.
    """
    # Add a new index level to the multi-index DataFrame for age_group
    df["age_group"] = (
        df.index.get_level_values("age_start").astype(str)
        + "_to_"
        + df.index.get_level_values("age_end").astype(str)
    )
    df = df.set_index("age_group", append=True)
    # Drop the old index levels
    df = df.droplevel(["age_start", "age_end"])

    return df


def rebin_dataframe(
    df: pd.DataFrame,
    target_age_schema: AgeSchema,
):
    """
    Rebin a DataFrame to match the target age schema.

    Parameters:
    - df: DataFrame with multi-index including 'age_group' level
    - target_age_schema: AgeSchema instance for the target age buckets

    Returns a new DataFrame with values redistributed to new age buckets
    """
    source_age_schema = AgeSchema.from_dataframe(df)
    source_age_schema.validate_compatible(target_age_schema)

    # Get the mapping between source and target schemas
    converter = target_age_schema.get_converter(source_age_schema)

    # Reset index to work with age_group as a column
    df_reset = df.reset_index()

    # Identify index columns (all multi-index levels except age_group)
    idx_cols = [
        col for col in df_reset.columns if col in df.index.names and col != "age_group"
    ]

    # Identify value columns (all non-index columns)
    value_cols = [
        col for col in df_reset.columns if col not in df.index.names and col != "age_group"
    ]

    # Group by all index columns except age_group
    groupby_cols = idx_cols.copy()
    grouped = df_reset.groupby(groupby_cols)

    # Create a list to hold new rows
    result_rows = []

    # Process each group
    for group_key, group_df in grouped:
        # Convert group_key to a tuple for consistent handling
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # Get all the unique source bucket names in this group
        source_buckets = group_df["age_group"].unique()

        # For each target age bucket, calculate the contributed values
        for target_idx, target_bucket in enumerate(target_age_schema.age_buckets):
            new_row = {}

            # Set index columns
            for i, col in enumerate(groupby_cols):
                new_row[col] = group_key[i]

            # Set the target age bucket name
            new_row["age_group"] = target_bucket.name

            # Initialize value columns
            for col in value_cols:
                new_row[col] = 0.0

            # Find all converter entries for this target bucket
            for t_idx, s_idx, fraction in converter:
                if t_idx == target_idx:
                    # Get the corresponding source bucket
                    source_bucket = source_age_schema.age_buckets[s_idx]

                    # Find the source row in the group dataframe
                    source_rows = group_df[group_df["age_group"] == source_bucket.name]

                    if not source_rows.empty:
                        # Get the source values
                        source_row = source_rows.iloc[0]

                        # Add proportional values for each value column
                        for col in value_cols:
                            new_row[col] += source_row[col] * fraction

            # Only add the row if at least one value column has a non-zero value
            if any(new_row[col] != 0.0 for col in value_cols):
                result_rows.append(new_row)

    # Create new dataframe from the results
    result_df = pd.DataFrame(result_rows)

    # Check if we actually got data
    if result_df.empty:
        warnings.warn(
            "No data was generated. Check that the schemas are compatible and the converter is working correctly."
        )
        return pd.DataFrame()

    # Set multi-index back
    index_cols = groupby_cols + ["age_group"]
    result_df = result_df.set_index(index_cols)

    return result_df
