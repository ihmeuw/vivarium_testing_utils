from __future__ import annotations

import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd


class AgeGroup:
    """
    Class to represent an age group with start and end ages.
    """

    def __init__(self, name, start: int | float, end: int | float):
        self.name = name
        self.start = start
        self.end = end
        self.span = end - start
        if self.span < 0:
            raise ValueError("End age must be greater than or equal to start age.")

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
        converter = defaultdict(dict)
        for target_bucket in self.age_buckets:
            for source_bucket in other.age_buckets:
                # Calculate what fraction of the source bucket should go into the target bucket
                fraction = source_bucket.fraction_contained_by(target_bucket)
                if fraction > 0:
                    converter[target_bucket.name][source_bucket.name] = fraction
        return converter

    @classmethod
    def from_dict(cls, age_buckets: dict[str, tuple[int | float, int | float]]):
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

    converter = target_age_schema.get_converter(source_age_schema)
    # Get new categories from mapping
    new_ages = list(converter.keys())

    # Create transformation matrix
    old_ages = df.index.get_level_values("age_group").unique()
    transform_matrix = pd.DataFrame(0, index=new_ages, columns=old_ages)

    # Fill transformation matrix with weights
    for new_age, weights in converter.items():
        for old_age, weight in weights.items():
            transform_matrix.loc[new_age, old_age] = weight

    # Get index structure
    idx_names = df.index.names
    other_levels = [name for name in idx_names if name != "age_group"]

    # Get other index levels
    other_idx_values = [df.index.get_level_values(name).unique() for name in other_levels]

    # Create cartesian product of other index values
    other_combos = pd.MultiIndex.from_product(other_idx_values, names=other_levels)

    # Create new index for result
    new_idx_components = [[new_ages[i] for i in range(len(new_ages))]]
    new_idx_components.extend(
        [df.index.get_level_values(name).unique() for name in other_levels]
    )
    new_idx_names = ["age_group"] + other_levels

    new_idx = pd.MultiIndex.from_product(new_idx_components, names=new_idx_names)

    # Initialize result DataFrame
    result = pd.DataFrame(0.0, index=new_idx, columns=df.columns)

    # Process each combination of other indices
    for combo in other_combos:
        # Create mask for this combination
        mask = True
        for i, level_name in enumerate(other_levels):
            if isinstance(combo, tuple):
                val = combo[i]
            else:
                val = combo  # Single value
            mask = mask & (df.index.get_level_values(level_name) == val)

        # Extract subset with this combination
        subset = df[mask]

        # Create values array (age_groups x columns)
        values = np.zeros((len(old_ages), len(df.columns)))

        # Fill values from subset
        for i, age_group in enumerate(old_ages):
            # Find matching indices with this age_group
            age_mask = subset.index.get_level_values("age_group") == age_group
            if age_mask.any():
                # If there are matches, sum the values (in case of duplicates)
                values[i, :] = subset[age_mask].sum().values

        # Apply transformation
        result_values = np.dot(transform_matrix, values)

        # Update result DataFrame
        for i, new_age in enumerate(new_ages):
            # Create new index tuple
            if isinstance(combo, tuple):
                new_idx_tuple = (new_age,) + combo
            else:
                new_idx_tuple = (new_age, combo)

            # Assign transformed values
            result.loc[new_idx_tuple, :] = result_values[i, :]

    return result.reorder_levels(df.index.names)
