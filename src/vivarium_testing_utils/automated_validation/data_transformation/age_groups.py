from __future__ import annotations

import re

import pandas as pd


class AgeGroup:
    """
    Class to represent an age group with start and end ages.
    """

    def __init__(self, name: str, start: int | float, end: int | float):
        self.name = name
        if start < 0:
            raise ValueError(f"Negative start age.")
        self.start = float(start)
        if end < 0:
            raise ValueError(f"Negative end age.")
        self.end = float(end)
        if end - start <= 0:
            raise ValueError("End age must be greater than start age.")
        self.span = float(end - start)

    def __eq__(self, other: object) -> bool:
        """True if two age groups have the same start and end ages."""
        if not isinstance(other, AgeGroup):
            return NotImplemented
        return (self.start, self.end) == (other.start, other.end)

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
    def from_string(cls, name: str) -> AgeGroup:
        """
        Parse age group names to extract start and end ages and their units.
        Supports formats like '0_to_6_months', '12_to_15_years', '0_to_8_days', '14_to_17'
        """
        # Extract numbers and unit from the group name
        pattern = r"(\d+(?:\.\d+)?)_to_(\d+(?:\.\d+)?)(?:_(\w+))?"
        match = re.match(pattern, name.lower())

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

    @classmethod
    def from_range(cls, start: float | int, end: float | int) -> AgeGroup:
        return cls(f"{start}_to_{end}", start, end)


class AgeSchema:
    """
    Class to represent a schema of age groups.
    """

    def __init__(self, age_groups: list[AgeGroup]) -> None:
        self.age_groups = age_groups
        self.age_groups.sort(key=lambda x: x.start)
        self._validate()
        self.range = (self.age_groups[0].start, self.age_groups[-1].end)
        self.span = self.range[1] - self.range[0]

    def __getitem__(self, index: int) -> AgeGroup:
        """Get an age group by index."""
        return self.age_groups[index]

    def __len__(self) -> int:
        """Get the number of age groups."""
        return len(self.age_groups)

    def __eq__(self, other: object) -> bool:
        """True if two schemas have the same age groups."""
        if not isinstance(other, AgeSchema):
            return NotImplemented
        if len(self.age_groups) != len(other.age_groups):
            return False
        for i in range(len(self.age_groups)):
            if self.age_groups[i] != other.age_groups[i]:
                return False
        return True

    def __contains__(self, item: AgeGroup) -> bool:
        """
        Check if an age group is contained in the schema.
        """
        return any(item == group for group in self.age_groups)

    def is_subset(self, other: AgeSchema) -> bool:
        """
        Check if this schema is a subset of another schema.
        """
        return all(group in other for group in self.age_groups)

    @classmethod
    def from_tuples(cls, age_tuples: list[tuple[str, int | float, int | float]]) -> AgeSchema:
        """group
        Create an AgeSchema from a dictionary of age groups.
        """
        age_groups = []
        for group_tuple in age_tuples:
            age_groups.append(AgeGroup(*group_tuple))
        return cls(age_groups)

    @classmethod
    def from_ranges(cls, age_ranges: list[tuple[int | float, int | float]]) -> AgeSchema:
        """
        Create an AgeSchema from a list of age ranges.
        """
        age_groups = []
        for start, end in age_ranges:
            age_groups.append(AgeGroup.from_range(start, end))
        return cls(age_groups)

    @classmethod
    def from_strings(cls, age_strings: list[str]) -> AgeSchema:
        """
        Create an AgeSchema from a list of age group names.
        """
        age_groups = []
        for name in age_strings:
            age_groups.append(AgeGroup.from_string(name))
        return cls(age_groups)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> AgeSchema:
        """
        Create an AgeSchema from a DataFrame with age group names.
        """
        has_names = "age_group" in df.index.names
        has_ranges = "age_start" in df.index.names and "age_end" in df.index.names
        if has_names and has_ranges:
            levels = ["age_group", "age_start", "age_end"]
            age_groups = list(
                df.index.droplevel(list(set(df.index.names) - set(levels)))
                .reorder_levels(levels)
                .unique()
            )

            return cls.from_tuples(age_groups)
        elif has_ranges:
            levels = ["age_start", "age_end"]
            age_groups = (
                df.index.droplevel(list(set(df.index.names) - set(levels)))
                .reorder_levels(levels)
                .unique()
            )
            return cls.from_ranges(age_groups)
        elif has_names:
            levels = ["age_group"]
            age_groups = list(
                df.index.droplevel(list(set(df.index.names) - set(levels))).unique()
            )
            return cls.from_strings(age_groups)
        else:
            raise ValueError(
                "DataFrame must have either 'age_group' or 'age_start' and 'age_end' index levels."
            )

    def _validate(self) -> None:
        """
        Validate the age groups to ensure they are non-overlapping and ordered.
        """
        if len(self.age_groups) == 0:
            raise ValueError("No age groups provided.")

        for i in range(len(self.age_groups) - 1):
            if self.age_groups[i].end > self.age_groups[i + 1].start:
                raise ValueError(
                    f"Overlapping age groups: {self.age_groups[i]} and {self.age_groups[i + 1]}"
                )
            if self.age_groups[i].end < self.age_groups[i + 1].start:
                raise ValueError(
                    f"Gap between consecutive age groups: {self.age_groups[i]} and {self.age_groups[i + 1]}"
                )

    def validate_compatible(self, other: AgeSchema) -> None:
        """
        Validate that two age schemas are compatible, that is, they span the same contiguous range.
        """
        if self.range != other.range:
            raise ValueError(
                f"Age schemas have different ranges: {self.range} and {other.range}"
            )

    def get_transform_matrix(self, other: AgeSchema) -> pd.DataFrame:
        """
        Get a converter mapping between this schema (target) and another schema (source).
        Returns a datafrme with the target age groups as rows and the source age groups as columns,
        with the values representing the fraction of the source age group that should go into the target age group.
        """
        source_age_groups = [group.name for group in other.age_groups]
        target_age_groups = [group.name for group in self.age_groups]

        transform_matrix = pd.DataFrame(
            0.0, index=target_age_groups, columns=source_age_groups
        )
        for target_group in self.age_groups:
            for source_group in other.age_groups:
                # Calculate what fraction of the source group should go into the target group
                fraction = source_group.fraction_contained_by(target_group)
                if fraction > 0:
                    transform_matrix.loc[target_group.name, source_group.name] = fraction
        return transform_matrix

    def rebin_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rebin a DataFrame to match the target age schema.

        Parameters:
        - df: DataFrame with multi-index including 'age_group' level
        - target_age_schema: AgeSchema instance for the target age groups

        Returns a new DataFrame with values redistributed to new age groups
        """
        source_age_schema = AgeSchema.from_dataframe(df)
        source_age_schema.validate_compatible(self)

        transform_matrix = self.get_transform_matrix(source_age_schema)

        original_index_names = list(df.index.names)

        all_results_series = []

        for val_col in df.columns.tolist():

            # Unstack the DataFrame to get the age groups as columns
            unstacked_series = (
                df[val_col]
                .unstack(level="age_group", fill_value=0)
                .reindex(columns=transform_matrix.columns, fill_value=0)
            )

            # Perform the dot product
            result_matrix_for_col = unstacked_series.dot(transform_matrix.T)

            # Name the column "age_group" for re-stacking
            result_matrix_for_col.columns.name = "age_group"

            # Stack the new age group columns into the index
            stacked_series_for_col = result_matrix_for_col.stack(
                level="age_group", dropna=False
            )
            stacked_series_for_col.name = val_col

            all_results_series.append(stacked_series_for_col)

        output_df = pd.concat(all_results_series, axis=1).reorder_levels(original_index_names)

        return output_df.sort_index()
