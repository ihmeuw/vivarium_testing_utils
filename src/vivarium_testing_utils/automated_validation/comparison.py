from abc import ABC, abstractmethod
from typing import Collection, Literal

import pandas as pd
from vivarium_testing_utils.automated_validation.data_loader import DataSource
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    Measure,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    stratify,
    align_indexes,
)


class Comparison(ABC):
    """A Comparison is the basic testing unit to compare two datasets, a "test" dataset and a
    "reference" dataset. The test dataset is the one that is being validated, while the reference
    dataset is the one that is used as a benchmark. The comparison operates on a *measure* of the two datasets,
    typically a derived quantity of the test data such as incidence rate or prevalence."""

    measure: Measure
    test_source: DataSource
    test_data: pd.DataFrame
    reference_source: DataSource
    reference_data: pd.DataFrame
    stratifications: list[str]

    @property
    @abstractmethod
    def metadata(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def verify(self, stratifications: Collection[str] = ()):
        pass

    @abstractmethod
    def get_frame(
        self,
        stratifications: Collection[str] = (),
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "percent_error",
        ascending: bool = False,
    ) -> pd.DataFrame:
        pass


class FuzzyComparison(Comparison):
    """A FuzzyComparison is a comparison that requires statistical hypothesis testing
    to determine if the distributions of the datasets are the same. We require both the numerator and
    denominator for the test data, to be able to calculate the statistical power."""
    def __init__(
        self,
        measure: RatioMeasure,
        test_source: DataSource,
        test_data: pd.DataFrame,
        reference_source: DataSource,
        reference_data: pd.DataFrame,
        stratifications: Collection[str] = (),
    ):
        self.measure = measure
        self.test_source = test_source
        self.test_data = test_data
        self.reference_source = reference_source
        self.reference_data = reference_data.rename(columns={"value": "reference_rate"})
        self.stratifications = stratifications

    @property
    def metadata(self) -> pd.DataFrame:
        """A summary of the test data and reference data, including:
        - the measure key
        - source
        - index columns
        - size
        - number of draws
        - a sample of the input draws.
        """
        measure_key = self.measure.measure_key
        test_info = self._data_info(self.test_source, self.test_data)
        reference_info = self._data_info(self.reference_source, self.reference_data)
        return _format_metadata_pandas(measure_key, test_info, reference_info)

    def verify(self, stratifications: Collection[str] = ()):
        raise NotImplementedError

    def get_frame(
        self,
        stratifications: list[str],
        num_rows: int | Literal["all"] = 10,
        sort_by: str = "percent_error",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Get a DataFrame of the comparison data, with naive comparison of the test and reference.

        Parameters:
        -----------
        stratifications
            The stratifications to use for the comparison
        num_rows
            The number of rows to return. If "all", return all rows.
        sort_by
            The column to sort by. Default is "percent_error".
        ascending
            Whether to sort in ascending order. Default is False.
        Returns:
        --------
        A DataFrame of the comparison data.
        """
        converted_test_data = self.measure.get_measure_data_from_ratio(self.test_data).rename(
            columns={"value": "test_rate"}
        )
        stratified_test_data = stratify(converted_test_data, stratifications, agg="mean")
        stratified_reference_data = stratify(self.reference_data, stratifications, agg="mean")

        merged_data = pd.concat([stratified_test_data, stratified_reference_data], axis=1)
        merged_data["percent_error"] = (
            (merged_data["test_rate"] - merged_data["reference_rate"])
            / merged_data["reference_rate"]
        ) * 100
        sorted_data = merged_data.sort_values(
            by=sort_by,
            ascending=ascending,
        )
        if num_rows == "all":
            return sorted_data
        else:
            return sorted_data.head(n=num_rows)

    def _data_info(self, source: DataSource, dataframe: pd.DataFrame) -> dict[str, str]:
        """Organize the data information into a dictionary for display by a styled pandas DataFrame.

        Parameters:
        -----------
        source
            The source of the data (i.e. sim, artifact, or gbd)
        dataframe
            The DataFrame containing the data to be displayed
        Returns:
        --------
        A dictionary containing the data information.

        """
        data_info: dict[str, str] = {}
        data_info["source"] = source.value
        data_info["index_columns"] = dataframe.index.names
        data_info["size"] = dataframe.shape
        if "input_draw" in dataframe.index.names:
            data_info["num_draws"] = dataframe.index.get_level_values("input_draw").nunique()
            data_info["input_draw"] = dataframe.index.get_level_values("input_draw").unique()
        else:
            data_info["num_draws"] = 0
        if source == DataSource.SIM:
            data_info["num_seeds"] = dataframe.index.get_level_values("random_seed").nunique()
        return data_info


def _format_metadata_pandas(
    measure_key: str, test_info: dict[str, str], reference_info: dict[str, str]
):
    """
    Format the comparison data as a styled pandas DataFrame

    Parameters:
    -----------
    measure_key
        The key of the measure being compared
    test_info
        Information about the test data to be displayed
    reference_info
        Information about the reference data to be displayed
    Returns:
    --------
        Styled DataFrame for display
    """
    # Extract necessary data

    def get_display_formatting(data_info):
        source = data_info.get("source", "Unknown")
        size = data_info.get("size", (0, 0))
        num_draws = data_info.get("num_draws", 0)
        index_cols = data_info.get("index_columns", [])

        return [
            measure_key,
            source,
            ", ".join(str(col) for col in index_cols),
            f"{size[0]:,} rows × {size[1]:,} columns",
            f"{num_draws:,}",
            _format_draws_sample(data_info.get("input_draw", [])),
        ]

    # Create data for summary table
    data = {
        "Property": [
            "Measure Key",
            "Source",
            "Index Columns",
            "Size",
            "Number of Draws",
            "Draw Sample",
        ],
        "Test Data": get_display_formatting(test_info),
        "Reference Data": get_display_formatting(reference_info),
    }

    # Create and style DataFrame
    df = pd.DataFrame(data)

    # Apply styling
    styled_df = df.style.set_properties(
        **{"text-align": "left", "padding": "10px", "border": "1px solid #dddddd"}
    )

    # Add title as caption
    styled_df = styled_df.set_caption("Comparison Summary").set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-size", "16px"),
                    ("font-weight", "bold"),
                ],
            }
        ]
    )

    # Color headers
    styled_df = styled_df.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1E1E1E"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "left"),
                    ("padding", "10px"),
                ],
            }
        ]
    )

    # Alternate row colors
    styled_df = styled_df.apply(
        lambda x: ["background-color: #E6F0FF" if i % 2 == 0 else "" for i in range(len(x))],
        axis=0,
    )

    return styled_df


def _format_draws_sample(draw_index, max_display=5) -> str:
    """Helper function to format draw samples for display.

    Parameters:
    -----------
    draw_index
        The index of the draws to be formatted
    max_display
        The maximum number of draws to display. If the number of draws exceeds this, the
        function will display the first and last max_display draws, separated by ellipses.
    Returns:
    --------
        A string representation of the draws sample.
    """
    if hasattr(draw_index, "__iter__"):
        # Convert to list if it's any iterable
        draw_list = list(draw_index)

        if len(draw_list) <= max_display * 2:
            return str(draw_list)
        else:
            first = draw_list[:max_display]
            last = draw_list[-max_display:]
            return f"{first} ... {last}"
    else:
        return str(draw_index)
