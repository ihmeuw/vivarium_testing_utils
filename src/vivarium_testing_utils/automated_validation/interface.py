from __future__ import annotations

import base64
import io
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Collection, Literal, Mapping, cast

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import HTML, display
from loguru import logger
from matplotlib.figure import Figure
from vivarium_inputs import utilities as vi

from vivarium_testing_utils.automated_validation.bundle import RatioMeasureDataBundle
from vivarium_testing_utils.automated_validation.comparison import Comparison, FuzzyComparison
from vivarium_testing_utils.automated_validation.constants import DAYS_PER_YEAR
from vivarium_testing_utils.automated_validation.data_loader import DataLoader, DataSource
from vivarium_testing_utils.automated_validation.data_transformation import report
from vivarium_testing_utils.automated_validation.data_transformation.calculations import (
    filter_data,
)
from vivarium_testing_utils.automated_validation.data_transformation.measures import (
    CategoricalRelativeRisk,
    Measure,
    MeasureMapper,
    RatioMeasure,
)
from vivarium_testing_utils.automated_validation.data_transformation.utils import (
    add_comparison_metadata_levels,
    drop_extra_columns,
    get_measure_index_names,
    set_gbd_index,
    set_validation_index,
)
from vivarium_testing_utils.automated_validation.results import VerificationResults
from vivarium_testing_utils.automated_validation.visualization import plot_utils
from vivarium_testing_utils.fuzzy_checker import TestResult


class ValidationContext:
    @property
    def verifications(self) -> VerificationResults:
        """Get the verification results dynamically from comparisons."""
        results = VerificationResults()
        for comparison_dict in self.comparisons.values():
            for comparison in comparison_dict.values():
                # Check if comparison has been verified by checking for test results
                if (
                    not hasattr(comparison, "proportion_test_results")
                    or not comparison.proportion_test_results
                ):
                    continue

                # Categorize based on test results
                _, _, result = self._gather_comparison_test_results(comparison)
                source_key = f"{comparison.test_bundle.source.name.lower()}_{comparison.reference_bundle.source.name.lower()}"

                if not any(result):
                    results.passing[comparison.comparison_key][source_key] = comparison
                else:
                    results.failing[comparison.comparison_key][source_key] = comparison
        return results

    def __init__(self, results_dir: str | Path, scenario_columns: Collection[str] = ()):
        self.results_dir = Path(results_dir)
        self.data_loader = DataLoader(self.results_dir)
        self.comparisons: defaultdict[str, defaultdict[str, Comparison]] = defaultdict(
            lambda: defaultdict(Comparison)
        )
        self.age_groups = self._get_age_groups()
        self.scenario_columns = scenario_columns
        self.location = self.data_loader.location
        self.measure_mapper = MeasureMapper()
        self.model_spec = self.data_loader.model_spec.configuration

    def get_sim_outputs(self) -> list[str]:
        """Get a list of the datasets available in the given simulation output directory."""
        return sorted(self.data_loader.get_sim_outputs())

    def get_artifact_keys(self) -> list[str]:
        """Get a list of the artifact keys available to compare against."""
        return sorted(self.data_loader.get_artifact_keys())

    def get_raw_data(self, data_key: str, source: str) -> Any:
        """Return a copy of the data for manual inspection."""
        return self.data_loader.get_data(data_key, DataSource.from_str(source))

    def upload_custom_data(
        self, data_key: str, data: pd.DataFrame | pd.Series[float]
    ) -> None:
        """Upload a custom DataFrame or Series to the context given by a data key."""
        if isinstance(data, pd.Series):
            data = data.to_frame(name="value")
        self.data_loader.upload_custom_data(data_key, data)

    def add_comparison(
        self,
        measure_key: str,
        test_source: str,
        ref_source: str,
        test_scenarios: dict[str, str] = {},
        ref_scenarios: dict[str, str] = {},
    ) -> None:
        """Add a comparison to the context given a measure key and data sources."""
        if measure_key.endswith(".relative_risk"):
            raise ValueError(
                f"For relative risk measures, use 'add_relative_risk_comparison' instead. "
                f"Got measure_key='{measure_key}'"
            )

        measure = self.measure_mapper.get_measure_from_key(
            measure_key, list(self.scenario_columns)
        )
        self._add_comparison_with_measure(
            measure,
            test_source,
            ref_source,
            test_scenarios,
            ref_scenarios,
        )

    def add_relative_risk_comparison(
        self,
        risk_factor: str,
        affected_entity: str,
        affected_measure: str,
        test_source: str,
        ref_source: str,
        test_scenarios: dict[str, str] = {},
        ref_scenarios: dict[str, str] = {},
        risk_stratification_column: str | None = None,
        risk_state_mapping: dict[str, str] | None = None,
    ) -> None:
        """Add a relative risk comparison to the context.

        Parameters
        ----------
        risk_factor
            The risk factor name (e.g., 'child_stunting')
        affected_entity
            The entity affected by the risk factor (e.g., 'cause.diarrheal_diseases')
        affected_measure
            The measure to calculate (e.g., 'excess_mortality_rate', 'incidence_rate')
        risk_stratification_column
            The column to use for stratifying the risk factor in simulation data (e.g., 'risk_factor')
        test_source
            Source for test data ('sim', 'artifact', or 'custom')
        ref_source
            Source for reference data ('sim', 'artifact', or 'custom')
        test_scenarios
            Dictionary of scenario filters for test data
        ref_scenarios
            Dictionary of scenario filters for reference data
        """

        measure = CategoricalRelativeRisk(
            risk_factor,
            affected_entity,
            affected_measure,
            risk_stratification_column,
            risk_state_mapping,
        )
        self._add_comparison_with_measure(
            measure, test_source, ref_source, test_scenarios, ref_scenarios
        )

    def _add_comparison_with_measure(
        self,
        measure: Measure,
        test_source: str,
        ref_source: str,
        test_scenarios: dict[str, str] = {},
        ref_scenarios: dict[str, str] = {},
    ) -> None:
        """Internal method to add a comparison with a pre-constructed measure."""

        test_source_enum = DataSource.from_str(test_source)
        ref_source_enum = DataSource.from_str(ref_source)

        # Check if the measure is a RatioMeasure for FuzzyComparison
        if not isinstance(measure, RatioMeasure):
            raise NotImplementedError(
                f"Measure {measure.measure_key} is not a RatioMeasure. Only RatioMeasures are currently supported for comparisons."
            )

        for source, scenarios in (
            (test_source_enum, test_scenarios),
            (ref_source_enum, ref_scenarios),
        ):
            if source == DataSource.SIM and set(scenarios.keys()) != set(
                self.scenario_columns
            ):
                raise ValueError(
                    f"Each simulation comparison subject must choose a specific scenario. "
                    f"You are missing scenarios for: {set(self.scenario_columns) - set(scenarios.keys())}."
                )

        test_data_bundle = RatioMeasureDataBundle(
            measure=measure,
            source=test_source_enum,
            data_loader=self.data_loader,
            age_group_df=self.age_groups,
            scenarios=test_scenarios,
        )
        ref_data_bundle = RatioMeasureDataBundle(
            measure=measure,
            source=ref_source_enum,
            data_loader=self.data_loader,
            age_group_df=self.age_groups,
            scenarios=ref_scenarios,
        )

        comparison = FuzzyComparison(
            test_bundle=test_data_bundle, reference_bundle=ref_data_bundle
        )
        self.comparisons[measure.measure_key][
            f"{test_source_enum.name.lower()}_{ref_source_enum.name.lower()}"
        ] = comparison

    def verify(
        self,
        comparison: str,
        stratifications: Collection[str] | Literal["all"] = "all",
        test_source: str = "sim",
        ref_source: str = "artifact",
    ) -> bool:
        """Verify a single comparison, log the result, and return True if successful, False otherwise.

        Parameters
        ----------
        comparison
            The key of the comparison to verify.
        stratifications
            The stratifications to use for the comparison. Default is "all".
        test_source
            The source of the test data (e.g., 'sim', 'artifact', 'custom'). Default is "sim".
        ref_source
            The source of the reference data (e.g., 'sim', 'artifact', 'custom'). Default is "artifact".

        Returns
        -------
            True if the comparison passes validation, False otherwise.
        """
        return self._verify(
            self.comparisons[comparison][f"{test_source}_{ref_source}"], stratifications
        )

    def _verify(
        self,
        comparison: Comparison,
        stratifications: Collection[str] | Literal["all"] = "all",
    ) -> bool:
        if "step_size" in self.model_spec.time:
            step_size = self.model_spec.time.step_size / DAYS_PER_YEAR
        else:
            step_size = None
            logger.warning(
                "Step size is not defined in the model specification. This may result "
                "in inaccurate verification results."
            )

        comparison.verify(step_size, stratifications)
        overall_result, stratified_results, result = self._gather_comparison_test_results(
            comparison
        )
        if not any(result):
            logger.info(f"Comparison {comparison.comparison_key} passed!")
            return True
        else:
            logger.warning(f"Comparison {comparison.comparison_key} failed.")
            if overall_result.reject_null:
                logger.warning(f"Overall comparison for {comparison.comparison_key} failed.")
            # stratified_results is dict[str, dict[str, TestResult]]
            for group_dict in stratified_results.values():
                for group in group_dict.values():
                    if group.reject_null:
                        logger.warning(f"Group {group.name}_{group.name_additional} failed.")
            return False

    def metadata(
        self, comparison_key: str, test_source: str, ref_source: str
    ) -> pd.DataFrame:
        """Get the metadata for a given comparison and specified sources.

        Parameters
        ----------
        comparison_key
            The key of the comparison for which to get the metadata
        test_source
            The source of the test data (e.g., 'sim', 'artifact', 'custom')
        ref_source
            The source of the reference data (e.g., 'sim', 'artifact', 'custom')

        Returns
        -------
            A DataFrame containing the metadata for the comparison in tabular format.
        """
        comparison_metadata = self.comparisons[comparison_key][
            f"{test_source}_{ref_source}"
        ].metadata
        directory_metadata = self._get_directory_metadata()

        data = pd.concat([comparison_metadata, directory_metadata])
        # Display draw values on multiple lines if necessary
        display_df = data.copy()
        display_df["Test Data"] = display_df["Test Data"].str.wrap(30, break_long_words=False)
        display_df["Reference Data"] = display_df["Reference Data"].str.wrap(
            30, break_long_words=False
        )

        display(HTML(display_df.to_html().replace("\\n", "<br>")))  # type: ignore[no-untyped-call]
        return data

    def _get_directory_metadata(self) -> pd.DataFrame:
        """Add model run metadata to the dictionary."""
        sim_run_time = self.results_dir.name
        sim_dt = datetime.strptime(sim_run_time, "%Y_%m_%d_%H_%M_%S").strftime(
            "%b %d %H:%M %Y"
        )
        artifact_run_time = self._get_artifact_creation_time()
        directory_metadata = pd.DataFrame(
            {
                "Property": ["Run Time"],
                "Test Data": [sim_dt],
                "Reference Data": [artifact_run_time],
            }
        )

        return directory_metadata.set_index("Property")

    def _get_artifact_creation_time(self) -> str:
        """Get the artifact creation time from the artifact file."""
        artifact_path = Path(
            yaml.safe_load((self.results_dir / "model_specification.yaml").open("r"))[
                "configuration"
            ]["input_data"]["artifact_path"]
        )
        os_time = os.path.getmtime(artifact_path)
        artifact_time = datetime.fromtimestamp(os_time).strftime("%b %d %H:%M %Y")

        return artifact_time

    def get_frame(
        self,
        comparison_key: str,
        test_source: str,
        ref_source: str,
        stratifications: Collection[str] | Literal["all"] = "all",
        num_rows: int | Literal["all"] = "all",
        sort_by: str = "",
        filters: Mapping[str, str | list[str]] | None = None,
        ascending: bool = False,
        aggregate_draws: bool = False,
    ) -> pd.DataFrame:
        """Get a DataFrame of the comparison data, with naive comparison of the test and reference.

        Parameters:
        -----------
        comparison_key
            The key of the comparison for which to get the data
        test_source
            The source of the test data (e.g., 'sim', 'artifact', 'custom')
        ref_source
            The source of the reference data (e.g., 'sim', 'artifact', 'custom')
        stratifications
            The stratifications to use for the comparison. If "all", no aggregation will happen and
            all existing stratifications will remain. If an empty list is passed, no stratifications
            will be retained.
        num_rows
            The number of rows to return. If "all", return all rows.
        filters
            A mapping of index levels to filter values. Only rows matching the filter will be included.
        sort_by
            The column to sort by. Default is "percent_error" for non-aggregated data, and no sorting for aggregated data.
        ascending
            Whether to sort in ascending order. Default is False.
        aggregate_draws
            If True, aggregate over draws to show means and 95% uncertainty intervals.

        Returns:
        --------
            A DataFrame of the comparison data.
        """
        if not aggregate_draws and not sort_by:
            sort_by = "percent_error"

        if (isinstance(num_rows, int) and num_rows > 0) or num_rows == "all":
            data = self.comparisons[comparison_key][f"{test_source}_{ref_source}"].get_frame(
                stratifications, num_rows, sort_by, ascending, aggregate_draws
            )
            data = self.format_ui_data_index(data, comparison_key)
            return (
                filter_data(data, filters, drop_singles=False)
                if filters is not None
                else data
            )
        else:
            raise ValueError("num_rows must be a positive integer or literal 'all'")

    def plot_comparison(
        self,
        comparison_key: str,
        test_source: str,
        ref_source: str,
        type: str,
        condition: dict[str, Any] = {},
        stratifications: Collection[str] | Literal["all"] = "all",
        **kwargs: Any,
    ) -> Figure | list[Figure]:
        """Create a plot for the given comparison.

        Parameters
        ----------
        comparison_key
            The comparison object to plot.
        test_source
            The source of the test data (e.g., 'sim', 'artifact', 'custom')
        ref_source
            The source of the reference data (e.g., 'sim', 'artifact', 'custom')
        type
            Type of plot to create.
        condition
            Conditions to filter the data by, by default {}
        stratifications
            Stratifications to retain in the plotted dataset, by default "all"
        **kwargs
            Additional keyword arguments for specific plot types.

        Returns
        -------
            The generated figure or list of figures.
        """
        return plot_utils.plot_comparison(
            self.comparisons[comparison_key][f"{test_source}_{ref_source}"],
            type,
            condition,
            stratifications,
            **kwargs,
        )

    def generate_comparisons(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def verify_all(self) -> bool:
        """Verify all comparisons in the context and capture results.

        Returns
        -------
            True if all comparisons pass validation, False otherwise.

        """
        for comparison_dict in self.comparisons.values():
            for comparison in comparison_dict.values():
                # TODO: MIC-6840 - Infer set of stratifications to iterate through with verify
                self._verify(comparison)

        # Return True if no failing results (property dynamically computes results)
        return not any(self.verifications.failing.values())

    def _gather_comparison_test_results(
        self, comparison: Comparison
    ) -> tuple[TestResult, dict[tuple[str, ...], dict[str, TestResult]], list[bool]]:
        overall_result = cast(TestResult, comparison.proportion_test_results["overall"])
        stratified_results = cast(
            dict[tuple[str, ...], dict[str, TestResult]],
            comparison.proportion_test_results["stratified"],
        )
        # Collect all reject_nulls from the nested structure
        reject_nulls = [
            test_result.reject_null
            for group in stratified_results.values()
            for test_result in group.values()
        ]
        result = [overall_result.reject_null] + reject_nulls
        return overall_result, stratified_results, result

    def plot_all(self, type: str, **kwargs: Any) -> dict[tuple[str, str, str], list[Figure]]:
        """Plot all comparisons and return dict of (measure_key, test_source, ref_source) -> list[Figure]."""
        figures_dict: dict[tuple[str, str, str], list[Figure]] = {}
        for comparison_dict in self.comparisons.values():
            for comparison in comparison_dict.values():
                test_source = comparison.test_bundle.source.name.lower()
                ref_source = comparison.reference_bundle.source.name.lower()
                fig = self.plot_comparison(
                    comparison.comparison_key,
                    test_source,
                    ref_source,
                    type,
                    **kwargs,
                )
                figs = fig if isinstance(fig, list) else [fig]
                figures_dict[(comparison.comparison_key, test_source, ref_source)] = figs
        return figures_dict

    def _figures_to_base64_dict(
        self, figures_dict: dict[tuple[str, str, str], list[Figure]]
    ) -> dict[tuple[str, str, str], list[str]]:
        """Convert a dict of (measure_key, test_source, ref_source) -> list[Figure] to base64 images."""
        plot_images: dict[tuple[str, str, str], list[str]] = {}
        for key, figs in figures_dict.items():
            images: list[str] = []
            for figure in figs:
                buf = io.BytesIO()
                figure.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                images.append(img_b64)
                buf.close()
                # Handle both matplotlib Figure and seaborn FacetGrid
                underlying_fig = getattr(figure, "fig", figure)
                plt.close(underlying_fig)
            plot_images[key] = images
        return plot_images

    def generate_results(
        self,
        output_path: str | Path | None = None,
        plot_type: str = "line",
        display_in_notebook: bool = True,
        **kwargs: Any,
    ) -> None:
        """Generate an HTML report of validation results.

        This method runs verification for all comparisons, generates plots,
        and creates an interactive HTML report with tabbed navigation.

        Parameters
        ----------
        output_path
            Optional path to save the HTML report. If None, returns HTML string only.
        plot_type
            Type of plots to generate for comparisons (default: "line").
            Options: "line", "bar", "box", "heatmap"
        display_in_notebook
            If True (default), automatically displays the report in Jupyter notebooks.
            Set to False if you only want the HTML string returned without display.
        kwargs
            Additional keyword arguments to pass to the plotting function.

        Examples
        --------
        >>> # Auto-display in Jupyter (default behavior)
        >>> context = ValidationContext(results_dir="path/to/results")
        >>> context.add_comparison("cause.diarrhea.incidence_rate", "sim", "artifact")
        >>> context.generate_results()  # Automatically displays!

        >>> # Save to file and display
        >>> context.generate_results(output_path="report.html")

        """
        # Generate all plots as Figures using plot_all
        figures_dict = self.plot_all(plot_type, **kwargs)
        plot_images = self._figures_to_base64_dict(figures_dict)
        html_content = self._generate_report(plot_images=plot_images)

        if output_path is not None:
            saved_path = report.save_html_report(html_content, Path(output_path))
            logger.info(f"Report saved to: {saved_path}")

        # Auto-display in Jupyter notebooks if requested
        if display_in_notebook:
            try:
                # Check if we're in a Jupyter environment
                get_ipython()  # type: ignore[name-defined]
                display(HTML(html_content))  # type: ignore
            except NameError:
                # Not in Jupyter, skip display
                logger.info("Not in a Jupyter environment, cannot display the report.")
                pass

    def _generate_report(
        self, plot_images: dict[tuple[str, str, str], list[str]] | None = None
    ) -> str:
        """Generate HTML report content from validation results.

        This is the core report generation logic that collects data from
        comparisons and renders the HTML template.

        Returns
        -------
            HTML string of the generated report
        """
        # Run verification on all comparisons to populate results
        self.verify_all()

        # Collect summary statistics from verifications
        passing_count = sum(len(sources) for sources in self.verifications.passing.values())
        failing_count = sum(len(sources) for sources in self.verifications.failing.values())
        total_count = passing_count + failing_count

        # Calculate pass rate
        pass_rate = (passing_count / total_count * 100) if total_count > 0 else 0.0

        # Build list of comparison details for extended summary
        comparisons_list: list[dict[str, Any]] = []
        comparisons_list.extend(
            self._compile_comparison_results(self.verifications.passing, True)
        )
        comparisons_list.extend(
            self._compile_comparison_results(self.verifications.failing, False)
        )

        # Prepare the report data
        report_data = {
            "summary": {
                "total": total_count,
                "passing": passing_count,
                "failing": failing_count,
                "pass_rate": round(pass_rate, 1),
            },
            "comparisons": comparisons_list,
            "plot_images": plot_images or {},
            "filtered_failing_results": self._gather_filtered_test_results(),
        }

        # Generate the HTML report using the template
        html_content = report.create_html_report(report_data)

        return html_content

    def _compile_comparison_results(
        self, comparison_dict: Mapping[str, Mapping[str, Comparison]], passed: bool
    ) -> list[dict[str, Any]]:
        """Compile comparison results for report generation."""

        results = []
        for measure_key, source_dict in comparison_dict.items():
            for source_key, comparison in source_dict.items():
                test_source = comparison.test_bundle.source.name.lower()
                ref_source = comparison.reference_bundle.source.name.lower()
                overall_metadata = comparison.proportion_test_results["overall"].to_dict()
                all_test_results = self._extract_all_test_results(comparison)
                results.append(
                    {
                        "measure_key": measure_key,
                        "test_source": test_source,
                        "ref_source": ref_source,
                        "passed": passed,
                        "overall_testresult": overall_metadata,
                        "all_testresults": all_test_results,
                        "passing_count": sum(
                            1 for tr in all_test_results if not tr["reject_null"]
                        ),
                        "failing_count": sum(
                            1 for tr in all_test_results if tr["reject_null"]
                        ),
                    }
                )
        return results

    def _extract_all_test_results(self, comparison: Comparison) -> list[dict[str, Any]]:
        """Collect all TestResults (overall and stratified) as a flat list of dicts."""
        results = []
        overall = comparison.proportion_test_results.get("overall")
        if overall:
            results.append(overall)
        stratified = comparison.proportion_test_results.get("stratified", {})
        if isinstance(stratified, dict):
            for group in stratified.values():
                for test_result in group.values():
                    results.append(test_result)
        return [test_result.to_dict() for test_result in results]

    def _gather_filtered_test_results(self) -> list[dict[str, Any]]:
        """Collect and filter test results using lattice drill-down algorithm.

        For each failing comparison, starts from the overall result and drills
        down through the lattice of stratification levels:

        - If a node failed and has no failing descendants: display it
        - If a node failed and a single child contains all failing descendants:
          skip the parent and recurse into that child
        - If a node failed and failures span multiple children: display the node
        - If a node passed: recurse into each child

        Results are sorted by bayes_factor descending (highest first).

        Returns
        -------
            Sorted list of filtered failing TestResult dicts.
        """
        all_filtered: list[dict[str, Any]] = []

        for source_dict in self.verifications.failing.values():
            for comparison in source_dict.values():
                overall = comparison.proportion_test_results.get("overall")
                stratified = comparison.proportion_test_results.get("stratified", {})

                if not overall:
                    continue

                # Collect all TestResults for this comparison
                all_results: list[TestResult] = [overall]
                if isinstance(stratified, dict):
                    for group in stratified.values():
                        for test_result in group.values():
                            all_results.append(test_result)

                # Run the lattice drill-down starting from overall
                displayed: list[TestResult] = []
                self._process_lattice_node(overall, all_results, displayed)
                all_filtered.extend(r.to_dict() for r in displayed)

        # Sort by bayes_factor descending (highest first)
        all_filtered.sort(key=lambda result: result["bayes_factor"], reverse=True)
        return all_filtered

    def _process_lattice_node(
        self,
        node: TestResult,
        all_results: list[TestResult],
        displayed: list[TestResult],
    ) -> None:
        """Process a single node in the lattice drill-down algorithm.

        Parameters
        ----------
        node
            The current TestResult node being evaluated.
        all_results
            All TestResults for the current comparison.
        displayed
            Accumulator list of TestResults to display.
        """
        children = self._get_lattice_children(node, all_results)

        if node.reject_null:
            failing_beneath = self._get_failing_descendants(node, all_results)

            if not failing_beneath:
                # No failing checks beneath: display this node, stop
                displayed.append(node)
                return

            # Check if a single child contains ALL failing descendants
            failing_ids = {id(r) for r in failing_beneath}

            for child in children:
                # Compute the set of failing result ids beneath (or equal to) this child
                child_coverage: set[int] = set()
                if child.reject_null:
                    child_coverage.add(id(child))
                for r in self._get_failing_descendants(child, all_results):
                    child_coverage.add(id(r))

                if failing_ids.issubset(child_coverage):
                    # Single child contains all failures — recurse, don't display parent
                    self._process_lattice_node(child, all_results, displayed)
                    return

            # No single child contains all failures: display parent, don't recurse
            displayed.append(node)
        else:
            # Node passed: recurse into each child
            for child in children:
                self._process_lattice_node(child, all_results, displayed)

    @staticmethod
    def _get_lattice_children(
        node: TestResult, all_results: list[TestResult]
    ) -> list[TestResult]:
        """Get direct children of a node in the lattice.

        A direct child has exactly one additional stratification dimension
        and matching values on all dimensions shared with the parent.
        """
        node_info = node.index_info or {}
        node_dims = frozenset(node_info.keys())
        target_dim_count = len(node_dims) + 1

        children = []
        for result in all_results:
            result_info = result.index_info or {}
            result_dims = frozenset(result_info.keys())

            if len(result_dims) != target_dim_count:
                continue
            if not node_dims.issubset(result_dims):
                continue
            if not all(result_info.get(k) == v for k, v in node_info.items()):
                continue

            children.append(result)

        return children

    @staticmethod
    def _is_lattice_descendant(candidate: TestResult, ancestor: TestResult) -> bool:
        """Check if candidate is a strict descendant of ancestor in the lattice.

        A descendant has strictly more dimensions and matching values on all
        shared dimensions.
        """
        ancestor_info = ancestor.index_info or {}
        candidate_info = candidate.index_info or {}

        if len(candidate_info) <= len(ancestor_info):
            return False

        return all(candidate_info.get(key) == value for key, value in ancestor_info.items())

    @classmethod
    def _get_failing_descendants(
        cls, node: TestResult, all_results: list[TestResult]
    ) -> list[TestResult]:
        """Get all failing TestResults that are strict descendants of node."""
        return [
            r for r in all_results if r.reject_null and cls._is_lattice_descendant(r, node)
        ]

    # TODO MIC-6047 Let user pass in custom age groups
    def _get_age_groups(self) -> pd.DataFrame:
        """Get the age groups from the given DataFrame or from the artifact."""
        from vivarium.framework.artifact.artifact import ArtifactException

        try:
            age_groups: pd.DataFrame = self.data_loader.get_data(
                "population.age_bins", DataSource.ARTIFACT
            )
        # If we can't find the age groups in the artifact, get them directly from vivarium inputs
        except ArtifactException:
            from vivarium_inputs import get_age_bins

            age_groups = get_age_bins()

        # mypy wants this to do type narrowing
        if age_groups is None:
            raise ValueError(
                "No age groups found. Please provide a DataFrame or use the artifact."
            )
            # relabel index level age_group_name to age_group

        return age_groups.rename_axis(index={"age_group_name": "age_group"})

    def cache_gbd_data(
        self,
        data_key: str,
        data: pd.DataFrame | dict[str, str] | str,
        overwrite: bool = False,
    ) -> None:
        """Upload the output of a get_draws call to the context given by a data key."""
        formatted_data: pd.DataFrame | dict[str, str] | str
        if isinstance(data, pd.DataFrame):
            formatted_data = self._format_to_vivarium_inputs_conventions(data, data_key)
            formatted_data = set_validation_index(formatted_data)
        else:
            formatted_data = data
        self.data_loader.cache_gbd_data(data_key, formatted_data, overwrite=overwrite)

    def _format_to_vivarium_inputs_conventions(
        self, data: pd.DataFrame, data_key: str
    ) -> pd.DataFrame:
        """Format the output of a get_draws call to data schema conventions for the validation context."""
        if "relative_risk" in data_key:
            data = vi.get_affected_measure_column(data)
        data = drop_extra_columns(data, data_key)
        data = set_gbd_index(data, data_key=data_key)
        data = vi.scrub_gbd_conventions(data, self.location)
        data = vi.split_interval(data, interval_column="age", split_column_prefix="age")
        data = vi.split_interval(data, interval_column="year", split_column_prefix="year")
        formatted_data: pd.DataFrame = vi.sort_hierarchical_data(data)
        return formatted_data

    @staticmethod
    def format_ui_data_index(data: pd.DataFrame, comparison_key: str) -> pd.DataFrame:
        """Format and sort the data for UI display.

        Parameters
        ----------
        data
            The DataFrame to sort.
        comparison_key
            The comparison key for logging purposes.

        Returns
        -------
            The sorted DataFrame.
        """

        expected_order = get_measure_index_names(comparison_key, "vivarium")
        ordered_cols = [col for col in expected_order if col in data.index.names]
        extra_idx_cols = [col for col in data.index.names if col not in ordered_cols]
        sorted_index = ordered_cols + extra_idx_cols
        sorted = data.reorder_levels(sorted_index).sort_index()
        return add_comparison_metadata_levels(sorted, comparison_key)

    def add_new_measure(self, measure_key: str, measure_class: type[Measure]) -> None:
        """Add a new measure class to the context's measure mapper.

        Parameters
        ----------
        measure_key
            The measure key in format 'entity_type.entity.measure_key' or 'entity_type.measure_key'.
        measure_class
            The class implementing the measure.
        """

        self.measure_mapper.add_new_measure(measure_key, measure_class)
