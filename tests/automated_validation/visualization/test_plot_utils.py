import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import os
from vivarium_testing_utils.automated_validation.visualization.plot_utils import (
    line_plot,
    RELPLOT_KWARGS,
)
import seaborn as sns
from vivarium_testing_utils.automated_validation.data_loader import DataSource


# Fixture for test data
@pytest.fixture
def sample_data():
    """Create a sample comparison with test and reference data."""
    # Create index with multiple levels: age_group, sex, region
    idx = pd.MultiIndex.from_product(
        [
            [
                "early_neonatal",
                "late_neonatal",
                "1-5_months",
                "6-11_months",
                "12_to_23_months",
                "2_to_4",
            ],  # age_group
            ["Male", "Female"],  # sex
            ["North", "South"],  # region
        ],
        names=["age_group", "sex", "region"],
    )

    # Create test data with random values
    np.random.seed(42)  # For reproducibility
    test_data = pd.DataFrame({"value": np.random.uniform(0.01, 0.06, len(idx))}, index=idx)

    # Create reference data with slightly different values
    np.random.seed(43)  # Different seed for reference data
    reference_data = pd.DataFrame(
        {"value": np.random.uniform(0.01, 0.06, len(idx))}, index=idx
    )

    return test_data, reference_data


@pytest.fixture
def combined_data(sample_data):
    """Combine test and reference data for plotting."""
    test_data, reference_data = sample_data

    # Add source column to each DataFrame
    test_data["source"] = "Simulation"
    reference_data["source"] = "Artifact"

    # Combine the two DataFrames
    combined_data = pd.concat([test_data, reference_data]).reset_index()
    return combined_data


@pytest.fixture
def sample_title():
    """Create a sample title for the comparison."""
    return "Sample Comparison Title"


@pytest.fixture()
def reference_dir(tmpdir):
    """Create a temporary directory for reference plots."""
    output_dir = str(tmpdir.mkdir("reference_plots"))
    return output_dir


# Individual reference figure fixtures
@pytest.fixture
def reference_no_stratification(combined_data, sample_title, reference_dir):
    """Generate reference plot for no stratification scenario."""
    output_path = os.path.join(reference_dir, "reference_no_strat.png")
    fig = sns.relplot(
        data=combined_data,
        x="age_group",
        y="value",
        hue="source",
        kind="line",
        **RELPLOT_KWARGS,
    )

    fig.set_axis_labels("age_group", "Proportion")
    # Add overall title
    fig.figure.suptitle(sample_title, y=1.02, fontsize=16)
    fig.tight_layout()
    fig.map(plt.grid, alpha=0.5, color="gray")

    # Save the reference plot
    fig.savefig(output_path)
    return output_path


@pytest.fixture
def reference_one_stratification(combined_data, sample_title, reference_dir):
    """Generate reference plot for one stratification (sex) scenario."""
    output_path = os.path.join(reference_dir, "reference_one_strat.png")
    fig = sns.relplot(
        data=combined_data,
        x="age_group",
        y="value",
        hue="source",
        kind="line",
        col="sex",
        col_wrap=3,
        **RELPLOT_KWARGS,
    )
    fig.set_axis_labels("age_group", "Proportion")
    # Add overall title
    fig.figure.suptitle(sample_title, y=1.02, fontsize=16)
    fig.tight_layout()
    fig.map(plt.grid, alpha=0.5, color="gray")
    # Save the reference plot
    fig.savefig(output_path)

    return output_path


@pytest.fixture
def reference_two_stratifications(combined_data, sample_title, reference_dir):
    """Generate reference plot for two stratifications (sex and region) scenario."""
    output_path = os.path.join(reference_dir, "reference_two_strat.png")
    fig = sns.relplot(
        data=combined_data,
        x="age_group",
        y="value",
        hue="source",
        kind="line",
        col="sex",
        row="region",
        **RELPLOT_KWARGS,
    )
    fig.set_axis_labels("age_group", "Proportion")
    # Add overall title
    fig.figure.suptitle(sample_title, y=1.02, fontsize=16)
    fig.map(plt.grid, alpha=0.5, color="gray")
    fig.tight_layout()
    # Save the reference plot
    fig.savefig(output_path)

    return output_path


# Individual tests for each scenario
def test_line_plot_no_stratification(
    sample_data, sample_title, reference_no_stratification, tmpdir
):
    """Test line_plot with no stratifications."""
    test_data, reference_data = sample_data

    output_path = os.path.join(str(tmpdir), "test_no_strat.png")

    # Call the function to test
    fig = line_plot(
        title=sample_title,
        test_data=test_data,
        test_source=DataSource.SIM,
        reference_data=reference_data,
        reference_source=DataSource.ARTIFACT,
        x_axis="age_group",
        stratifications=[],
    )
    fig.savefig(output_path)

    # Compare with reference
    result = compare_images(reference_no_stratification, output_path, tol=10)
    assert result is None, f"Images differ: {result}"


def test_line_plot_one_stratification(
    sample_data, sample_title, reference_one_stratification, tmpdir
):
    """Test line_plot with one stratification (sex)."""
    test_data, reference_data = sample_data

    output_path = os.path.join(str(tmpdir), "test_one_strat_sex.png")

    # Call the function to test
    fig = line_plot(
        title=sample_title,
        test_data=test_data,
        test_source=DataSource.SIM,
        reference_data=reference_data,
        reference_source=DataSource.ARTIFACT,
        x_axis="age_group",
        stratifications=["sex"],
    )
    fig.savefig(output_path)

    # Compare with reference
    result = compare_images(reference_one_stratification, output_path, tol=10)
    assert result is None, f"Images differ: {result}"


def test_line_plot_two_stratifications(
    sample_data, sample_title, reference_two_stratifications, tmpdir
):
    """Test line_plot with two stratifications."""
    test_data, reference_data = sample_data
    output_path = os.path.join(str(tmpdir), "test_two_strat.png")

    # Call the function to test
    fig = line_plot(
        title=sample_title,
        test_data=test_data,
        test_source=DataSource.SIM,
        reference_data=reference_data,
        reference_source=DataSource.ARTIFACT,
        x_axis="age_group",
        stratifications=["sex", "region"],
    )
    fig.savefig(output_path)

    # Compare with reference
    result = compare_images(reference_two_stratifications, output_path, tol=10)
    assert result is None, f"Images differ: {result}"


def test_line_plot_too_many_stratifications(sample_data, sample_title):
    """Test that line_plot raises an error with more than two stratifications."""

    with pytest.raises(ValueError, match="Maximum of 2 stratifications supported"):
        line_plot(
            title=sample_title,
            test_data=sample_data[0],
            test_source=DataSource.SIM,
            reference_data=sample_data[1],
            reference_source=DataSource.ARTIFACT,
            x_axis="age_group",
            stratifications=["sex", "region", "extra_strat"],
        )


def test_line_plot_invalid_stratification(sample_data, sample_title):
    """Test that line_plot handles invalid stratification names gracefully."""

    with pytest.raises(KeyError):
        line_plot(
            title=sample_title,
            test_data=sample_data[0],
            test_source=DataSource.SIM,
            reference_data=sample_data[1],
            reference_source=DataSource.ARTIFACT,
            x_axis="age_group",
            stratifications=["invalid_strat"],
        )
