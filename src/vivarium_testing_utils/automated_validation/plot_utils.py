from vivarium_testing_utils.automated_validation.comparison import Comparison
import pandas as pd


def plot_comparison(comparison: Comparison, type: str, kwargs):
    pass


def plot_data(dataset: pd.DataFrame, type: str, kwargs):
    pass


def line_plot(comparison: Comparison, x_axis: str, stratifications: list[str]):
    pass


def bar_plot(comparison: Comparison, x_axis: str, stratifications: list[str]):
    pass


def box_plot(comparison: Comparison, cat: str, stratifications: list[str]):
    pass


def heatmap(comparison: Comparison, row: str, col: str):
    pass


def save_plot(fig, name, format):
    pass
