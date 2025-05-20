import pandas as pd

from vivarium_testing_utils.automated_validation.comparison import Comparison


def plot_comparison(comparison: Comparison, type: str, kwargs):
    raise NotImplementedError


def plot_data(dataset: pd.DataFrame, type: str, kwargs):
    raise NotImplementedError


def line_plot(comparison: Comparison, x_axis: str, stratifications: list[str]):
    raise NotImplementedError


def bar_plot(comparison: Comparison, x_axis: str, stratifications: list[str]):
    raise NotImplementedError


def box_plot(comparison: Comparison, cat: str, stratifications: list[str]):
    raise NotImplementedError


def heatmap(comparison: Comparison, row: str, col: str):
    raise NotImplementedError


def save_plot(fig, name, format):
    raise NotImplementedError
