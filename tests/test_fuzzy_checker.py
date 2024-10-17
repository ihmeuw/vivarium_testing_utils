import pytest

from vivarium_testing_utils.fuzzy_checker import FuzzyChecker


@pytest.mark.parametrize(
    "numerator, denominator, target_proportion", 
    [(10_008, 100_000, 0.1), (976, 1_000_000, 0.001), (1_049, 50_000, (0.0198, 0.0202))]
)
def test_pass_fuzzy_assert_proportion(numerator, denominator, target_proportion) -> None:
    FuzzyChecker().fuzzy_assert_proportion(numerator, denominator, target_proportion)


@pytest.mark.parametrize(
    "numerator, denominator, target_proportion, match",
    [(901, 100_000, 0.05, "is significantly less than expected"),
     (1_150, 50_000, 0.02, "is significantly greater than expected")]
)
def test_fail_fuzzy_assert_proportion(numerator, denominator, target_proportion, match) -> None:
    with pytest.raises(AssertionError, match=match):
        FuzzyChecker().fuzzy_assert_proportion(numerator, denominator, target_proportion)


def test_small_sample_size_fuzzy_assert_proportion(caplog) -> None:
    FuzzyChecker().fuzzy_assert_proportion(1, 10, 0.1)
    assert "Sample size too small" in caplog.text


def test_not_conclusive_fuzzy_assert_proportion(caplog) -> None:
    FuzzyChecker().fuzzy_assert_proportion(110, 1_000, 0.1)
    assert "is not conclusive" in caplog.text


def test__calculate_bayes_factor() -> None:
    pass


def test_zero_division__calculate_bayes_factor() -> None:
    pass


def test__fit_beta_distribution_to_uncertainty_interval() -> None:
    pass


def test__no_best_fit_beta_distribution() -> None:
    pass


def test__ui_squared_error() -> None:
    pass


def test__quantile_squared_error() -> None:
    pass


def test_save_diagnotic_file() -> None:
    pass
