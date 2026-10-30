import pytest

from evalh.metrics import spearman, weighted_kappa, within_one


def test_weighted_kappa_perfect_agreement():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_weighted_kappa_penalizes_big_gaps_more():
    truth = [1, 2, 3, 4, 5, 3, 2, 4]
    off_by_one = [2, 3, 4, 5, 4, 4, 3, 5]
    off_by_two = [3, 4, 5, 2, 3, 5, 4, 2]
    assert weighted_kappa(truth, off_by_one) > weighted_kappa(truth, off_by_two)


def test_weighted_kappa_known_value():
    # Hand-computed on a 1-3 scale (weights (i-j)^2 / 4):
    # observed weighted disagreement = 1/3 * 1/4 = 1/12
    # expected from marginals [1/3, 1/3, 1/3] x [1/3, 0, 2/3] = 5/12
    # kappa = 1 - (1/12) / (5/12) = 0.8
    assert weighted_kappa([1, 2, 3], [1, 3, 3], categories=(1, 2, 3)) == pytest.approx(0.8, abs=1e-9)


def test_spearman_handles_ties_and_direction():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_within_one():
    assert within_one([1, 3, 5], [2, 5, 5]) == pytest.approx(2 / 3)
