"""`build_design_matrix`'s term filter: a subset must keep the term-order contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malt.engine.factors import Factor, decode_design
from malt.engine.glm import (
    ALL_TERMS,
    build_design_matrix,
    design_jacobian,
    predict_mu,
    predict_mu_grad,
)

FACTORS = (Factor("a", 0, 10), Factor("b", 0.1, 10, scale="log"), Factor("c", 1, 2))
DATA = pd.DataFrame({"a": [0.0, 5, 10, 2], "b": [0.1, 1, 10, 3], "c": [1, 1.5, 2, 1.2]})


def test_default_is_the_full_quadratic_in_contract_order():
    assert build_design_matrix(DATA, FACTORS).term_names == (
        "a", "b", "c", "a^2", "b^2", "c^2", "a:b", "a:c", "b:c",
    )


@pytest.mark.parametrize(
    "terms", [("linear",), ("linear", "interaction"), ("interaction", "linear"), ("quadratic",)]
)
def test_subset_is_the_matching_columns_of_the_full_design(terms):
    full = build_design_matrix(DATA, FACTORS)
    subset = build_design_matrix(DATA, FACTORS, terms=terms)
    idx = [full.term_names.index(name) for name in subset.term_names]
    assert idx == sorted(idx)  # order follows the contract, not the argument
    assert set(subset.term_kinds) == set(terms)
    np.testing.assert_array_equal(subset.X, full.X[:, idx])


@pytest.mark.parametrize("terms", [(), ("cubic",)])
def test_rejects_empty_or_unknown_terms(terms):
    with pytest.raises(ValueError, match="terms must be"):
        build_design_matrix(DATA, FACTORS, terms=terms)


# The design Jacobian


@pytest.mark.parametrize(
    "factors",
    [
        (Factor("a", 1.0, 10.0), Factor("b", 0.1, 10.0, scale="log")),
        (Factor("a", 0.0, 5.0), Factor("b", 0.1, 10.0, scale="log"), Factor("c", -2.0, 2.0)),
    ],
    ids=["two-factor", "three-factor"],
)
@pytest.mark.parametrize(
    "terms",
    [ALL_TERMS, ("linear",), ("linear", "quadratic"), ("quadratic", "interaction")],
)
def test_design_jacobian_matches_finite_differences_of_the_design_matrix(factors, terms):
    """The guard on the term-order contract: a permuted column would show up here.

    Both come from `_term_layout`, so this is what stops the design matrix and
    its derivative drifting apart — a mismatch that produces a confidently
    wrong gradient with no error.
    """
    k = len(factors)
    z = np.random.default_rng(0).uniform(-0.9, 0.9, (6, k))
    X = lambda zz: build_design_matrix(decode_design(factors, zz), factors, terms=terms).X

    h = 1e-6
    expected = np.stack(
        [(X(z + h * np.eye(k)[d]) - X(z - h * np.eye(k)[d])) / (2 * h) for d in range(k)],
        axis=-1,
    )
    jacobian = design_jacobian(z, factors, terms=terms)
    assert jacobian.shape == expected.shape
    np.testing.assert_allclose(jacobian, expected, atol=1e-7)


def test_design_jacobian_is_pointwise():
    factors = (Factor("a", 1.0, 10.0), Factor("b", 1.0, 10.0))
    z = np.random.default_rng(1).uniform(-1.0, 1.0, (8, 2))
    np.testing.assert_allclose(design_jacobian(z, factors)[3:5], design_jacobian(z[3:5], factors))


def test_design_jacobian_rejects_a_column_count_mismatch():
    factors = (Factor("a", 1.0, 10.0), Factor("b", 1.0, 10.0))
    with pytest.raises(ValueError, match="columns but 2 factors"):
        design_jacobian(np.zeros((4, 3)), factors)


def test_predict_mu_grad_is_mu_times_the_linear_predictor_gradient():
    factors = (Factor("a", 1.0, 10.0), Factor("b", 0.1, 10.0, scale="log"))
    rng = np.random.default_rng(2)
    z = rng.uniform(-0.8, 0.8, (5, 2))
    x = decode_design(factors, z)
    intercept, beta = rng.normal(size=7), rng.normal(size=(7, 5))

    mu = predict_mu(intercept, beta, build_design_matrix(x, factors).X)
    got = predict_mu_grad(mu, beta, design_jacobian(z, factors))

    # Finite differences of `predict_mu` itself, in coded units.
    h = 1e-6
    expected = np.stack(
        [
            (
                predict_mu(intercept, beta, build_design_matrix(decode_design(factors, z + h * np.eye(2)[d]), factors).X)
                - predict_mu(intercept, beta, build_design_matrix(decode_design(factors, z - h * np.eye(2)[d]), factors).X)
            )
            / (2 * h)
            for d in range(2)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_predict_mu_grad_rejects_mismatched_shapes():
    mu, beta, jacobian = np.ones((4, 3)), np.ones((4, 5)), np.ones((3, 5, 2))
    predict_mu_grad(mu, beta, jacobian)  # the consistent case
    with pytest.raises(ValueError, match="draws"):
        predict_mu_grad(np.ones((6, 3)), beta, jacobian)
    with pytest.raises(ValueError, match="does not match"):
        predict_mu_grad(mu, beta, np.ones((3, 4, 2)))
