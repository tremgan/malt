"""`build_design_matrix`'s term filter: a subset must keep the term-order contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malt.engine.factors import Factor
from malt.engine.glm import build_design_matrix

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
