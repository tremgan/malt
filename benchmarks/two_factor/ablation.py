"""Is the q-NEI vs RSM gap the acquisition, or the model? Cross both and see.

    uv run python -m benchmarks.two_factor.ablation --replicates 20 --workers 8

`regret.py`'s arms differ in two things at once: q-NEI runs on the Gamma GLM and
iterative RSM on the Bayesian linear baseline. That confounds the headline, and
"BO beats DOE" would be a much weaker claim if the gap were really "a Gamma GLM
beats a linear model on strictly positive, heteroscedastic data" — which it
would be, since that comparison is rigged from the start.

Crossing the two factors separates them. Same surfaces, same seeds and the same
paired noise streams as `regret.py`, so the cells are directly comparable to it.
Two factors rather than three because the answer does not need the extra
dimension and this way it costs a couple of minutes.

Writes `ablation_results/regret.csv` and a figure per colour scheme.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness import Benchmark, main
from malt.active_learning.acquisitions import CentralComposite, QNoisyExpectedImprovement
from malt.active_learning.surrogates import BayesianLinearRegression, GammaGLMSurrogate
from malt.simulation.regret import Arm

from benchmarks.two_factor.regret import FACTORS

BENCHMARK = Benchmark(
    factors=FACTORS,
    arms=(
        Arm("q-NEI, Gamma GLM", GammaGLMSurrogate(FACTORS), QNoisyExpectedImprovement(FACTORS)),
        Arm("q-NEI, BLR", BayesianLinearRegression(FACTORS, "quadratic"), QNoisyExpectedImprovement(FACTORS)),
        Arm("CCD, Gamma GLM", GammaGLMSurrogate(FACTORS), CentralComposite(FACTORS)),
        Arm("CCD, BLR", BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS)),
    ),
    batch=10,
    rounds=4,
    results=Path(__file__).parent / "ablation_results",
)

if __name__ == "__main__":
    main(BENCHMARK, __doc__)
