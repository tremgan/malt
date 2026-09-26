"""Regret benchmark in 2 factors: BO (Gamma GLM + q-NEI) against iterative RSM and random.

    uv run python -m benchmarks.two_factor.regret --replicates 20

Glucose and nitrogen, as in `rounds.py`; an 11-run local CCD seed around the
house recipe, then 4 rounds of 10 runs (a 2-factor CCD is 8 runs; two centre
replicates fill the batch). See `benchmarks.harness` for how arms are paired
and scored.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness import Benchmark, main
from malt.active_learning.acquisitions import CentralComposite, QNoisyExpectedImprovement, RandomBatch
from malt.active_learning.surrogates import BayesianLinearRegression, GammaGLMSurrogate
from malt.engine.factors import Factor
from malt.simulation.regret import Arm

FACTORS = (
    Factor("glucose", 0.5, 20.0, units="g/L"),
    Factor("nitrogen", 0.1, 10.0, units="g/L", scale="log"),
)

BENCHMARK = Benchmark(
    factors=FACTORS,
    arms=(
        Arm("q-NEI, Gamma GLM", GammaGLMSurrogate(FACTORS), QNoisyExpectedImprovement(FACTORS)),
        Arm("RSM, BLR + CCD", BayesianLinearRegression(FACTORS, "quadratic"), CentralComposite(FACTORS)),
        Arm("Random, Gamma GLM", GammaGLMSurrogate(FACTORS), RandomBatch(FACTORS)),
    ),
    batch=10,
    rounds=4,
    results=Path(__file__).parent / "results",
)

if __name__ == "__main__":
    main(BENCHMARK, __doc__)
