# malt🌾

**M**edia **A**ctive-**L**earning **T**oolkit.

malt picks which media compositions to test next. It fits a Bayesian Gamma
model to growth data and proposes each next batch of experiments by Bayesian
optimization. Nothing reaches the lab until a person approves the batch.

**Status:** early and changing fast. Simulated campaigns run end to end; the
state store and the agent-facing tools come next.

## Layout

- `malt.engine`: factors, the Gamma GLM, acquisition maths, continuous search
- `malt.active_learning`: the campaign loop, surrogate models, acquisition rules, stopping rules
- `malt.simulation`: simulated labs with a known truth, and regret scoring
- `benchmarks/`: benchmark scripts and their results

## Install

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/tremgan/malt.git
cd malt
uv sync
uv run pytest
```

## Benchmarks

```bash
uv run python -m benchmarks.two_factor.regret --replicates 20 --workers 8
uv run python -m benchmarks.three_factor.regret --replicates 20 --workers 8
```
