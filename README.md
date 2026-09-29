# malt🌾

**M**edia **A**ctive-**L**earning **T**oolkit.

malt picks which media compositions to test next. It fits a Bayesian Gamma GLM
model to growth data (not Gaussian since growth is bounded from below by 0) and proposes the next batch by Bayesian optimization.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/quadratic_oracle_dark.svg">
  <img alt="A synthetic growth surface: mean biomass against glucose and nitrogen, with a diagonal ridge where the two interact" src="docs/figures/quadratic_oracle.svg">
</picture>

The problem in two components. Real media have six or eight, the surface is not
this smooth, and every point costs a culture.

Candidates are space-filling draws rather than a grid, redrawn each round, and
q-NEI treats them only as starting points. It then optimizes each pick with
L-BFGS-B on the model's analytic derivative. The gain grows with the number of
components, since a pool of n points covers only n^(1/k) levels per component.
The rule is BoTorch's q-NEI under sequential greedy selection, in numpy over
PyMC's posterior draws, so nothing here depends on torch. UCB and Thompson
sampling still pick from the draws.

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

The regret benchmarks score Bayesian optimization against iterative
response-surface DOE and against random batches, by cumulative regret on a
simulated lab whose optimum is known:

```bash
uv run python -m benchmarks.two_factor.regret --replicates 20 --workers 8
uv run python -m benchmarks.three_factor.regret --replicates 20 --workers 8
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/three_factor/results/regret_dark.svg">
  <img alt="Cumulative regret by round over three factors: q-NEI flattens while iterative RSM and random batches stay linear" src="benchmarks/three_factor/results/regret.svg">
</picture>

Three factors, 16 runs a round, 20 paired campaigns per arm. q-NEI starts level
with random in the first round, when it has only the seed to go on, then
flattens to about 0.19 of a run's biomass lost per run on the quadratic surface
while random stays near 0.86. Crossing model against acquisition on the
two-factor benchmark puts that gap in the acquisition rather than the Gamma GLM:
swapping q-NEI for a central composite costs a median 5.7 to 8.7 regret with
either model, while swapping the model moves it by at most 2.6, and by under 0.1
in half the cells.

Both baselines are iterative. A one-shot Box-Behnken or CCD, which is what a lab
would more often run, is not in the benchmark yet, so nothing here yet supports
"fewer experiments than a fixed design".

What the campaign is doing between those numbers, round by round. Two factors
here so the surface can be drawn; each row is one round, showing the truth, what
the model believes, and where it is still unsure.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/two_factor/results/rounds_qnei_quadratic_dark.svg">
  <img alt="Five rounds on the quadratic surface: true mean biomass, posterior mean, and posterior uncertainty, with runs so far, the next proposed batch and the true optimum" src="benchmarks/two_factor/results/rounds_qnei_quadratic.svg">
</picture>

```bash
uv run python -m benchmarks.two_factor.rounds --arm qnei --surface quadratic
```

On the quadratic surface the model is in family, and the seed alone is not
enough: after 11 runs the posterior mean is a broad gradient and uncertainty is
high nearly everywhere. By round 2 the ridge is in the right place and
uncertainty has collapsed into an island around the optimum. The far corner
stays dark to the end, because nothing out there would change the answer.

The GP-sampled surface is the harder case. No quadratic fits it, so the
posterior mean stays a smooth approximation of a surface that is not smooth, and
the campaign is still descending at round 4 rather than converged:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/two_factor/results/rounds_qnei_gp_dark.svg">
  <img alt="Five rounds on the GP-sampled surface: true mean biomass, posterior mean, and posterior uncertainty, with runs so far, the next proposed batch and the true optimum" src="benchmarks/two_factor/results/rounds_qnei_gp.svg">
</picture>

```bash
uv run python -m benchmarks.two_factor.rounds --arm qnei --surface gp
```

`polish_tradeoff` isolates the continuous optimization: what it gains over
picking from the pool, and what it costs, in 2 to 8 factors. `--campaign` times
it against a fitted model instead of a synthetic one.

```bash
uv run python -m benchmarks.polish_tradeoff
uv run python -m benchmarks.polish_tradeoff --campaign
```

`ablation` crosses the model against the acquisition, so the gap above can be
attributed to one or the other:

```bash
uv run python -m benchmarks.two_factor.ablation --replicates 20 --workers 8
```
