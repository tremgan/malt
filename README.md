# malt🌾

**M**edia **A**ctive-**L**earning **T**oolkit.

malt characterizes how cell cultures respond to their media, using Bayesian
modeling and active learning to get there in fewer experiments than a fixed
design would need.

The classical approach is design of experiments and response surface
methodology (Montgomery, *Design and Analysis of Experiments*): ordinary least
squares on linear, quadratic, and interaction terms. OLS carries a Gaussian
likelihood, whose support runs from negative to positive infinity. Growth is
bounded below by zero, so that likelihood puts mass where no measurement can
land and predicts negative biomass near the edges of a design. malt fits a
Gamma likelihood instead, which has support on the positive reals and lets
variance grow with the mean the way these assays do.

The second departure is what happens after a round finishes. Rather than
committing to one design up front, malt fits what you have measured, works out
where the response surface is least known, and proposes the next batch from
there.

## Status

Early. Two modules exist:

- `malt.engine.factors` declares the variables a campaign may vary and their ranges
- `malt.engine.glm` fits a Gamma/log-link response surface and reports whether the sampler converged

Nothing else is written yet. The uncertainty split, the acquisition functions,
the batch-effect model, the state store, and the MCP server are all still ahead.

## Install

Needs Python 3.12 or newer.

```bash
uv sync
```

### macOS 26 and later need one flag

Sampling dies with `ld: library 'd64' not found` until you set it. PyTensor
hardcodes a `-ld64` linker argument for macOS 15+ in
`pytensor/link/c/cmodule.py` and offers no config switch to turn it off. The
current linker reads that argument as a request to link a library called `d64`,
which does not exist.

```bash
PYTENSOR_FLAGS='cxx=' uv run python your_script.py
```

Set it once per machine instead, in `~/.pytensorrc`:

```ini
[global]
cxx =
```

Turning off the C backend costs almost nothing here. Sampling runs through
nutpie, which compiles the log-density with numba, so the C backend only ever
handled small helper graphs. A 15-run fit takes about two seconds with or
without it. Leave the flag alone.

One warning: `PYTENSOR_FLAGS='ldflags='` looks like it works and does not.
PyTensor does not recognize `ldflags` as a flag, ignores it, and falls back to
a slow Python interpreter for graphs it cannot compile. You get a working
script and wrong performance.

## Fitting a model

Your data is one row per run, a column per factor in real units, and a column
for what you measured:

```
   glucose  nitrogen  phosphate  biomass
0      0.1      0.50       0.55    0.584
1      0.1      4.00       0.55    1.077
2     10.0      0.50       0.55    1.507
3     10.0      4.00       0.55   10.112
4      0.1      2.25       0.10    0.560
```

Declare what each column means, then fit:

```python
from malt.engine.factors import Factor
from malt.engine.glm import fit_gamma_glm

factors = [
    Factor("glucose", 0.1, 10.0, units="g/L", scale="log"),
    Factor("nitrogen", 0.5, 4.0, units="g/L"),
    Factor("phosphate", 0.1, 1.0, units="g/L"),
]

fit = fit_gamma_glm(runs, response="biomass", factors=factors, random_seed=0)

print(fit.convergence.summary())                                  # converged
print(fit.posterior["posterior"]["beta"].mean(("chain", "draw"))) # the surface
```

```
glucose               0.72     glucose^2            -0.65
nitrogen              0.54     nitrogen^2           -0.24
phosphate             0.25     phosphate^2          -0.50
                               glucose:nitrogen      0.30
```

Every factor gets a linear term, a squared term, and one interaction per pair.
Three factors make nine coefficients plus an intercept, so ten runs is the
floor before a fit means anything. `fit_gamma_glm` refuses fewer.

Negative squared terms are the interesting ones. Each says the response peaks
inside the range you declared rather than running off an edge, which is what
makes an optimum worth searching for. Read one directly with
`fit.posterior["posterior"]["beta"].sel(term="glucose^2")`.

### Defining and sampling separately

`fit_gamma_glm` wraps two steps. Defining a model costs milliseconds and
sampling it costs seconds, so split them when you want to see the
specification before paying for a posterior:

```python
import pymc as pm
from malt.engine.glm import build_gamma_glm, sample_gamma_glm

glm = build_gamma_glm(runs, response="biomass", factors=factors)
pm.sample_prior_predictive(draws=200, model=glm.model)   # do the priors make sense?
fit = sample_gamma_glm(glm, random_seed=0)
```

Arguments follow the split. `alpha_prior_sigma` shapes the model, while
`draws`, `chains`, `target_accept` and `random_seed` shape the sampling.

## Why the log link

The intro covers the Gamma likelihood. The link deserves its own note.

A log link makes the coefficients multiplicative: doubling glucose scales
biomass by a factor rather than adding a fixed amount, which is how media
components behave. It also pairs with the Gamma to hold the coefficient of
variation constant, so a run yielding 10 g/L carries proportionally more
absolute noise than one yielding 1. Fit those on a Gaussian and the
high-yield runs look like outliers, so the model chases them.

One consequence worth knowing: the posterior over the mean is lognormal
shaped, not normal. Closed-form Gaussian formulas for expected improvement do
not apply, so the acquisition functions will compute by Monte Carlo over the
draws instead.

## Why coded units, not z-scores

`Factor.encode` maps each value to `[-1, +1]` against the range you declared.
It does not standardize against the observed data, and the difference matters
twice.

Empirical means and standard deviations move as a campaign progresses. Round
one spreads across the whole space; round four clusters near an optimum. If the
scaling follows the data, a coefficient from round one and a coefficient from
round four are measured in different units, and the prior on the quadratic
terms silently becomes a different prior each round.

The second reason applies to any factor you declare as `scale="log"`. Glucose
from 0.1 to 10 g/L has design levels at 0.1, 1, and 10. Z-scoring those
arithmetically puts them at -0.658, -0.493, and 1.151, which crushes two of the
three levels together and leaves the curvature estimated from almost nothing.
Encoding in log space puts them back at -1, 0, and +1. On a simulated fit that
change moved effective sample size from 1113 to 3740 and cut mean coefficient
error by half.

## Convergence

Every fit returns a `ConvergenceReport` next to the draws. The gate is zero
divergences, r-hat below 1.01, and at least 1000 effective samples in both bulk
and tail.

```python
fit.convergence.converged   # False if any gate failed
fit.convergence.failures    # ("max r_hat 1.526 >= 1.01", ...)
```

A failed fit still comes back, because the diverged draws are what you need to
work out why. `fit_gamma_glm` raises a `ConvergenceWarning` so a person notices
and sets `converged=False` so a program can.

Tail effective sample size is gated alongside bulk because the acquisition
functions read the upper tail of the posterior. A chain can mix well through
the middle of a distribution and still wander in the tail, which would move the
recommended next experiment for no reason but sampler noise.
