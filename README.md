# malt

**M**edia **A**ctive-**L**earning **T**oolkit.

malt uses approaches from Bayesian Machine Learning and Active Learnign to efficienctly characterize the response surface of cell cultures to varying media components.


## Status

Early. Two modules exist:

- `malt.engine.factors` declares the variables a campaign may vary and their ranges
- `malt.engine.glm` fits a Gamma/log-link response surface and reports whether the sampler converged

Nothing else is written yet. The uncertainty split, the acquisition functions,
the batch-effect model, the state store, and the MCP server are all still ahead.

## Install

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

Declare the factors, then pass them to `fit_gamma_glm` along with a dataframe
of runs:

```python
from malt.engine.factors import Factor
from malt.engine.glm import fit_gamma_glm

factors = [
    Factor("glucose", 0.1, 10.0, units="g/L", scale="log"),
    Factor("nitrogen", 0.5, 4.0, units="g/L"),
    Factor("phosphate", 0.1, 1.0, units="g/L"),
]

# runs: one column per factor, plus the measured response
fit = fit_gamma_glm(runs, response="biomass", factors=factors, random_seed=0)

print(fit.convergence.summary())
print(fit.posterior["posterior"]["beta"].sel(term="glucose^2").mean().item())
```

The model is a quadratic response surface: a linear term for each factor, a
squared term for each, and one term per pair. Three factors give nine
coefficients plus an intercept, so you need at least ten runs before the fit
means anything.

Coefficients come back as an xarray coordinate indexed by term name, so
`.sel(term="glucose^2")` reads the curvature in glucose directly.

## Why Gamma with a log link

Biomass is positive and its variance grows with its mean. Ordinary least
squares assumes neither, so it will happily predict negative growth at the
edges of a design and will underweight the noisy high-yield runs you care most
about. A Gamma likelihood with a log link holds the coefficient of variation
constant instead, which is what these assays actually do.

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
