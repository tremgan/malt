# CLAUDE.md

Guidance for Claude (or any agent) working in this repository.

## Project name

MALT (Media Acquisition & Learning Toolkit). Package name `malt`,
namespaced as `malt.engine`, `malt.state`, `malt.mcp_server`.

## What this project is

An autonomous experimental-design engine for media/process optimization
(biomass growth vs. media composition — e.g. glucose/nitrogen/phosphate
concentrations). It fits a Bayesian GLM to noisy, strictly-positive,
heteroscedastic growth data, quantifies epistemic uncertainty via full
MCMC, and uses a Bayesian-optimization acquisition function to recommend
the next batch of experiments. The interface is a set of tools an agent
calls directly (MCP), with a human approval gate before any physical
execution — not a dashboard or CLI a human has to drive turn-by-turn.

## Architecture

```
malt/
  engine/          # the modeling core — no I/O, no orchestration, pure functions
    glm.py         # design matrix, model definition, NUTS sampling, convergence report
    diagnostics.py # residual plots, mean-variance check, OLS/negativity baseline
    uncertainty.py # posterior -> predictions: mu at query points, epistemic/aleatoric split
    acquisition.py # UCB / EI / PI / Thompson, single-point and batch variants
    batch_effects.py # random-intercept batch model, marginal predictive draws
    factors.py     # Factor (name/range/units/scale) — the feasible region

  state/           # persistence layer — the source of truth for a running loop
    schema.py      # rounds / batches / experiments / model_versions
    store.py       # read/write against the state store (swap backend freely)

  mcp_server/      # the agent-facing interface — thin wrappers around engine/ + state/
    tools.py       # suggest_batch, ingest_results, get_diagnostics, get_status, ...
    server.py      # MCP server entrypoint

demo/            # simulated end-to-end loop (no real lab hardware assumed)
  simulate_loop.py

reports/         # generated, not hand-maintained — see "Reporting" below
```

**Dependency direction is one-way**: `malt.mcp_server` depends on
`malt.engine` and `malt.state`; `malt.engine` never imports from
`malt.mcp_server` or `malt.state`. `engine` functions are pure — data in,
results out, no side effects — so they're independently testable and
reusable outside the MCP context (e.g. directly in a notebook).

The one sanctioned exception to "no side effects": `engine` may raise a
`warnings.warn` to flag a result the caller must not trust silently (see
`glm.ConvergenceWarning`). A warning mutates no state, touches no I/O, and
is suppressible — and it's the lesser evil against handing back an unmixed
chain that looks like a normal return value. Don't extend this to logging,
file writes, or progress output.

**`glm.py` owns the fit; `uncertainty.py` owns what you do with it.** The
model specification, sampling, and convergence checking all live in
`glm.py`, but **defining a model and sampling it are separate functions**:
`build_gamma_glm` returns a `GammaGLM` handle, `sample_gamma_glm` turns that
into a `GammaGLMFit`, and `fit_gamma_glm` is a thin wrapper over both. Keep
them separate — defining costs milliseconds and sampling costs seconds, and
the split is what makes prior predictive checks affordable. Arguments follow
the split: `alpha_prior_sigma` is a definition parameter, `draws`/`chains`/
`target_accept`/`random_seed` are sampling parameters.
`sample_gamma_glm` returns a `GammaGLMFit` value carrying posterior draws, the
term-name contract, the `Factor` declarations, and a `ConvergenceReport`.
`uncertainty.py` consumes that value to predict mu at unobserved query
points and split epistemic from aleatoric variance; it should refuse or
loudly warn on a fit whose `convergence.converged` is `False` rather than
re-deriving the check (`glm.check_convergence` is public for this).

## Design principles (don't violate these without discussion)

- **Agent-native, not human-native-with-an-API-bolted-on.** The MCP tools
  are the primary interface. A human can call the same functions through
  chat or a thin script, but nothing should require a GUI to operate.
- **Stateless functions + a persisted state store**, not a stateful
  long-running app. Any tool call should be resumable — an agent picking
  up mid-loop after a restart should be able to reconstruct exactly where
  things stand from the state store alone.
- **Physical execution is always gated.** A proposed batch is written with
  `status: proposed`. Nothing in this repo transitions a batch to `approved`
  automatically. That transition is a deliberate human checkpoint (however
  lightweight) before any real reagents/time are committed. Do not build a
  path that skips this, even for a "just for testing" convenience — write a
  simulator instead (see `demo/`).
- **Every batch is a first-class object**, carrying a batch ID from
  proposal through results. This is what makes the batch-effect random
  intercept model possible after the fact — don't flatten batches into a
  single long dataframe without preserving batch membership.
- **Uncertainty fed to acquisition functions must be marginal, not just
  fixed-effect.** Once `batch_effects.py` is in play, `get_uncertainty_map`
  and the acquisition functions must integrate over the batch random
  effect's estimated variance (a new batch hasn't been observed yet, so its
  offset is itself uncertain) — not just the coefficient posterior. Using
  fixed-effect-only uncertainty here is a silent correctness bug, not a
  simplification.

## Modeling conventions

- Default family/link: **Gamma with log link** for strictly positive
  biomass with no true zeros. Don't default to OLS or to a log-transform +
  OLS workaround — see `diagnostics.py` mean-variance check for why.
- Confirm family choice per-dataset via the mean-variance diagnostic
  (`diagnostics.mean_variance_slope`) rather than assuming Gamma always
  applies — if the estimated power is far from 2, flag it rather than
  silently proceeding.
- Posterior inference is **full MCMC via PyMC (NUTS)** by default. Dataset
  sizes in this domain (tens of design points per round, not millions) make
  the compute cost of real sampling negligible, so there's no good reason
  to settle for an approximation — use the actual posterior, not a
  normal/Laplace stand-in. `glm.py` returns genuine posterior draws over
  the GLM coefficients, and `uncertainty.py` turns those into draws over mu
  at any query point — sampled from the model, not derived from
  `cov_params()`. A
  Laplace/normal approximation on this dataset underestimated epistemic
  uncertainty at the edge of the design space relative to the full
  posterior (90% CI width on mu ~15 vs. ~72 at the same grid point), which
  changed the UCB-recommended next point — so treat the approximation as
  unreliable for feeding acquisition functions, not just slower to obtain.
  A Laplace/normal approximation may still be useful as a **fast sanity
  check or a fallback when PyMC/sampling isn't available in the runtime**,
  but it is not the default and must be clearly labeled as an
  approximation wherever it's used instead.
- **Encode predictors to coded units before fitting, always.** Raw-scale
  media concentrations (e.g. 0-10) make a quadratic linear predictor badly
  collinear — glucose and glucose² are nearly perfectly correlated over a
  bounded positive range — which wrecks NUTS's sampling geometry. This is
  not a theoretical concern: fitting on raw scale on the BB seed dataset
  produced 2500+ divergences and r_hat up to ~3.4 (complete
  non-convergence); centering first, with no other changes, gave 0
  divergences and max r_hat 1.002 on the same data.
- **The encoding is declared, not empirical.** `Factor.encode` maps each
  value to `[-1, +1]` against the factor's declared range; it does not
  z-score against the observed data. Two reasons, both load-bearing:

  1. Empirical mean/sd drift as a campaign concentrates near an optimum
     (glucose went 3.16/4.29 in round 1 to 2.30/3.30 after round 2 in a
     simulated campaign). That makes coefficients incomparable across model
     versions and — worse — silently turns the `Normal(-0.5, 0.5)` quadratic
     prior into a different prior every round.
  2. It destroys the balance of a `scale="log"` factor. Levels 0.1/1/10
     z-score to -0.658/-0.493/1.151, crushing two of three levels together,
     so curvature is estimated from almost nothing. Log-space encoding
     restores -1/0/+1. Measured effect on a simulated fit: ESS 1113 → 3740,
     mean coefficient error halved. Both parameterizations converge cleanly,
     so **the convergence gate cannot catch this** — it is an estimation
     bug, not a sampling one.

  Because `engine` can't import `state`, the division of labor is: `engine`
  returns the `Factor` tuple on the fit, `state/` persists it alongside the
  model version. Query grids must be encoded through **the fit's own
  factors**, never re-derived.

  Note the ordering: predictors are encoded *first*, then the squared and
  interaction terms are formed from the coded values. Squaring raw values and
  encoding afterward is not the same thing and reintroduces the
  collinearity.
- **Quadratic-term priors are centered negative** (e.g. `Normal(-0.5, 0.5)`
  on the standardized scale), reflecting the working assumption that
  growth response surfaces are peaked (diminishing then declining returns)
  rather than monotonic or explosive. This is a soft prior, not a hard
  concavity constraint — the data can and should override it if the
  evidence doesn't support a peak. Linear and interaction terms get
  weakly-informative zero-centered priors (`Normal(0, 1)` / `Normal(0, 0.5)`
  on the coded scale); the Gamma shape parameter (`alpha`) gets a
  weak `HalfNormal`.
- **Two priors that aren't arbitrary, and why.** The log-link intercept is
  `Normal(log(mean(y)), 2)` — it sits on whatever scale the response uses
  (biomass could be 0.5 or 5000), so a fixed zero-centered prior is
  informative-and-wrong for most datasets. `alpha` is `HalfNormal(10)` by
  default and is **exposed as `alpha_prior_sigma`** because it is weakly
  identified at these dataset sizes and its prior materially moves the
  posterior — measured on a 27-point fixture against a truth of `alpha=20`:
  `HalfNormal(5)` → 12.3, `HalfNormal(10)` → 19.7, `HalfNormal(25)` → 29.3,
  `HalfNormal(50)` → 33.0. This is not cosmetic: `alpha` *is* the aleatoric
  variance (`Var(y) = mu²/alpha`), so overestimating it understates noise
  and makes acquisition under-weight replication. Prefer erring low
  (more assumed noise) until real assay CVs are known.
- **Sampling environment:** don't assume multiple CPU cores are available.
  PyMC's automatic core allocation divides by `cores`, which raises a
  `ZeroDivisionError` on a single-core container/runtime — `glm.py` pins
  `cores=1` for this reason; it's a constraint, not a knob. Prefer
  `target_accept=0.95-0.99` and a higher `max_treedepth` as a first
  response to divergences before reparameterizing further; coded encoding
  (above) should already resolve the geometry for this model family.

  Sampling uses `nuts_sampler="nutpie"`, which compiles the log-density
  with numba. On macOS 26+ every fit **requires `PYTENSOR_FLAGS='cxx='`**
  or it dies with `ld: library 'd64' not found` — PyTensor hardcodes a
  `-ld64` linker flag with no config switch, and the current linker reads
  it as a request to link a library named `d64`. Disabling the C backend
  costs ~nothing here because nutpie never used it for the heavy work.
  Note `ldflags=` is *not* a PyTensor flag and does not fix this; it
  silently falls back to the Python VM and looks like it worked.
- **Always check convergence before trusting a fit.** The gate is strict
  0 divergences, max r_hat < 1.01, and min ESS >= 1000 — **bulk and tail**.
  Tail ESS is gated because acquisition is Monte Carlo over draws and reads
  the tails of mu's posterior; bulk ESS can look healthy while the 95th
  percentile still jitters and the recommended next point moves for no
  reason but sampler noise. Max-treedepth hits are reported but not gated
  (biased exploration, not a broken chain).

  `glm.fit_gamma_glm` surfaces this as a `ConvergenceReport` alongside the
  draws and raises `ConvergenceWarning` on failure — it does **not** raise,
  because the diverged draws are exactly what you need to diagnose the fit.
  A caller (including an agent driving the AL loop) must be able to tell a
  trustworthy fit from a broken one without re-deriving the check.
- Quadratic response-surface terms (linear + squared + pairwise
  interactions) are the default functional form for the linear predictor,
  matching Box-Behnken-style designs. Don't add higher-order terms without
  a design that can identify them. Term order is a **contract**
  (`build_design_matrix`): all linear terms in the caller's predictor
  order, then all squared terms in the same order, then all pairwise
  interactions in `itertools.combinations` order. Anything rebuilding a
  query row must reproduce that order exactly — getting it wrong produces
  a confidently wrong surface with no error.
- **A `Factor` is the unit of experimental design**: name, low/high range,
  units, and `scale` (`"linear"` or `"log"`). The scale field is load-
  bearing, not decoration — media concentrations routinely span an order of
  magnitude, and a factor over 0.1–10 g/L has arithmetic center 5.05 but
  geometric center 1.0. Designs replicate their center runs, so getting this
  wrong spends real reagents on an uninformative point.
- **Seed designs are not first-class here, deliberately.** The seed round
  happens once per campaign and isn't part of the loop — the value of this
  repo is rounds 2..N — and a real campaign often arrives with historical
  data or a house design anyway. So `engine/` ingests a seed dataset rather
  than generating one: no design-generator module, and no DOE dependency.
  Building one ad hoc is two lines (`pyDOE3.bbdesign` plus `Factor.decode`,
  recipe in the `factors.py` docstring); if `demo/` later needs one, add the
  dependency there rather than here. `Factor.decode` exists so that mapping
  puts center points in the right place under log scale.
- Acquisition functions are computed by **Monte Carlo directly over
  posterior draws**, not closed-form Gaussian EI/PI formulas — mu's
  posterior is lognormal-shaped under the log link, not normal, so the
  closed-form Gaussian formulas don't apply cleanly.

## Reporting

`reports/` content is generated by `get_diagnostics` / round-completion
hooks, not hand-edited. Each round should produce: the diagnostics grid,
the epistemic uncertainty map, the acquisition surface, and a short
plain-language summary (a paragraph, not a dashboard) of what changed and
what's recommended next. These are meant to be pushed to wherever people
already look (chat, doc, email) — don't build a page someone has to
remember to check.

## Testing

- `engine/` functions get unit tests against simulated data with a known
  ground truth (reuse the simulation approach from the exploratory work:
  simulate a peaked Gamma/log-link surface, fit, check recovered
  coefficients are close to truth).
- `mcp_server/` tools get tested against the state store with a fixture
  loop (a few rounds of proposed → approved → ingested) to confirm
  resumability — kill the "loop" mid-round and confirm state reconstructs
  correctly.
- `glm.py` needs a test that fits on the actual BB seed dataset (or an
  equivalent fixture) and asserts convergence diagnostics pass (0
  divergences, max r_hat < 1.01) — this is a regression test against the
  coded-encoding requirement above, since reverting to raw-scale
  predictors will silently break convergence without raising an error.
  Pair it with the negative control, which is what makes the test mean
  something: the same data with raw-scale quadratic terms gives ~2000
  divergences, r_hat ~1.53, and min ESS ~7, so all gates fire.
- `uncertainty.py` needs a test that a query grid encoded through the fit's
  own `Factor` tuple reproduces the training design matrix when handed back
  the training points — the cheapest guard against a term-order or
  re-encoding mismatch, both of which fail silently.
- Any new acquisition function needs a test confirming it: (a) never
  recommends a point outside the declared feasible region, (b) degrades
  gracefully (doesn't error) when posterior draws are unavailable yet
  (e.g. before the first model fit).

## What NOT to do

- Don't add a persistent dashboard/webapp as the primary interface. A
  thin generated-report view is fine; a maintained stateful UI is not
  the point of this repo.
- Don't let any component call out to real lab hardware or a real LIMS
  directly from `engine/` or `mcp_server/tools.py` without going through
  the `approved` gate in `state/`.
- Don't fit plain OLS as the "default" model anywhere in `engine/` —
  it exists only as an explicit comparison baseline in `diagnostics.py`
  for the negativity/heteroscedasticity demonstration.
- Don't add a seed-design generator back to `engine/` (see above), and
  don't reimplement the convergence check — `glm.check_convergence` is
  public precisely so `uncertainty.py` and `diagnostics.py` don't.
- Don't reach for RL. This is a static-function, expensive-evaluation,
  no-persistent-state optimization problem — Bayesian optimization is the
  right tool. If a future extension genuinely introduces sequential state
  (e.g. real-time feed-rate control during a fermentation run), that's a
  different repo, not a mode of this one.

## Commands

The project is uv-managed (`uv sync`), with a src layout built by hatchling.
**Every command that samples needs `PYTENSOR_FLAGS='cxx='` on macOS 26+** —
see the sampling-environment note above. Set it once per machine in
`~/.pytensorrc` to avoid prefixing every invocation.

```bash
uv sync                                              # install, incl. the package itself
PYTENSOR_FLAGS='cxx=' uv run pytest                  # not yet scaffolded — no tests, no pytest dep
PYTENSOR_FLAGS='cxx=' uv run python -m demo.simulate_loop      # not yet written
PYTENSOR_FLAGS='cxx=' uv run python -m malt.mcp_server.server  # not yet written
```

Current state: only `engine/glm.py` exists. A fit of a 27-point, 9-term
Gamma GLM takes ~2s, so verification is cheap — prefer actually running a
fit over reasoning about whether one would work.