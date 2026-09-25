# CLAUDE.md

Guidance for Claude (or any agent) working in this repository.

## Project name

MALT (Media Active-Learning Toolkit). Package name `malt`, namespaced as
`malt.engine`, `malt.active_learning`, `malt.benchmark` (and, later,
`malt.state`, `malt.mcp_server`).

## What this project is

An autonomous experimental-design engine for media/process optimization
(biomass growth vs. media composition — e.g. glucose/nitrogen/phosphate
concentrations). It fits a Bayesian GLM to noisy, strictly-positive,
heteroscedastic growth data, quantifies epistemic uncertainty via full
MCMC, and uses a Bayesian-optimization acquisition function to recommend
the next batch of experiments. The interface is a set of tools an agent
calls directly (MCP), with a human approval gate before any physical
execution — not a dashboard or CLI a human has to drive turn-by-turn.

## Current state

Built:

- `engine/factors.py`, `engine/glm.py` — the `Factor` declaration and the
  Gamma GLM (definition, sampling, convergence report).
- `active_learning/` — the loop's abstractions, the loop itself, the lab
  journal, and termination rules. **Nothing implements the actor ABCs yet**
  except test dummies and the simulated oracle.
- `benchmark/oracle.py` — a simulated environment with a known ground truth,
  including `gp_sampled_latent` (a fixed RBF-GP surface via random Fourier features).
- `benchmark/surrogates.py` — baseline surrogates: Bayesian linear regression
  (linear or quadratic features, reference prior) and `GammaGLMSurrogate`
  (any `terms` subset; the default is the main quadratic model).
- `tests/active_learning/` — 50 tests of loop-level properties, using dummy
  actors (no PyMC).

Next, in order: a candidate grid in `factors.py`; a conjugate Bayesian linear
regression surrogate and Thompson sampling, with a `y = x1 + x2` demo as the
first end-to-end run; then the Gamma GLM surrogate, Monte Carlo UCB/EI,
fixed-design/RSM acquisitions, and an OLS baseline for benchmarking BO against
iterative DOE. `uncertainty.py`, `batch_effects.py`, `state/` and `mcp_server/`
are not started.

Known debts: `oracle.quadratic_latent`
peaks at the raw-unit origin with no linear or cross terms, so its optimum sits
in a corner; `glm.py:224,232` have 4 pyright errors from xarray's loose
`DataTree.__getitem__` typing (fix: `.to_dataset()` on `sample_stats`, not yet
applied); `benchmark/loop.py` is an empty placeholder.

## Architecture

```
src/malt/
  engine/            # the modeling core — no I/O, no orchestration, pure functions
    factors.py       # Factor (name/range/units/scale) — the feasible region
    glm.py           # design matrix, model definition, NUTS sampling, convergence report
    uncertainty.py   # (planned) posterior -> mu at query points, epistemic/aleatoric split
    acquisition.py   # (planned) MC UCB/EI/PI scores over posterior draws
    batch_effects.py # (planned) random-intercept batch model, marginal predictive draws

  active_learning/   # the loop: abstractions + orchestration, no modeling maths
    actors.py        # SurrogateModel, Acquisition, Environment ABCs; Experimenter
    loop.py          # loop_step, run_loop, LabJournal, Seed, RoundEntry
    termination.py   # TerminationRule ABC, composition, and the rule library
    surrogates.py    # (planned) SurrogateModel implementations wrapping engine/
    acquisitions.py  # (planned) Acquisition implementations wrapping engine/

  benchmark/         # simulation and evaluation — test infrastructure, not product
    oracle.py        # Oracle(Environment): latent surface + likelihood
    surrogates.py    # baseline SurrogateModels for the 2x2 likelihood x features ablation

  state/             # (planned) persistence — the source of truth for a real campaign
  mcp_server/        # (planned) agent-facing tools, thin wrappers over the above

tests/active_learning/  # conftest.py holds dummy actors; test_*.py hold the tests
demo/                   # (planned) simulated end-to-end loop
reports/                # generated, not hand-maintained — see "Reporting"
```

**Dependency direction is one-way**: `engine` imports nothing from the rest
of `malt`. `active_learning` imports `engine`. `benchmark` imports
`active_learning` (the oracle implements `Environment`). `state` and
`mcp_server` sit on top. Never import upward.

**Maths in `engine/`, loop classes in `active_learning/`.** A surrogate's
fitting code and an acquisition's scoring function are pure functions in
`engine/`, usable from a notebook with no loop. The classes that implement
the actor ABCs — holding data seen so far, advancing state, drawing seeds from
an `rng` — live in `active_learning/`.

`engine` functions are pure — data in, results out, no side effects. The one
sanctioned exception: `engine` may raise a `warnings.warn` to flag a result
the caller must not trust silently (see `glm.ConvergenceWarning`). A warning
mutates no state, touches no I/O, and is suppressible — and it's the lesser
evil against handing back an unmixed chain that looks like a normal return
value. Don't extend this to logging, file writes, or progress output.

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

## The active-learning loop

```
Experimenter = SurrogateModel + Acquisition  --propose x-->  Environment
             <------------------- observe (x, y) --------------
```

- **Actors are ABCs, not Protocols.** `SurrogateModel`, `Acquisition`,
  `Environment` (and `benchmark.oracle.Likelihood`) are abstract base classes
  with `__slots__ = ()`, so `slots=True` dataclass subclasses stay slotted.
  Implementations subclass explicitly. An ABC only checks that methods exist,
  not their signatures — pyright (`typeCheckingMode = "standard"`) catches an
  incompatible override such as a `query` missing `rng`.
- **Surrogates and acquisitions are immutable values.** `condition` returns a
  new model; `propose` returns `(x, successor)`. This is what lets the journal
  keep one snapshot per round by reference. `Experimenter` is the single
  mutable holder: `propose(n, rng)` swaps in the successor rule, `observe(data,
  rng)` swaps in the conditioned model.
- **`condition` must be associative**: `m.condition(d1).condition(d2)` equals
  `m.condition(concat(d1, d2))`. Models without a closed-form update (the MCMC
  GLM) meet this by keeping their prior config and all data seen, and refitting
  on everything. Don't "optimize" with a Laplace/normal carry-forward — see the
  modeling conventions below.
- **`sample(x)` returns joint draws of mu, shape `(n_draws, len(x))`, and row
  `s` is the same surface on every call and every `x`.** Thompson sampling takes
  a row's argmax across candidates, which is only meaningful if the row is one
  surface. A point-estimate model (OLS) is a legal surrogate with `n_draws == 1`;
  model-driven acquisitions must degrade gracefully on it, not error.
- **`reliable` means "the fit worked"**, e.g. passed the convergence gate — not
  "has uncertainty".
- **`Environment.query(x, rng)` returns a DataFrame**, one row per input row by
  position: a `y` column plus any covariates the environment knows (batch,
  operator, inoculum). The loop joins it to `x` **positionally**
  (`x.join(obs.set_axis(x.index))`), because proposals picked off a candidate
  grid carry non-contiguous index labels and an index-aligned join silently
  pairs y with the wrong x. `set_axis` raises on a row-count mismatch; `join`
  raises on a column clash with a factor.
- **Batch labels come from the environment, never the loop.** A batch is the
  classic DOE block — runs sharing session, operator, cell state — and only the
  lab knows it. A synthetic environment chooses its own convention. The
  surrogate decides which columns to group by. The loop's round UUID is the
  round's identity in the journal, not a batch label.
- **Randomness enters only through `run_loop(random_seed=...)`**, which spawns
  three independent streams (`default_rng(seed).spawn(3)`): experimenter,
  environment, round IDs. Never use numpy's global state or `uuid4()`. Never
  give two generators the same seed (identical sequences, correlated with each
  other). The environment's own stream is what makes benchmark arms paired: two
  experimenters with the same seed see identical observation noise however much
  randomness they consume.
- **The seed is not a round.** `run_loop` takes `seed_data`, conditions on it,
  and records it as `LabJournal.seed`. A benchmark gives every arm the same seed.
- **`LabJournal` is the campaign's complete record**: `random_seed`,
  `batch_size`, the prior, the starting acquisition, the `Seed`, the
  `termination_rule`, one `RoundEntry` per round (`id`, `data`, and the model
  and acquisition *after* the round, so the last entry is the current state),
  and `stopped_by`. Termination rules read it; the loop needs it by design.
- **Termination rules are pure functions of the journal**, subclass
  `TerminationRule`, and compose with `&`/`|` **only with other rules** —
  anything else is a `TypeError`, so every part of a criterion is a named,
  comparable, picklable object. Purity is load-bearing: `fired(journal)`
  re-evaluates the rule on the finished journal to fill `stopped_by`, which is
  only correct if the rule gives the same answer twice. A rule that needs
  randomness derives it from the journal (see `RandomStopRule`). `run_loop` takes
  a single rule, default `MaxRoundsRule(10)`; combine with `UnreliableFitRule()`
  so a run can't propose from a failed fit.
- **`run_loop` is the simulation path only.** It queries synchronously. A real
  campaign proposes, waits for approval and results, then observes — the same
  two `Experimenter` methods, days apart, driven by `state/` and the MCP tools.

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
  simulator instead (see `benchmark/`).
- **Every batch is a first-class object**, carrying a batch ID from
  proposal through results. This is what makes the batch-effect random
  intercept model possible after the fact — don't flatten batches into a
  single long dataframe without preserving batch membership.
- **Uncertainty fed to acquisition functions must be marginal, not just
  fixed-effect.** Once `batch_effects.py` is in play, `get_uncertainty_map`
  and the acquisition functions must integrate over the batch random
  effect's estimated variance for any batch level not yet observed (a new
  batch's offset is itself uncertain) — not just the coefficient posterior.
  For a level already seen and reused (e.g. a known operator), condition on
  its estimated offset instead. Using fixed-effect-only uncertainty for an
  unseen level is a silent correctness bug, not a simplification.

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
  `cov_params()`. A Laplace/normal approximation on this dataset
  underestimated epistemic uncertainty at the edge of the design space
  relative to the full posterior (90% CI width on mu ~15 vs. ~72 at the same
  grid point), which changed the UCB-recommended next point — so treat the
  approximation as unreliable for feeding acquisition functions, not just
  slower to obtain. A Laplace/normal approximation may still be useful as a
  **fast sanity check or a fallback when PyMC/sampling isn't available in
  the runtime**, but it is not the default and must be clearly labeled as an
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
  silently falls back to the Python VM and looks like it worked. (This is
  documented here only, not in the README.)
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
  trustworthy fit from a broken one without re-deriving the check. In the
  loop, the GLM surrogate exposes this as `reliable`, and
  `UnreliableFitRule` stops a campaign on it.
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

## Simulated environments

`benchmark.oracle.Oracle(latent, likelihood)` splits the truth in two:
`latent(x)` is the surface on the link scale (log biomass under a log link),
and a `Likelihood` maps it to a mean and draws around it. `GammaLikelihood`
uses the same parameterization as the GLM, so a recovery test isolates
inference from family mismatch — to simulate misspecification, vary the
latent, not the likelihood. `Oracle.query` returns `DataFrame({"y": ...})`;
`Oracle.mean` is the ground truth a benchmark scores against (regret is
measured on the true mean at the recommended point, never on the best
observed y, which rewards noise).

## Reporting

`reports/` content is generated by `get_diagnostics` / round-completion
hooks, not hand-edited. Each round should produce: the diagnostics grid,
the epistemic uncertainty map, the acquisition surface, and a short
plain-language summary (a paragraph, not a dashboard) of what changed and
what's recommended next. These are meant to be pushed to wherever people
already look (chat, doc, email) — don't build a page someone has to
remember to check.

## Testing

- **Loop tests use dummy actors** (`tests/active_learning/conftest.py`) so they
  run in under a second without PyMC. Each dummy exists to make one loop
  property observable; the tests target failures that would be *silent*:
  rows paired with the wrong results, dropped acquisition state, broken
  reproducibility, unpaired noise across arms, journal entries aliasing one
  mutable object, termination truth tables and `stopped_by`. When changing the
  loop, confirm a test fails if you reintroduce the bug it guards.
- **Contract tests for implementations** (planned): one parametrized suite run
  against every `SurrogateModel` and `Acquisition` — associativity of
  `condition`, stable joint draws from `sample`, `self` unchanged by
  `condition`, proposals inside the declared `Factor` ranges, graceful
  handling of single-draw models.
- `engine/` functions get unit tests against simulated data with a known
  ground truth (simulate a peaked Gamma/log-link surface, fit, check recovered
  coefficients are close to truth).
- `glm.py` needs a test that fits on the actual BB seed dataset (or an
  equivalent fixture) and asserts convergence diagnostics pass (0
  divergences, max r_hat < 1.01) — a regression test against the
  coded-encoding requirement above, since reverting to raw-scale predictors
  will silently break convergence without raising an error. Pair it with the
  negative control, which is what makes the test mean something: the same
  data with raw-scale quadratic terms gives ~2000 divergences, r_hat ~1.53,
  and min ESS ~7, so all gates fire.
- `uncertainty.py` needs a test that a query grid encoded through the fit's
  own `Factor` tuple reproduces the training design matrix when handed back
  the training points — the cheapest guard against a term-order or
  re-encoding mismatch, both of which fail silently.
- `mcp_server/` tools get tested against the state store with a fixture
  loop (a few rounds of proposed → approved → ingested) to confirm
  resumability — kill the "loop" mid-round and confirm state reconstructs
  correctly.
- In tests, narrow ABC-typed values to the dummy class with `isinstance`
  (the `narrow` helper) or `cast`, rather than relaxing pyright for `tests/`.

## What NOT to do

- Don't add a persistent dashboard/webapp as the primary interface. A
  thin generated-report view is fine; a maintained stateful UI is not
  the point of this repo.
- Don't let any component call out to real lab hardware or a real LIMS
  directly from `engine/`, `active_learning/` or `mcp_server/tools.py`
  without going through the `approved` gate in `state/`. `run_loop` is for
  simulated environments only.
- Don't fit plain OLS as the "default" model anywhere in `engine/` —
  it exists only as an explicit comparison baseline (the RSM benchmark arm,
  and the negativity/heteroscedasticity demonstration in `diagnostics.py`).
- Don't add a seed-design generator back to `engine/` (see above), and
  don't reimplement the convergence check — `glm.check_convergence` is
  public precisely so `uncertainty.py` and `diagnostics.py` don't.
- Don't draw randomness anywhere but from the `rng` a method is handed —
  no `np.random.*` global calls, no `uuid4()`, no unseeded generators.
- Don't compose termination rules with bare callables, and don't give a rule
  hidden state (counters, clocks) — it breaks `stopped_by`.
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
uv sync                                          # install, incl. the package itself
uv run pytest                                    # all tests; the one module that samples sets the flag itself
uv run --with pyright pyright src tests          # type check in the project env
PYTENSOR_FLAGS='cxx=' uv run python -m demo.simulate_loop      # not yet written
PYTENSOR_FLAGS='cxx=' uv run python -m malt.mcp_server.server  # not yet written
```

pyright skips directories whose names start with `.` — code placed there is
silently unchecked. A fit of a 27-point, 9-term Gamma GLM takes ~2s, so
verification is cheap — prefer actually running a fit over reasoning about
whether one would work.
