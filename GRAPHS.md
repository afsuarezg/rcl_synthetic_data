# Graphs reference

What every figure produced by `plot_specs.py` is *for* — what question it answers,
what's on each axis, and how to read it.

Run it with:

```bash
uv run python plot_specs.py
uv run python plot_specs.py --root output/multiple_specs --basin-threshold 2.0
```

## Output locations

| Scope | Path |
| --- | --- |
| Per-(seed, iv) figures (01–15, 20–35) | `output/multiple_specs/seed_<X>/iv_<Y>/graphs/*.png` |
| Cross-seed figures (16–19, only when ≥2 seeds exist for an iv-mode) | `output/multiple_specs/graphs/*.png` (or `.../graphs/iv_<Y>/` when multiple iv-modes) |

Figures are numbered, not sequential by file order: **01–15** are single-run diagnostics,
**16–19** are cross-seed, **20–35** are elasticity / post-estimation (rendered only when
`compute_elasticities.py` has produced the elasticity CSVs). Numbers 29 and 32 are intentionally
unused.

## Shared conventions

- **Spec labels** are abbreviated by `_abbrev()`, e.g. `x2-x1_x2_x3__demos-income_age_hh_size_education` → `x123 | inc,age,hh,edu`.
- **"Perturbed start"** = an optimizer restart from a randomly jittered initial point. **"Truth-warm reference"** = a start initialized at the known ground-truth parameters; it's a best-case yardstick, never counted as a real estimate.
- **"Best per spec"** = the perturbed start with the lowest GMM objective for that spec.
- **"Best perturbed start"** = the random restart with the **lowest (best) GMM objective** for that spec, among those that converged, **excluding** the truth-warm start. The truth-warm start is held out because it has near-zero recovery error by construction — handing the optimizer the answer — so it would dominate any multistart comparison. The best perturbed start is therefore the estimate you'd report *in practice*, where the objective is the only selection criterion available without knowing the truth. (If no start converged, it falls back to the lowest objective overall.)
- **Parameter groups** are color-coded throughout: `beta`, `sigma`, `pi`, `gamma`.
- The synthetic market has 10 products in 5 firms (`FIRM_PATTERN = 1,1,2,2,3,3,4,4,5,5`), which is why cross-elasticity heatmaps draw firm-block lines at 2/4/6/8.

---

## Single-run diagnostics (01–15)

These read one `(seed, iv_mode)` long-format CSV and answer: *did the optimizer behave, and
did this spec recover the truth?*

### 01 — Objective ranking across specifications
`01_objective_ranking.png` · horizontal bars
Best (lowest) GMM objective per spec, sorted, with the truth-warm reference marked as a tick.
**Reads as:** which specifications fit the data best, and how close the best perturbed start
gets to the truth-warm baseline.

### 02 — Multistart objective spread per spec
`02_multistart_stability.png` · stripplot, sorted by spread
Every perturbed start's objective, one row per spec, ordered by max−min spread.
**Reads as:** optimizer reliability. A tight cluster = the spec converges to the same optimum
regardless of start; a wide spread = the surface is multimodal and the answer depends on luck.

### 03 — Convergence audit
`03_convergence_audit.png` · stacked bars (converged vs non-converged), failing specs first
Count of converged vs failed perturbed starts per spec; title reports the global failure tally.
**Reads as:** which specs are numerically fragile and how many starts are wasted.

### 04 — Distance from global minimum
`04_global_minimum.png` · horizontal bars
For each spec, (its best converged objective) − (the single global-best objective across all specs).
**Reads as:** how far each spec's best fit is from the overall winner. Zero = this spec *is* the
global best.

### 05 — Two-basin classification
`05_two_basin_analysis.png` · stacked histogram
All converged perturbed-start objectives, split into Basin A (within `--basin-threshold` of the
global best) and Basin B (beyond it). Vertical lines mark the global best and the threshold.
**Reads as:** is the objective landscape one good basin plus noise, or two genuinely separate
attractors? Tune the split with `--basin-threshold` (default 2.0).

### 06 — Per-start wall-clock runtime
`06_runtime.png` · boxplot, sorted by median time
Elapsed seconds per start, per spec.
**Reads as:** which specs are expensive; useful for budgeting `--n-starts` and SLURM walltime.

### 07 — Price coefficient across perturbed starts
`07_price_coef.png` · stripplot
The `beta_1` (price coefficient) estimate from every perturbed start, per spec, with the truth
line.
**Reads as:** stability and bias of the single most economically important parameter.

### 08 — Recovery RMSE stacked by parameter group
`08_recovery_rmse_by_group.png` · stacked horizontal bars
For each spec's best start, RMSE of `abs_error` vs truth, decomposed into beta/sigma/pi/gamma
contributions.
**Reads as:** which specs recover the truth best overall, and *which kind* of parameter drives the
remaining error.

### 09 — Recovery vs. objective
`09_recovery_vs_objective.png` · scatter + OLS line
Best-start GMM objective (x) vs total parameter RMSE (y) per spec; title shows Pearson & Spearman.
**Reads as:** the key identification check — does a better in-sample fit (lower objective) actually
mean better recovery of true parameters? A weak/negative correlation warns that the objective is
not a reliable guide to truth.

### 10 — Per-parameter recovery error
`10_param_level_error.png` · horizontal bars with ±std whiskers
Mean |error| for each physical parameter, averaged over the specs that include it, sorted worst-first.
**Reads as:** which individual parameters are hard to pin down regardless of spec.

### 11 — Phantom pi interactions (truth = 0)
`11_pi_zero_false_positive.png` · boxplot
For demographic-interaction (`pi`) terms whose true value is 0, the distribution of |estimate|.
**Reads as:** false-positive rate — how often the model invents demographic interactions that
don't exist. Larger boxes = more spurious structure.

### 12 — Recovery MAE by X2 inclusion pattern
`12_omitted_x2_bias.png` · grouped bars
Mean |error| by parameter group, grouped by which X2 (random-coefficient) variables the spec
includes.
**Reads as:** the cost of omitting (or adding) random coefficients — i.e., omitted-variable bias on
the surviving parameters.

### 13 — Recovery error vs. # demographic vars
`13_demo_overfit.png` · line plot
Mean |error| per parameter group as a function of how many demographic variables the spec carries,
plus a dashed line for the truth-zero `pi` terms only.
**Reads as:** overfitting check — does piling on demographic controls reduce error, or just inflate
phantom interactions?

### 14 — Within-spec estimate std across perturbed starts
`14_param_stability_within_spec.png` · heatmap (specs × parameters)
Std of each parameter's estimate across perturbed starts.
**Reads as:** a fingerprint of estimation stability — dark cells flag (spec, parameter) pairs that
are poorly identified / start-dependent.

### 15 — RMSE: best-by-objective vs best-by-RMSE
`15_best_vs_truth_start.png` · scatter vs 45° line
Per spec: lowest RMSE achievable among starts (x) vs RMSE at the start the objective *actually*
selects (y). Color marks whether the two agree; title counts matches.
**Reads as:** does picking the lowest-objective start also pick the most-accurate one? Points above
the diagonal mean the objective chose a worse-recovering optimum than was available.

---

## Cross-seed diagnostics (16–19)

Rendered only when an iv-mode has ≥2 seeds. These separate *sampling noise* from *systematic bias*
by pooling across the synthetic datasets.

### 16 — Mean bias across seeds
`16_estimate_across_seeds.png` · heatmap (specs × parameters), red/blue diverging
Mean (estimate − truth) per parameter, averaged over seeds.
**Reads as:** systematic bias that survives averaging — blue/red cells are parameters consistently
under/over-estimated, not just noisy.

### 17 — Recovery RMSE per spec, across seeds
`17_recovery_across_seeds.png` · boxplot
Distribution of per-spec total RMSE over seeds.
**Reads as:** which specs recover the truth *reliably* (low and tight) vs only sometimes.

### 18 — Best-spec consistency across seeds
`18_best_spec_consistency.png` · horizontal bars
Count of seeds in which each spec was the global-best (lowest objective).
**Reads as:** is there a single robust winning specification, or does the winner flip with the
draw?

### 19 — Price coefficient across seeds
`19_price_coef_across_seeds.png` · stripplot
Within-seed mean `beta_1` per spec, one point per seed, with the truth line.
**Reads as:** cross-seed stability and bias of the price coefficient.

---

## Elasticity & post-estimation (20–35)

Rendered only when elasticity CSVs exist (run `compute_elasticities.py` first). These move from
parameters to economically meaningful quantities — own/cross-price elasticities and merger
predictions. Several focus on the **rank-1 spec** (the best-objective spec) or a representative
best-mean-objective spec.

### 20 — Own-price elasticity distribution per spec
`20_elasticity_own_summary.png` · boxplot, sorted by GMM objective
Spread of own-price elasticities (best start) per spec, with the truth mean line.
**Reads as:** do better-fitting specs produce more sensible / accurate own-price elasticities?

### 21 — Own-price elasticity stability across perturbed starts
`21_elasticity_multistart_stability.png` · stripplot, top-10 most unstable specs
Per-product own-elasticity across perturbed starts, for the 10 specs with the widest spread.
**Reads as:** start-dependence of the elasticities that matter for downstream policy — these are
the specs where the elasticity you report depends on the optimizer's luck.

### 22 — Cross-elasticity matrix (rank-1 spec)
`22_elasticity_top_substitutes.png` · 10×10 annotated heatmap, firm blocks marked
Full product-by-product elasticity matrix for the best spec's best start.
**Reads as:** the substitution structure — diagonal = own-price, off-diagonal = who steals from
whom; firm-block lines show within- vs between-firm substitution.

### 23 — Cross-elasticity asymmetry (rank-1 spec)
`23_elasticity_asymmetry.png` · scatter vs 45° line
e_jk (x) vs e_kj (y) for every unordered product pair.
**Reads as:** how asymmetric substitution is. Points off the diagonal are pairs where j responds to
k very differently than k responds to j.

### 24 — Spearman correlation of own-price elasticities across specs
`24_elasticity_spec_spearman.png` · heatmap (specs × specs)
Pairwise rank correlation of the per-product own-elasticity vector between specs.
**Reads as:** do different specifications agree on the *ordering* of products by elasticity, even if
levels differ? Clusters = specs that tell the same substitution story.

### 25 — Within/between-firm mean cross-elasticity (rank-1 spec)
`25_elasticity_firm_substitution.png` · 5×5 annotated heatmap
Mean cross-elasticity aggregated to firm × firm.
**Reads as:** firm-level substitution — the diagonal (within-firm) vs off-diagonal (between-firm)
is exactly what drives merger price effects.

### 26 — Own-price elasticity per product across specs
`26_elasticity_own_cross_spec_stability.png` · stripplot, colored by firm
Each product's own-elasticity (best start) across all specs.
**Reads as:** which products have a stable elasticity regardless of spec vs which swing widely.

### 27 — Top-k substitute pair elasticities across specs
`27_elasticity_cross_cross_spec_stability.png` · stripplot (default k=5)
For the rank-1 spec's strongest substitute pairs, those same pairs' cross-elasticities across all
specs.
**Reads as:** are the headline substitution relationships robust across specs, or an artifact of one?

### 28 — Pairwise MAD of own-price elasticities between specs
`28_elasticity_pairwise_mad.png` · heatmap (specs × specs)
Mean absolute difference of own-elasticity vectors between every pair of specs, ordered so the
most "central" specs sit first.
**Reads as:** how much specs disagree on elasticity *levels* (complements 24's rank view); dark =
far apart.

### 30 — Pair (j,k) across perturbed starts
`30_elasticity_pair_across_sims.png` · 4 panels (e_jj, e_kk, e_jk, e_kj)
For the best-mean-objective spec, the four elasticities of products 0 and 1 across starts, with
truth lines.
**Reads as:** within-spec start-to-start stability of a concrete elasticity quartet.

### 31 — Pair (j,k) across specs
`31_elasticity_pair_best_sim_across_specs.png` · 4 panels vs GMM objective
The same four elasticities (best start each) plotted against the spec's objective, with truth lines.
**Reads as:** does a lower objective drive these specific elasticities toward truth across specs?

### 33 — Own-elasticity recovery vs. truth
`33_elasticity_recovery_own.png` · horizontal bars, colored by bias sign
Per-spec RMSE of own-price elasticity vs truth; bar color encodes the sign of the mean bias.
**Reads as:** which specs recover own-price elasticities accurately, and in which direction they err.

### 34 — Cross-elasticity recovery: same-firm vs different-firm
`34_elasticity_recovery_cross.png` · grouped horizontal bars
Per-spec MAE on cross-elasticities, split into same-firm and different-firm pairs.
**Reads as:** where cross-elasticity error concentrates — same-firm error is especially relevant
because it governs unilateral merger price effects.

### 35 — Post-estimation recovery vs. parameter recovery
`35_post_estimation_recovery.png` · 3-panel scatter
Parameter total RMSE (x) vs error in three downstream outputs (y): mean own-elasticity, mean
markup, and merger Δ-HHI.
**Reads as:** the bottom line — does better parameter recovery translate into better economic
predictions (especially the merger Δ-HHI), or do the policy-relevant outputs go wrong even when
parameters look fine?
