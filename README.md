# BLP Synthetic Data

Generate and recover a random coefficients logit (BLP) dataset, end to end:

- `simulate.py` builds product-market data with equilibrium Bertrand–Nash
  prices implied by a known parameter vector.
- `estimate.py` reloads the dataset into `pyblp.Problem` and runs a
  multistart round-trip estimation — recovery near truth is the sanity
  check that the synthetic data is "BLP-shaped".

## Model

Indirect utility of consumer *i* for product *j* in market *t*:

```
U_ijt  = δ_jt + μ_ijt + ε_ijt           ε_ijt ~ Type-I EV
δ_jt   = α · price_jt + β'·x_jt + ξ_jt
μ_ijt  = X2_jt · (Σ ν_i + Π d_i)
```

Supply (multi-product Bertrand FOC):

```
p = c + Δ(p)⁻¹ s(p)        c = γ' w_jt + ω_jt
```

with ownership matrix `H` and `Δ = − H ⊙ ∂s/∂p'`. Equilibrium prices are
solved by `pyblp.Simulation.replace_endogenous()`.

The default DGP: T=200 markets × J=10 products × 5 firms, 5 demand
characteristics (3 carry random coefficients alongside the constant and
price), 2 cost shifters, and 4 demographics (income, age, household
size, education). Identification leans on BLP rivals-sum instruments
together with Gandhi–Houde differentiation instruments.

## Install

```
pip install -r requirements.txt
```

## Run

```
python simulate.py                       # default: T=200, J=10, seed=0
python estimate.py                       # reads output/seed_0/

python simulate.py --seed 1
python estimate.py --output-dir output/seed_1

python simulate.py --T 20 --J 5 --seed 42      # small smoke test
python estimate.py --output-dir output/seed_42 --n-starts 3
```

Each `simulate.py` run writes `product_data.csv`, `agent_data.csv`, and
`truth.pkl` into `output/seed_{seed}/`. `estimate.py` runs `--n-starts`
optimizer starts (start 0 = truth, the rest perturb truth by ~50 % of
its magnitude) and keeps the lowest-objective fit.

## Known gotchas

- **Pin `numpy < 2.0`.** With numpy 2.0 + scipy 1.13, `scipy.linalg.pinv`
  emits spurious divide-by-zero warnings inside matmul which pyblp
  treats as 2SLS failures. The `requirements.txt` here pins to
  `numpy>=1.26,<2.0` and `scipy<1.14`.
- **`error_behavior='warn'` is intentional.** Pyblp's default is to
  raise on the first transient numpy overflow inside the ζ-contraction
  even when the fixed point recovers; `simulate.py` switches to `warn`
  and instead gates on `sim_results.fp_converged.all()`.
- **Demographics must be standardized.** Π·d on raw scales (e.g. age
  with SD = 5) blows up the random-coefficient contribution for tail
  agents, saturating shares and breaking the FOC. `simulate.py`
  centers/scales `age` and `hh_size`; if you add demographics, do the
  same.
- **Marginal cost must stay positive.** Cost shifters `w1, w2` are
  drawn from `U(0,1)` (not `N(0,1)`) so that `c = γ'X3 + ω` is
  comfortably above zero for every product — `c < 0` lets the firm set
  `p < 0` and the contraction diverges.
- **Drop the constant from IV formulations.** `build_blp_instruments`
  and `build_differentiation_instruments` return sums of rivals' /
  other-firm characteristics; combining them with an explicit constant
  gives a collinear matrix.
