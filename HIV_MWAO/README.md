# HIV MWAO — Mode-Wise Adaptive Observer for the switched HIV mutation model

HIV case study: a Mode-Wise Adaptive Observer (MWAO) for the 4-genotype HIV mutation model under SWATCH therapy switching.

Three therapy modes drive the switching signal `σ(t)`:

| σ | mode | shading |
|---|---|---|
| 0 | no therapy | green |
| 1 | drug therapy 1 | orange |
| 2 | drug therapy 2 | blue |

Each mode gets its own **full 14-state adaptive observer** `Π_ℓ`, estimating
`θ_ℓ = [K_T^1, K_T^2, K_T^3, K_T^4]_ℓ` for that mode. At any instant only
`Π_σ(t)` integrates; the other two **freeze**, so a therapy switch never injects
one mode's parameter error into another.

- State vector (14): `[T*_i, M*_i, V_i]` for `i = 1..4`, plus `T` and `M`.
- Measurements (5): `y = [T, V₁, V₂, V₃, V₄]`.
- E-scaling: `zhat = xhat/E`, `yhat = y/E`.
- Timing: plant step `dt = 1` d, observer sampling `Ts = 1` d, observer starts at
  `T_obs = 3` yr, therapy starts at `T_cont = 4` yr, final time `t_f = 6` yr,
  SWATCH switching period `dts = 1` yr.
- Integration: `scipy.solve_ivp` (RK45) for both plant and observers.

## Results

Parameter RMSLE at the end of the run:

| | K_T^1 | K_T^2 | K_T^3 | K_T^4 |
|---|---|---|---|---|
| RMSLE | 1.65e-07 | 2.66e-07 | 2.48e-07 | 2.76e-07 |

Excitation / dwell-time diagnostics, all read off the **single global threshold**
`δ = 4.52e-05`:

| mode | excited element | λ_min(N_ℓ) at AT | T_min,ℓ | AT_ℓ |
|---|---|---|---|---|
| 0 (no therapy) | K_T^1 | 1.16e-04 | 137 d | 366 d |
| 1 (therapy 1) | K_T^2 | 9.03e-05 | 189 d | 365 d |
| 2 (therapy 2) | K_T^4 | 9.70e-05 | 182 d | 365 d |

Each therapy excites essentially one genotype's parameter — the one it selects
for — so `λ_min` is taken over the excited subspace (`REL_TOL = 1e-2`), and the
dwell-time condition `AT_ℓ ≥ T_min,ℓ` holds in every mode with roughly a 2×
margin.

## Files

- `hiv_mw_ao.py` — the simulation: plant + three mode observers, `run()`, and
  `save_data()` / `load_data()`.
- `params.py` — HIV model parameters and dimensions.
- `hiv_plant.py` — the 14-state HIV mutation model right-hand side.
- `mw_ao_rhs.py` — mode-wise adaptive observer right-hand side.
- `combined_rhs.py` — stacks plant + active observer for a single `solve_ivp` call.
- `get_drug_eff.py` — per-genotype drug efficacy of each therapy.
- `get_treatment.py` — the SWATCH switching signal `σ(t)`.
- `hiv_figures.py` — all three figures, plus the PE diagnostics `compute_pe()`.
- `hiv_data.npz` — the saved `run()` result (~1.2 MB).
- `hiv_figures/` — generated figures (PDF + PNG; the PE figure also writes EPS).
- `requirements.txt` — numpy, scipy, matplotlib.

## Reproducing

Simulation and plotting are separated, so restyling a figure never re-runs the
integrator — and never silently plots a *different* run.

```
python hiv_mw_ao.py     # simulate once (~13 s) -> writes hiv_data.npz
python hiv_figures.py   # plot          (~10 s) -> writes hiv_figures/
```

`hiv_figures.py` loads `hiv_data.npz` when it exists and otherwise simulates
once and saves it, so a bare `python hiv_figures.py` works from a fresh clone.
Delete `hiv_data.npz`, or re-run `hiv_mw_ao.py`, after changing the model or its
tunings.

From a session or notebook:

```python
import hiv_mw_ao as sim, hiv_figures as F

res = sim.load_data()          # or: res = sim.run(); sim.save_data(res)
pe = F.compute_pe(res)         # PE / dwell-time diagnostics per mode

F.make_stacked_figure(res); F.make_kt_figure(res); F.make_pe_figure(res)
```

## Figures

| file | content |
|---|---|
| `fig_states_stacked` | viral loads `V_i`, infected T cells `T*_i`, infected macrophages `M*_i`, per-observer error norm `Π₀ Π₁ Π₂`, and `σ(t)` |
| `fig_KT_convergence` | per-genotype `K_T^i` error contraction against the therapy bands |
| `fig_PE` | per-mode excitation `N_{ℓ,ii}`, `λ_min`, threshold `δ`, dwell time `T_min,ℓ` |



## Contact

- Juan Sereno — jeserenom@gmail.com
- E. A. Hernandez-Vargas — esteban@uidaho.edu

## Cite

> J. Sereno and E. A. Hernandez-Vargas, "A Mode-wise Adaptive Observer for
> Autonomous Switched Nonlinear Systems," arXiv:2608.18251, 2026.
