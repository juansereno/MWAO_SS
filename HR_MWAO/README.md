# A Mode-wise Adaptive Observer for Autonomous Switched Nonlinear Systems

This repository provides the reference implementation of the **mode-wise adaptive
observer (MWAO)** for autonomous switched nonlinear systems in state-affine form,
together with the code that reproduces the switched Hindmarsh–Rose (HR) neuron
case study of the paper.

The observer assigns a *dedicated* adaptive observer `Π_ℓ` to each mode `ℓ`, so
that a switch never injects the estimation error of one parameter realization
into another. Convergence rests on two verifiable conditions, both of which the
code evaluates directly from a simulation:

- a **finite-window persistence-of-excitation** condition on the accumulated
  information matrix `N_ℓ(t_k, t_k + T)`;
- a **minimum dwell-time** condition `AT_ℓ ≥ T_min,ℓ`, where `T_min,ℓ` is the
  first time the excitation crosses the threshold `δ`.

The example is a switched Hindmarsh–Rose neuron whose three firing regimes
correspond to three unknown parameter vectors `θ_ℓ = (a, b, d)`, with only the
membrane potential measured.

## Table of contents

- [Installation](#installation)
- [Usage](#usage)
- [What the figures show](#what-the-figures-show)
- [Contact](#contact)
- [Cite](#cite)

## Installation

Requires Python 3.9+ and three standard scientific packages:

```
pip install -r requirements.txt
```

(`numpy`, `scipy` for the integration, `matplotlib` for the figures.)

## Usage

Simulation and plotting are deliberately separated, so restyling a figure never
re-runs the integrator — and never silently plots a *different* run.

```
python hr_mw_ao.py      # simulate once (~40 s) -> writes hr_data.npz
python hr_figures.py    # plot          (~10 s) -> writes hr_figures/*.{pdf,png}
```

`hr_figures.py` loads `hr_data.npz` when it exists and falls back to running the
simulation (then saving it) when it does not, so a bare `python hr_figures.py`
works from a fresh clone. Delete `hr_data.npz`, or re-run `hr_mw_ao.py`, after
changing anything in the model or its tunings.

From a session or notebook:

```python
import hr_mw_ao as M, hr_figures as F

res = M.run()                 # or M.load_data() to reuse a saved run
M.save_data(res)
pe = M.compute_pe(res)        # PE / dwell-time diagnostics per mode

F.fig_convergence(res); F.fig_states(res); F.fig_pe(res)
```

### Parameter setting

Switched unknown parameters `θ_ℓ = (a, b, d)`, one realization per firing mode:

| mode ℓ | a_ℓ | b_ℓ | d_ℓ |
|---|---|---|---|
| 1 | 1.00 | 3.00 | 5.00 |
| 2 | 0.95 | 2.60 | 4.70 |
| 3 | 0.88 | 3.30 | 4.00 |

Estimates are initialized 30 % away from truth — above for every entry except
`a_1`, which starts 30 % *below* it (`SIGN_INIT[0,0] = -1`), so the figures show
error contraction from both directions. Modes are cycled `1 → 2 → 3` twice with
dwell `D = 400` units at `Ts = 0.1`; time is dimensionless throughout.

Known HR constants: `c = 1.0`, `s = 4.0`, `x₁ = -1.6`, `r = 0.006`, `I = 3.2`.
Observer tunings: `Q = 5I`, `ρ_x = 8.0`, `ρ_θ = 0.03`, `S₀ = 5I`, `Γ₀ = I`,
`Λ₀ = 0`, information-matrix regularization `1e-3`.

## What the figures show

`hr_figures.py` writes three figures into `hr_figures/`:

| file | content |
|---|---|
| `fig_HR_convergence` | per-parameter error contraction for `a`, `b`, `d` against the switching signal |
| `fig_HR_states` | state estimation `x₁, x₂, x₃`, per-observer parameter-error norm `Π₁, Π₂, Π₃`, and `σ(t)` |
| `fig_HR_PE` | per-mode excitation levels `N_{ℓ,·}`, `λ_min`, the threshold `δ` and the dwell time `T_min,ℓ` |

Final per-mode relative parameter errors are below 0.6 % in every mode, and the
dwell-time condition holds with a wide margin:

| mode | T_min,ℓ | AT_ℓ | max rel. error |
|---|---|---|---|
| 1 | 29 | 400 | 0.54 % |
| 2 | 174 | 400 | 0.39 % |
| 3 | 11 | 400 | 0.51 % |

The excitation threshold follows a **global** policy
(`DELTA_POLICY = "global"` in `hr_mw_ao.py`): a single
`δ = DELTA_FRAC · min_ℓ λ_min(N_ℓ(AT))` is shared by all modes, capped by the
least-excited one, so the three `T_min,ℓ` are first crossings of the *same* line
and are directly comparable. Set `DELTA_POLICY = "permode"` for the per-mode
threshold, or `DELTA_ABS` to pin a value by hand.

## Contact

- Juan Sereno — jeserenom@gmail.com
- E. A. Hernandez-Vargas — esteban@uidaho.edu

## Cite

If you use this code, please cite:

> J. Sereno and E. A. Hernandez-Vargas, "A Mode-wise Adaptive Observer for
> Autonomous Switched Nonlinear Systems," arXiv:2608.18251, 2026.

```bibtex
@article{sereno2026modewise,
  title   = {A Mode-wise Adaptive Observer for Autonomous Switched Nonlinear Systems},
  author  = {Sereno, Juan and Hernandez-Vargas, Esteban A.},
  journal = {arXiv preprint arXiv:2608.18251},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.18251}
}
```
