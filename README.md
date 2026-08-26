# A Mode-wise Adaptive Observer for Autonomous Switched Nonlinear Systems

Reference implementation of the **mode-wise adaptive observer (MWAO)** for
autonomous switched nonlinear systems in state-affine form, with the two case
studies of the paper.

The observer assigns a *dedicated* adaptive observer `Π_ℓ` to each mode `ℓ`. At
any instant only `Π_σ(t)` integrates and the others freeze, so a switch never
injects the estimation error of one parameter realization into another.
Convergence rests on two conditions, both evaluated directly from a simulation
by the code here:

- a **finite-window persistence-of-excitation** condition on the accumulated
  information matrix `N_ℓ(t_k, t_k + T)`;
- a **minimum dwell-time** condition `AT_ℓ ≥ T_min,ℓ`, where `T_min,ℓ` is the
  first time the excitation crosses the threshold `δ`.

## Case studies

| folder | system | modes | unknown parameters | measured |
|---|---|---|---|---|
| [`HR_MWAO/`](HR_MWAO) | switched Hindmarsh–Rose neuron | 3 firing regimes | `θ_ℓ = (a, b, d)` | membrane potential |
| [`HIV_MWAO/`](HIV_MWAO) | HIV mutation model, 4 genotypes, SWATCH therapy | 3 therapies (none / 1 / 2) | `θ_ℓ = (K_T^1..K_T^4)` | `T` and the four viral loads |

Each folder is self-contained and has its own README with the parameter
setting, results, and figure conventions.

## Installation

Python 3.9+ and three standard scientific packages:

```
pip install -r requirements.txt
```

## Usage

Both case studies follow the same two-step pattern: simulate once, then plot as
often as you like. Separating them means restyling a figure never re-runs the
integrator — and never silently plots a *different* run.

```
cd HR_MWAO
python hr_mw_ao.py      # simulate  -> hr_data.npz
python hr_figures.py    # plot      -> hr_figures/
```

```
cd HIV_MWAO
python hiv_mw_ao.py     # simulate  -> hiv_data.npz
python hiv_figures.py   # plot      -> hiv_figures/
```

The figure script loads the saved run when one exists and otherwise simulates
once and saves it, so a bare `python *_figures.py` works from a fresh clone.
The generated `*_data.npz` and `*_figures/` are git-ignored; delete the `.npz`,
or re-run the simulation script, after changing a model or its tunings.

## Results at a glance

**Hindmarsh–Rose.** Final per-mode relative parameter errors below 0.6 % in
every mode, with the dwell-time condition satisfied by a wide margin
(`T_min,ℓ` = 29 / 174 / 11 units against `AT_ℓ` = 400).

**HIV.** Parameter RMSLE of order `1e-7` on all four `K_T^i`. Each therapy
excites essentially the one genotype it selects for, so `λ_min` is taken over
the excited subspace; `T_min,ℓ` = 137 / 189 / 182 days against `AT_ℓ` ≈ 365.

In both case studies the excitation threshold follows a **global** policy: one
`δ = DELTA_FRAC · min_ℓ λ_min(N_ℓ(AT_ℓ))` shared by every mode and capped by the
least-excited one, so the `T_min,ℓ` are first crossings of the same line and are
comparable across modes.

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
