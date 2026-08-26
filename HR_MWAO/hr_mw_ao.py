"""
Switched Hindmarsh-Rose mode-wise adaptive observer (fast toy validation).

Autonomous switched nonlinear system (no control input); the switching signal
selects one of three parameter realizations theta_l = (a, b, d).  The plant is
already in the state-affine form required by the mode-wise observer:

    z_dot = A z + beta + Psi(y) theta_l ,   y = C z = x  (membrane potential)

with A constant and (A,C) observable; the unknown parameters multiply
output-only regressors.  Mirrors the manuscript's Eq. (state-affine) / Eq.(13).
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

# ----- known HR parameters -----
c_, s_, x1_, r_, I_ = 1.0, 4.0, -1.6, 0.006, 3.2
# switched unknown parameters theta_l = (a, b, d) per mode (3 firing regimes)
# DEFINITIVE SETTING: a_1 = 1.00 (baseline), a_2 = 0.95, a_3 = 0.88;
# b and d unchanged.  With a_3 near 1.0 mode 3 sits on the quiescent side of the
# HR firing map -- its burst dies out mid-window, N_3 saturates and the first
# visit is not visibly contractive.  Lowering a_3 keeps the neuron bursting for
# the whole active interval.  Sweeping a_3 with everything else fixed, measured
# over the first mode-3 window [800, 1200]:
#     0.98 -> |a~_3| 0.294 -> 0.255 (-13 %, reads flat)
#     0.92 -> 0.276 -> 0.221 (-20 %)
#     0.90 -> 0.270 -> 0.165 (-39 %, rho_3 = 0.689)
#     0.88 -> 0.264 -> 0.084 (-68 %, rho_3 = 0.505)   <-- chosen
#     0.86 -> 0.258 -> 0.008 (-97 %, visit 1 already done)
# 0.88 makes visit 1 clearly contractive while leaving real work for visit 2
# (cf. a3_study_notes.md).
THETA = np.array([[1.00, 3.00, 5.00],
                  [0.95, 2.60, 4.70],
                  [0.88, 3.30, 4.00]])
NX, NP, NY, NS = 3, 3, 1, 3
THETA_NAMES = ["a", "b", "d"]

A = np.array([[0., 1., -1.], [0., -1., 0.], [r_ * s_, 0., -r_]])
beta = np.array([I_, c_, -r_ * s_ * x1_])
C = np.array([[1., 0., 0.]])


def Psi(o):
    return np.array([[-o ** 3, o ** 2, 0.], [0., 0., -o ** 2], [0., 0., 0.]])


def hr_plant(x, th):
    X, Y, Z = x; a, b, d = th
    return np.array([Y - a * X ** 3 + b * X ** 2 - Z + I_,
                     c_ - d * X ** 2 - Y,
                     r_ * (s_ * (X - x1_) - Z)])

# ----- observer tunings -----
Q = 5.0 * np.eye(1)
RHO_X = 8.0
RHO_TH = np.array([0.03, 0.03, 0.03])
S0 = 5.0 * np.eye(NX)
G0 = 1.0 * np.eye(NP)
L0 = np.zeros((NX, NP))
GREG = 1e-3                 # information-matrix regularisation (keeps Gamma PD)
EPS_INIT = 0.3             # offset prior on theta_hat (30%), unchanged
# SCENARIO: sign of that 30 % offset, per mode (row) and per parameter (column).
# Only a_1 -- the a parameter of mode 1 -- is initialised BELOW its true value;
# every other entry keeps the original offset from above.
SIGN_INIT = np.ones((NS, NP))
SIGN_INIT[0, 0] = -1.0
KT_INIT = 1.0 + SIGN_INIT * EPS_INIT        # ns x np
CtQC = C.T @ Q @ C

_zl = slice(0, 3); _Sl = slice(3, 12); _Ll = slice(12, 21)
_Gl = slice(21, 30); _thl = slice(30, 33)


def _pack(z, S, L, G, th):
    return np.concatenate([z, S.flatten('F'), L.flatten('F'), G.flatten('F'), th])


def _obs_rhs(so, o, ell):
    z = so[_zl]; S = so[_Sl].reshape(3, 3, order='F'); L = so[_Ll].reshape(3, 3, order='F')
    G = so[_Gl].reshape(3, 3, order='F'); th = so[_thl]
    innov = o - (C @ z)[0]
    lu = lu_factor(S)
    CtQinnov = C.T[:, 0] * Q[0, 0] * innov
    Greg = G + GREG * np.eye(NP)
    term1 = lu_solve(lu, CtQinnov)
    term2 = L @ np.linalg.solve(Greg, L.T @ CtQinnov)
    dz = A @ z + beta + Psi(o) @ th + term1 + term2
    dS = -RHO_X * S - A.T @ S - S @ A + CtQC
    dL = A @ L - lu_solve(lu, CtQC @ L) + Psi(o)
    dG = -RHO_TH[ell] * G + L.T @ CtQC @ L
    dth = np.linalg.solve(Greg, L.T @ CtQinnov)
    return _pack(dz, dS, dL, dG, dth)


def _combined(t, yv, ell):
    xp = yv[:3]; so = yv[3:]
    return np.concatenate([hr_plant(xp, THETA[ell]), _obs_rhs(so, xp[0], ell)])


def run(D=400.0, Ts=0.1, ncyc=2, verbose=True):
    """Run the switched-HR mode-wise adaptive observer; return a results dict."""
    order = [0, 1, 2] * ncyc
    x_plant = np.array([-1.3, -7.0, 3.0])
    zhat0 = x_plant + np.array([0.5, 1.0, -0.5])
    modes = [_pack(zhat0.copy(), S0.copy(), L0.copy(), G0.copy(), KT_INIT[m] * THETA[m])
             for m in range(3)]
    th_all = np.array([KT_INIT[m] * THETA[m] for m in range(3)], float)  # per-mode theta_hat

    to = []; sig = []; XP = []; XH = []
    KThat = []; KTtrue = []; KThatAll = []; LA = []
    tcur = 0.0
    for ell in order:
        tspan = np.arange(tcur, tcur + D + Ts / 2, Ts)
        sol = solve_ivp(_combined, [tspan[0], tspan[-1]],
                        np.concatenate([x_plant, modes[ell]]),
                        t_eval=tspan, args=(ell,), rtol=1e-6, atol=1e-8, max_step=0.5)
        Yv = sol.y
        for k in range(len(tspan)):
            so = Yv[3:, k]
            th_all[ell] = so[_thl]                       # active mode updates
            to.append(tspan[k]); sig.append(ell)
            XP.append(Yv[:3, k]); XH.append(so[_zl])
            KThat.append(so[_thl].copy()); KTtrue.append(THETA[ell].copy())
            KThatAll.append(th_all.copy())
            LA.append(so[_Ll].reshape(3, 3, order='F'))
        x_plant = Yv[:3, -1].copy(); modes[ell] = Yv[3:, -1].copy()
        for m in range(3):
            if m != ell:
                modes[m][_zl] = modes[ell][_zl]          # share state estimate
        tcur = tspan[-1]
    if verbose:
        print("Switched-HR mode-wise AO done. final per-mode rel. errors:")
    res = dict(to=np.array(to), Ts=Ts, ns_modes=NS, np=NP, nx=NX, Q=Q, C=C,
               sigma_arr=np.array(sig), xp=np.array(XP).T, xhat=np.array(XH).T,
               KThatArray=np.array(KThat).T, KTtrue_arr=np.array(KTtrue).T,
               KThatAll=np.transpose(np.array(KThatAll), (2, 1, 0)),  # np x ns x N
               KT_true_per_mode=THETA.T, LambdaActive=np.transpose(np.array(LA), (1, 2, 0)),
               THETA_NAMES=THETA_NAMES)
    if verbose:
        sg = res["sigma_arr"]
        for m in range(NS):
            idx = np.where(sg == m)[0]
            rel = np.abs(res["KThatArray"][:, idx[-1]] - THETA[m]) / np.abs(THETA[m])
            print(f"  mode {m + 1}: theta={THETA[m]}  hat={np.round(res['KThatArray'][:,idx[-1]],3)}"
                  f"  relerr={np.round(rel,4)}")
    return res


# ----- per-element / per-mode PE diagnostics (per active window) -----
REL_TOL = 1e-2
# Excitation threshold policy for Assumption 2 / Inequality (14):
#   "global"  -> ONE delta shared by every mode, so the three T_min,l reported in
#                Fig. 4 are first crossings of the SAME horizontal line and are
#                therefore comparable across modes.
#   "permode" -> the original delta_l = DELTA_FRAC * lam_l(AT_l), one line/panel.
# Inequality (14) needs delta <= lam_min(N_l(AT_l)) for EVERY l, so a global delta
# is capped by the least-excited mode (here mode 2, lam_2 = 4.72e-3).  DELTA_FRAC
# is the fraction of that cap actually used: larger -> larger T_min, smaller
# margin.  DELTA_ABS, when not None, overrides both and fixes delta by hand.
DELTA_POLICY = "global"
DELTA_FRAC = 0.5          # -> delta = 2.36e-3, T_min = 29 / 174 / 11
DELTA_ABS = None


def _lam_min_psd(M):
    M = 0.5 * (M + M.T)
    return float(np.linalg.eigvalsh(M)[0])


def _windows(sigma, ell):
    idx = np.where(sigma == ell)[0]
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)


def compute_pe(res):
    """Per-element / per-mode PE diagnostics, accumulated within each window.

    Pass 1 builds the accumulated information N_l and the reduced lambda_min
    curves.  Pass 2 fixes the excitation threshold and locates T_min,l as its
    first crossing.  With DELTA_POLICY == "global" the SAME delta is used for
    every mode, delta = DELTA_FRAC * min_l lambda_min(N_l(t_k, t_k + AT_l)),
    i.e. it is set by the least-excited mode, so Inequality (14) is still met by
    all of them and the three T_min,l are read off one common line.
    """
    Lam = res["LambdaActive"]; sigma = res["sigma_arr"].astype(int)
    to = res["to"]; Ts = res["Ts"]; M = (C.T @ Q @ C)
    out = {"modes": {}, "npar": NP}

    # ---- pass 1: accumulated information per mode / per window ----
    raw = {}
    for ell in range(NS):
        wd = []
        for w in _windows(sigma, ell):
            el = to[w] - to[w[0]]
            Nseq = np.array([Lam[:, :, i].T @ M @ Lam[:, :, i] for i in w])
            cum = np.cumsum(Nseq * Ts, axis=0)
            diag = np.diagonal(cum, axis1=1, axis2=2)
            fd = diag[-1]; mx = fd.max()
            exc = np.where(fd >= REL_TOL * mx)[0] if mx > 0 else np.array([], int)
            if exc.size:
                sub = cum[:, exc[:, None], exc[None, :]]
                lam = np.array([_lam_min_psd(sub[j]) for j in range(len(w))])
            else:
                lam = np.zeros(len(w))
            wd.append(dict(elapsed=el, diag=diag, excited=exc, lam_red=lam,
                           AT=el[-1] + Ts, final_diag=fd))
        raw[ell] = wd

    # ---- excitation threshold ----
    if DELTA_ABS is not None:
        delta_of = lambda ell, d: float(DELTA_ABS)
    elif DELTA_POLICY == "global":
        d_glob = DELTA_FRAC * min(raw[e][0]["lam_red"][-1] for e in range(NS))
        delta_of = lambda ell, d: d_glob
    else:
        delta_of = lambda ell, d: DELTA_FRAC * d["lam_red"][-1]

    # ---- pass 2: T_min as the first crossing of that threshold ----
    for ell in range(NS):
        wd = []
        for d in raw[ell]:
            dl = delta_of(ell, d)
            cr = np.where(d["lam_red"] >= dl)[0]
            wd.append({**d, "delta": dl,
                       "Tmin": d["elapsed"][cr[0]] if cr.size else np.nan})
        rep = wd[0]
        out["modes"][ell] = {**rep, "windows": wd, "n_windows": len(wd),
                             "Tmin_all": [w["Tmin"] for w in wd]}
    return out


# ----- persisted run data -----
# The simulation takes ~30 s; the figures take under a second.  run() is
# therefore executed once and its complete result dict is written to HR_DATA,
# after which hr_figures.py can be re-run (restyled, re-laid out, re-exported)
# as often as needed without touching the integrator.  Everything the three
# figures and compute_pe() need is in the file -- states, per-mode estimates,
# the switching signal and the Lambda history -- so the figures are guaranteed
# to show the same run, not a fresh one.
HR_DATA = "hr_data.npz"


def save_data(res, path=HR_DATA):
    """Write a run() result to `path` (npz).  Returns the path written."""
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in res.items()})
    return path


def load_data(path=HR_DATA):
    """Read back a run() result written by save_data(); inverse of save_data().

    np.load returns every entry as an array, so the handful of scalar / list
    entries are restored to the native types the figure code expects.
    """
    with np.load(path, allow_pickle=False) as z:
        res = {k: z[k] for k in z.files}
    res["Ts"] = float(res["Ts"])
    for k in ("ns_modes", "np", "nx"):
        res[k] = int(res[k])
    res["THETA_NAMES"] = [str(s) for s in res["THETA_NAMES"]]
    return res


if __name__ == "__main__":
    res = run(verbose=True)
    save_data(res)
    print(f"Saved run data to {HR_DATA} -- run 'python hr_figures.py' to plot it.")
