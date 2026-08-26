"""Figures for the HIV mode-wise adaptive observer.

Three publication figures, all driven from one ``run()`` result:

  fig_states_stacked  -- viral loads, infected T cells, infected macrophages,
                         per-observer parameter-error norm, switching signal
  fig_KT_convergence  -- per-genotype K_T error contraction
  fig_PE              -- per-mode excitation, lambda_min, dwell time T_min

Each figure keeps the exact rcParams it was published with: the three settings
differ (font sizes above all), so they are applied per figure through
``plt.rc_context`` rather than globally at import time.

Run ``python hiv_figures.py`` to write all three into ``hiv_figures/``.
"""
from __future__ import annotations

import functools
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import hiv_mw_ao as sim

TimeScale = 365.0   # days -> years
USETEX = False      # usetex off for portability (as published)

# --- per-figure styles (each figure was tuned with its own sizes) -----------
RC_STATES = {
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": False, "font.size": 12,
    "axes.labelsize": 13, "legend.fontsize": 14,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 120,
}
RC_KT = {
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": False, "font.size": 15,
    "axes.labelsize": 17, "legend.fontsize": 14,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 120,
}
RC_PE = {
    "text.usetex": USETEX,
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 15,   # matches fig_states_stacked.py
    "ytick.labelsize": 15,   # matches fig_states_stacked.py
    "axes.linewidth": 0.9,
    "lines.linewidth": 2.0,
    "savefig.dpi": 600,
    "figure.dpi": 120,
}

# --- mode shading, shared by the states and K_T figures --------------------
mode_colors = [np.array([0.20, 0.60, 0.20]),   # mode 0 - green
               np.array([0.85, 0.33, 0.10]),   # mode 1 - orange
               np.array([0.10, 0.30, 0.80])]   # mode 2 - blue


def _with_style(rc):
    """Run one figure under its own rcParams, leaving the others untouched."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with plt.rc_context(rc):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def plot_switched_style(ax, t, y, active_mask, color, lw):
    """Plot y(t) solid where active, dotted where frozen.

    Returns a Line2D suitable as a legend handle (prefers a solid segment).
    """
    t = np.asarray(t).ravel()
    y = np.asarray(y).ravel()
    active = np.asarray(active_mask).ravel().astype(bool)
    n = t.size
    if n == 0:
        (h,) = ax.plot([np.nan], [np.nan], "-", color=color, linewidth=lw)
        return h
    flips = np.where(np.diff(active.astype(int)) != 0)[0] + 1
    runs = np.concatenate([[0], flips, [n]])
    h_solid = []
    for k in range(runs.size - 1):
        s = runs[k]
        e = runs[k + 1] - 1
        e_plot = min(e + 1, n - 1)      # overlap by one sample for continuity
        sl = slice(s, e_plot + 1)
        if active[s]:
            (h,) = ax.plot(t[sl], y[sl], "-", color=color, linewidth=lw)
            h_solid.append(h)
        else:
            ax.plot(t[sl], y[sl], ":", color=color, linewidth=lw)
    if h_solid:
        return h_solid[0]
    (h,) = ax.plot([np.nan], [np.nan], ":", color=color, linewidth=lw)
    return h


def get_results(verbose=True):
    """The saved run when there is one, else simulate once and save it."""
    if os.path.exists(sim.HIV_DATA):
        if verbose:
            print(f"[hiv_figures] loading {sim.HIV_DATA}")
        return sim.load_data()
    if verbose:
        print(f"[hiv_figures] {sim.HIV_DATA} not found -- running the simulation once")
    res = sim.run(verbose=verbose)
    sim.save_data(res)
    return res


# =========================================================== states (stacked)
GENO_T = [np.minimum(1.10 * np.array([0.0, 0.470, 0.760]), 1),
          np.minimum(1.10 * np.array([0.850, 0.330, 0.100]), 1),
          np.minimum(1.07 * np.array([0.930, 0.690, 0.130]), 1),
          np.minimum(1.10 * np.array([0.570, 0.210, 0.790]), 1)]
GENO_E = [np.array([0.0, 0.250, 0.540]), np.array([0.630, 0.110, 0.080]),
          np.array([0.670, 0.500, 0.090]), np.array([0.330, 0.130, 0.430])]
MODE_SHADE = [np.array([0.20, 0.60, 0.20]), np.array([0.85, 0.33, 0.10]),
              np.array([0.10, 0.30, 0.80])]
TimeScale = 365.0
TI = [0, 3, 6, 9]; MI = [1, 4, 7, 10]; VI = [2, 5, 8, 11]


def _phase_spans(t, sigma):
    sigma = np.asarray(sigma, int); spans = []; s = 0
    for k in range(1, len(sigma)):
        if sigma[k] != sigma[s]:
            spans.append((t[s], t[k], int(sigma[s]))); s = k
    spans.append((t[s], t[-1], int(sigma[s])))
    return spans


def _mask_log(y, floor):
    """Mask non-positive / sub-floor samples with NaN so the line breaks
    instead of drawing a vertical spike down to the axis floor."""
    y = np.asarray(y, dtype=float).copy()
    y[~(y > floor)] = np.nan
    return y


def _exp_ticks(ax, ylim):
    """Label the log y-axis with the base-10 exponent only.
    Negatives/zero every 2 decades, positives every 1 (e.g. -12,-10,...,0,1,2)."""
    hi = int(np.ceil(np.log10(ylim[1])))
    exps = [k for k in range(-8, 0, 2)] + [k for k in range(0, hi + 1,2)]
    exps = [k for k in exps if 10.0 ** k >= ylim[0] * 0.999]
    ax.set_yticks([10.0 ** k for k in exps])
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f"{int(np.round(np.log10(y)))}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


@_with_style(RC_STATES)
def make_stacked_figure(results=None, savedir="hiv_figures", show=False):
    r = results if results is not None else get_results(verbose=True)
    tt = r["tp_full"] / TimeScale
    te = r["to"] / TimeScale
    XT, XE = r["xp_full"], r["xhatArray"]
    spans = _phase_spans(tt, r["sigma_full"])

    def shade(ax):
        for (t0, t1, mode) in spans:
            ax.axvspan(t0, t1, color=MODE_SHADE[mode], alpha=0.10, lw=0)

    HEIGHT_RATIOS = [1, 1, 1, 1, 0.58]
    # --- individual vertical gaps between consecutive panels (figure fraction) ---
    GAP_AB = 0.032
    GAP_BC = 0.032
    GAP_CD = 0.032
    GAP_DE = 0.032
    GAPS = [GAP_AB, GAP_BC, GAP_CD, GAP_DE]
    LEFT, WIDTH = 0.16, 0.81
    Y_TOP, BOTTOM = 0.98, 0.09
    _u = (Y_TOP - BOTTOM - sum(GAPS)) / sum(HEIGHT_RATIOS)
    heights = [_u * rr for rr in HEIGHT_RATIOS]
    bottoms = []; _b = Y_TOP
    for _k in range(5):
        _b -= heights[_k]; bottoms.append(_b)
        if _k < 4:
            _b -= GAPS[_k]

    fig, axes = plt.subplots(5, 1, figsize=(8.4, 11.6), sharex=True,
                             gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    axV, axT, axM, axTh, axS = axes

    def plot_group_log(ax, rows, names, ylabel, ylim):
        floor = ylim[0]
        for k, row in enumerate(rows):
            ax.plot(tt, _mask_log(XT[row], floor), color=GENO_T[k], lw=1.7,
                    label=names[k])
            ax.plot(te, _mask_log(XE[row], floor), color=GENO_E[k], lw=1.5, ls="--")
        shade(ax)
        ax.set_yscale("log"); ax.set_ylim(*ylim)
        _exp_ticks(ax, ylim)
        ax.set_ylabel(ylabel, fontsize=18); ax.grid(True, which="major", alpha=0.2)

    plot_group_log(axV, VI, [f"$V_{i+1}$" for i in range(4)], r"$V_i\ [\mathrm{copies/mL}]$", (1e-8, 1e4))
    plot_group_log(axT, TI, [f"$T^*_{i+1}$" for i in range(4)], r"$T^{*}_i\ [\mathrm{cells/mm^{3}}]$", (1e-8, 1e3))
    plot_group_log(axM, MI, [f"$M^*_{i+1}$" for i in range(4)], r"$M^*_i \ [\mathrm{cells/mm^{3}}]$", (1e-8, 1e2))

    # ---- theta-error norm panel (as in fig8) -----------------------------
    KThatAll = r["KThatAll"]; KTc = r["KT_true_per_mode"]
    sig = r["sigma_arr"].astype(int)
    for m in range(r["ns_modes"]):
        err = KTc[:, m][:, None] - KThatAll[:, m, :]
        rel = np.sqrt(np.sum(err ** 2, axis=0)) / np.sqrt(np.sum(KTc[:, m] ** 2))
        plot_switched_style(axTh, te, rel, sig == m, mode_colors[m], 2.0)
    shade(axTh)
    axTh.set_ylim(0, 1); axTh.set_ylabel(r"$\|\tilde{\theta}_{\ell}\|/\|\theta_{\ell}\|$", fontsize=18)
    #axTh.set_yticks([0, 0.25, 0.5, 0.75, 1]);
    axTh.grid(True, alpha=0.2)
    theta_handles = [Line2D([0], [0], color=mode_colors[m], lw=2.0,
                            label=r"$\Pi_{%d}$" % m) for m in range(r["ns_modes"])]
    axTh.legend(handles=theta_handles, loc="upper right", ncol=1, framealpha=0.9, fontsize=16)

    # ---- sigma(t) --------------------------------------------------------
    axS.step(tt, r["sigma_full"], where="post", color="black", lw=1.6)
    shade(axS)
    axS.set_yticks([0, 1, 2]); axS.set_ylim(-0.4, 2.4)
    axS.set_ylabel(r"$\sigma(t)$", fontsize=18); axS.grid(True, alpha=0.2)
    axS.set_xlabel(r"time (years)", fontsize=18)
    axS.set_xticks([3, 4, 5, 6])

    axV.legend(loc="lower right", ncol=2, title=r"Virus", framealpha=0.9, columnspacing=1.0)
    style_handles = [Line2D([0], [0], color="0.3", lw=1.7, ls="-"),
                     Line2D([0], [0], color="0.3", lw=1.5, ls="--")]
    axT.legend(style_handles, ["Plant", "Mode-wise AO"], loc="lower right", framealpha=0.9)

    axV.set_xlim(3.0, 6.0)
    for _k in range(5):
        axes[_k].set_position([LEFT, bottoms[_k], WIDTH, heights[_k]])
    fig.align_ylabels(axes)

    # ---- bottom-centered captions (Fig. 5 style) -------------------------
    def _cap(s):
        return re.sub(r"^([a-zA-Z])\)\s*", r"$\\mathbf{\1)}$ ", s)
    caps = ["a) Viral load estimation",
            "b) Infected T-cell estimation",
            "c) Infected macrophage estimation",
            "d) Per-mode norm parameter error",
            "e) Switching signal"]
    xc = LEFT + WIDTH / 2.0
    for _k in range(4):
        yc = (bottoms[_k] + bottoms[_k + 1] + heights[_k + 1]) / 2.0
        fig.text(xc, yc, _cap(caps[_k]), ha="center", va="center", fontsize=16)
    fig.text(xc, bottoms[4] - 0.085, _cap(caps[4]),
             ha="center", va="center", fontsize=16)

    if savedir and savedir != "None":
        os.makedirs(savedir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(savedir, f"fig_states_stacked.{ext}"),
                        bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ============================================== K_T parameter convergence
# mode shading colours -- identical to fig8_theta_norm (green / orange / blue)
MODE_COLORS = [np.array([0.20, 0.60, 0.20]),
               np.array([0.85, 0.33, 0.10]),
               np.array([0.10, 0.30, 0.80])]
MODE_LABELS = ["no therapy", "therapy 1", "therapy 2"]


@_with_style(RC_KT)
def make_kt_figure(results=None, savedir="hiv_figures", show=False):
    r = results if results is not None else get_results(verbose=True)
    t = r["to"] / TimeScale
    KTt, KTh = r["KTtrue_arr"], r["KThatArray"]
    spans = _phase_spans(t, r["sigma_arr"])
    panel = ["(a)", "(b)", "(c)", "(d)"]

    letters = ["a", "b", "c", "d"]
    # --- individual vertical gaps between the four panels (figure fraction) ---
    GAP_AB = 0.05
    GAP_BC = 0.05
    GAP_CD = 0.05
    GAPS = [GAP_AB, GAP_BC, GAP_CD]
    LEFT, WIDTH = 0.15, 0.82
    Y_TOP, BOTTOM = 0.98, 0.11
    H = (Y_TOP - BOTTOM - sum(GAPS)) / 4.0
    bottoms = []; _b = Y_TOP
    for _k in range(4):
        _b -= H; bottoms.append(_b)
        if _k < 3:
            _b -= GAPS[_k]

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 10.6), sharex=True)
    for i in range(4):
        ax = axes[i]
        for (t0, t1, mode) in spans:
            ax.axvspan(t0, t1, color=MODE_COLORS[mode], alpha=0.12, lw=0)
        ax.plot(t, KTt[i] * 1e5, color="black", lw=2.2, drawstyle="steps-post")
        ax.plot(t, KTh[i] * 1e5, color="red", lw=2.0, ls="--")
        ax.set_ylabel(r"$K_T^{%d}$  [$\times10^{-5}$]" % (i + 1), fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.margins(y=0.18)

    axes[-1].set_xlabel(r"time (years)", fontsize=18)
    axes[-1].set_xticks([3, 4, 5, 6])
    axes[0].set_xlim(t[0], t[-1])

    # ---- separate legend frames -----------------------------------------
    h_true = Line2D([0], [0], color="black", lw=2.2, label="actual")
    h_est = Line2D([0], [0], color="red", lw=2.0, ls="--", label="estimate")
    mode_patches = [Patch(facecolor=MODE_COLORS[m], alpha=0.40, label=MODE_LABELS[m])
                    for m in range(3)]

    # frame 1: actual + estimate together (panel a, top-right)
    leg_lines = axes[0].legend([h_true, h_est], ["actual", "estimate"],
                               loc="upper right", framealpha=0.95, handlelength=1.8)
    axes[0].add_artist(leg_lines)
    # frame 2: therapy modes (panel b, top-right)
    axes[1].legend(mode_patches, MODE_LABELS, loc="upper right",
                   framealpha=0.95, ncol=1, handlelength=1.4)

    # ---- manual positioning + bottom-centered captions (Fig. 4 style) ----
    for _k in range(4):
        axes[_k].set_position([LEFT, bottoms[_k], WIDTH, H])
    fig.align_ylabels(axes)

    def _cap(s):
        return re.sub(r"^([a-zA-Z])\)\s*", r"$\\mathbf{\1)}$ ", s)
    caps = [r"%s) parameter $K_T^{%d}$ error contraction" % (letters[i], i + 1)
            for i in range(4)]
    xc = LEFT + WIDTH / 2.0
    for _k in range(3):
        yc = (bottoms[_k] + bottoms[_k + 1] + H) / 2.0
        fig.text(xc, yc, _cap(caps[_k]), ha="center", va="center", fontsize=16)
    fig.text(xc, bottoms[3] - 0.095, _cap(caps[3]),
             ha="center", va="center", fontsize=16)

    if savedir and savedir != "None":
        os.makedirs(savedir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(savedir, f"fig_KT_convergence.{ext}"),
                        bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ========================================================== PE / dwell time
# Per-genotype colours (K_T^1..K_T^4) -- consistent with Figs 2-3 palette.
GENO_COLORS = [
    np.array([0.000, 0.450, 0.740]),   # genotype 1 - blue
    np.array([0.850, 0.330, 0.100]),   # genotype 2 - orange
    np.array([0.930, 0.690, 0.130]),   # genotype 3 - yellow
    np.array([0.490, 0.180, 0.560]),   # genotype 4 - purple
]
MODE_NAMES = ["mode 0  (no therapy)", "mode 1  (therapy 1)", "mode 2  (therapy 2)"]

# ---------------------------------------------------------------------------
# Tunable validation parameters
# ---------------------------------------------------------------------------
REL_TOL = 1e-2       # excited element if cal_N_ii(end) >= REL_TOL * max_j cal_N_jj(end)
# Excitation threshold policy for Assumption 3 / the dwell-time inequality
# (mirrors hr_mw_ao.py so both models are diagnosed the same way):
#   "global"  -> ONE delta shared by every mode, delta = DELTA_FRAC *
#                min_l lambda_min(cal_N_l(t_k, t_k + AT_l)), i.e. set by the
#                LEAST-excited mode.  The three T_min,l are then first crossings
#                of the SAME horizontal line and are comparable across modes.
#   "permode" -> the original delta_l = DELTA_FRAC * lambda_min_l(AT_l), one
#                threshold per panel.  Since that threshold is read off the very
#                curve being tested, and lambda_min is non-decreasing from 0, the
#                crossing always exists: T_min then measures the SHAPE of the
#                accumulation, not whether its LEVEL is adequate.
# The inequality needs delta <= lambda_min(cal_N_l(AT_l)) for EVERY l, which caps
# a global delta at the least-excited mode; DELTA_FRAC is the fraction of that cap
# actually used (larger -> larger T_min, smaller margin).  DELTA_ABS, when not
# None, overrides both and fixes delta by hand -- use it to check the trajectory
# against a level required by the observer design rather than one derived from
# the data.
DELTA_POLICY = "global"
DELTA_FRAC = 0.5
DELTA_ABS = None


def _C_matrix(nx=14):
    """Output matrix C (5 x 14): rows pick [T, V1, V2, V3, V4]."""
    C = np.zeros((5, nx))
    C[0, 12] = 1.0   # T
    C[1, 2] = 1.0    # V1
    C[2, 5] = 1.0    # V2
    C[3, 8] = 1.0    # V3
    C[4, 11] = 1.0   # V4
    return C


def _lam_min_psd(M):
    M = 0.5 * (M + M.T)
    return float(np.linalg.eigvalsh(M)[0])


def _active_windows(sigma, ell):
    """Return contiguous active windows (arrays of sample indices) for mode ell."""
    idx = np.where(sigma == ell)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)


def _window_pe(w, to, Ts, Lam, M):
    """Accumulated information WITHIN a single active window ``w``.

    Returns the raw (threshold-free) diagnostics; ``delta`` and ``Tmin`` are
    added afterwards by :func:`compute_pe`, once the threshold policy is known.
    """
    npar = Lam.shape[1]
    elapsed = to[w] - to[w[0]]
    Nseq = np.array([Lam[:, :, i].T @ M @ Lam[:, :, i] for i in w])   # (len,np,np)
    cum = np.cumsum(Nseq * Ts, axis=0)                               # reset at t_k
    diag = np.diagonal(cum, axis1=1, axis2=2)
    final_diag = diag[-1]
    mx = final_diag.max()
    excited = np.where(final_diag >= REL_TOL * mx)[0] if mx > 0 else np.array([], int)
    if excited.size:
        sub = cum[:, excited[:, None], excited[None, :]]
        lam_red = np.array([_lam_min_psd(sub[j]) for j in range(len(w))])
    else:
        lam_red = np.zeros(len(w))
    return {"elapsed": elapsed, "diag": diag, "excited": excited,
            "lam_red": lam_red,
            "AT": elapsed[-1] + Ts, "final_diag": final_diag, "t_k": to[w[0]]}


def compute_pe(results):
    """Return per-mode, per-element PE diagnostics, computed PER ACTIVE WINDOW.

    Pass 1 accumulates the information matrix and the reduced lambda_min curve
    of every window; pass 2 fixes the excitation threshold (see DELTA_POLICY)
    and locates T_min as its first crossing.  With DELTA_POLICY == "global" the
    SAME delta is used for every mode, delta = DELTA_FRAC *
    min_l lambda_min(cal_N_l(t_k, t_k + AT_l)), so it is set by the least-excited
    mode, every mode still satisfies the inequality, and the three T_min,l are
    read off one common line.

    The top-level fields of ``modes[ell]`` mirror the FIRST active window (so
    existing callers keep working); ``modes[ell]["windows"]`` holds the list of
    all per-window diagnostics and ``Tmin_all`` the per-window dwell times.
    """
    Lam = results["LambdaActive"]          # nx x np x (Nsteps+1)
    sigma = results["sigma_arr"].astype(int)
    Q = results["Q"]
    to = results["to"]                     # absolute time grid [days]
    Ts = results["Ts"]
    ns_modes = results["ns_modes"]
    nx, npar, nT = Lam.shape

    C = _C_matrix(nx)
    M = C.T @ Q @ C                        # C' Q C  (PSD)

    out = {"to": to, "Ts": Ts, "npar": npar, "modes": {}}

    # ---- pass 1: accumulated information per mode / per window ----------
    raw = {ell: [_window_pe(w, to, Ts, Lam, M)
                 for w in _active_windows(sigma, ell)]
           for ell in range(ns_modes)}

    # ---- excitation threshold -------------------------------------------
    if DELTA_ABS is not None:
        delta_of = lambda ell, d: float(DELTA_ABS)
    elif DELTA_POLICY == "global":
        d_glob = DELTA_FRAC * min(raw[e][0]["lam_red"][-1]
                                  for e in range(ns_modes))
        delta_of = lambda ell, d: d_glob
    else:
        delta_of = lambda ell, d: DELTA_FRAC * d["lam_red"][-1]

    # ---- pass 2: T_min as the first crossing of that threshold ----------
    for ell in range(ns_modes):
        win_data = []
        for d in raw[ell]:
            dl = delta_of(ell, d)
            cross = np.where(d["lam_red"] >= dl)[0]
            win_data.append({**d, "delta": dl,
                             "Tmin": d["elapsed"][cross[0]] if cross.size
                             else np.nan})
        rep = win_data[0]
        out["modes"][ell] = {**rep,
                             "windows": win_data,
                             "n_windows": len(win_data),
                             "Tmin_all": [wd["Tmin"] for wd in win_data]}
    return out


@_with_style(RC_PE)
def make_pe_figure(results, savedir="hiv_figures", show=False):
    pe = compute_pe(results)
    ns = len(pe["modes"])
    npar = pe["npar"]

    fig, axes = plt.subplots(1, ns, figsize=(13.2, 4.7), sharey=True)
    letters = ["a", "b", "c"]
    # panel labels carry the mode index only; the therapy each mode stands for
    # is stated in the caption instead of on every panel
    tags = [f"{letters[ell]}) switching mode {ell}" for ell in range(ns)]

    def _cap(s):
        return re.sub(r"^([a-zA-Z])\)\s*", r"$\\mathbf{\1)}$ ", s)

    def _sublabel(ax, text, y=-0.22, fontsize=17):
        ax.text(0.5, y, _cap(text), transform=ax.transAxes, ha="center",
                va="top", fontsize=fontsize)

    for ell in range(ns):
        ax = axes[ell]
        d = pe["modes"][ell]
        el = d["elapsed"]
        excited = set(d["excited"].tolist())

        # Per-element accumulated excitation.  Legend labels carry the symbol
        # only, as in fig_HR_PE (hr_figures_scenario.py): solid = excited,
        # dashed/faint = weak, which the caption states once instead of
        # repeating "(excited)"/"(weak)" on every entry.
        for i in range(npar):
            col = GENO_COLORS[i]
            yy = np.maximum(d["diag"][:, i], 1e-300)
            if i in excited:
                ax.plot(el, yy, color=col, lw=2.2,
                        label=fr"$\mathcal{{N}}_{{{ell},{i+1}{i+1}}}$")
            else:
                ax.plot(el, yy, color=col, lw=1.4, ls="--", alpha=0.6,
                        label=fr"$\mathcal{{N}}_{{{ell},{i+1}{i+1}}}$")

        # faint overlay of repeated activations of this mode (per-window reset)
        for wd in d["windows"][1:]:
            ax.plot(wd["elapsed"], np.maximum(wd["lam_red"], 1e-300),
                    color="black", lw=1.0, alpha=0.30)

        # Matrix PE on the excited subspace (first window) + threshold + T_min.
        # Here every mode excites a single element, so lambda_min coincides
        # exactly with the corresponding diagonal curve; the black line is dashed
        # so the coloured curve underneath stays visible through the gaps.
        ax.plot(el, np.maximum(d["lam_red"], 1e-300), color="black", lw=2.4,
                ls=(0, (6, 3)), label=r"$\lambda_{\min}$")
        ax.axhline(max(d["delta"], 1e-300), color="black", ls=":", lw=1.2)
        if np.isfinite(d["Tmin"]):
            ax.axvline(d["Tmin"], color="red", ls="-.", lw=1.2)
            _toff = (16, -26) if ell == 2 else (6, -14)
            ax.annotate(fr"$T_{{\min,{ell}}}\approx{d['Tmin']:.0f}$ d",
                        xy=(d["Tmin"], max(d["delta"], 1e-300)),
                        xytext=_toff, textcoords="offset points",
                        fontsize=13, color="0.2")
            _ty, _tva = 0.975, "top"
            ax.annotate("Dwell-time",
                        xy=(d["Tmin"], _ty), xycoords=ax.get_xaxis_transform(),
                        xytext=(-3, 0), textcoords="offset points",
                        rotation=90, va=_tva, ha="right", fontsize=12, color="0.25")

        ax.set_yscale("log")
        ax.set_ylim(1e-12, 3.0)
        ax.set_xlabel(r"Active time interval (days)", fontsize=16)
        if ell == 0:
            ax.set_ylabel(r"Excitation level", fontsize=16)
        # Major y ticks EVERY TWO decades (10^-12, 10^-10, ..., 10^0): this panel
        # spans ~12.5 decades, so one label per decade crowds the axis.  Set
        # explicitly rather than leaving it to matplotlib's automatic thinning,
        # which is resolution-dependent.  Minor ticks keep the 2..9 subdivisions.
        ax.set_yticks([10.0 ** k for k in range(-12, 1, 2)])
        # Label the ticks with the base-10 exponent only (0, -2, -4, ...) instead
        # of 10^{k}, matching the stacked-states figure (fig_states_stacked.py).
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{int(np.round(np.log10(y)))}"))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1),
                               numticks=100))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        # this panel spans ~12.5 decades against ~5.7 in Fig. 4, so the minor
        # lines are drawn lighter to keep the same visual weight per decade.
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", alpha=0.10, lw=0.5)
        ax.legend(loc="lower right", fontsize=12, framealpha=0.9, ncol=1)
        _sublabel(ax, tags[ell], y=-0.22, fontsize=17)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    # Match the inter-panel spacing of Fig. 4 (HR).  tight_layout leaves a wider
    # gap here because this figure uses a smaller base font, so the gap is reset
    # explicitly while the outer margins found by tight_layout are preserved.
    PANEL_GAP = 0.0233          # figure fraction, as measured on Fig. 4
    _pos = [ax.get_position() for ax in axes]
    _left, _right = _pos[0].x0, _pos[-1].x1
    _w = (_right - _left - PANEL_GAP * (ns - 1)) / ns
    for _k, ax in enumerate(axes):
        ax.set_position([_left + _k * (_w + PANEL_GAP), _pos[_k].y0,
                         _w, _pos[_k].height])

    if savedir and savedir != "None":
        os.makedirs(savedir, exist_ok=True)
        for ext in ("pdf", "png", "eps"):
            fig.savefig(os.path.join(savedir, f"fig_PE.{ext}"), bbox_inches="tight")

    if show:
        plt.show()

    # ---- console summary --------------------------------------------------
    print("\n=== PE validation (per-element / per-mode, per-window) ===")
    _d0 = pe["modes"][0]["delta"]
    if DELTA_ABS is not None:
        print(f"  threshold policy: absolute, delta = {_d0:.3e} (DELTA_ABS)")
    elif DELTA_POLICY == "global":
        print(f"  threshold policy: global, delta = {DELTA_FRAC} * "
              f"min_l lambda_min_l(AT_l) = {_d0:.3e} (same line in all panels)")
    else:
        print(f"  threshold policy: per-mode, delta_l = {DELTA_FRAC} * "
              f"lambda_min_l(AT_l)")
    for ell in range(ns):
        d = pe["modes"][ell]
        exc = [i + 1 for i in d["excited"]]
        tmins = ", ".join("nan" if not np.isfinite(t) else f"{t:.0f}" for t in d["Tmin_all"])
        print(f"  mode {ell}: excited K_T = {exc} | n_windows={d['n_windows']} | "
              f"AT={d['AT']:.0f} d | delta_{ell}={d['delta']:.3e} | "
              f"lambda_min(end,excited)={d['lam_red'][-1]:.3e}")
        print(f"          T_min per window [d]: {tmins}")
        per = ", ".join(f"K_T^{i+1}={d['final_diag'][i]:.2e}" for i in range(npar))
        print(f"          per-element final excitation (1st window): {per}")
    return pe, fig


if __name__ == "__main__":
    res = get_results(verbose=True)
    make_stacked_figure(res)
    make_kt_figure(res)
    make_pe_figure(res)
    print("Saved hiv_figures/fig_{states_stacked,KT_convergence,PE}.{pdf,png}")
