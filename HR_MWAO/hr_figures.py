"""Figures for the switched Hindmarsh-Rose mode-wise adaptive observer."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as _mtrans
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import hr_mw_ao as M

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": False, "font.size": 14,
    # legend / tick sizes match the HIV figures (fig_states_stacked.py,
    # fig_kt_convergence.py); the PE figure overrides ticks locally
    "axes.labelsize": 16, "legend.fontsize": 14,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
    "axes.linewidth": 0.9, "savefig.dpi": 600, "figure.dpi": 120,
})
MODE_COLORS = [np.array([0.20, 0.60, 0.20]), np.array([0.85, 0.33, 0.10]),
               np.array([0.10, 0.30, 0.80])]
# Modes are displayed 1-indexed (manuscript uses \mathcal{M} = {1, 2, 3}),
# while the code keeps 0-based indices internally.
MODE_IDS = [1, 2, 3]
MODE_LABELS = ["mode %d" % m for m in MODE_IDS]
# Pi_l is the mode-wise adaptive OBSERVER of mode l (see hr_mw_ao.py): the
# theta-error panel tracks one observer per curve, including while it is
# frozen, so it is labelled Pi_l rather than "mode l" -- which the shading
# and the sigma(t) panel already convey.
OBS_LABELS = [r"$\Pi_{%d}$" % m for m in MODE_IDS]
STATE_T = [np.array([0.0, 0.45, 0.74]), np.array([0.85, 0.33, 0.10]),
           np.array([0.20, 0.55, 0.20])]
STATE_E = [np.array([0.0, 0.25, 0.54]), np.array([0.63, 0.11, 0.08]),
           np.array([0.10, 0.35, 0.10])]


def _spans(t, sigma):
    sigma = np.asarray(sigma, int); sp = []; s = 0
    for k in range(1, len(sigma)):
        if sigma[k] != sigma[s]:
            sp.append((t[s], t[k], int(sigma[s]))); s = k
    sp.append((t[s], t[-1], int(sigma[s]))); return sp


def _shade(ax, sp):
    for (t0, t1, m) in sp:
        ax.axvspan(t0, t1, color=MODE_COLORS[m], alpha=0.12, lw=0)


def _switched(ax, t, y, active, color, lw):
    a = np.asarray(active, bool); n = len(t)
    flips = np.where(np.diff(a.astype(int)) != 0)[0] + 1
    runs = np.concatenate([[0], flips, [n]])
    for k in range(len(runs) - 1):
        sl = slice(runs[k], min(runs[k + 1] + 1, n))
        ax.plot(t[sl], y[sl], "-" if a[runs[k]] else ":", color=color, lw=lw,
                alpha=1.0 if a[runs[k]] else 0.8)


import re as _re


def _sublabel(ax, text, y=-0.14, fontsize=15):
    """Bottom-centered panel label ('a) ...'); only the 'a)' marker is bold."""
    text = _re.sub(r"^([a-zA-Z])\)\s*", r"$\\mathbf{\1)}$ ", text)
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize)


# ---------------------------------------------------------------- convergence
def fig_convergence(res, savedir="hr_figures"):
    t = res["to"]; KTt = res["KTtrue_arr"]; KTh = res["KThatArray"]
    sp = _spans(t, res["sigma_arr"]); names = res["THETA_NAMES"]
    letters = ["a", "b", "c"]
    sw_times = [t[0]] + [s[1] for s in sp]          # exact switching instants

    # --- individual vertical spacing between the three graphs (figure fraction) ---
    GAP_AB = 0.055        # space between graph a and graph b (increase -> more space)
    GAP_BC = 0.055        # space between graph b and graph c
    LEFT, WIDTH = 0.12, 0.85
    Y_TOP, BOTTOM = 0.97, 0.13
    H = (Y_TOP - BOTTOM - GAP_AB - GAP_BC) / 3.0      # height of each graph
    bottoms = [Y_TOP - H, Y_TOP - 2 * H - GAP_AB, Y_TOP - 3 * H - GAP_AB - GAP_BC]

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.4), sharex=True)
    for i in range(3):
        ax = axes[i]; _shade(ax, sp)
        # raw logged estimate: no smoothing, no decimation
        ax.plot(t, KTt[i], color="black", lw=2.2, drawstyle="steps-post")
        ax.plot(t, KTh[i], color="red", lw=2.0, ls="--")
        # axis labels at 18 pt, as in the HIV parameter figure
        # (fig_kt_convergence.py); the y labels are pure math ($a_{\sigma(t)}$
        # etc.), whose lowercase italic glyphs read smaller than the upright
        # "time (units)" of the x label, so they get one extra point
        ax.set_ylabel(r"$%s_{\sigma(t)}$" % names[i], fontsize=20)
        ax.grid(True, alpha=0.25); ax.margins(y=0.25)
        ax.set_position([LEFT, bottoms[i], WIDTH, H])
    axes[-1].set_xlabel(r"time (units)", fontsize=18); axes[0].set_xlim(t[0], t[-1])
    axes[-1].set_xticks(sw_times)                    # ticks at exact switch times
    h = [Line2D([0], [0], color="black", lw=2.2, label="actual"),
         Line2D([0], [0], color="red", lw=2.0, ls="--", label="estimate")]
    # actual/estimate legend pinned to the upper-right corner of panel a)
    axes[0].legend(handles=h, loc="upper right", framealpha=0.95)
    mp = [Patch(facecolor=MODE_COLORS[m], alpha=0.4, label=MODE_LABELS[m]) for m in range(3)]
    axes[1].legend(handles=mp, loc="upper right", ncol=3, framealpha=0.95)
    _y0, _y1 = axes[1].get_ylim()                     # more headroom for the legend
    axes[1].set_ylim(_y0, _y1 + 0.18 * (_y1 - _y0))
    # --- put the three y labels on one vertical line ---------------------
    # A rotated y label is anchored on its text baseline, so glyphs with an
    # ascender ("b", "d") push ink further left than "a", which has none: even
    # with fig.align_ylabels the three read as misaligned.  Anchor all three at
    # a common x, measure the rendered ink, then shift each label so the LEFT
    # INK EDGES coincide (on the rightmost of the three, i.e. "a").
    _X0 = 0.045                                    # figure fraction
    def _set_ylabel_x(ax, x):
        ax.yaxis.set_label_coords(
            x, 0.5,
            transform=_mtrans.blended_transform_factory(fig.transFigure,
                                                        ax.transAxes))
    for ax in axes:
        _set_ylabel_x(ax, _X0)
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _x0 = [ax.yaxis.label.get_window_extent(_r).x0 / fig.bbox.width for ax in axes]
    _tgt = max(_x0)
    for ax, x0 in zip(axes, _x0):
        _set_ylabel_x(ax, _X0 + (_tgt - x0))

    # captions centered in each gap (a, b) and below the bottom graph (c)
    def _cap(s):
        return _re.sub(r"^([a-zA-Z])\)\s*", r"$\\mathbf{\1)}$ ", s)
    xc = LEFT + WIDTH / 2.0
    lab = [r"%s) parameter $%s_{\sigma(t)}$ error contraction" % (letters[i], names[i])
           for i in range(3)]
    # panel captions at 16 pt, as in fig_kt_convergence.py (HIV)
    fig.text(xc, (bottoms[0] + bottoms[1] + H) / 2.0, _cap(lab[0]),
             ha="center", va="center", fontsize=16)
    fig.text(xc, (bottoms[1] + bottoms[2] + H) / 2.0, _cap(lab[1]),
             ha="center", va="center", fontsize=16)
    fig.text(xc, bottoms[2] - 0.095, _cap(lab[2]),
             ha="center", va="center", fontsize=16)
    os.makedirs(savedir, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{savedir}/fig_HR_convergence.{e}", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- states
def fig_states(res, savedir="hr_figures"):
    t = res["to"]; XP = res["xp"]; XH = res["xhat"]; sp = _spans(t, res["sigma_arr"])
    sig = res["sigma_arr"].astype(int)
    HEIGHT_RATIOS = [1, 1, 1, 1, 0.58]
    # --- individual vertical gaps between consecutive graphs (figure fraction) ---
    GAP_AB = 0.032        # a - b
    GAP_BC = 0.032        # b - c
    GAP_CD = 0.032        # c - d
    GAP_DE = 0.032        # d - e
    GAPS = [GAP_AB, GAP_BC, GAP_CD, GAP_DE]
    LEFT, WIDTH = 0.12, 0.85
    Y_TOP, BOTTOM = 0.98, 0.09
    _u = (Y_TOP - BOTTOM - sum(GAPS)) / sum(HEIGHT_RATIOS)
    heights = [_u * r for r in HEIGHT_RATIOS]
    bottoms = []; _b = Y_TOP
    for _k in range(5):
        _b -= heights[_k]; bottoms.append(_b)
        if _k < 4:
            _b -= GAPS[_k]

    fig, axes = plt.subplots(5, 1, figsize=(8.0, 11.4), sharex=True,
                             gridspec_kw={"height_ratios": HEIGHT_RATIOS})
    # states named x_1 (membrane potential / output), x_2, x_3 to match the
    # HIV state figure's subscripted notation
    labs = [r"$x_1(t)$", r"$x_2(t)$", r"$x_3(t)$"]
    panel_labs = [r"a) State $x_1$ (output) estimation",
                  r"b) State $x_2$ estimation",
                  r"c) State $x_3$ estimation",
                  r"d) Per-mode norm parameter error",
                  r"e) Switching signal"]
    for i in range(3):
        ax = axes[i]; _shade(ax, sp)
        ax.plot(t, XP[i], color=STATE_T[i], lw=1.6, label="actual")
        ax.plot(t, XH[i], color=STATE_E[i], lw=1.4, ls="--", label="estimate")
        # axis labels at 18 pt, as in fig_states_stacked.py (HIV)
        ax.set_ylabel(labs[i], fontsize=18); ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right", ncol=2, framealpha=0.95)
    # theta-error norm per mode
    axe = axes[3]; KTc = res["KT_true_per_mode"]; KThA = res["KThatAll"]
    for m in range(3):
        err = KTc[:, m][:, None] - KThA[:, m, :]
        rel = np.sqrt(np.sum(err ** 2, 0)) / np.sqrt(np.sum(KTc[:, m] ** 2))
        _switched(axe, t, rel, sig == m, MODE_COLORS[m], 2.0)
    _shade(axe, sp); axe.set_ylim(0, 0.5); axe.set_ylabel(r"$\|\tilde{\theta}_{\ell}\|/\|\theta_{\ell}\|$", fontsize=18)
    axe.grid(True, alpha=0.2)
    # single column at 16 pt, as in the HIV theta-error panel
    # (fig_states_stacked.py)
    axe.legend(handles=[Line2D([0], [0], color=MODE_COLORS[m], lw=2, label=OBS_LABELS[m])
                        for m in range(3)], loc="upper right", ncol=1,
               framealpha=0.9, fontsize=16)
    axs = axes[4]; axs.step(t, sig + 1, where="post", color="black", lw=1.6); _shade(axs, sp)
    axs.set_yticks(MODE_IDS); axs.set_ylim(MODE_IDS[0] - 0.4, MODE_IDS[-1] + 0.4)
    axs.set_ylabel(r"$\sigma(t)$", fontsize=18)
    axs.set_xlabel(r"time (units)", fontsize=18); axs.grid(True, alpha=0.2)
    axes[0].set_xlim(t[0], t[-1])
    for _k in range(5):
        axes[_k].set_position([LEFT, bottoms[_k], WIDTH, heights[_k]])
    fig.align_ylabels(axes)
    # captions centered in each gap; e) below the bottom graph
    def _cap(s):
        return _re.sub(r"^([a-zA-Z])\)\s*", r"$\\mathbf{\1)}$ ", s)
    xc = LEFT + WIDTH / 2.0
    # panel captions at 16 pt, as in the HIV state figure (fig_states_stacked.py)
    for _k in range(4):
        yc = (bottoms[_k] + bottoms[_k + 1] + heights[_k + 1]) / 2.0
        fig.text(xc, yc, _cap(panel_labs[_k]), ha="center", va="center", fontsize=16)
    fig.text(xc, bottoms[4] - 0.072, _cap(panel_labs[4]),
             ha="center", va="center", fontsize=16)
    os.makedirs(savedir, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{savedir}/fig_HR_states.{e}", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- PE
def fig_pe(res, savedir="hr_figures"):
    pe = M.compute_pe(res); names = res["THETA_NAMES"]
    cols = [np.array([0.0, 0.45, 0.74]), np.array([0.85, 0.33, 0.10]), np.array([0.49, 0.18, 0.56])]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True)
    tag = ["%s) switching mode %d" % (lt, m)
           for lt, m in zip(["a", "b", "c"], MODE_IDS)]
    for ell in range(3):
        mid = MODE_IDS[ell]
        ax = axes[ell]; d = pe["modes"][ell]; el = d["elapsed"]; exc = set(d["excited"].tolist())
        for i in range(3):
            yy = np.maximum(d["diag"][:, i], 1e-300)
            if i in exc:
                ax.plot(el, yy, color=cols[i], lw=2.2, label=r"$\mathcal{N}_{%d,%s}$" % (mid, names[i]))
            else:
                ax.plot(el, yy, color=cols[i], lw=1.4, ls="--", alpha=0.6,
                        label=r"$\mathcal{N}_{%d,%s}$ (weak)" % (mid, names[i]))
        ax.plot(el, np.maximum(d["lam_red"], 1e-300), color="black", lw=2.4,
                label=r"$\lambda_{\min}$")
        # single excitation threshold shared by every mode (see M.DELTA_POLICY).
        # Drawn without a legend entry: the same delta now appears in all three
        # panels, so its value belongs in the caption rather than in each legend.
        _dl = max(d["delta"], 1e-300)
        ax.axhline(_dl, color="black", ls=":", lw=1.2)
        if np.isfinite(d["Tmin"]):
            ax.axvline(d["Tmin"], color="red", ls="-.", lw=1.2)
            _toff = (16, -26) if ell == 2 else (6, -14)
            ax.annotate(r"$T_{\min,%d}\approx%.0f$ units" % (mid, d["Tmin"]),
                        xy=(d["Tmin"], max(d["delta"], 1e-300)), xytext=_toff,
                        textcoords="offset points", fontsize=13, color="0.2")
            _ty, _tva = 0.975, "top"
            ax.annotate("Dwell-time",
                        xy=(d["Tmin"], _ty), xycoords=ax.get_xaxis_transform(),
                        xytext=(-3, 0), textcoords="offset points",
                        rotation=90, va=_tva, ha="right", fontsize=12, color="0.25")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 5e1)
        ax.set_xlabel(r"Active time interval (units)", fontsize=16)
        if ell == 0:
            ax.set_ylabel("Excitation level", fontsize=16)
        # Axis styling kept in sync with the HIV PE figure
        # (fig_pe_validation.py): axis labels at 16 pt, tick labels at 15 pt and
        # the log y-axis labelled with the base-10 exponent only (-4, -3, ..., 1)
        # instead of 10^{k}.  Set locally rather than in the module rcParams so
        # the states/convergence figures keep their own sizes.
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.set_yticks([10.0 ** k for k in range(-4, 2)])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{int(np.round(np.log10(y)))}"))
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
        _sublabel(ax, tag[ell], y=-0.22, fontsize=17)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    os.makedirs(savedir, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(f"{savedir}/fig_HR_PE.{e}", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Plot the saved run when there is one, so restyling the figures never
    # re-runs the integrator (and never silently plots a different run).
    # Delete hr_data.npz -- or run `python hr_mw_ao.py` -- to regenerate it.
    if os.path.exists(M.HR_DATA):
        res = M.load_data()
        print(f"Loaded {M.HR_DATA}")
    else:
        print(f"{M.HR_DATA} not found -- running the simulation once")
        res = M.run(verbose=True)
        M.save_data(res)
        print(f"Saved run data to {M.HR_DATA}")
    fig_convergence(res); fig_states(res); fig_pe(res)
    print("Saved hr_figures/fig_HR_{convergence,states,PE}.{pdf,png}")
