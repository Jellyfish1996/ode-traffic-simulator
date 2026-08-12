import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from load_pems_eb import load_segment, SEGMENT_MILES
from sim_4_3lane import path_laplacian

# Model 4 driven by the measured arrivals, against the observed occupancies.
# Eastbound deck, the segment paper.tex cites.
#
# The deck carries three lanes. Fitting only two leaves lambda unidentifiable,
# since exchange with lane 3 gets absorbed into k. Outflow is linear
# throughout; Greenshields was tried and dropped, as the deck never leaves free
# flow and its jam density is unidentifiable here.
#
#     python plot_sim_4.py        writes model_4_pems.png
#     python plot_sim_4.py 3      writes model_4_3lane_pems.png

WINDOW = ("15:00", "18:00")
DATES = ("2026_08_03", "2026_01_06")

# Categorical slots 1 to 3 of the reference palette. Light mode CVD dE 24.7,
# normal vision dE 33.6. Slot 3 differs in lightness, not hue alone.
LANES = ("#2a78d6", "#eb6834", "#5b3a9e")
LANE_1, LANE_2 = LANES[0], LANES[1]
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"


def make_rhs(ts: np.ndarray, As, k: float, lam: float):
    lap = path_laplacian(len(As))

    def rhs(t: float, N: np.ndarray):
        A = np.array([np.interp(t, ts, a) for a in As])
        return A - k * N - lam * (lap @ N)

    return rhs


def predict(seg, k: float, lam: float, n_lanes: int = 2):
    ts = seg.t.to_numpy()
    As = [seg[f"A_{i+1}"].to_numpy() for i in range(n_lanes)]
    N0 = np.array([seg[f"N_{i+1}"].iloc[0] for i in range(n_lanes)])
    sol = solve_ivp(make_rhs(ts, As, k, lam), (ts[0], ts[-1]), N0, t_eval=ts,
                    rtol=1e-10, atol=1e-12)
    return sol.y


def fit(seg, n_lanes: int = 2):
    obs = [seg[f"N_{i+1}"].to_numpy() for i in range(n_lanes)]

    def residuals(p):
        Y = predict(seg, p[0], p[1], n_lanes)
        return np.concatenate([Y[i] - obs[i] for i in range(n_lanes)])

    k_guess = float(seg.A_1.mean() / seg.N_1.mean())
    # The default finite difference step is smaller than the solver's own error,
    # which flattens the Jacobian and strands the optimizer at its start.
    res = least_squares(residuals, [k_guess, 1.0], bounds=([1e-6, 0.0], np.inf),
                        diff_step=1e-5)
    rmse = np.sqrt(np.sum(res.fun**2) / len(res.fun))
    return res.x, rmse, standard_errors(res)


# One annotation for every model 4 figure, so the variants can be read side by
# side. The baseline is the standard deviation of the observations, which is
# what a mean-only prediction would score.
def annotate_fit(ax, k, lam, se, rmse, baseline):
    ax.annotate(f"k = {k:.2f} ($\\pm${se[0]:.2f})   "
                f"$\\lambda$ = {lam:.2f} ($\\pm${se[1]:.2f})   "
                f"rmse = {rmse:.1f} cars   (baseline {baseline:.1f})",
                (0, 1), xytext=(0, 6), xycoords="axes fraction",
                textcoords="offset points", color=INK_2, fontsize=8.5)


def baseline_sd(seg, n_lanes: int):
    obs = [seg[f"N_{i+1}"].to_numpy() for i in range(n_lanes)]
    return float(np.concatenate(obs).std())


def standard_errors(res):
    dof = len(res.fun) - len(res.x)
    try:
        cov = (np.sum(res.fun**2) / dof) * np.linalg.inv(res.jac.T @ res.jac)
        return np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        return np.full(len(res.x), np.nan)


def label_offsets(ends, span: float):
    """Nudge colliding end labels apart, spread about the group centre."""
    ys = sorted(e[0] for e in ends)
    if min(np.diff(ys), default=np.inf) >= 0.06 * span:
        return {y: 0.0 for y in ys}
    n = len(ys)
    return {y: 0.06 * span * (i - (n - 1) / 2) for i, y in enumerate(ys)}


def panel(ax, date: str, show_ylabel: bool, n_lanes: int = 2):
    seg = load_segment(date, *WINDOW)
    (k, lam), rmse, se = fit(seg, n_lanes)
    Y = predict(seg, k, lam, n_lanes)

    # Clock time reads better than the model's 5 minute index.
    hours = seg.index.hour + seg.index.minute / 60

    for i in range(n_lanes):
        ax.plot(hours, seg[f"N_{i+1}"], "o", ms=4.5, color=LANES[i], alpha=0.45,
                mew=0, zorder=2, label=f"Lane {i+1}, observed")
        ax.plot(hours, Y[i], "-", lw=2, color=LANES[i], zorder=3,
                label=f"Lane {i+1}, model")

    # Direct labels, nudged apart when lanes end within a few cars of each
    # other.
    ends = [(Y[i][-1], LANES[i], f"Lane {i+1}") for i in range(n_lanes)]
    offsets = label_offsets(ends, float(Y.max()))
    for y, colour, label in ends:
        ax.annotate(label, (hours[-1], y + offsets[y]), xytext=(6, 0),
                    textcoords="offset points", color=colour, fontsize=9,
                    va="center")

    ax.set_title(date.replace("_", "-"), color=INK, fontsize=11, loc="left",
                 pad=22)
    annotate_fit(ax, k, lam, se, rmse, baseline_sd(seg, n_lanes))

    ax.set_xlabel("Time of day", color=INK_2, fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Vehicles on the segment", color=INK_2, fontsize=9.5)
    # A feed can die early, so each panel ends where its own data does.
    first, last = float(hours[0]), float(hours[-1])
    ticks = np.arange(np.floor(first), np.ceil(last) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(h):02d}:00" for h in ticks])
    ax.set_xlim(first - 0.1, last + 0.55)
    ax.set_ylim(0, None)

    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=INK_3, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    return seg


def main(n_lanes: int = 2, out: str | None = None):
    assert n_lanes in (2, 3), "the deck carries three lanes"
    if out is None:
        out = "model_4_pems.png" if n_lanes == 2 else "model_4_3lane_pems.png"

    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                             facecolor=SURFACE)
    for ax, date in zip(axes, DATES):
        panel(ax, date, show_ylabel=ax is axes[0], n_lanes=n_lanes)

    # Separate text so the source line stays recessive.
    lanes_txt = "lanes 1 and 2" if n_lanes == 2 else "all three lanes"
    fig.text(0.008, 1.005, "Model 4 driven by measured arrivals, against "
             "observed occupancy", color=INK, fontsize=12.5, ha="left",
             va="bottom")
    fig.text(0.008, 0.975, f"Richmond-San Rafael bridge eastbound, VDS 421237 "
             f"to 421238, {SEGMENT_MILES} mi, {lanes_txt}", color=INK_2,
             fontsize=9.5, ha="left", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    # Two lanes fit one row; three need a column per lane to pair up.
    fig.legend(handles, labels, loc="lower center",
               ncol=4 if n_lanes == 2 else n_lanes, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
