import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from load_pems_eb import load_segment, SEGMENT_MILES

# Figure for model 4: the two lane ODE driven by the measured arrivals, against
# the occupancies actually observed at the downstream detector. Eastbound deck,
# the same segment paper.tex cites, so k is comparable with the other models.

WINDOW = ("15:00", "18:00")
DATES = ("2026_08_03", "2026_01_06")

# Categorical slots 1 and 2 of the reference palette. Validated all pairs, light
# mode: CVD dE 24.7, normal vision dE 33.6, both above the floor.
LANE_1, LANE_2 = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"


def make_rhs(ts: np.ndarray, A1: np.ndarray, A2: np.ndarray,
             k: float, lam: float):
    def rhs(t: float, N: np.ndarray):
        A = np.array([np.interp(t, ts, A1), np.interp(t, ts, A2)])
        exchange = lam * (N[0] - N[1])
        return A - k * N + np.array([-exchange, exchange])

    return rhs


def predict(seg, k: float, lam: float):
    ts = seg.t.to_numpy()
    N0 = np.array([seg.N_1.iloc[0], seg.N_2.iloc[0]])
    sol = solve_ivp(make_rhs(ts, seg.A_1.to_numpy(), seg.A_2.to_numpy(), k, lam),
                    (ts[0], ts[-1]), N0, t_eval=ts, rtol=1e-10, atol=1e-12)
    return sol.y


def fit(seg):
    def residuals(p):
        N1, N2 = predict(seg, p[0], p[1])
        return np.concatenate([N1 - seg.N_1.to_numpy(), N2 - seg.N_2.to_numpy()])

    k_guess = float(seg.A_1.mean() / seg.N_1.mean())
    # The default finite difference step is smaller than the solver's own error,
    # which flattens the Jacobian and strands the optimizer at its start.
    res = least_squares(residuals, [k_guess, 1.0], bounds=([1e-6, 0.0], np.inf),
                        diff_step=1e-5)
    rmse = np.sqrt(np.sum(res.fun**2) / len(res.fun))
    return res.x, rmse


def panel(ax, date: str, show_ylabel: bool):
    seg = load_segment(date, *WINDOW)
    (k, lam), rmse = fit(seg)
    N1, N2 = predict(seg, k, lam)

    # Clock time reads better than the model's 5 minute index.
    hours = seg.index.hour + seg.index.minute / 60

    for obs, model, colour, label in ((seg.N_1, N1, LANE_1, "Lane 1"),
                                      (seg.N_2, N2, LANE_2, "Lane 2")):
        ax.plot(hours, obs, "o", ms= 4.5, color=colour, alpha=0.45,
                mew=0, zorder=2, label=f"{label}, observed")
        ax.plot(hours, model, "-", lw=2, color=colour, zorder=3,
                label=f"{label}, model")

    # Direct labels instead of relying on the legend alone for identity. When
    # the two lanes end within a few cars of each other the labels collide, so
    # nudge them apart rather than letting one hide the other.
    ends = [(N1[-1], LANE_1, "Lane 1"), (N2[-1], LANE_2, "Lane 2")]
    ends.sort(key=lambda e: e[0])
    span = max(N1.max(), N2.max())
    if ends[1][0] - ends[0][0] < 0.06 * span:
        offsets = [-0.03 * span, 0.03 * span]
    else:
        offsets = [0.0, 0.0]
    for (y, colour, label), dy in zip(ends, offsets):
        ax.annotate(label, (hours[-1], y + dy), xytext=(6, 0),
                    textcoords="offset points", color=colour, fontsize=9,
                    va="center")

    ax.set_title(date.replace("_", "-"), color=INK, fontsize=11, loc="left",
                 pad=22)
    ax.annotate(f"k = {k:.2f}   $\\lambda$ = {lam:.2f}   rmse = {rmse:.1f} cars",
                (0, 1), xytext=(0, 6), xycoords="axes fraction",
                textcoords="offset points", color=INK_2, fontsize=8.5)

    ax.set_xlabel("Time of day", color=INK_2, fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Vehicles on the segment", color=INK_2, fontsize=9.5)
    # A feed can die before the window closes, so each panel ends where its own
    # data does rather than carrying empty axis to the nominal end time.
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


def main(out: str = "model_4_pems.png"):
    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                             facecolor=SURFACE)
    for ax, date in zip(axes, DATES):
        panel(ax, date, show_ylabel=ax is axes[0])

    # Title and subtitle as separate text so the source line stays recessive.
    fig.text(0.008, 1.005, "Model 4 driven by measured arrivals, against "
             "observed occupancy", color=INK, fontsize=12.5, ha="left",
             va="bottom")
    fig.text(0.008, 0.975, f"Richmond-San Rafael bridge eastbound, VDS 421237 "
             f"to 421238, {SEGMENT_MILES} mi, lanes 1 and 2", color=INK_2,
             fontsize=9.5, ha="left", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
