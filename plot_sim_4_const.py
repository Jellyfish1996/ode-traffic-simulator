import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from load_pems_eb import load_segment, SEGMENT_MILES
from plot_sim_4 import (WINDOW, DATES, LANE_1, LANE_2, INK, INK_2, INK_3,
                        SURFACE, annotate_fit, baseline_sd, standard_errors)

# Control for model 4: same segment, window and dates as plot_sim_4.py, but
# arrivals held at their window means. That is the constant A case the report
# solves in closed form, so the equilibrium comes from the formula rather than
# the solver. Shows how much of the agreement in model_4_pems.png survives
# without the real arrival signal.


def predict_const(seg, k: float, lam: float):
    ts = seg.t.to_numpy()
    A = np.array([seg.A_1.mean(), seg.A_2.mean()])
    N0 = np.array([seg.N_1.iloc[0], seg.N_2.iloc[0]])

    def rhs(t: float, N: np.ndarray):
        exchange = lam * (N[0] - N[1])
        return A - k * N + np.array([-exchange, exchange])

    sol = solve_ivp(rhs, (ts[0], ts[-1]), N0, t_eval=ts, rtol=1e-10, atol=1e-12)
    return sol.y


def fit_const(seg):
    def residuals(p):
        N1, N2 = predict_const(seg, p[0], p[1])
        return np.concatenate([N1 - seg.N_1.to_numpy(), N2 - seg.N_2.to_numpy()])

    k_guess = float(seg.A_1.mean() / seg.N_1.mean())
    # Same Jacobian caveat as the driven fit.
    res = least_squares(residuals, [k_guess, 1.0], bounds=([1e-6, 0.0], np.inf),
                        diff_step=1e-5)
    rmse = np.sqrt(np.sum(res.fun**2) / len(res.fun))
    return res.x, rmse, standard_errors(res)


def equilibria(seg, k: float, lam: float):
    """Closed form fixed point of the two lane system under constant arrivals."""
    A1, A2 = seg.A_1.mean(), seg.A_2.mean()
    S = (A1 + A2) / k
    D = (A1 - A2) / (k + 2 * lam)
    return 0.5 * (S + D), 0.5 * (S - D)


def panel(ax, date: str, show_ylabel: bool):
    seg = load_segment(date, *WINDOW)
    (k, lam), rmse, se = fit_const(seg)
    N1, N2 = predict_const(seg, k, lam)
    eq1, eq2 = equilibria(seg, k, lam)

    hours = seg.index.hour + seg.index.minute / 60

    for obs, model, eq, colour, label in (
            (seg.N_1, N1, eq1, LANE_1, "Lane 1"),
            (seg.N_2, N2, eq2, LANE_2, "Lane 2")):
        ax.plot(hours, obs, "o", ms=4.5, color=colour, alpha=0.45, mew=0,
                zorder=2, label=f"{label}, observed")
        ax.plot(hours, model, "-", lw=2, color=colour, zorder=4,
                label=f"{label}, model")
        # Span only the data, not the full axis, or it runs through the
        # labels.
        ax.plot([hours[0], hours[-1]], [eq, eq], color=colour, lw=0.9,
                ls=(0, (4, 3)), alpha=0.55, zorder=3)

    # The curves are flat and close, so label the equilibria and nudge them
    # apart.
    ends = sorted(((eq1, LANE_1, "Lane 1"), (eq2, LANE_2, "Lane 2")),
                  key=lambda e: e[0])
    span = max(N1.max(), N2.max())
    offsets = ([-0.035 * span, 0.035 * span]
               if ends[1][0] - ends[0][0] < 0.07 * span else [0.0, 0.0])
    for (y, colour, label), dy in zip(ends, offsets):
        ax.annotate(f"{label}, $N^*$ = {y:.0f}", (hours[-1], y + dy),
                    xytext=(6, 0), textcoords="offset points", color=colour,
                    fontsize=9, va="center")

    ax.set_title(date.replace("_", "-"), color=INK, fontsize=11, loc="left",
                 pad=22)
    annotate_fit(ax, k, lam, se, rmse, baseline_sd(seg, 2))

    ax.set_xlabel("Time of day", color=INK_2, fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel("Vehicles on the segment", color=INK_2, fontsize=9.5)
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


def main(out: str = "model_4_const.png"):
    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                             facecolor=SURFACE)
    for ax, date in zip(axes, DATES):
        panel(ax, date, show_ylabel=ax is axes[0])

    fig.text(0.008, 1.005, "Model 4 with arrivals held constant, against "
             "observed occupancy", color=INK, fontsize=12.5, ha="left",
             va="bottom")
    fig.text(0.008, 0.975, f"Richmond-San Rafael bridge eastbound, VDS 421237 "
             f"to 421238, {SEGMENT_MILES} mi, lanes 1 and 2. Dashed lines are "
             f"the closed form equilibria.", color=INK_2, fontsize=9.5,
             ha="left", va="bottom")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
