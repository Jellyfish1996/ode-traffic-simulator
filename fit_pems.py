import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from load_pems import load_segment, load_segment_raw

# Fits model 4 to the bridge data. The report assumes constant arrivals, but the
# measured arrivals climb all morning, so we feed the measured A_1(t), A_2(t)
# into a numerical solve rather than using the closed form.


def make_rhs(ts: np.ndarray, A1: np.ndarray, A2: np.ndarray, k: float, lam: float):
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


def residuals(seg, k: float, lam: float):
    N1, N2 = predict(seg, k, lam)
    return np.concatenate([N1 - seg.N_1.to_numpy(), N2 - seg.N_2.to_numpy()])


def fit(seg, fix_lam: float | None = None):
    # At equilibrium A = kN, so the mean ratio is a good starting guess for k.
    k_guess = float(seg.A_1.mean() / seg.N_1.mean())

    if fix_lam is None:
        f = lambda p: residuals(seg, p[0], p[1])
        p0, lo, hi = [k_guess, 0.1], [1e-6, 0.0], [np.inf, np.inf]
    else:
        f = lambda p: residuals(seg, p[0], fix_lam)
        p0, lo, hi = [k_guess], [1e-6], [np.inf]

    # The default finite difference step is smaller than the ODE solver's own
    # error, which makes the Jacobian collapse to zero and strands the optimizer
    # at its starting point.
    res = least_squares(f, p0, bounds=(lo, hi), diff_step=1e-5)
    n, p = len(res.fun), len(p0)
    rmse = np.sqrt(np.sum(res.fun**2) / n)

    # Standard errors from the Gauss-Newton approximation to the covariance.
    s2 = np.sum(res.fun**2) / (n - p)
    cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
    return res.x, np.sqrt(np.diag(cov)), rmse


def report(date: str, raw: bool = False):
    seg = load_segment_raw(date) if raw else load_segment(date)
    (k, lam), (k_se, lam_se), rmse = fit(seg)
    (k0,), (k0_se,), rmse0 = fit(seg, fix_lam=0.0)

    print(f"\n{date}  {'30 second' if raw else '5 minute'}  "
          f"({len(seg)} points, 05:00-08:00)")
    print(f"  full model    k = {k:.4f} +/- {k_se:.4f}   "
          f"lambda = {lam:.4f} +/- {lam_se:.4f}   rmse = {rmse:.2f} cars")
    print(f"  lambda = 0    k = {k0:.4f} +/- {k0_se:.4f}   "
          f"{'':<28} rmse = {rmse0:.2f} cars")
    print(f"  lane changing improves rmse by {100 * (1 - rmse / rmse0):.1f}%")
    print(f"  mean |N_1 - N_2| in the data: {np.abs(seg.N_1 - seg.N_2).mean():.2f} cars")
    return seg, (k, lam)


if __name__ == "__main__":
    args = sys.argv[1:]
    raw = "--raw" in args
    dates = [a for a in args if not a.startswith("--")] or ["2026_08_03", "2026_01_06"]
    for d in dates:
        report(d, raw=raw)
