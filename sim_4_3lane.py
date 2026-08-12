import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

from outflow import B, make_linear_B, B_max, vf, L

Outflow = Callable[[float | np.ndarray], float | np.ndarray]

# Model 4 widened from two lanes to any number of lanes in a row. Lanes 1 and
# 3 are not adjacent, so exchange runs neighbour to neighbour along a path.
# For two lanes the Laplacian reduces to the original +/- lam (N_1 - N_2).

# Arrival rates per lane. Unequal, otherwise the imbalances decay to zero and
# lambda has no visible effect.
arrivals_2 = np.array([0.20, 0.05])
arrivals_3 = np.array([0.20, 0.10, 0.05])

lam = 0.3
k = vf / L


def path_laplacian(n: int) -> np.ndarray:
    """Graph Laplacian of n lanes in a row, adjacent lanes connected."""
    A = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    return np.diag(A.sum(axis=1)) - A


# State is [N_1, ..., N_n]. Passing n = 2 reproduces make_two_lane_system.
def make_lane_system(A: np.ndarray, lam: float, B_fn: Outflow = B):
    A = np.asarray(A, float)
    assert A.ndim == 1 and A.size >= 2, "need one arrival rate per lane"
    lap = path_laplacian(A.size)

    def rhs(t: float, N: np.ndarray):
        return A - B_fn(N) - lam * (lap @ N)

    return rhs


# The closed form, valid only for B(N) = kN. The report's sum and difference
# substitution is the eigenbasis of the Laplacian: (1, 1) with eigenvalue 0 and
# (1, -1) with eigenvalue 2, hence k and k + 2*lam. Diagonalising for general n
# gives one decoupled mode per eigenvalue, relaxing at k + lam * mu.
def lane_closed_form(A: np.ndarray, lam: float, k: float,
                     N0: np.ndarray, ts: np.ndarray):
    A = np.asarray(A, float)
    mu, Q = np.linalg.eigh(path_laplacian(A.size))
    rates = k + lam * mu

    a = Q.T @ A
    c0 = Q.T @ np.asarray(N0, float)
    c_inf = a / rates

    # modes on the rows, time on the columns
    c = c_inf[:, None] + (c0 - c_inf)[:, None] * np.exp(-np.outer(rates, ts))
    return Q @ c


if __name__ == "__main__":
    N0_2, N0_3 = np.zeros(2), np.zeros(3)
    t_span = (0.0, 20.0)
    ts = np.linspace(*t_span, 400)

    # Greenshields has no equilibrium if a lane is fed faster than it can drain.
    assert arrivals_3.max() < B_max, \
        "arrival rate exceeds capacity, the lane will fill without bound"

    print(f"path Laplacian eigenvalues, 2 lanes: "
          f"{np.linalg.eigvalsh(path_laplacian(2)).round(3)}")
    print(f"path Laplacian eigenvalues, 3 lanes: "
          f"{np.linalg.eigvalsh(path_laplacian(3)).round(3)}")
    print(f"so the modes relax at k, k + {lam:.1f} and k + {3 * lam:.1f}\n")

    for name, A, N0 in (("2 lanes", arrivals_2, N0_2),
                        ("3 lanes", arrivals_3, N0_3)):
        lin = solve_ivp(make_lane_system(A, lam, make_linear_B(k)), t_span, N0,
                        t_eval=ts, rtol=1e-10, atol=1e-12)
        exact = lane_closed_form(A, lam, k, N0, ts)
        err = np.abs(lin.y - exact).max()
        green = solve_ivp(make_lane_system(A, lam, B), t_span, N0, t_eval=ts,
                          rtol=1e-10, atol=1e-12)
        print(f"{name}: max |numerical - closed form| = {err:.2e}")
        print(f"   linear equilibrium:       {exact[:, -1].round(4)}")
        print(f"   Greenshields equilibrium: {green.y[:, -1].round(4)}")
