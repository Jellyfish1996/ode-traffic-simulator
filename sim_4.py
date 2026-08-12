import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

from outflow import B, make_linear_B, B_max, vf, L

Outflow = Callable[[float | np.ndarray], float | np.ndarray]

# Arrival rates onto lane 1 and lane 2. Unequal, otherwise the imbalance decays
# to zero and lambda has no visible effect.
arrivals = np.array([0.20, 0.05])

# Lane changing rate.
lam = 0.3

# Slope of the linear outflow, the tangent to Greenshields at the origin.
k = vf / L


# Model 4. State is [N_1, N_2].
def make_two_lane_system(A: np.ndarray, lam: float, B_fn: Outflow = B):
    A = np.asarray(A, float)
    assert A.shape == (2,), "need one arrival rate per lane"

    def rhs(t: float, N: np.ndarray):
        exchange = lam * (N[0] - N[1])
        return A - B_fn(N) + np.array([-exchange, exchange])

    return rhs


# The closed form from the report. Only valid for B(N) = kN, where the sum
# relaxes at rate k and the difference at rate k + 2*lam.
def two_lane_closed_form(A: np.ndarray, lam: float, k: float,
                         N0: np.ndarray, ts: np.ndarray):
    S0, D0 = N0[0] + N0[1], N0[0] - N0[1]
    S_inf = (A[0] + A[1]) / k
    D_inf = (A[0] - A[1]) / (k + 2 * lam)

    S = S_inf + (S0 - S_inf) * np.exp(-k * ts)
    D = D_inf + (D0 - D_inf) * np.exp(-(k + 2 * lam) * ts)
    return 0.5 * (S + D), 0.5 * (S - D)


N0 = np.array([0.0, 0.0])
t_span = (0.0, 20.0)
ts = np.linspace(*t_span, 400)

# Greenshields has no equilibrium if a lane is fed faster than it can drain.
assert arrivals.max() < B_max + lam * abs(arrivals[0] - arrivals[1]), \
    "arrival rate exceeds capacity, the lane will fill without bound"

linear_sol = solve_ivp(make_two_lane_system(arrivals, lam, make_linear_B(k)),
                       t_span, N0, t_eval=ts, rtol=1e-8, atol=1e-10)
greenshields_sol = solve_ivp(make_two_lane_system(arrivals, lam, B),
                             t_span, N0, t_eval=ts, rtol=1e-8, atol=1e-10)

N1_exact, N2_exact = two_lane_closed_form(arrivals, lam, k, N0, ts)

# Checks the algebra in the report against an independent integration.
err = max(np.abs(linear_sol.y[0] - N1_exact).max(),
          np.abs(linear_sol.y[1] - N2_exact).max())
print(f"max |numerical - closed form| for the linear model: {err:.2e}")

print(f"linear equilibrium:       N_1 = {N1_exact[-1]:.4f}, N_2 = {N2_exact[-1]:.4f}")
print(f"Greenshields equilibrium: N_1 = {greenshields_sol.y[0, -1]:.4f}, "
      f"N_2 = {greenshields_sol.y[1, -1]:.4f}")
