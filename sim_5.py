import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

# Free flow velocity
vf = 1.0
# Length of road segment
L = 1.0
# Max traffic density
rho_m = 1.0

# Define turning fractions matrix. Shape is (n_out, n_in): column j is how
# inbound road j splits across the outbound roads, so columns sum to 1.
turning_fractions = np.array([[0.5, 0.5], [0.5, 0.5]])

# Constant arrival rates onto each inbound road. Length n_in.
arrivals = np.array([0.2, 0.2])

# Greenshields model for traffic flow
def B(x: float | np.ndarray):
    return vf * x/L * (1 - x/(L * rho_m))


# Gate function. Just pass durations. For the n=2 case discussed in the 
# report take a T and then just use [T, T].
def make_gate(durations: list[float]):
    durations = np.asarray(durations, float)
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    T = edges[-1]
    n = len(durations)
    def g(ts):
        idx = np.searchsorted(edges, np.asarray(ts, float) % T, side="right") - 1
        return (idx[..., None] == np.arange(n)).astype(float)
    return g

Gate = Callable[[float], np.ndarray]

# This is equivalent to [dN_1/dt, dN_2/dt] in the latex
def inflows_system(A: np.ndarray, N_in: np.ndarray, t: float, g: Gate):
    return A - g(t) * B(N_in)

# inflow is a vector. should be equal to inflows
def outflow_system(t: float, fracs: np.ndarray, inflow: np.ndarray, N_outs: np.ndarray):
    return fracs @ inflow - B(N_outs)


# The whole system, for any n_in and n_out. Both counts come from the shape of
# fracs. Returns the right-hand side and the state dimension n_in + n_out, with
# the state laid out as [N_in (n_in), N_out (n_out)].
def make_whole_system(A: np.ndarray, fracs: np.ndarray, g: Gate):
    A = np.asarray(A, float)
    fracs = np.asarray(fracs, float)
    n_out, n_in = fracs.shape
    assert A.shape == (n_in,), f"arrivals must have length n_in={n_in}"
    assert np.allclose(fracs.sum(axis=0), 1.0), "columns of fracs must sum to 1"
    assert g(0.0).shape == (n_in,), f"gate must return a vector of length n_in={n_in}"

    def whole_system(t: float, N: np.ndarray):
        N_in, N_outs = N[:n_in], N[n_in:]
        inflow_values = g(t) * B(N_in)

        inflows = inflows_system(A, N_in, t, g)
        outflows = outflow_system(t, fracs, inflow_values, N_outs)
        return np.concatenate([inflows, outflows])

    return whole_system, n_in + n_out

# Numerically solve it.
# Technically, the inflows system has a closed form, but coding it this way is
# easier.
# Default: the n=2 in / n=2 out case discussed in the report.
rhs, dim = make_whole_system(arrivals, turning_fractions, make_gate([1.0, 1.0]))
sol = solve_ivp(rhs, [0, 10], np.zeros(dim))