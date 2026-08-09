import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

# Free flow velocity
vf = 1.0
# Length of road segment
L = 1.0
# Max traffic density
rho_m = 1.0

# Define turning fractions matrix
turning_fractions = np.array([[0.5, 0.5], [0.5, 0.5]])

# Constant arrival rates onto each inbound road
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



def whole_system(t: float, N: np.ndarray, g: Gate):
    # Hardcoded to 2 inputs and 2 outputs for now. 
    N_in, N_outs = N[:2], N[2:]
    inflow_values = g(t) * B(N_in)
    
    inflows = inflows_system(arrivals, N_in, t, g)
    outflows = outflow_system(t, turning_fractions, inflow_values, N_outs)
    return np.concatenate([inflows, outflows])


sol = solve_ivp(whole_system, [0, 10], np.zeros(4), args=(make_gate([1.0, 1.0]),))