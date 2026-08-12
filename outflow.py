import numpy as np

# Free flow velocity
vf = 1.0
# Length of road segment
L = 1.0
# Max traffic density
rho_m = 1.0


# Greenshields model for traffic flow
def B(x: float | np.ndarray):
    return vf * x/L * (1 - x/(L * rho_m))


# Largest outflow Greenshields can sustain, attained at N = L * rho_m / 2.
B_max = vf * L * rho_m / 4


# Linear outflow B(N) = kN, used by the closed-form derivations. The default k
# is the tangent to the Greenshields curve at the origin.
def make_linear_B(k: float = vf / L):
    def B_linear(x: float | np.ndarray):
        return k * x

    return B_linear
