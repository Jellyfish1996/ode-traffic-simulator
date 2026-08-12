"""The Goal Of Model 4 is to model what would happen with 2 lanes.
The time step is 5 min this is because it is the minimum of timestep the traffic data has.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



N1_initial =
N2_initial =

time =
Y_data_N1 =
Y_data_N2 =

S0 = N1_initial + N2_initial
D0 = N1_initial - N2_initial

def analytical_solution_N1(t, A1, A2, k, lam):
    """Gives the solution to the first lane (N1) solution"""

    N1 = 0.5 * ((A1 + A2) / k + (S0 - (A1 + A2) / k) * np.exp(-k * t) + (A1 - A2) / (k + 2 * lam) + (D0 - (A1 - A2)
    / (k + 2 * lam)) * np.exp(-(k + 2 * lam) * t))

    return N1

def analytical_solution_N2(t, A1, A2, k, lam):
    """Gives the solution to the second lane (N2) solution"""
    N2 = 0.5 * ((A1 + A2) / k + (S0 - (A1 + A2) / k) * np.exp(-k * t) - (A1 - A2) / (k + 2 * lam)- (D0 - (A1 - A2) /
        (k + 2 * lam)) * np.exp(-(k + 2 * lam) * t))

    return N2

def coupled_solution(t, A1, A2, k, lam):

    N1 = analytical_solution_N1(t, A1, A2, k, lam)
    N2 = analytical_solution_N2(t, A1, A2, k, lam)

    return np.concatenate((N1, N2))

Y_data_combined = np.concatenate((Y_data_N1, Y_data_N2))


popt, pcov = curve_fit(coupled_solution, time, Y_data_combined, p0=[1, 1, 0.1, 0.1], bounds=([0, 0, 0.000001, 0],
            [np.inf, np.inf, np.inf, np.inf]), maxfev=10000)
unc = np.sqrt(np.diag(pcov))


# Plot measured data.
plt.scatter(time, Y_data_N1, label="Lane 1 data")
plt.scatter(time, Y_data_N2, label="Lane 2 data")

# Plotting the graph
plt.plot(time, analytical_solution_N1(time, *popt), 'r-', label="Lane 1")
plt.plot(time, analytical_solution_N2(time, *popt), 'b-', label="Lane 2")
plt.xlabel('Time measured in 5 minute increments')
plt.ylabel('Cars in Lanes')
plt.legend(loc='best')
plt.title("How cars switch between lanes 1 and 2")
plt.tight_layout()
plt.show()

print("A1, A2, k, lambda =", popt)
print("Parameter uncertainties =", unc)