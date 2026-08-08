import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

N1_values[0] = N1_intial
N2_values[0] = N2_intial
N3_values[0] = N3_intial
N4_values[0] = N4_intial




for i in range(len(time) - 1):

    t = time[i]

    N1 = N1_values[i]
    N2 = N2_values[i]
    N3 = N3_values[i]
    N4 = N4_values[i]

    B1 = B(N1)
    B2 = B(N2)
    B3 = B(N3)
    B4 = B(N4)

    G1 = g1(t)
    G2 = 1 - G1

    dN1_dt = A1 - G1 * B1
    dN2_dt = A2 - G2 * B2

    dN3_dt = p13 * G1 * B1 + p23 * G2 * B2 - B3
    dN4_dt = p14 * G1 * B1 + p24 * G2 * B2 - B4

    N1_values[i + 1] = N1 + dt * dN1_dt
    N2_values[i + 1] = N2 + dt * dN2_dt
    N3_values[i + 1] = N3 + dt * dN3_dt
    N4_values[i + 1] = N4 + dt * dN4_dt



    plt.plot(time, N1_values, label="N1")
    plt.plot(time, N2_values, label="N2")
    plt.plot(time, N3_values, label="N3")
    plt.plot(time, N4_values, label="N4")

    plt.xlabel("Time (minutes)")
    plt.ylabel("Number of vehicles")
    plt.title("Traffic Flow Through Signalized Intersection")

    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()