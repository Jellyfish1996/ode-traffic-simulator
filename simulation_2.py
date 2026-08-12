import math
import matplotlib.pyplot as plt
# So this will be our Linear outflow model where A is a constant function, B is proportional to the number of vehicles on
# the road (i.e. B(N) = kN(t))

# So we get an ODE like N'(t) = A - kN(t), solving we obtain: N(t) = Ce^{-kt} + A/k

def run_simulation_2(initial_vehicles: float, A: float, k: float) :
    # C = initial vehicles - A/k
    C = initial_vehicles - A/k
    time = []
    vehicles = []

    for t in range(13):
        N = C*math.exp(-k*t) + A/k
        time.append(t)
        vehicles.append(N)
        print(f"time = {t*5} min, vehicles = {C*math.exp(-k*t) + A/k}")

    return time, vehicles

# run_simulation_2()
time, vehicles = run_simulation_2(61.7, 470, 5.366)

plt.plot(time, vehicles, 'r-', label="Lane 1")
plt.xlabel('Time measured in 5 minute increments')
plt.ylabel('Cars in Lanes')
plt.legend(loc='best')
plt.title("How Many Cars in the Lane")
plt.tight_layout()
plt.show()
