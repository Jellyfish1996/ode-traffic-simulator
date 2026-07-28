import math
# So this will be our Linear outflow model where A is a constant function, B is proportional to the number of vehicles on
# the road (i.e. B(N) = kN(t))

# So we get an ODE like N'(t) = A - kN(t), solving we obtain: N(t) = Ce^{-kt} + A/k

def run_simulation_2(initial_vehicles: float, A: float, k: float) :
    # C = initial vehicles - A/k
    C = initial_vehicles - A/k

    for t in range(13):
        print(f"time = {t*5} min, vehicles = {C*math.exp(-k*t) + A/k}")

# run_simulation_2()
