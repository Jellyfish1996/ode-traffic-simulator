# We will assume a road has n lanes, vehicles are all identical in size and can only travel
# in one direction, recall before we begin theat density equals flow/speed
import matplotlib.pyplot as plt

# First consider the constant outflow model 

# 1) Solving this ode give a function y(t) = at + C for unknown constant a and C. We can then specifciy various inital
# conditons and see how they compare to a similar model.

def run_simulation_1(initial_vehicles: float, inflow: int, outflow: int):
    time = []
    number_of_vehicles = []
    for t in range(13):
        time.append(t)
        N = (inflow-outflow) * t + initial_vehicles
        number_of_vehicles.append(N)
        print (f"time = {t*5} min, Number of vehicles = {(inflow-outflow) * t + initial_vehicles}")

    return time, number_of_vehicles



time, number_of_vehicles = run_simulation_1(90, 511, 511)

plt.plot(time, number_of_vehicles, 'r-', label="Lane 1")
plt.xlabel('Time measured in 5 minute increments')
plt.ylabel('Cars in Lanes')
plt.legend(loc='best')
plt.title("How Many Cars in the lane")
plt.tight_layout()
plt.show()
