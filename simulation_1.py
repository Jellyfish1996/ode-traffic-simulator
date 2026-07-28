# We will assume a road has n lanes, vehicles are all identical in size and can only travel
# in one direction, recall before we begin theat density equals flow/speed

# First consider the constant outflow model 

# 1) Solving this ode give a function y(t) = at + C for unknown constant a and C. We can then specifciy various inital
# conditons and see how they compare to a similar model.

# 1)
def run_simulation_1(initial_vehicles: float, inflow: int, outflow: int):

    for t in range(13):
        print (f"time = {t*5} min, Number of vehicles = {(inflow-outflow) * t + initial_vehicles}")



run_simulation_1(81,482,482)

# For an example consider the road segment Mainline VDS 717404  to Mainline VDS 716295
# this has length approximately 0.9 miles we know the flow and avg speed of the upstream detector is 482 * 12 = 5784 vehicles per hour
# and 64.1 mph respectively, this gives approximately and average desnity of 90 vehicles per mile. Now we multiply by the length of 
# our segment so our segments initial vhehicles is approx 90 * .9 = 81 vehicles, so initial_vehicles = 81, inflow = 482 = outflow
# the result is 81.
