# This is the congestion model, so here B(N) decreases as road reaches maximum capcity,
# So here the ODE is: N'(t) = A - v_f * N/L*(1-N/(L*p_m))
# Let p = N/L be the density of vehicles at a certain time t, v_f is the free flow traffic speed.
import matplotlib.pyplot as plt
# So lets compute this numerically, so we need to fix a initial condition. 

def simulation_3(initial_condition: float, v_f: float, A: float, p_m: float, L: float, step_size: float):
    y_n = initial_condition
    time = []
    vehicles = []

    def ODE(N: float):
        return A - v_f*(N/L)*(1- (N/L) * (1/p_m))

    number_of_steps = int(12/step_size)

    for i in range(number_of_steps + 1):
        time.append(step_size*i)
        vehicles.append(y_n)
        print(f"time = {i*step_size *5} minutes, number of vehicles = {y_n}")
        y_n = y_n + step_size*(ODE(y_n))

    return time, vehicles

time, vehicles = simulation_3(30, 40, 60, 150, 1, 0.05)



plt.plot(time, vehicles, 'r-', label="Lane 1")
plt.xlabel('Time measured in 5 minute increments')
plt.ylabel('Cars in Lanes')
plt.legend(loc='best')
plt.title("How Many Cars Enter and Leave The Lane")
plt.tight_layout()
plt.show()
