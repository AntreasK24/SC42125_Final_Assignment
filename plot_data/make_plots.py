

import numpy as np
import matplotlib.pyplot as plt

# Define the variable names and corresponding file patterns
variables = ['x', 'y', 'z', 'yaw']
horizons = ["0,1", "1", "5", "10","20"]


plt.rcParams.update({
    "font.size": 16,         # Default font size
    "axes.titlesize": 20,    # Title font
    "axes.labelsize": 18,    # X/Y label font
    "xtick.labelsize": 14,   # X-axis ticks
    "ytick.labelsize": 14,   # Y-axis ticks
    "legend.fontsize": 14,   # Legend
    "figure.titlesize": 22   # Figure-level title
})


for var in variables:
    plt.figure(figsize=(10, 5))
    for N in horizons:
        data = np.load(f"{var}_R_{N}.npy")
        timesteps = np.arange(len(data)) * 0.1 
        
        # Always define label properly
        N_label = "0.1" if str(N) == "0,1" else N
        label = rf"$Q_{{i,i}} = {N_label}$"

        plt.plot(timesteps, data, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel(f"{var} [rad]" if var == "yaw" else f"{var} [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()