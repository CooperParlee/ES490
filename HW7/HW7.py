"""
Created on Fri April 18 12:59:08 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""
#%% HW7 Imports
import matplotlib.pyplot as plt;
import numpy as np;
from matplotlib.animation import FuncAnimation;

#%% HW7.1 
# Givens
l = 1;     # [m]
k = 401;   # [W/m^2K]
c = 0.386; # [J/gK]
dx = 0.1;  # [m]
dt = 5; # [s]
rho = 8.96E6; # [g/m^3]

t_max = 30*60;

x = np.arange(0, l + dx, dx);
t = np.arange(0, t_max + dt, dt);
alpha = k*dt/c/rho/dx**2;

# Make sure alpha < 0.5
if alpha > 0.5:
    print(f"Alpha: {alpha}") 
    raise ValueError("Alpha must be less than 0.5 for stability");

T = np.zeros((len(x), len(t)));
T[0, :] = 150;
T[-1, :] = 0;

for j in range(len(t)-1):
    for i in range(1, len(x)-1):
        T[i, j + 1] = alpha * T[i-1, j] + (1-2*alpha)*T[i, j] + alpha * T[i+1, j];

X, Tm = np.meshgrid(t, x);

fig = plt.figure();
ax = fig.add_subplot(111, projection="3d");

surf = ax.plot_surface(X, Tm, T, cmap="hot");

ax.set_title("HW 7.1a: Parabolic PDE Bar Conduction");
ax.set_xlabel("Time [s]");
ax.set_ylabel("Position [m]");
ax.set_zlabel("Temperature [C]");

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10);

plt.show();
#%% 7.1b Derivative = 0 Boundary Condition
T = np.zeros((len(x), len(t)));
T[0, :] = 150;
T[-1, :] = 0;

for j in range(len(t)-1):
    for i in range(1, len(x)-1):
        T[i, j + 1] = alpha * T[i-1, j] + (1-2*alpha)*T[i, j] + alpha * T[i+1, j];
    T[-1, j] = T[-2, j];

fig = plt.figure();
ax = fig.add_subplot(111, projection="3d");

surf = ax.plot_surface(X, Tm, T, cmap="hot");

ax.set_title("HW 7.1b: Derivative Boundary Condition");
ax.set_xlabel("Time [s]");
ax.set_ylabel("Position [m]");
ax.set_zlabel("Temperature [C]");

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10);

plt.show();