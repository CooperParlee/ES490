# -*- coding: utf-8 -*-
"""
Created on Fri March 21 12:59:08 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""

import matplotlib.pyplot as plt;
import numpy as np;
from bisect import bisect;

#%% HW6.1 ds/dt = -2/5 * S(t): t in minutes, S0 = 1
print("-=- HW6.1 -=-");
h = 0.01
t_upper = 20; # [min] upper range of differential

t = np.arange(0, t_upper + h, h); # array of times from 0 to 15 minutes
s_RK4 = np.empty_like(t);
s_RK4[0] = 1; # [mole MGO / mole composite] MGO fuel composition starts at 100%

# return differential of mole fraction function at s
def dMoleFrac (s): 
    return -2/5*s;

for i in range(len(t)-1):
    k1 = dMoleFrac(s_RK4[i]);
    k2 = dMoleFrac(s_RK4[i] + h/2 * k1);
    k3 = dMoleFrac(s_RK4[i] + h/2 * k2);
    k4 = dMoleFrac(s_RK4[i] + h*k3);

    s_RK4[i+1] = s_RK4[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4);

def moleFrac(t, s0): # exact solution found via separation of vars
    return s0*np.exp(-2/5*t);

plt.figure();
plt.plot(t, s_RK4, label="RK4", color="red");
plt.plot(t, moleFrac(t, s_RK4[0]), label="Exact", color="blue");
plt.title("Wartsila 6L20 Mixing Tank Fuel MDO-HFO Fuel Changeover");
plt.xlabel("Time (minutes)");
plt.ylabel(r"Mole Ratio $S=\frac{mol_{MDO}}{mol_{HFO}}$");
plt.legend();

plt.show();

i = np.where(s_RK4 <= 0.001)[0][0]; # Search the RK4 array for the first index where s is 0.1%
print(f"S=0.1% MDO: {str(t[i])} minutes");

#%% HW6.2
print("-=- HW6.2 -=-");
m = 1200; # [kg] mass of the car
k = 60000;# [N/m] spring constant of suspension
c = 6000; # [Ns/m] damping constant of the suspension
y0 = 0.1; # [m] road roughness amplitude
l = 5; #    [m] road roughness wavelength

v0 = 25; #   [m/s] car velocity for 6.2.1
x0 = 0; #   [m] initial car bounce height
u0 = 0; #   [m] initial car bounce velocity
dt = 0.001; # [s] time step

def roadFreq (v, l):
    """Returns the frequency of the road and suspension system given a
    velocity (v) and wavelength (l).
    Args:
        v (float): suspension velocity in m/s.
        l (float): road roughness bump wavelength.

    Returns:
        float: road bump frequency in angular velocity (w) rad/s.
    """
    return 2*np.pi*v/l;

def yRough (y0, t, w):
    
    """Returns the height of the road roughness at the given point in
    time (t) seconds.

    Args:
        y0 (float): the amplitude of the road bumps from equilibrium in meters.
        t (float): time delta in seconds since start of the simulation.
        w (float): road bump frequency in rad/s.

    Returns:
        float: height of road roughness at given temporal point t seconds.
    """
    return y0*np.sin(t*w);

def dRough(y0, t, w):
    return y0*w*np.cos(w*t);

w0 = roadFreq(v0, l);

def dxdt (t, x, u,
                  k=k, m=m, c=c, y0=y0, w=w0):
    """ Calculates the vertical acceleration at a given temporal point t (seconds) 
    with the vertical displacement x (meters) of the previous time step.

    Optional parameters may be provided of the system characteristics.

    Args:
        t (float): value of time at given point (s).
        x (float): value of vertical displacement (m) at previous time step.
        u (float): value of vertical velocity (m/s) at previous time step.
        k (float, optional): spring stiffness (N/m). Defaults to k.
        m (float, optional): system mass (kg). Defaults to m.
        c (float, optional): dampening constant (Ns/m). Defaults to c.
        y0 (float, optional): system roughness (m). Defaults to y0.
        w (float, optional): system roughness frequency (rad/s). Defaults to w0.

    Returns:
        float: acceleration at given time and vertical displacement.
    """
    y = yRough(y0, t, w);
    y_dot = dRough(y0, t, w);

    return k/m*y + c/m * y_dot - c/m * u - k/m * x, u;

def suspensionDyn (t, x0=x0, u0=u0, w=w0):
    """Return an array of displacement and velocities given the provided system arguments.

    Args:
        t (np.float64): array of times for the suspension calculations.
        x0 (float, optional): initial displacement condition. Defaults to x0.
        u0 (float, optional): initial velocity condition. Defaults to u0.
        w (float, optional): system frequency condition in rad/s. Defaults to w0.

    Returns:
        tuple[ np.float64, np.float64 ]:
            - x (np.ndarray): Displacement of the car over time (m).
            - u (np.ndarray): Velocity of the car over time (m/s).
    """
    u = np.empty_like(t);
    x = np.empty_like(t);
    #a = np.empty_like(t);
    
    # Define initial conditions
    x[0] = x0;
    u[0] = u0;

    for i in range(len(t)-1):

        k1_u, k1_x = dxdt(t[i], x[i], u[i], w=w);
        k2_u, k2_x = dxdt(t[i] + dt/2, x[i] + dt/2 * k1_x, u[i] + dt/2 * k1_u, w=w);
        k3_u, k3_x = dxdt(t[i] + dt/2, x[i] + dt/2 * k2_x, u[i] + dt/2 * k2_u, w=w);
        k4_u, k4_x = dxdt(t[i] + dt, x[i] + dt * k3_x, u[i] + dt * k3_u, w=w);

        x[i+1] = x[i] + dt/6 * (k1_x + 2*k2_x + 2*k3_x + k4_x);
        u[i+1] = u[i] + dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u);

    return x, u;

t = np.arange(0, 5, dt);
susP, susV = suspensionDyn(t);

fig, ax = plt.subplots(2);

fig.suptitle(f"6.2.1 Suspension Dynamics: k={k} N/m, c = {c} Ns/m")
ax[0].plot(t, susP, color = "blue");
ax[0].set_title("Suspension Position [m]");
ax[1].plot(t, susV, color = "red");
ax[1].set_title("Suspension Velocity [m/s]");
ax[1].set_xlabel("Time [s]");

fig.subplots_adjust(hspace=0.4);
plt.show();

# Find the max amplitude

vList = np.linspace(1, 30, 100); # [m/s] array of velocities to test
xMax = np.empty_like(vList); # [m] array of maximum displacements

for i in range(len(vList)):
    v = vList[i];
    wi = roadFreq(v, l);

    dyn = suspensionDyn(t, w=wi);
    xMax[i] = np.max(np.abs(dyn[0]));

plt.figure();
plt.plot(roadFreq(vList, l), xMax);
plt.title("6.2.2 Maximum Displacement Amplitude as a function of Angular Velocity");
plt.xlabel("System Frequency [rad/s]");
plt.ylabel("Maximum Displacement [m]");
vMax = vList[np.where(xMax == np.max(xMax))[0]][0];
wMax = roadFreq(vMax, l);
print(f"Maximum oscilation occurs at: {vMax} m/s");
print(f"System resonant frequency: {wMax} rad/s");

plt.show();

#%% 6.2.3 Different Initial Conditions
bumpDyn = suspensionDyn(t, x0=0.2, w=w0);

fig2, ax2 = plt.subplots(2);

fig2.suptitle(f"6.2.3 Suspension Dynamics: x0=0.2m")
ax2[0].plot(t, bumpDyn[0], color = "blue");
ax2[0].set_title("Suspension Position [m]");
ax2[1].plot(t, bumpDyn[1], color = "red");
ax2[1].set_title("Suspension Velocity [m/s]");
ax2[1].set_xlabel("Time [s]");
fig2.subplots_adjust(hspace=0.4);

plt.show();