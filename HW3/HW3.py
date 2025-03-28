# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:23:18 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""
#%% Imports
import matplotlib.pyplot as plt;
import numpy as np;

#%% HW3.1 Sailboat Numerical Integration

height = 30;    # [ft] total height of mast.
N = 13;         # [] number of Simpson's points.

def mastForce (z, H):
    """Function that returns the force at given point on a vertical 
    sailboat mast at a height (z, ft) over a total mast height of (h, ft).

    Args:
        z (float): [ft] distance above the deck
        H (float): [ft] total height of the mast

    Returns:
        (float): [lbf] force at that point on the mast.
    """
    return 200 * z / (5 + z) * np.exp(-2 * z / H);

def midpointNumerical (f, h, N):
    """Function returns the midpoint rule integral sum of the provided function.
    This is used for error-testing with a very small dx.

    Args:
        f (function): function to numerically integrate
        h (float): [ft] maximum x to integrate too
        N (int): [#] of subintervals

    Returns:
        float: integral of provided function f.
    """
    dx = h/(N);
    x_range = np.linspace(0, h, N+1);
    y_range = f(x_range, h);
    return np.sum(y_range) * dx;

def trapezoidNumerical (f, h, N):
    """Function returns the trapezoid rule integral sum of the mastForce function.

    Args:
        f (function): function to numerically integrate
        h (float): [ft] maximum height to integrate too
        N (int): [#] of subintervals

    Returns:
        float: force acting on the mast, integrated.
    """
    dx = h/(N);
    x_range = np.linspace(0, h, N+1);
    y_range = f(x_range, h);

    area = f(x_range[0], h) + f(x_range[N], h);
    for i in range(1, N):
        area = area + 2*(y_range[i]);
    
    return area * dx/2;

def generateSimpArray(n):
    """Generates an array of coefficients in Simpson's rule. Should be in
    the form [1 4 2 4 2 4 ... 4 1]

    Args:
        n (int): number of Simpson subintervals

    Returns:
        [n]: array of length n coefficients
    """
    if n % 2 == 0:
        raise ValueError("n parameter to generateSimpArray should be odd");
    simp = np.full(n, 1);
    for i in range(1, n-1):
        if i % 2 != 0: # If array index is odd, set array 4.
            simp[i] = 4;
        else: simp[i] = 2;
    return simp;


def mastSimpson (f, h, N):
    """Function returns the Simpson's rule integral sum of the mastForce function.

    Args:
        f (function): function to numerically integrate
        h (float): [ft] maximum height to integrate too
        N (int): [#] of subintervals

    Returns:
        float: force acting on the mast, integrated.
    """
    dx = h/(N-1);
    x_range = np.linspace(0, h, N);
    y_range = f(x_range, h);

    return np.sum(dx/3*(generateSimpArray(N) * y_range));

print ("-=-=- 3.1 -=-=- ")
print("Midpoint (precise): " + str(midpointNumerical(mastForce, height, int(1E2)))); # Use to test for errors
print("Trapezoid: " + str(trapezoidNumerical(mastForce, height, N)));
print("Simpson: " + str(mastSimpson(mastForce, height, N)));
#%% HW3.2 Solar Panel Optimization Calculator
import SolarEnergy as Solar

print ("-=-=- 3.2 -=-=- ")

# Define some default parameters:

w = 1.361;      # [kW/m2]
beta = 20;      # [Deg]
gamma = 0;      # [Deg]
lat = 44.3889;  # [Deg]    
long = 68.7990; # [Deg]    
lst = 75;       # [Deg]
n = 1;          # [day] of the year corresponding numerically to 365 days

times = np.linspace(0, 24, 24); # array of 50 timestamps from 0h to 24h
# %% HW3.2 Familiarization
wabs = Solar.SolarEnergy(w, beta, gamma, lat, long, lst, n, times);

plt.plot(times, wabs);
plt.title("HW3.2: Hourly Solar Energy Potential 01-Jan");
plt.ylabel("Energy Potential [kW/m^2]");
plt.xlabel("Clock Time [Hours]");
# %% HW3.2 Integration: Yearly Potential 

num_samples = 501; # in each 24 hours

def dayEnergy (day, times, beta, w=w, gamma=gamma, lat=lat, long=long, lst=lst):
    num = len(times);
    simpArr = generateSimpArray(num);
    dt = 24/(num - 1);

    return np.sum(simpArr * Solar.SolarEnergy(w, beta, gamma, lat, long, lst, day, times) * dt / 3);


days = np.arange(1, 366, 1);
times = np.linspace(0, 24, num_samples);

daily_energy = np.zeros([365]);

for day in days:
    daily_energy[day - 1] = dayEnergy(day, times, beta);

print(f"Energy 1yr: {np.sum(daily_energy)} KWh/m^2");

# %% HW3.2 Daily Energy Plot
plt.plot (days, daily_energy);
plt.title("HW3.2: Daily Solar Energy Potential");
plt.ylabel("Energy Potential [kWh/m^2]");
plt.xlabel("Day of Year");

# %% HW3.2 Create Function
days = np.arange(1, 366, 1);

def totalEnergy(w, beta, gamma, lat, long, lst, days):
    
    times = np.linspace(0, 24, num_samples);

    daily_energy = np.zeros([len(days)]);
    
    for i in range(0, len(days), 1):
        day = days[i];
        daily_energy[i] = dayEnergy(day, times, beta);
    return np.sum(daily_energy);
#%% HW3.2 Optimize Beta
from scipy.optimize import minimize_scalar;

g = lambda beta: -totalEnergy(w, beta, gamma, lat, long, lst, days);
res = minimize_scalar(g, bounds=(0, 90));

betaOpt = res.x;
Wtotopt = -res.fun;
print("Optimization Complete;");
print("Beta: " + str(betaOpt));
print(f"Total Energy: {str(Wtotopt)} " + "KWh/m^2");

# %% HW3.2 Beta Optimization Plotting Confirmation
betaCount = 15;

betas = np.linspace(0.1, 90, betaCount);
totals = np.empty_like(betas);

for i in range(betaCount):
    totals[i] = totalEnergy(w, betas[i], gamma, lat, long, lst, days);

plt.plot(betas, totals);
plt.title(f"Yearly Solar Panel Energy Potential via Incline Angle \u03B2")
plt.xlabel(f"Solar Panel Incline Angle \u03B2 [\N{DEGREE SIGN}]");
plt.ylabel("Energy Potential [KWh/m^2]");

#%% HW3.2 Winter Optimization
winterDays = np.concatenate((np.arange(1, 92, 1), np.arange(306, 366, 1)));
h = lambda beta: -totalEnergy(w, beta, gamma, lat, long, lst, winterDays);
resW = minimize_scalar(h, bounds=(0, 90));

betaOpt = resW.x;
eTotalOpt = -resW.fun;

print("Winter Optimization Complete;");
print("Beta: " + str(betaOpt) + " degrees");
print(f"Total Energy: {str(eTotalOpt)} KWh/m^2");

# %% HW3.2 Winter Beta Optimization Plotting Confirmation
betaCount = 15;

betas = np.linspace(0.1, 90, betaCount);
totalsWinter = np.empty_like(betas);

for i in range(betaCount):
    totalsWinter[i] = totalEnergy(w, betas[i], gamma, lat, long, lst, winterDays);

plt.plot(betas, totalsWinter);
plt.title(f"Winter Solar Panel Energy Potential via Incline Angle \u03B2")
plt.xlabel(f"Solar Panel Incline Angle \u03B2 [\N{DEGREE SIGN}]");
plt.ylabel("Energy Potential [KWh/m^2]");



# %%
