# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:32:08 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""

import matplotlib.pyplot as plt;
import matplotlib.ticker as mtick;
import numpy as np;
import time;

#%% HW2.1 NOx Reduction - Lambda Function

# Technically this is an improper function definition as it
# poses challenges for readibility. It's better to explicitly
# use a "def" operator with a return, but I'll use it anyway since
# that's what's suggested in lecture.
# See: https://peps.python.org/pep-0008/#:~:text=Always%20use%20a,a%20larger%20expression)
g = lambda x, T: ((6.61) * (1-x) - (3.83 * 10**20) * np.power(float(T), -2.0) * np.exp(-69160.0/float(T)));

print ("-=-=- 2.1 -=-=- ")

def bisection (g, x, tl, tu, tol):
    tm = (tu + tl) / 2;
    margin = abs(g(x, tm));
    if margin < tol:
        #print(f"Within acceptable margin: {margin}/{tol}");
        #print(f"Solution: {tm}; Took {i} iterations.");
        return tm;
    else: 
        _prod = g(x, tl) * g(x, tm);
        if _prod < 0:
            tu=tm;
        elif _prod > 0:
            tl=tm;
        else:
            # Address the edge case where the guess range doesn't contain a root.
            raise ValueError("Value not found within provided range.");
        tm = (tu + tl) / 2;
        
    # Rather than a loop of some sort, recursion:
    return bisection(g, x, tl, tu, tol);
            
# %% HW2.1 - Solve combustion temperature for a NOx reduction of 50%
start = time.time();

x = 0.5; # NOx fraction (50%)
tl = 500;  # [*K] lower bound of bisection function
tu = 2400; # [*K] upper bound of bisection function
tol = 0.000001; # Acceptable margin of bisection function

temp_50 = bisection(g, x, tl, tu, tol);

print(f"Combustion temperature to reduce NOx production by 50%: {temp_50} *K");
print(f"Function completion took {time.time()-start} seconds.");

#%% HW2.1 - NOx Reduction Ranges

r_upper = 1 - 9/1180; # NOx reduction percent to achieve the 9ppm requirement
r_lower = 0.5;        # Lower reduction percent
qty = 1000;           # How many values to interspace between these ranges

start = time.time();
reductions = np.linspace(r_lower, r_upper, qty);
comb_temperatures = np.zeros(qty);

for i in range(qty):
    comb_temperatures[i] = bisection(g, reductions[i], tl, tu, tol); 

fig, ax = plt.subplots();

ax.plot(comb_temperatures, reductions);
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0));

plt.title("HW2.1: NOx Reduction as Function of Temperature");
plt.ylabel("NOx Reduction [%]");
plt.xlabel("Combustion Temperature [\N{DEGREE SIGN}K]");

print(f"Produced graph of kerosene combustion temperatures. Function completion took {time.time()-start} seconds.");

#%% HW2.2 - Mixture Composite Viscosity

print ("-=-=- 2.2 -=-=- ")

t = 70;      # [*C] Temperature of mixture
x_mgo = 0.4; #[dec] portion of the mixture that is MGO
x_rmg = 0.6; #[dec] portion of the mixture that is RMG

v_mgo = 8.6145 * np.exp(-0.036605*t); # [m^2/s] kinematic viscosity of MGO
v_rmg = 3115.6 * np.exp(-0.040802*t); # [m^2/s] kinematic viscosity of RMG

tol = 0.0001 # relative tolerance to yield bisection search

def VBI (viscosity): # [unitless] returns component viscosity blend index
    return np.log(viscosity)/np.log(1000*viscosity);

# Find the VBIs of the components
vbi_mgo = VBI(v_mgo);
vbi_rmg = VBI(v_rmg);

# VBI of the mix is the sum of the component fractions
vbi_mix = vbi_mgo * x_mgo + vbi_rmg * x_rmg;

def h (vbi_mix, v_mix): return vbi_mix - VBI(v_mix);

def viscosityBisection (h, vbi_mix, tl, tu, tol, i):
    i = i + 1;

    tm = (tl + tu) / 2;
    margin = abs(h(vbi_mix, tm));
    if (margin < tol):
        print(f"Within acceptable margin: {margin}/{tol}");
        print(f"Solution: {tm}; Took {i} iterations.");
        return tm;
    else: 
        prod = h(vbi_mix, tl) * h(vbi_mix, tm);
        if prod < 0:
            tu=tm;
        elif prod > 0:
            tl=tm;
        else:
            # Address the edge case where the guess range doesn't contain a root.
            raise ValueError("Value not found within provided range.");
        tm = (tu + tl) / 2;
        
    # Rather than a loop of some sort, recursion:
    return viscosityBisection(h, vbi_mix, tl, tu, tol, i);

comp_viscosity = viscosityBisection (h, vbi_mix, 
                    # use the min and max viscosities of MGO and RMG 
                    # respectively for upper and lower limits, because 
                    # composite *should* fall within this range.
                    min(v_mgo, v_rmg), max(v_mgo, v_rmg), 
                    tol, 0);
# %% HW2.3 - Beam Deflection

print ("-=-=- 2.3 -=-=- ")

a = 800; #      [cm] - distance to concentrated load
l = 1000; #     [cm] - overall length of beam
E = 60000; #    [kN/cm^2] - Young's modulus
I = 25000; #    [cm^4] - moment of inertia
p = 100; #      [kN] - magnitude of the beam load

b = l-a; #      [cm] - length of the second half of the beam


x_a = np.linspace(0, a, 500); # [cm] - x values for first half of beam 
x_b = np.linspace(a, l, 500); # [cm] - x values for second half of beam

# Calculate deflection for the beam as a piecewise function
def deflect (x):

    return np.where(x < a, 
                    # If beam location is less than a, return:
                    (p*b*x)/(6*E*I*l)*(l**2 - np.power(x, 2) - b**2),
                    # Otherwise, return:
                    (p*b)/(6*E*I*l)*(l/b * np.power(x-a, 3) 
                    + x*(l**2 - b**2) 
                    - np.power(x, 3)));

fig, ax = plt.subplots();
ax.plot(x_a, deflect(x_a), color="blue", label="Side-A Deflection");
ax.plot(x_b, deflect(x_b), color="red", label="Side-B Deflection");
ax.legend();
plt.xlabel("Displacement [cm]");
plt.ylabel("Deflection [cm]");
plt.title("HW2.3: Beam Deflection in Simply Supported Beam")
# %% HW2.3 - Golden Section Search
from math import sqrt
g_r = (sqrt(5) - 1)/2;

start = time.time();
# Since there is 1 maximum in this range, use golden ratio optimization
def goldenMax (f, xl, xu, i=0, tol=0.0001):
    i = i + 1; #            [#] iterations that this optimization has run
    d = g_r * (xu - xl); #      distance of one golden ratio in scale
    x1 = xu - d; #              lower ratio checkpoint
    x2 = d + xl; #              upper ratio checkpoint
    y1 = f(x1); #               deflection value at lower checkpoint
    y2 = f(x2); #               deflection value at upper checkpoint

    perc_margin = (xu - xl)/xu;

    if (perc_margin < tol):
        # If the % error is within the tolerance, return the location
        # of the maximum of the next golden iteration.
        print(f"Optimization complete. Took {i} iterations.");
        if (y1 > y2): return x1;
        else: return x2;
    if (y1 < y2):
        # If the value of the right golden value is greater than the left,
        # remove the left the range (xl to x1)
        xl = x1;
    # Otherwise, move the right range to the upper golden rule value.
    else: xu = x2;

    # Then run the algorithm again
    return goldenMax(f, xl, xu, i);

max_defx = goldenMax(deflect, 0, l, tol=1E-14);
max_defy = deflect(max_defx);

plt.scatter(max_defx, max_defy, marker="x", s=50, color = "black");

print(f"Maximum deflection occurs at {max_defx} with a value of {deflect(max_defx)}.");
print(f"Computation took {time.time() - start}s");
