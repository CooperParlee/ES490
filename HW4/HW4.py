# -*- coding: utf-8 -*-
"""
Created on Tue Mar 4 11:44:47 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""
#%% Imports
import matplotlib.pyplot as plt;
import numpy as np;

#%% Silence Lint DONT RUN THIS
raise Exception("dont run this (less nice edition)");
ca = ca;
P1 = P1;

#%% 4.1a Plot Pressure vs Crank Angle
plt.plot(ca, P1);
plt.title("Cylinder Pressure as Function of Crank Angle");
plt.xlabel(f"Crank Angle [\N{DEGREE SIGN}]");
plt.ylabel("Pressure [kPa?]");

#%% 4.1b Pressure Derivative

def forwardDerivative (xArray, yArray):
    dFWD = np.zeros(len(ca));
    for i in range(0, len(xArray)-1, 1):
        dFWD[i] = (yArray[i+1] - yArray[i])/(xArray[i+1]-xArray[i]);
    return dFWD;

def centralDerivative (xArray, yArray):
    dCTR = np.zeros(len(ca));
    
    for i in range(1, len(ca)-1, 1):
        dCTR[i] = (yArray[i+1] - yArray[i-1])/(xArray[i+1]-xArray[i-1]);
    return dCTR;

dFWD = forwardDerivative(ca, P1);
dCTR = centralDerivative(ca, P1);

plt.figure();
plt.title("Derivative Cylinder Pressure as Function of Crank Angle (FWD)");
plt.plot(ca, dFWD);
plt.figure();
plt.title("Derivative Cylinder Pressure as Function of Crank Angle (CTR)");
plt.plot(ca, dCTR);

# 4.1c: Both derivative methods produce a significant amount of noise.
# However, the center differencing method appears to produce less noise than
# forward differencing.

#%% 4.1d-f Second Pressure Derivative
ddFWD = forwardDerivative(ca, dFWD);
ddCTR = centralDerivative(ca, dCTR);

plt.figure();
plt.title("2nd Derivative Cylinder Pressure as Function of Crank Angle (FWD)");
plt.plot(ca, ddFWD);
plt.figure();
plt.title("2nd Derivative Cylinder Pressure as Function of Crank Angle (CTR)");
plt.plot(ca, ddCTR);

# 4.1e: Yes, a large positive peak can be seen in these graphs, but they do
# not have the same positive peak. I guess that makes sense though.

#%% 4.1g Finding Ignition Start
injectionStart = 350;

# Since data was very discontinuous and noisy, had to use a brute force
# algorithm to truly find the global maximum where combustion occurred.

def bruteMax (xArray, yArray):
    lastMax = -1;
    lastMaxX = -1;
    for i in range(0, len(yArray)):
        x = xArray[i];
        y = yArray[i];
        
        if y > lastMax:
            lastMax = y;
            lastMaxX = x;
            
    return (lastMaxX[0], lastMax);

res = bruteMax(ca, ddCTR)
print("Combustion Occurs at: " + str(res[0]) + " degrees.");
print("Combustion Delay: " + str(round(10*(res[0] - injectionStart))/10) + " degrees.");

