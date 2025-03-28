# -*- coding: utf-8 -*-
"""
Created on Sun Mar 9 11:54:47 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""
#%% Imports
import matplotlib.pyplot as plt;
import numpy as np;

#%% HW 5.1 Cauchy Stress Tensor Analysis
print("-=- HW5.1 -=-");

sigmaX = 8030; # [ksi] 
sigmaY = 2045; # [ksi] 
sigmaZ = 2870; # [ksi] 

tauXY = 2870; #  [ksi] 
tauXZ = 1450; #  [ksi] 
tauYZ = 4575; #  [ksi] 

sigma = [
    [sigmaX, tauXY, tauXZ],    
    [tauXY, sigmaY, tauYZ],
    [tauXZ, tauYZ, sigmaZ],
];

#%% HW 5.1 Eigenvalue Matrix & Visualization
eigen = np.linalg.eig(sigma);
v = eigen.eigenvectors;
print(v);
ax = plt.figure().add_subplot(projection='3d');
# Plotting the old coordinate system
ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3),
          np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), color='blue');

# Principle axes from Eigenvectors
ax.quiver(np.zeros(3),np.zeros(3),np.zeros(3), v[:,0],v[:,1],v[:,2], color='red');
ax.set_xlim([-1, 1]);
ax.set_ylim([-1, 1]);
ax.set_zlim([-1, 1]);

plt.show();

#%% HW 5.1 Safety Factor
stressYield = 29000; # [ksi] yield stress of steel

sigmaP0 = np.abs(eigen.eigenvalues[0]);
sigmaP1 = np.abs(eigen.eigenvalues[1]);
sigmaP2 = np.abs(eigen.eigenvalues[2]);

sigmaVM = np.sqrt(2)/2*np.sqrt((sigmaP0-sigmaP1)**2 + (sigmaP1-sigmaP2)**2 + (sigmaP2-sigmaP0)**2);
safetyFactor = stressYield / sigmaVM;
print("Safety Factor: " + str(safetyFactor));

#%% HW 5.2 Spring-Mass Building System
print("-=- HW5.2 -=-");
k1 = 3000; # [kN/m]
k2 = 2400; # [kN/m]
k3 = 1800; # [kN/m]

# Convert all kN to N
k1 *= 1000; 
k2 *= 1000; 
k3 *= 1000; 

m1 = 12000; # [kg]
m2 = 10000; # [kg]
m3 =  8000; # [kg]

coeffMatrix = [
    [-k1-k2, k2,    0],    
    [k1,    -k2-k3, k3],
    [0,      k3,   -k3],    
];

eigen2 = np.linalg.eig(coeffMatrix);

# The Eigenvalues represent the square of fundamental frequencies
print("Eigenvalues");
print(eigen2.eigenvalues);
# The Eigenvectors represent the oscilitory mode of the building
print("Eigenvector Matrix");
print(eigen2.eigenvectors);

f1, f2, f3 = np.sqrt(abs(eigen2.eigenvalues));
print("Characteristic Frequencies: ")
print(str(f1) + " " + str(f2) + " " + str(f3));

#%% HW 5.2 Fundamental Frequency Plotting

fig = plt.figure()
plt.subplot(1, 3, 1)

# Format the graph window to not be too small
window = fig.canvas.get_tk_widget().master; 
window.minsize(width=1200, height=800);
w = eigen2.eigenvectors;

plt.plot(np.append([0], w[:,2]), [0, 1, 2, 3])
plt.title(f'Fundamental Freq {str(f3)[:5]} (Hz)', fontweight='bold', fontsize=12)
plt.subplot(1, 3, 2)
plt.plot(np.append([0], w[:,1]), [0, 1, 2, 3])
plt.title(f'Fundamental Freq {str(f2)[:5]} (Hz)', fontweight='bold', fontsize=12)
plt.subplot(1, 3, 3)
plt.plot(np.append([0], w[:,0]), [0, 1, 2, 3])
plt.title(f'Fundamental Freq {str(f1)[:5]} (Hz)', fontweight='bold', fontsize=12)

plt.show();

# %%
