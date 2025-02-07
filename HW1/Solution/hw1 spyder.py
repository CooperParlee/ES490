# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:17:16 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""
import numpy as np;
from math import cos, sin, radians;

"""
    Problem 2: Solution of large matricies using linalg.solve() from Numpy.
"""
r45 = radians(45);

##           AB         AE          BC          BD          BE          CD          DE          Ax          Ay          Cy
A = np.array([[1,       cos(r45),   0,          0,          0,          0,          0,          1,          0,          0],
             [0,        sin(r45),   0,          0,          0,          0,          0,          0,          1,          0],
             [-1,       0,          1,          cos(r45),   -cos(r45),  0,          0,          0,          0,          0],
             [0,        0,          0,          sin(r45),   sin(r45),   0,          0,          0,          0,          0],
             [0,        0,          -1,         0,          0,          -cos(r45),  0,          0,          0,          0],
             [0,        0,          0,          0,          0,          sin(r45),   0,          0,          0,          1],
             [0,        -cos(r45),  0,          0,          cos(r45),   0,          1,          0,          0,          0],
             [0,        -sin(r45),  0,          0,          -sin(r45),  0,          0,          0,          0,          0],
             [0,        0,          0,          -cos(r45),  0,          cos(r45),   -1,         0,          0,          0],
             [0,        0,          0,          -sin(r45),  0,          -sin(r45),  0,          0,          0,          0]], dtype=float);
b = np.array([0, 0, 0, 0, 0, 0, 0, 400, 0, 600], dtype=float); 

x1 = np.linalg.solve(A, b);

print(x1);

"""
    Problem 3: Heat transfer solution
"""
## Givens:
T0 = 70; # [*F]
T4 = 10; # [*F]

h0 = 2;      # [Btu/hr-ft^2-*F]
h4 = 5;      # [Btu/hr-ft^2-*F]
kA = 0.3;    # [Btu/hr-ft-*F] 
kB = 0.02;   # [Btu/hr-ft-*F]

m = np.array([[-(h0 + kA/8), kA/8, 0],
             [kA/8, -(kA/8 + kB/2), kB/2],
             [0, kB/2, -(kB/2 + h4)]]);

q = np.array([-140, 0, -50]);

T = np.linalg.solve(m, q);