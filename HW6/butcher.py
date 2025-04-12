# -*- coding: utf-8 -*-
"""
Created on Fri April 11 08:43:18 2025

@author: Cooper Parlee (cooper@cooperparlee.com)
"""

import numpy as np;

# A Butcher Tableau is represented in the form:
#   c|AAAAAA
#   c|AAAAAA
#   c|AAAAAA
#   c|AAAAAA
#   --------
#     BBBBBB

class Butcher:

    def numerical_jacobian(self, f, t, y, eps=1e-8):
        def wrapped(y_):
            return f(t, y_)
        n = len(y)
        J = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = eps
            J[:, i] = (wrapped(y + e_i) - wrapped(y - e_i)) / (2 * eps)
        return J
    
    def __init__(self, A, b, c, f, jac, m_y0, t0, tf, h, tol=1e-2, max_iter=10):
        self.a = A;
        self.b = b;
        self.c = c;
        self.f = f;
        self.h = h;
        self.jac = jac;

        self.d = len(m_y0);        
        self.t = np.arange(t0, tf + h, h);
        self.y = np.zeros((len(self.t), self.d));
        self.y[0, :] = m_y0;
    
        self.tol = tol;
        self.max_iter = max_iter;

    def calc(self):
        I_d = np.eye(self.d);
        s = len(self.b);
        h = self.h;

        for n in range(1, len(self.t)):
            t = self.t[n-1];
            y = self.y[n-1];

            K = np.tile(self.f(t, y), (s, 1));
            for iteration in range (self.max_iter):
                F = np.zeros_like(K);
                J = np.zeros((s*self.d, s*self.d));

                for i in range(s):
                    y_stage = y + h * sum(self.a[i, j] * K[j] for j in range(s));
                    t_stage = t + self.c[i] * h;

                    F[i] = K[i] - self.f(t_stage, y_stage);

                    J_i = self.jac(t_stage, y_stage);

                    for j in range(s):
                        J_block = I_d if i==j else np.zeros((self.d, self.d));
                        J_block -= h * self.a[i, j] * J_i;
                        J[i*self.d:(i+1)*self.d, j*self.d:(j+1)*self.d] = J_block;

                delta = np.linalg.solve(J, F.flatten());
                K -= delta.reshape((s, self.d));

                if np.linalg.norm(delta) < self.tol:
                    break;
            else: 
                raise RuntimeError(f"Newton-Raphson did not converge at step {n}");
            deltaY = h * np.dot(self.b, K);
            self.y[n] = self.y[n-1] + deltaY;
        return t, y;
