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

#%% HW 7.2 Greenhouse Problem

g_l = 20 * 0.3048; #         [m]
g_w = 10 * 0.3048; #         [m]
g_hnorth = 10 * 0.3048; #    [m]
g_hsouth = 6 * 0.3048; #     [m]
c_thick = 6/12 * 0.3048;  #  [m]


# b
rho_floor = 2300; # [kg/m^3]
m_floor = g_w * g_l * c_thick * rho_floor; # [kg]

rho_air = 1.293; #[kg/m^3]
m_air = g_w * g_l * (g_hnorth + g_hsouth)/2 * rho_air; # [kg]

m_water = 100; #[kg] starting guess for water mass

c_water =       4.184e3;   # [J/kg-K]
c_air =         1.0035e3;  # [J/kg-K]
c_concrete =    0.950e3;   # [J/kg-K]


a_concrete = g_w * g_l; # [m^2]
a_glass = (np.sqrt((g_hnorth - g_hsouth)**2 + g_w**2)*g_l # roof area
        + g_hsouth * g_l # south side area
        + (g_hnorth + g_hsouth)*g_w); # side trapezoid areas
a_water = g_hnorth * g_l; # [m^2] assume tank is as tall as roof

u_glass = 5.5; # [W/m^2-K]
h_air = 15; # [W/m^2-K]

# d
#Time varying outdoor temperature [degC] (32 to -4 degF)

def Tout (t):
    return 10 * np.sin(2 * np.pi / 86400 * t - np.pi) - 10;

#%% 7.2e Solar Energy Function

from SolarEnergy import SolarEnergy;

W = 882; # [W/m^2]
gamma = 0;
numSamples = 501;
ct=np.linspace(0, 24, numSamples);
dt = 24/numSamples * 3600; # [s]

#qsolar water [W/m2]
beta_concrete = 90;
beta_water = 0;
n = np.arange(0, 8, 1);

n_exch = 1.44 / 3600; # [#] of exchanges per hour converted into exchanges per second

def solarPwr (beta, sun, n, ct):
    # From HW3:
    gamma = 0;      # [Deg]
    lat = 44.3889;  # [Deg]    
    long = 68.7990; # [Deg]    
    lst = 75;       # [Deg] 
    
    Wabs=np.zeros([len(n),len(ct)])

    for i in range(len(n)):
        Wabs[i,:] = SolarEnergy(W,beta,gamma,lat,long,lst,n[i],ct)
        qsolarw=Wabs[0,:]
        if sun==1: #sun is out every day
            for j in range(1,len(n)):
                qsolarw=np.concatenate((qsolarw,Wabs[j,1:]))
        else: #sun is only out for the first day
            for j in range(1,len(n)):
                qsolarw=np.concatenate((qsolarw,np.zeros(len(ct)-1)))

    return Wabs;

def greenhouseDerivatives (t, y, q_water, q_concrete):
        T_f, T_w, T_a = y;
        

        T_outside = Tout(t);

        # Solve for the delta-T for draft
        dT_air_exch = m_air * c_air * n_exch * (T_a - T_outside);

        return np.array([
            # For the floor:
            (q_concrete*a_concrete - h_air * a_concrete * (T_f - T_a))/m_floor/c_concrete,
            # For the water:
            (q_water * a_water - h_air * a_water * (T_w - T_a))/m_water/c_water,
            # For the air:
            (h_air * a_water * (T_f - T_a) + 
            h_air * a_water * (T_w - T_a) - 
            u_glass * a_glass * (T_a - T_outside) - 
            dT_air_exch)/m_air/c_air,
        ]);

Wabs_water = solarPwr(beta_water, 1, n, ct);
Wabs_concrete = solarPwr(beta_concrete, 1, n, ct);

#plt.plot(ct, Wabs_water[1, :]);
#plt.plot(ct, Wabs_concrete[1, :]);
#
#plt.show();

T_temporal = np.zeros((3, len(ct) * len(n)-6));

T_temporal[:, 0] = 18; # Start everything off at 18*C
for day in n:
    print(day);
    for i in range(len(ct)):
        i_t = i + day * (numSamples-1);
        
        k1 = greenhouseDerivatives(ct[i], T_temporal[:, i_t], Wabs_water[day, i], Wabs_concrete[day, i]);
        k2 = greenhouseDerivatives(ct[i] + dt/2, T_temporal[:, i_t] + k1 * dt/2, Wabs_water[day, i], Wabs_concrete[day, i]);
        k3 = greenhouseDerivatives(ct[i] + dt/2, T_temporal[:, i_t] + k2 * dt/2, Wabs_water[day, i], Wabs_concrete[day, i]);
        k4 = greenhouseDerivatives(ct[i] + dt, T_temporal[:, i_t] + k3 * dt, Wabs_water[day, i], Wabs_concrete[day, i]);

        T_temporal[:, i_t + 1] = T_temporal[:, i_t] + (k1 + 2*k2 + 2*k3 + k4)/6;

t = np.linspace(0, len(n), len(T_temporal[0, :]));

plt.plot(t, T_temporal[0, :], label="Floor");
plt.plot(t, T_temporal[1, :], label=f"Water");
plt.plot(t, T_temporal[2, :], label="Air");
plt.legend();

plt.show();