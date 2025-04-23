import numpy as np;

class Greenhouse:
    # Greenhouse Class Conditions:
    c_water =       4.184e3;   # [J/kg-K]
    c_air =         1.0035e3;  # [J/kg-K]
    c_concrete =    0.950e3;   # [J/kg-K]

    rho_air = 1.293; #[kg/m^3]
    rho_floor = 2300; # [kg/m^3]

    u_glass = 5.5; # [W/m^2-K]
    h_air = 15; # [W/m^2-K]

    def __init__ (self, length, width, heightNorth, heightSouth, m_water, ct,  
                  c_thick = 6/12 * 0.3048, n_exch = 1.44):
        g_l = length;
        g_w = width;
        g_hnorth = heightNorth;
        g_hsouth = heightSouth;
        
        self.m_water = m_water;
        self.ct = ct;
        
        self.m_floor = g_w * g_l * c_thick * Greenhouse.rho_floor; # [kg]

        self.m_air = g_w * g_l * (g_hnorth + g_hsouth)/2 * Greenhouse.rho_air; # [kg]

        self.a_glass = (np.sqrt((g_hnorth - g_hsouth)**2 + g_w**2)*g_l # roof area
                + g_hsouth * g_l # south side area
                + (g_hnorth + g_hsouth)*g_w); # side trapezoid areas
        self.a_tanks = g_hnorth * g_l; # [m^2] assume tank is as tall as roof
        self.a_concrete = g_l * g_w; # [m^2]
    
        self.n_exch = n_exch / 3600;

    #Time varying outdoor temperature [degC] (32 to -4 degF)
    def Tout (self, t):
        return 10 * np.sin(2 * np.pi / 86400 * t - np.pi) - 10;

    def greenhouseDerivatives (self, t, y, q):
        T_f, T_w, T_a = y;

        m_air = self.m_air;
        m_floor = self.m_floor;
        m_water = self.m_water;

        c_air = Greenhouse.c_air;
        c_concrete = Greenhouse.c_concrete;
        c_water = Greenhouse.c_water;

        n_exch = self.n_exch;

        a_concrete = self.a_concrete;
        a_tanks = self.a_tanks;
        a_glass = self.a_glass;
        h_air = Greenhouse.h_air;
        u_glass = Greenhouse.u_glass;
        

        T_outside = self.Tout(t);

        # Solve for the delta-T for draft
        dT_air_exch = m_air * c_air * n_exch * (T_a - T_outside);

        return [
            # For the floor:
            (q*a_concrete - h_air * a_concrete * (T_f - T_a))/m_floor/c_concrete,
            # For the water:
            (q - h_air * a_tanks * (T_w - T_a))/m_water/c_water,
            # For the air:
            (h_air * a_tanks * (T_f - T_a) + 
            h_air * a_tanks * (T_w - T_a) - 
            u_glass * a_glass * (T_a - T_outside) - 
            dT_air_exch)/m_air/c_air,
        ];

    
    def SolveGreenhouse (self):

        ct = self.ct;
        q = self.q;
        dt = (ct[1] - ct[0])*3600;
        T_temporal = np.zeros((3, len(ct)));
        T_temporal[:, 0] = [21, 21, 21];

        for i in len(ct)-1:
            k1 = self.greenhouseDerivatives(ct[i], T_temporal[:, i], q[i]);
            k2 = self.greenhouseDerivatives(ct[i] + dt/2, T_temporal[:, i] + k1 * dt/2, q[i]);
            k3 = self.greenhouseDerivatives(ct[i] + dt/2, T_temporal[:, i] + k2 * dt/2, q[i]);
            k4 = self.greenhouseDerivatives(ct[i] + dt, T_temporal[:, i] + k3 * dt, q[i]);

            T_temporal[:, i+1] = T_temporal[:, i] + (k1 + 2*k2 + 2*k3 + k4)/6;
    


    def setQ (self, q):
        self.q = q;
    def setWater (self, m_water):
        self.m_water = m_water;