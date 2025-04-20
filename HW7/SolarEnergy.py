#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 22:15:01 2023

@author: Brendyn
"""
import numpy as np

def SolarEnergy(W, beta, gamma, Lat, Long, Lst, n, ct):
    # Wabs=SolarEnergy(W,n,beta,gamma,Lat,Long,Lst,ct) models solar flux relative 
    # to a panel at any fixed orientation according to sun position in the
    # sky throughout a day at a specified location according to latitude and longitude. 
    #   W=solar constant flux density [kW/m^2] 
    #   beta=tilt angle of PV panel [deg]
    #   gamma=PV surface azimuth angle [deg] (east of south negative, west of south positive)
    #   Lat=latitude of local location [deg]
    #   Long=longitude of local location [deg]
    #   Lst=standard meridian for local time zone (75 for EST) [deg]
    #   n=day number starting at n=1 for January 1st to n=365 for December 31st
    #   ct=clock time (Can be any time from 0-24 or an array of times
    #   (4:20 would correspond to times from 4am to 8pm) [hour]

    te = 0.1645*np.sin(4*np.pi*(n-81)/365) + 0.12783*np.sin(2*np.pi*(n-1)/365) # Time equation [hour]
    st = 4/60*(Lst-Long) + te + ct # Solar time [hour]
    w = 15*(st-12) # Hour angle [deg]
    delta = 23.45*np.sin(2*np.pi*(284+n)/365) # Declination angle [deg]
    # Solar altitude angle [deg]
    alphas = np.degrees(np.arcsin(np.sin(np.radians(Lat))*np.sin(np.radians(delta)) \
            + np.cos(np.radians(Lat))*np.cos(np.radians(delta))*np.cos(np.radians(w))))
    # Solar azimuth angle [deg]
    gammas = np.degrees(np.arcsin(np.cos(np.radians(delta))*np.sin(np.radians(w))/np.cos(np.radians(alphas))))
    I1 = np.argmin(np.abs(np.abs(gammas[:len(ct)//2])-90)) #Finding -90deg to east
    I2 = np.argmin(np.abs(gammas-90)) # Finding 90deg to west
    gammas[:I1-1] = -180-gammas[:I1-1] # Replacing east beyond -90deg with angles up to -180
    gammas[I2+1:] = 180-gammas[I2+1:] # Replacing west beyond 90 deg with angles up to 180
    alphas[alphas<0] = np.nan # Negating values before and after sunrise and sunset
    I3 = np.argmin(np.abs(np.abs(gammas[:len(ct)//2]-gamma)-90)) # Finding -90deg east of panel
    I4 = np.argmin(np.abs(gammas-gamma-90)) # Finding 90deg west of panel
    gammas[:I3-1] = np.nan # Negating east of panel beyond -90deg
    gammas[I4+1:] = np.nan # Negating west of panel beyond 90 deg
    costheta = np.sin(np.radians(alphas))*np.cos(np.radians(beta)) \
        + np.cos(np.radians(alphas))*np.sin(np.radians(beta))*np.cos(np.radians(gammas-gamma))
    # Calculate solar flux available to solar panel [kW/m^2]
    Wabs = W*costheta
    Wabs[np.isnan(Wabs)] = 0 # Replacing NaN with zero when light not incident on panel
    return Wabs