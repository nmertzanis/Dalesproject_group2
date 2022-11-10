# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:11:45 2022

@author: Alessandro Pieruzzi
"""

import numpy as np
import matplotlib.pyplot as plt



profiles = np.genfromtxt('0304092012 (min)/prof.inp.001', skip_header = 1)
profiles1 = np.genfromtxt('0601092012 (max)/prof.inp.001', skip_header = 1)



z = profiles[:,0]
thl = profiles[:,1]
qt = profiles[:,2]
u = profiles[:,3]
v = profiles[:,4]
TKE = profiles[:,5]

z1 = profiles1[:,0]
thl1 = profiles1[:,1]
qt1 = profiles1[:,2]
u1 = profiles1[:,3]
v1 = profiles1[:,4]
TKE1 = profiles1[:,5]


#Whole profile

plt.figure(figsize=(8,6))
plt.plot(thl,z, label="06/01/09")
plt.plot(thl1,z, label="03/04/09")
plt.xlabel('thl [K]')
plt.ylabel('Height [m]')
plt.title("Liquid potential temperature")
plt.legend()

plt.figure(figsize=(8,6))
plt.plot(qt,z, label="06/01/09")
plt.plot(qt1,z, label="03/04/09")
plt.xlabel('qt [kg/kg]')
plt.ylabel('Height [m]')
plt.legend()
plt.title("Total specific humidity")

plt.figure(figsize=(8,6))
plt.plot(u,z, label="06/01/09")
plt.plot(u1,z, label="03/04/09")
plt.legend()
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Height [m]')
plt.title("Wind speed (U)")

plt.figure(figsize=(8,6))
plt.plot(v,z, label="06/01/09")
plt.plot(v1,z, label="03/04/09")
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Height [m]')
plt.legend()
plt.title("Wind speed (V)")

plt.figure()
plt.plot(TKE,z, label="day_min")
plt.plot(TKE1,z, label="day_max")
plt.legend()
plt.title("TKE")