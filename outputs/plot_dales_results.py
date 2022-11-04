# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:46:13 2022

@author: Alessandro Pieruzzi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:30:01 2022

@author: Alessandro Pieruzzi
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import seaborn as sns

sns.set()


cp = 1005
g = 9.8
Lv = 2.5*10**6
R = 287 #[J/(kg*K)] dry air
Rv = 461 #[J/(kg*K)] vapour
Lv = 2.5*10**6 #J/kg latent heat of vaporization
e0 = 0.611 #kPa
g = 9.8
cp = 1005 #J/kg*K
p0=10*5 #Pa this can be more precise
cd = np.array([50, 1000]) #Cloud droplet numbers

#######################################################################################################################################################

"DALES OUTPUTS"


"DAY MIN"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final proj\Dalesproject_group2\outputs\day_min'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
profiles = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt = np.array(profiles['qt'])
ql = np.array(profiles['ql'])
thv =  np.array(profiles['thv'])
pres = np.array(profiles['presh'])
thl = np.array(profiles['thl'])
z = np.array(profiles['zm'])
u = np.array(profiles['u'])
v = np.array(profiles['v'])
swu = np.array(profiles['swu'])
swd = np.array(profiles['swd'])
lwu = np.array(profiles['lwu'])
lwd = np.array(profiles['lwd'])

T = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)



"DAY MAX"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final proj\Dalesproject_group2\outputs\day_max'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
profiles1 = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries1 = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt1 = np.array(profiles['qt'])
ql1 = np.array(profiles['ql'])
thv1 =  np.array(profiles['thv'])
pres1 = np.array(profiles['presh'])
thl1 = np.array(profiles['thl'])
z1 = np.array(profiles['zm'])
u1 = np.array(profiles['u'])
v1 = np.array(profiles['v'])
swu1 = np.array(profiles1['swu'])
swd1 = np.array(profiles1['swd'])
lwu1 = np.array(profiles1['lwu'])
lwd1 = np.array(profiles1['lwd'])

T1 = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)



#######################################################################################################################################################

"OBSERVATIONS"
#Files to open are in outputs/obs
obs_min = np.genfromtxt('obs/obs_min.txt')
obs_max = np.genfromtxt('obs/obs_max.txt')


"VARIABLES: DAY MIN"
thl = obs_min[:,1]
q = obs_min[:,2]
P = obs_min[:,-1]
T = thl * (P/(p0/100)**(R/cp))

"VARIABLES: DAY MAX"
thl1 = obs_max[:,1]
q1 = obs_max[:,2]
P1 = obs_max[:,-1]
T1 = thl * (P/(p0/100)**(R/cp))

"PLOTTING"

#Plot the upward and downward fluxes of shortwave and longwave radiation, 
# profiles: T, thl, q  



plt.figure(figsize=(7,6))
plt.plot(swu[-1],z, label = "Shortwave upwards radiation")
plt.plot(swd[-1],z, label = "Shortwave downwards radiation")
plt.xlabel("Radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()



plt.figure(figsize=(7,6))
plt.plot(lwu[-1],z, label = "Longwave upwards radiation")
plt.plot(lwd[-1],z, label = "Longwave downwards radiation")
plt.xlabel("Radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()





plt.figure(figsize=(7,6))
plt.plot(swu[-1],z, label = "DC = 50 cm^-3")
plt.plot(swu1[-1],z, label = "DC = 100 cm^-3")
plt.plot(swu2[-1],z, label = "DC = 1000 cm^-3")
plt.plot(swd[-1],z, label = "DC = 50 cm^-3")
plt.plot(swd1[-1],z, label = "DC = 100 cm^-3")
plt.plot(swd2[-1],z, label = "DC = 1000 cm^-3")
plt.xlabel("Shortwave radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()




plt.figure(figsize=(7,6))
plt.plot(lwu[-1],z, label = "DC = 50 cm^-3")
plt.plot(lwu1[-1],z, label = "DC = 100 cm^-3")
plt.plot(lwu2[-1],z, label = "DC = 1000 cm^-3")
plt.plot(lwd[-1],z, label = "DC = 50 cm^-3")
plt.plot(lwd1[-1],z, label = "DC = 100 cm^-3")
plt.plot(lwd2[-1],z, label = "DC = 1000 cm^-3")
plt.xlabel("Longwave radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()

























