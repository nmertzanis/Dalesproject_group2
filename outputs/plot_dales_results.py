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

# - plots for whole profiles at the end of the simulation (done)
# Plots for model every hour (first 200m)
# Plots for Cabauw every hour (first 200m)
# radiation timeseries (GHEYLLA)



cp = 1005
g = 9.8
Lv = 2.5*10**6
R = 287 #[J/(kg*K)] dry air
Rv = 461 #[J/(kg*K)] vapour
Lv = 2.5*10**6 #J/kg latent heat of vaporization
e0 = 0.611 #kPa
g = 9.8
cp = 1005 #J/kg*K
p0=10**5 #Pa this can be more precise
cd = np.array([50, 1000]) #Cloud droplet numbers

#######################################################################################################################################################

"DALES OUTPUTS"


"DAY MIN"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final_proj\Dalesproject_group2\outputs\day_min'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt = np.array(profiles['qt'])
ql = np.array(profiles['ql'])
thv =  np.array(profiles['thv'])
pres = np.array(profiles['presh'])
thl = np.array(profiles['thl'])
z = np.array(profiles['zm'])
swu = np.array(profiles['swu'])
swd = np.array(profiles['swd'])
lwu = np.array(profiles['lwu'])
lwd = np.array(profiles['lwd'])

T = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)



"DAY MAX"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final_proj\Dalesproject_group2\outputs\day_max'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles1 = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries1 = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt1 = np.array(profiles['qt'])
ql1 = np.array(profiles['ql'])
thv1 =  np.array(profiles['thv'])
pres1 = np.array(profiles['presh'])
thl1 = np.array(profiles['thl'])
z1 = np.array(profiles['zm'])
swu1 = np.array(profiles1['swu'])
swd1 = np.array(profiles1['swd'])
lwu1 = np.array(profiles1['lwu'])
lwd1 = np.array(profiles1['lwd'])

T1 = thv1/(1+0.608*(qt1-ql1)) * ((pres1)/p0)**(287/cp)



#######################################################################################################################################################

"OBSERVATIONS"
#Files are already cabauw and radiosondes merged together
#Files to open are in outputs/obs: they are already correct for 7/1/09 at 00 and 4/4/09 at 00 (12h after the start of the simulation)
obs_min = np.genfromtxt('obs/obs_min.txt')
obs_max = np.genfromtxt('obs/obs_max.txt')


"VARIABLES: DAY MIN"
thl_obs = obs_min[1:,1]
q_obs = obs_min[1:,2]
P_obs = obs_min[1:,-1]
#We can't have T since we don't have ql measurements


"VARIABLES: DAY MAX"
thl1_obs = obs_max[1:,1]
q1_obs = obs_max[1:,2]
P1_obs = obs_max[1:,-1]
#We can't have T since we don't have ql measurements



#######################################################################################################################################################

"PLOTTING"


"FINAL TIME STEP"
#Radiation profiles:

plt.figure(figsize=(7,6))
plt.plot(swu[-1],z, label = "day min")
plt.plot(swu1[-1],z, label = "day max")
plt.plot(swd[-1],z, label = "day min")
plt.plot(swd1[-1],z, label = "day max")
plt.xlabel("Shortwave radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure(figsize=(7,6))
plt.plot(lwu[-1],z, label = "day min")
plt.plot(lwu1[-1],z, label = "day max")
plt.plot(lwd[-1],z, label = "day min")
plt.plot(lwd1[-1],z, label = "day max")
plt.xlabel("Longwave radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()


#Observation and modeled profiles 

plt.figure()
plt.plot(thl_obs, z, label = 'Observed profile' )
plt.plot(thl[-1], z,  '--' , label = 'Modeled profile')
plt.title("Day_min")
plt.xlabel("Thl [K]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure()
plt.plot(thl1_obs, z, label = 'Observed profile' )
plt.plot(thl1[-1], z,  '--' , label = 'Modeled profile')
plt.title("Day_max")
plt.xlabel("Thl [K]")
plt.ylabel("Height [m]")
plt.legend()


plt.figure()
plt.plot(q_obs, z, label = 'Observed profile' )
plt.plot(qt[-1], z,  '--' , label = 'Modeled profile')
plt.title("Day_min")
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure()
plt.plot(q1_obs, z, label = 'Observed profile' )
plt.plot(qt1[-1], z,  '--' , label = 'Modeled profile')
plt.title("Day_max")
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.legend()


"EVERY HOUR"


#retrieve profiles for every hour: repeat interpolation procedure with cabauw data or save to file from the other script and re import it
# or: run the other script first and then paste everything in the console (fast way)

#Variables: thl_c1, thl_c4, q_c1, q_c4


#for cycle from h12 t h00 every hour plots of:
    #theta modeled vs observed day min
    #theta modeled vs observed day max
    #qt modeled vs observed day min
    #qt modeled vs observed day max
















