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
p0=10**5 #Pa this can be more precise
cd = np.array([50, 1000]) #Cloud droplet numbers

#######################################################################################################################################################

"DALES OUTPUTS"


"DAY MIN"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final proj\Dalesproject_group2\outputs\day_min'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt = np.array(profiles['qt'])[-1]
ql = np.array(profiles['ql'])[-1]
thv =  np.array(profiles['thv'])[-1]
pres = np.array(profiles['presh'])[-1]
thl = np.array(profiles['thl'])[-1]
z = np.array(profiles['zm'])
swu = np.array(profiles['swu'])[-1]
swd = np.array(profiles['swd'])[-1]
lwu = np.array(profiles['lwu'])[-1]
lwd = np.array(profiles['lwd'])[-1]

T = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)



"DAY MAX"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final proj\Dalesproject_group2\outputs\day_max'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles1 = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries1 = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt1 = np.array(profiles['qt'])[-1]
ql1 = np.array(profiles['ql'])[-1]
thv1 =  np.array(profiles['thv'])[-1]
pres1 = np.array(profiles['presh'])[-1]
thl1 = np.array(profiles['thl'])[-1]
z1 = np.array(profiles['zm'])
swu1 = np.array(profiles1['swu'])[-1]
swd1 = np.array(profiles1['swd'])[-1]
lwu1 = np.array(profiles1['lwu'])[-1]
lwd1 = np.array(profiles1['lwd'])[-1]

T1 = thv1/(1+0.608*(qt1-ql1)) * ((pres1)/p0)**(287/cp)



#######################################################################################################################################################

"OBSERVATIONS"
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

#Plot the upward and downward fluxes of shortwave and longwave radiation, 
# profiles: thl, q  


#Radiation profiles:

plt.figure(figsize=(7,6))
plt.plot(swu,z, label = "DC = 50 cm^-3")
plt.plot(swu1,z, label = "DC = 100 cm^-3")
plt.plot(swd,z, label = "DC = 50 cm^-3")
plt.plot(swd1,z, label = "DC = 100 cm^-3")
plt.xlabel("Shortwave radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure(figsize=(7,6))
plt.plot(lwu,z, label = "DC = 50 cm^-3")
plt.plot(lwu1,z, label = "DC = 100 cm^-3")
plt.plot(lwd,z, label = "DC = 50 cm^-3")
plt.plot(lwd1,z, label = "DC = 100 cm^-3")
plt.xlabel("Longwave radiation flux [W/m^2]")
plt.ylabel("Height [m]")
plt.legend()


#Observation and modeled profiles 

plt.figure()
plt.plot(thl_obs, z, label = 'Observed profile' )
plt.plot(thl, z,  '--' , label = 'Modeled profile')
plt.title("Day_min")
plt.xlabel("Thl [K]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure()
plt.plot(thl1_obs, z, label = 'Observed profile' )
plt.plot(thl1, z,  '--' , label = 'Modeled profile')
plt.title("Day_max")
plt.xlabel("Thl [K]")
plt.ylabel("Height [m]")
plt.legend()


plt.figure()
plt.plot(q_obs, z, label = 'Observed profile' )
plt.plot(qt, z,  '--' , label = 'Modeled profile')
plt.title("Day_min")
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure()
plt.plot(q1_obs, z, label = 'Observed profile' )
plt.plot(qt1, z,  '--' , label = 'Modeled profile')
plt.title("Day_max")
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.legend()


















