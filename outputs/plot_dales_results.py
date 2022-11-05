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

time = np.array(profiles['time'])

T = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)



"DAY MAX"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final_proj\Dalesproject_group2\outputs\day_max'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles1 = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries1 = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt1 = np.array(profiles1['qt'])
ql1 = np.array(profiles1['ql'])
thv1 =  np.array(profiles1['thv'])
pres1 = np.array(profiles1['presh'])
thl1 = np.array(profiles1['thl'])
z1 = np.array(profiles1['zm'])
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
T_obs = obs_min[1:,1]
q_obs = obs_min[1:,2]
P_obs = obs_min[1:,-1]
#We can't calculate thl since we don't have ql measurements


"VARIABLES: DAY MAX"
T1_obs = obs_max[1:,1]
q1_obs = obs_max[1:,2]
P1_obs = obs_max[1:,-1]
#We can't calculate thl since we don't have ql measurements



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
plt.plot(T_obs, z, label = 'Observed profile' )
plt.plot(T[-1], z,  '--' , label = 'Modeled profile')
plt.title("Day_min")
plt.xlabel("T [K]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure()
plt.plot(T1_obs, z, label = 'Observed profile' )
plt.plot(T1[-1], z,  '--' , label = 'Modeled profile')
plt.title("Day_max")
plt.xlabel("T [K]")
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

#fix times in the for cycle to have a plot per hour 
#fix heights to 200 only

cabauw_hourly = nc.Dataset('cabauw_hourly.nc')

T_obsh = np.array(cabauw_hourly['temp_jan'])
T1_obsh = np.array(cabauw_hourly['temp_apr'])

q_obsh = np.array(cabauw_hourly['q_jan'])
q1_obsh = np.array(cabauw_hourly['q_apr'])  

cabauw_hourly.close()

#Observation and modeled profiles 
for i in range(len(T_obsh)):
    
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    
    ax[0,0].plot(T_obsh[i], z[0:9], label = 'Observed profile' )
    ax[0,0].plot(T[i], z,  '--' , label = 'Modeled profile')
    ax[0,0].set_title("Temperature day_min")
    ax[0,0].set_xlabel("T [K]")
    ax[0,0].set_ylabel("Height [m]")
    ax[0,0].legend()
    
    ax[0,1].plot(T1_obsh[i], z[0:9], label = 'Observed profile' )
    ax[0,1].plot(T1[i], z,  '--' , label = 'Modeled profile')
    ax[0,1].set_title("Temperature day_max")
    ax[0,1].set_xlabel("T [K]")
    ax[0,1].set_ylabel("Height [m]")
    ax[0,1].legend()
    
    ax[1,0].plot(q_obsh[i], z[0:9], label = 'Observed profile' )
    ax[1,0].plot(qt[i], z,  '--' , label = 'Modeled profile')
    ax[1,0].set_title("Specific humidity day_min")
    ax[1,0].set_xlabel("qt [kg/kg]")
    ax[1,0].set_ylabel("Height [m]")
    ax[1,0].legend()
    
    ax[1,1].plot(q1_obsh[i], z[0:9], label = 'Observed profile' )
    ax[1,1].plot(qt1[i], z,  '--' , label = 'Modeled profile')
    ax[1,1].set_title("Specific humidity day_min")
    ax[1,1].set_xlabel("qt [kg/kg]")
    ax[1,1].set_ylabel("Height [m]")
    ax[1,1].legend()
    
    plt.suptitle(f"Profiles at time {i}")









