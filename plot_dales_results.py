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


"DAY MIN"

"IMPORT FILES"

exp_ind = 0

folder_path = r'E:\TUDELFT\2year\DALES\Final proj\Dalesproject_group2\outputs'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)

cd = np.array([50, 1000]) #Cloud droplet number 
file_index = np.array([0, 1]) + exp_ind 

profiles = nc.Dataset(f' {folder_path}\{list_of_files[file_index[0]]}')
timeseries = nc.Dataset(f' {folder_path}\{list_of_files[file_index[1]]}')

"RETRIEVE VARIABLES"
qt = np.array(profiles['qt'])
ql = np.array(profiles['ql'])
thv =  np.array(profiles['thv'])
pres = np.array(profiles['presh'])
thl = np.array(profiles['thl'])
z = np.array(profiles['zm'])
u = np.array(profiles['u'])
v = np.array(profiles['v'])
p0=10*5 #Pa

T = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)

swu = np.array(profiles['swu'])
swd = np.array(profiles['swd'])
lwu = np.array(profiles['lwu'])
lwd = np.array(profiles['lwd'])


"DAY MAX"

"IMPORT FILES"

exp_ind = 1

file_index = np.array([0, 3]) + exp_ind #fix numbers

profiles1 = nc.Dataset(f' {folder_path}\{list_of_files[file_index[0]]}')
timeseries1 = nc.Dataset(f' {folder_path}\{list_of_files[file_index[1]]}')


"RETRIEVE VARIABLES"
qt = np.array(profiles['qt'])
ql = np.array(profiles['ql'])
thv =  np.array(profiles['thv'])
pres = np.array(profiles['presh'])
thl = np.array(profiles['thl'])
z = np.array(profiles['zm'])
u = np.array(profiles['u'])
v = np.array(profiles['v'])
p0=10*5 #Pa

T = thv/(1+0.608*(qt-ql)) * ((pres)/p0)**(287/cp)

swu1 = np.array(profiles1['swu'])
swd1 = np.array(profiles1['swd'])
lwu1 = np.array(profiles1['lwu'])
lwd1 = np.array(profiles1['lwd'])




"OTHER DAY"

"IMPORT FILES"
exp_ind = 2

file_index = np.array([0, 3]) + exp_ind 

profiles2 = nc.Dataset(f' {folder_path}\{list_of_files[file_index[0]]}')
timeseries2 = nc.Dataset(f' {folder_path}\{list_of_files[file_index[1]]}')

"RETRIEVE VARIABLES"
swu2 = np.array(profiles2['swu'])
swd2 = np.array(profiles2['swd'])
lwu2 = np.array(profiles2['lwu'])
lwd2 = np.array(profiles2['lwd'])




"PLOTTING"

#Plot the upward and downward fluxes of shortwave and longwave radiation, 
# profiles: T, thl, qt, ql,  



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

























