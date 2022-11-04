# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:12:18 2022

@author: User
"""

import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import datetime

path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_surface_radiation_lc1_t10_v1.0_200904.nc'
data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
plt.figure()
data.SWU.plot()
day = data['time.day'].values
days = np.unique(day)

for i in range(len(days) - 1):
    day = days[i]

    SWU_specific = data.SWU[np.where(data['time.day'] == day)]
    plt.figure()
    SWU_specific.plot()
    plt.title('Upwelling Shortwave radiation')
    
    
    SWD_specific = data.SWD[np.where(data['time.day'] == day)]
    plt.figure()
    SWD_specific.plot()
    plt.title("Downwelling shortwave radiation")
    
# =============================================================================
#     LWU_specific = data.LWU[np.where(data['time.day'] == day)]
#     plt.figure()
#     LWU_specific.plot()
#     plt.title('Upwelling longwave radiation')
#     
#     
#     LWD_specific = data.LWD[np.where(data['time.day'] == day)]
#     plt.figure()
#     LWD_specific.plot()
#     plt.title('Downwelling longwave radiation')
#     
#     
#     #net 
#     plt.figure()
#     net_short = data.SWD[np.where(data['time.day'] == day)]-data.SWU[np.where(data['time.day'] == day)] 
#     net_short.plot()
#     net_long = data.LWU[np.where(data['time.day'] == day)]-data.LWD[np.where(data['time.day'] == day)] 
#     net_long.plot()
#     plt.title('Net long and shortwave radiation')
# 
# =============================================================================

path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_surface_radiation_lc1_t10_v1.0_200901.nc'
data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
plt.figure()
data.SWU.plot()
day = data['time.day'].values
days = np.unique(day)

for i in range(len(days) - 1):
    day = days[i]

    SWU_specific = data.SWU[np.where(data['time.day'] == day)]
    plt.figure()
    SWU_specific.plot()
    plt.title('Upwelling Shortwave radiation')
    
    
    SWD_specific = data.SWD[np.where(data['time.day'] == day)]
    plt.figure()
    SWD_specific.plot()
    plt.title("Downwelling shortwave radiation")
    
path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_surface_radiation_lc1_t10_v1.0_200902.nc'
data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
plt.figure()
data.SWU.plot()
day = data['time.day'].values
days = np.unique(day)

for i in range(len(days) - 1):
    day = days[i]

    SWU_specific = data.SWU[np.where(data['time.day'] == day)]
    plt.figure()
    SWU_specific.plot()
    plt.title('Upwelling Shortwave radiation')
    
    
    SWD_specific = data.SWD[np.where(data['time.day'] == day)]
    plt.figure()
    SWD_specific.plot()
    plt.title("Downwelling shortwave radiation")
    
path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_surface_radiation_lc1_t10_v1.0_200903.nc'
data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
plt.figure()
data.SWU.plot()
day = data['time.day'].values
days = np.unique(day)

for i in range(len(days)- 1):
    day = days[i]

    SWU_specific = data.SWU[np.where(data['time.day'] == day)]
    plt.figure()
    SWU_specific.plot()
    plt.title('Upwelling Shortwave radiation')
    
    
    SWD_specific = data.SWD[np.where(data['time.day'] == day)]
    plt.figure()
    SWD_specific.plot()
    plt.title("Downwelling shortwave radiation")