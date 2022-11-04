# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:58:07 2022

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

#jan
path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200901.nc'
data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
plt.figure()
data.counts_per_sample.plot()

plt.figure()
data_day = data.counts_per_sample[np.where(data['time.day'] == 6)]
data_day.plot()


plt.figure()
data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
data_day_hour_00.plot()

data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
max_hour_00 = max(data_day_hour_val_00)
min_hour_00 = min(data_day_hour_val_00)
print('max 00 day 6', max_hour_00 )
print('min00 day 6', min_hour_00)


plt.figure()
data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
data_day_hour_12.plot()

data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
max_hour_12 = max(data_day_hour_val_12)
min_hour_12 = min(data_day_hour_val_12)
print('max12 day 6', max_hour_12)
print('min12 day 6', min_hour_12)




path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200904.nc'

data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
plt.figure()
data.counts_per_sample.plot()

data_mean_day = data.counts_per_sample.groupby('time.hour').mean()
plt.figure()
data_mean_day.plot()
plt.title('Mean diurnal nuclei concentration for April 2009')


plt.figure()
data_day = data.counts_per_sample[np.where(data['time.day'] == 3)]
data_day.plot()


plt.figure()
data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
data_day_hour_00.plot()

data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
max_hour_00 = max(data_day_hour_val_00)
min_hour_00 = min(data_day_hour_val_00)
print('max00 day 3 ', max_hour_00)
print('min00 day 3', min_hour_00)

plt.figure()
data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
data_day_hour_12.plot()

data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
max_hour_12 = max(data_day_hour_val_12)
min_hour_12 = min(data_day_hour_val_12)
print('max12 day 3', max_hour_12)
print('min12 day 3', min_hour_12)

plt.figure()
data_day = data.counts_per_sample[np.where(data['time.day'] == 18)]
data_day.plot()


plt.figure()
data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
data_day_hour_00.plot()

data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
max_hour_00 = max(data_day_hour_val_00)
min_hour_00 = min(data_day_hour_val_00)
print('max00 day 18 ', max_hour_00)
print('min00 day 18', min_hour_00)

plt.figure()
data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
data_day_hour_12.plot()

data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
max_hour_12 = max(data_day_hour_val_12)
min_hour_12 = min(data_day_hour_val_12)
print('max12 day 18', max_hour_12)
print('min12 day 18', min_hour_12)

plt.figure()
data_day = data.counts_per_sample[np.where(data['time.day'] == 24)]
data_day.plot()


plt.figure()
data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
data_day_hour_00.plot()

data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
max_hour_00 = max(data_day_hour_val_00)
min_hour_00 = min(data_day_hour_val_00)
print('max00 day 24 ', max_hour_00)
print('min00 day 24', min_hour_00)

plt.figure()
data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
data_day_hour_12.plot()

data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
max_hour_12 = max(data_day_hour_val_12)
min_hour_12 = min(data_day_hour_val_12)
print('max12 day 24', max_hour_12)
print('min12 day 24', min_hour_12)

#=====================================================
 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # 
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max', max_hour_12)
# # print('min', min_hour_12)
# # =============================================================================
# 
# 
# #feb
# path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200902.nc'
# data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
# plt.figure()
# data.counts_per_sample.plot()
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 17)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # =============================================================================
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 20)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max', max_hour_12)
# # print('min', min_hour_12)
# # =============================================================================
# 
# 
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 28)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max ', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max ultimate', max_hour_12)
# # print('min', min_hour_12)
# # =============================================================================
# 
# 
# 
# #march
# path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200903.nc'
# data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
# plt.figure()
# data.counts_per_sample.plot()
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 6)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max', max_hour_12)
# # print('min', min_hour_12)
# # =============================================================================
# 
# 
# 
# #april
# path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200904.nc'
# data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
# plt.figure()
# data.counts_per_sample.plot()
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 27)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max', max_hour_12)
# # print('min', min_hour_12)
# # =============================================================================
# 
# 
# #mei
# path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200905.nc'
# data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
# plt.figure()
# data.counts_per_sample.plot()
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 28)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max', max_hour_12)
# # print('min ultimate', min_hour_12)
# # =============================================================================
# 
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 31)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# 
# 
# 
# 
# # =============================================================================
# # data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# # max_hour_00 = max(data_day_hour_val_00)
# # min_hour_00 = min(data_day_hour_val_00)
# # print('max', max_hour_00)
# # print('min', min_hour_00)
# # =============================================================================
# 
# # =============================================================================
# # data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# # max_hour_12 = max(data_day_hour_val_12)
# # min_hour_12 = min(data_day_hour_val_12)
# # print('max', max_hour_12)
# # print('min', min_hour_12)
# # =============================================================================
# 
# 
# 
# #june
# path = r'C:\Users\User\Desktop\TU DELFT\Master_2022-2023\Observations_to_modelling\final\cesar_ccncounter_nucleiconcentration_la1_t00_v1.0_200906.nc'
# data = xr.open_mfdataset(path, combine = 'by_coords', engine = 'netcdf4')
# plt.figure()
# data.counts_per_sample.plot()
# 
# 
# plt.figure()
# data_day = data.counts_per_sample[np.where(data['time.day'] == 27)]
# data_day.plot()
# 
# plt.figure()
# data_day_hour_00 = data_day[np.where(data_day['time.hour'] == 00)]
# data_day_hour_00.plot()
# 
# plt.figure()
# data_day_hour_12 = data_day[np.where(data_day['time.hour'] == 12)]
# data_day_hour_12.plot()
# =============================================================================

# =============================================================================
# data_day_hour_val_00 = data_day[np.where(data_day['time.hour'] == 00)].values
# max_hour_00 = max(data_day_hour_val_00)
# min_hour_00 = min(data_day_hour_val_00)
# print('max', max_hour_00)
# print('min', min_hour_00)
# =============================================================================

# =============================================================================
# data_day_hour_val_12 = data_day[np.where(data_day['time.hour'] == 12)].values
# max_hour_12 = max(data_day_hour_val_12)
# min_hour_12 = min(data_day_hour_val_12)
# print('max', max_hour_12)
# print('min', min_hour_12)
# =============================================================================




# =============================================================================
# days = data.groupby('time.day')
# days = np.linspace(3, 28, 26)
# 
# minimum = [] 
# maximum = []
# day_date = []
# #determine minimum in a couple of hour 6 in the morning
# for i in range(len(days)):
#     #plt.figure()
#     day = days[i]
#     data_day = data.counts_per_sample[np.where(data['time.day'] == day)]
#     #data_day.plot()
#     data_day_val = data.counts_per_sample[np.where(data['time.day'] == day)].values
# 
#     data_day_hour = data_day[np.where(data_day['time.hour'] == 6)]
#     #plt.figure()
#     #data_day_hour.plot()
#     data_day_hour_val = data_day[np.where(data_day['time.hour'] == 6)].values
#     max_hour_6 = data_day_hour_val.max()
#     maximum.append(max_hour_6)
#     min_hour_6 = data_day_hour_val.min()
#     minimum.append(min_hour_6)  
#     
# 
# #determine the absolute minimum and maximum out of all of the days at hour 6 
# abs_min = min(minimum)
# abs_max = max(maximum)
# 
# #determine the day of the abs min and max 
# data_day_min = data.time[np.where(data.counts_per_sample == abs_min)]
# data_day_max = data.time[np.where(data.counts_per_sample == abs_max)] #gives all values, check whether it is at 6 am. 
# =============================================================================


# =============================================================================
# 
# #do the same for hour 12
# 
# minimum_12 = [] 
# maximum_12 = []
# day_date_12 = []
# 
# days = data.groupby('time.day')
# days = np.linspace(3, 27, 25) #goes to 27 because day 28 does not have 12 hour values
# 
# 
# #determine minimum in a couple of hour 12 in the afternoon
# for i in range(len(days)):
#     plt.figure()
#     day_12 = days[i]
#     data_day_12 = data.counts_per_sample[np.where(data['time.day'] == day_12)]
#     data_day_12.plot()
#     data_day_val_12 = data.counts_per_sample[np.where(data['time.day'] == day_12)].values
# 
#     data_day_hour_12 = data_day_12[np.where(data_day_12['time.hour'] == 12)]
#     plt.figure()
#     data_day_hour_12.plot()
#     
#     data_day_hour_val_12 = data_day_12[np.where(data_day_12['time.hour'] == 12)].values
#     max_hour_12 = data_day_hour_val_12.max()
#     maximum_12.append(max_hour_12)
#     min_hour_12 = data_day_hour_val_12.min()
#     minimum_12.append(min_hour_12)  
#     
# 
# #determine the absolute minimum and maximum out of all of the days at hour 6 
# abs_min_12 = min(minimum_12)
# abs_max_12 = max(maximum_12)
# 
# #determine the day of the abs min and max 
# data_day_min_12 = data.time[np.where(data.counts_per_sample == abs_min_12)]
# data_day_max_12 = data.time[np.where(data.counts_per_sample == abs_max_12)] #gives all values, check whether it is at 6 am. 
# 
# =============================================================================
#do the same for hour 00

# =============================================================================
# minimum_00 = [] 
# maximum_00 = []
# day_date_00 = []
# 
# days = data.groupby('time.day')
# days = np.linspace(3, 28, 26) #goes to 27 because day 28 does not have 12 hour values
# 
# 
# #determine minimum in a couple of hour 12 in the afternoon
# for i in range(len(days)):
#     plt.figure()
#     day_00 = days[i]
#     data_day_00 = data.counts_per_sample[np.where(data['time.day'] == day_00)]
#     data_day_00.plot()
#     plt.title('Daily diurnal cycle of condensation nuclei')
#     data_day_val_00 = data.counts_per_sample[np.where(data['time.day'] == day_00)].values
# 
#     data_day_hour_00 = data_day_00[np.where(data_day_00['time.hour'] == 00)]
#     plt.figure()
#     data_day_hour_00.plot()
#     plt.title('Condensation nuclei only for hour 00')
#     
#     data_day_hour_val_00 = data_day_00[np.where(data_day_00['time.hour'] == 00)].values
#     max_hour_00 = data_day_hour_val_00.max()
#     maximum_00.append(max_hour_00)
#     min_hour_00 = data_day_hour_val_00.min()
#     minimum_00.append(min_hour_00)  
#     
# 
# #determine the absolute minimum and maximum out of all of the days at hour 6 
# abs_min_00 = min(minimum_00)
# abs_max_00 = max(maximum_00)
# 
# #determine the day of the abs min and max 
# data_day_min_00 = data.time[np.where(data.counts_per_sample == abs_min_00)]
# data_day_max_00 = data.time[np.where(data.counts_per_sample == abs_max_00)] #gives all values, check whether it is at 6 am. 
# 
# 
# =============================================================================
