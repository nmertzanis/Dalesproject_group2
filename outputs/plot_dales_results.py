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
import datetime
import imageio
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
p0=10**5 #Pa


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


"DAY MIN_FAKE"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final_proj\Dalesproject_group2\outputs\day_min_fake'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qtf = np.array(profiles['qt'])
qlf = np.array(profiles['ql'])
thvf =  np.array(profiles['thv'])
presf = np.array(profiles['presh'])
thlf = np.array(profiles['thl'])
zf = np.array(profiles['zm'])
swuf = np.array(profiles['swu'])
swdf = np.array(profiles['swd'])
lwuf = np.array(profiles['lwu'])
lwdf = np.array(profiles['lwd'])


Tf = thvf/(1+0.608*(qtf-qlf)) * ((presf)/p0)**(287/cp)



"DAY MAX_FAKE"

"IMPORT FILES"

folder_path = r'E:\TUDELFT\2year\DALES\Final_proj\Dalesproject_group2\outputs\day_max_fake'  #enter your file path in this line
list_of_files = os.listdir(folder_path)
list_of_files = sorted(list_of_files)
profiles1 = nc.Dataset(f' {folder_path}\{list_of_files[0]}')
timeseries1 = nc.Dataset(f' {folder_path}\{list_of_files[1]}')

"RETRIEVE VARIABLES"
qt1f = np.array(profiles1['qt'])
ql1f = np.array(profiles1['ql'])
thv1f =  np.array(profiles1['thv'])
pres1f = np.array(profiles1['presh'])
thl1f = np.array(profiles1['thl'])
z1f = np.array(profiles1['zm'])
swu1f = np.array(profiles1['swu'])
swd1f = np.array(profiles1['swd'])
lwu1f = np.array(profiles1['lwu'])
lwd1f = np.array(profiles1['lwd'])


T1f = thv1f/(1+0.608*(qt1f-ql1f)) * ((pres1f)/p0)**(287/cp)




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


"FINAL TIME STEP, WHOLE PROFILES"

#Observation and modeled profiles 

plt.figure(figsize=(10,7))
plt.plot(T_obs, z, label = 'Observed profile' )
plt.plot(T[-1], z,  '--' , label = 'Nc = 8e6')
plt.plot(Tf[-1], z,  '--' , label = 'Nc = 80e6')
plt.title("07/01/2009 00:00")
plt.xlabel("T [K]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure(figsize=(10,7))
plt.plot(T1_obs, z, label = 'Observed profile' )
plt.plot(T1[-1], z,  '--' , label = 'Nc = 200e6')
plt.plot(T1f[-1], z,  '--' , label = 'Nc = 2000e6')
plt.title("04/04/2009 00:00")
plt.xlabel("T [K]")
plt.ylabel("Height [m]")
plt.legend()


plt.figure(figsize=(10,7))
plt.plot(q_obs, z, label = 'Observed profile' )
plt.plot(qt[-1], z,  '--' , label = 'Nc = 8e6')
plt.plot(qtf[-1], z,  '--' , label = 'Nc = 80e6')
plt.title("07/01/2009 00:00")
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.legend()

plt.figure(figsize=(10,7))
plt.plot(q1_obs, z, label = 'Observed profile' )
plt.plot(qt1[-1], z,  '--' , label = 'Nc = 200e6')
plt.plot(qt1f[-1], z,  '--' , label = 'Nc = 2000e6')
plt.title("04/04/2009 00:00")
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.legend()


"EVERY HOUR"

#fix times in the for cycle to have a plot per hour 


cabauw_hourly = nc.Dataset('cabauw_hourly.nc')

T_obsh = np.array(cabauw_hourly['temp_jan'])[:,0:9] #only up to 200m
T1_obsh = np.array(cabauw_hourly['temp_apr'])[:,0:9]

q_obsh = np.array(cabauw_hourly['q_jan'])[:,0:9]
q1_obsh = np.array(cabauw_hourly['q_apr'])[:,0:9]

cabauw_hourly.close()

hours = np.arange(12,25,1)
minute = np.arange(0,15)
t = np.arange(datetime.datetime(2009,1,6,12), datetime.datetime(2009,1,7,00), datetime.timedelta(minutes=10)).astype(datetime.datetime)
dt = 180
dtmin = 180

#CAlculate NSE on the whole profiles for these variables:

#Define function calculate TKE and run for every iterations

# NSE_T_min = np.zeros(72)
# NSE_T_minf = np.zeros(72)
# NSE_T_max = np.zeros(72)
# NSE_T_maxf = np.zeros(72)

# NSE_qt_min = np.zeros(72)
# NSE_qt_minf = np.zeros(72)
# NSE_qt_max = np.zeros(72)
# NSE_qt_maxf = np.zeros(72)


# def NSE(mod,obs):
#     obsAv=np.mean(obs)
#     ErrUp=sum((mod[0:9]-obs)**2)
#     ErrDo=sum((obs-obsAv)**2)
#     NSE=1-ErrUp/ErrDo
#     return NSE

# def MSE(mod,obs):
#     MSE = np.sqrt((np.sum((obs-mod[0:9])**2))/(np.sum(mod[0:9]**2)))
#     return MSE


#Observation and modeled profiles 
# for i in range(0,72,1):
    
    # NSE_T_min[i] = MSE(T[i],T_obsh[i])
    # NSE_T_minf[i] = MSE(Tf[i],T_obsh[i])
    # NSE_T_max[i] = MSE(T1[i],T1_obsh[i])
    # NSE_T_maxf[i] = MSE(T1f[i],T1_obsh[i])

    # NSE_qt_min[i] = MSE(qt[i],q_obsh[i])
    # NSE_qt_minf[i] = MSE(qtf[i],q_obsh[i])
    # NSE_qt_max[i] = MSE(qt1[i],q1_obsh[i])
    # NSE_qt_maxf[i] = MSE(qt1f[i],q1_obsh[i])


"FIRST 200m"
fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):    
    ax[0].plot(T_obsh[i], z[0:9], label = 'Observed profile' )
    ax[0].plot(T[i*dtmin][0:9], z[0:9],  '--' , label = 'Nc = 8e6')
    ax[0].plot(Tf[i*dtmin][0:9], z[0:9],  '--' , label = 'Nc = 80e6')
    ax[0].set_title("Temperature day_min")
    ax[0].set_xlabel("T [K]")
    ax[0].set_xlim([np.min([np.min(T),np.min(T_obs)]),np.max([np.max(T),np.max(T_obsh)])])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend()
    
    ax[1].plot(T1_obsh[i], z[0:9], label = 'Observed profile' )
    ax[1].plot(T1[i*dt][0:9], z[0:9],  '--' , label = ' Nc = 200e6')
    ax[1].plot(T1f[i*dt][0:9], z[0:9],  '--' , label = 'Nc = 2000e6')
    ax[1].set_title("Temperature day_max")
    ax[1].set_xlabel("T [K]")
    ax[1].set_xlim([np.min([np.min(T1),np.min(T1_obsh)]),np.max([np.max(T1_obsh),np.max(T1)])])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/temp/plot' + str(i))


fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):
    ax[0].plot(q_obsh[i][0:9], z[0:9], label = 'Observed profile' )
    ax[0].plot(qt[dtmin*i][0:9], z[0:9],  '--' , label = 'Nc = 8e6')
    ax[0].plot(qtf[dtmin*i][0:9], z[0:9],  '--' , label = 'Nc = 80e6')
    ax[0].set_title("Specific humidity day_min")
    ax[0].set_xlabel("qt [kg/kg]")
    ax[0].set_xlim([np.min([np.min(q_obsh),np.min(qt)]),np.max([np.max(q_obsh),np.max(qt)])])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(q1_obsh[i], z[0:9], label = 'Observed profile' )
    ax[1].plot(qt1[dt*i][0:9], z[0:9],  '--' , label = 'Nc = 200e6')
    ax[1].plot(qt1f[dt*i][0:9], z[0:9],  '--' , label = 'Nc = 2000e6')
    ax[1].set_title("Specific humidity day_max")
    ax[1].set_xlabel("qt [kg/kg]")
    ax[1].set_xlim([np.min([np.min(q1_obsh),np.min(qt1)]),np.max([np.max(q1_obsh),np.max(qt1)])])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/q/plot' + str(i))



fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):    
    ax[0].plot(ql[dtmin*i][0:9], z[0:9],  '--' , label = 'Nc = 8e6')
    ax[0].plot(qlf[dtmin*i][0:9], z[0:9],  '--' , label = 'Nc = 80e6')
    ax[0].set_title("Specific liquid humidity day_min")
    ax[0].set_xlabel("ql [kg/kg]")
    ax[0].set_xlim([-0.5e-5,np.max(ql)])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(ql1[dt*i][0:9], z[0:9],  '--' , label = 'Nc = 200e6')
    ax[1].plot(ql1f[dt*i][0:9], z[0:9],  '--' , label = 'Nc = 2000e6')
    ax[1].set_title("Specific liquid humidity day_max")
    ax[1].set_xlabel("ql [kg/kg]")
    ax[1].set_xlim([-0.5e-5,np.max(ql1)+0.5e-5])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/ql/plot' + str(i))


fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):
    ax[0].plot(swu[dtmin*i], z,  '--' , label = 'Upwards (Nc = 8e6)')
    ax[0].plot(swuf[dtmin*i], z,  '--' , label = 'Upwards (Nc = 80e6)')
    ax[0].plot(abs(swd[dtmin*i]), z,  '--' , label = 'Downwards (Nc = 80e6)')
    ax[0].plot(abs(swdf[dtmin*i]), z,  '--' , label = 'Downwards (Nc = 80e6)')
    ax[0].set_title("Shortwave radiation day_min")
    ax[0].set_xlabel("Radiation flux [W/m^2]")
    ax[0].set_xlim([-50,np.max([np.max(swu),np.max(abs(swd))])])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(swu1[dt*i], z,  '--' , label = 'Upwards (Nc = 200e6)')
    ax[1].plot(abs(swd1[dt*i]), z,  '--' , label = 'Downwards (Nc = 200e6)')
    ax[1].plot(swu1f[dt*i], z,  '--' , label = 'Upwards (Nc = 2000e6)')
    ax[1].plot(abs(swd1f[dt*i]), z,  '--' , label = 'Downwards (Nc = 2000e6)')
    ax[1].set_title("Shortwave radiation day_max")
    ax[1].set_xlabel("Radiation flux [W/m^2]")
    ax[1].set_xlim([-50,np.max([np.max(swu1),np.max(abs(swd1))])])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/shortwave/plot' + str(i))


fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):    
    ax[0].plot(lwu[dtmin*i], z,  '--' , label = 'Upwards (Nc = 8e6)')
    ax[0].plot(abs(lwd[dtmin*i]), z,  '--' , label = 'Downwards (Nc = 8e6)')
    ax[0].plot(lwuf[dtmin*i], z,  '--' , label = 'Upwards (Nc = 80e6)')
    ax[0].plot(abs(lwdf[dtmin*i]), z,  '--' , label = 'Downwards (Nc = 80e6)')
    ax[0].set_title("Longwave radiation day_min")
    ax[0].set_xlim([0,np.max([np.max(lwu),np.max(lwd)])])
    ax[0].set_xlabel("Radiation flux [W/m^2]")
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(lwu1[dt*i], z,  '--' , label = 'Upwards (Nc = 200e6)')
    ax[1].plot(abs(lwd1[dt*i]), z,  '--' , label = 'Downwards (Nc = 200e6)')
    ax[1].plot(lwu1f[dt*i], z,  '--' , label = 'Upwards (Nc = 2000e6)')
    ax[1].plot(abs(lwd1f[dt*i]), z,  '--' , label = 'Downwards (Nc = 2000e6)')
    ax[1].set_title("Longwave radiation day_max")
    ax[1].set_xlabel("Radiation flux [W/m^2]")
    ax[1].set_xlim([0,np.max([np.max(lwu1),np.max(lwd1)])])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/longwave/plot' + str(i))



"WHOLE PROFILES"

fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):    
    ax[0].plot(T[i*dtmin], z,  '--' , label = 'Nc = 8e6')
    ax[0].plot(Tf[i*dtmin], z,  '--' , label = 'Nc = 80e6')
    ax[0].set_title("Temperature day_min")
    ax[0].set_xlabel("T [K]")
    ax[1].set_xlim([np.min([np.min(T),np.min(Tf)]),np.max([np.max(T),np.max(Tf)])])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(T1[i*dt], z,  '--' , label = 'Nc = 200e6')
    ax[1].plot(T1f[i*dt], z,  '--' , label = 'Nc = 2000e6')
    ax[1].set_title("Temperature day_max")
    ax[1].set_xlabel("T [K]")
    ax[1].set_xlim([np.min([np.min(T1),np.min(T1f)]),np.max([np.max(T1f),np.max(T1)])])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/fullprofiles/temp/plot' + str(i))


fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):    
    ax[0].plot(qt[dtmin*i], z,  '--' , label = 'Nc = 8e6')
    ax[0].plot(qtf[dtmin*i], z,  '--' , label = 'Nc = 80e6')
    ax[0].set_title("Specific humidity day_min")
    ax[0].set_xlabel("qt [kg/kg]")
    ax[0].set_xlim([np.min([np.min(qtf),np.min(qt)]),np.max([np.max(qtf),np.max(qt)])])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(qt1[dt*i], z,  '--' , label = 'Nc = 200e6')
    ax[1].plot(qt1f[dt*i], z,  '--' , label = 'Nc = 2000e6')
    ax[1].set_title("Specific humidity day_max")
    ax[1].set_xlabel("qt [kg/kg]")
    ax[1].set_xlim([np.min([np.min(qt1f),np.min(qt1)]),np.max([np.max(qt1f),np.max(qt1)])])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
plt.suptitle(f"Profiles at time {t[i]}")
plt.savefig(r'plots/fullprofiles/q/plot' + str(i))



fig, ax = plt.subplots(1,2, figsize = (10,10))
for i in range(0,4,1):    
    ax[0].plot(ql[dtmin*i], z,  '--' , label = 'Nc = 8e6')
    ax[0].plot(qlf[dtmin*i], z,  '--' , label = 'Nc = 80e6')
    ax[0].set_title("Specific liquid humidity day_min")
    ax[0].set_xlabel("ql [kg/kg]")
    ax[0].set_xlim([-0.5e-5,np.max(ql)])
    ax[0].set_ylabel("Height [m]")
    ax[0].legend(loc='lower left')
    
    ax[1].plot(ql1[dt*i], z,  '--' , label = 'Nc = 200e6')
    ax[1].plot(ql1f[dt*i], z,  '--' , label = 'Nc = 2000e6')
    ax[1].set_title("Specific liquid humidity day_max")
    ax[1].set_xlabel("ql [kg/kg]")
    ax[1].set_xlim([-0.5e-5,np.max(ql1)+0.5e-5])
    ax[1].set_ylabel("Height [m]")
    ax[1].legend(loc='lower left')
    
    plt.suptitle(f"Profiles at time {t[i]}")
    plt.savefig(r'plots/fullprofiles/ql/plot' + str(i))
     
    
    # plt.close('all')

# "NSE PLOTS"

# x = np.arange(0,72,1)


# fig, ax = plt.subplots(2,1, figsize = (10,10))

# ax[0].plot(x,NSE_T_min, label = "Nc = 8e6")
# ax[0].plot(x,NSE_T_minf, label = "Nc = 80e6")
# ax[0].set_title("NSE Temperature day_min")
# ax[0].set_xlabel('Time [10min]')
# ax[0].set_ylabel('NSE [-]')
# ax[0].legend()

# ax[1].plot(x,NSE_T_max, label = "Nc = 200e6")
# ax[1].plot(x,NSE_T_maxf, label = "Nc = 2000e6")
# ax[1].set_title("NSE Temperature day_max")
# ax[1].set_xlabel('Time [10min]')
# ax[1].set_ylabel('NSE [-]')
# ax[1].legend()

# fig, ax = plt.subplots(2,1, figsize = (10,10))

# ax[0].plot(x,NSE_qt_min, label = "Nc = 8e6")
# ax[0].plot(x,NSE_qt_minf, label = "Nc = 80e6")
# ax[0].set_title("NSE qt day_min")
# ax[0].set_xlabel('Time [10min]')
# ax[0].set_ylabel('NSE [-]')
# ax[0].legend()

# ax[1].plot(x,NSE_qt_max, label = "Nc = 200e6")
# ax[1].plot(x,NSE_qt_maxf, label = "Nc = 2000e6")
# ax[1].set_title("NSE qt day_max")
# ax[1].set_xlabel('Time [10min]')
# ax[1].set_ylabel('NSE [-]')
# ax[1].legend()




# "GIFS"

# folder_path = r'plots\shortwave'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/fullprofiles/shortwave.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/shortwave/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)
    

# folder_path = r'plots\temp'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/temp.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/temp/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)


# folder_path = r'plots\longwave'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/fullprofiles/longwave.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/longwave/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)

# folder_path = r'plots\q'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/q.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/q/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)
        

# folder_path = r'plots\ql'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/ql.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/ql/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)


# "FULL PROFILES"

# folder_path = r'plots/fullprofiles/temp'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/fullprofiles/temp.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/fullprofiles/temp/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)


# folder_path = r'plots/fullprofiles/q'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/fullprofiles/q.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/fullprofiles/q/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)
        

# folder_path = r'plots/fullprofiles/ql'
# list_of_files = os.listdir(folder_path)

# with imageio.get_writer('plots/fullprofiles/ql.gif', mode='I') as writer:
#     for i in range(len(list_of_files)):
#         filename = str('plots/fullprofiles/ql/') + list_of_files[i]
#         image = imageio.imread(filename)
#         writer.append_data(image)

# TODO:
#     add plots of fake runs in the same graphs for both cases
#     calculate NSE of modeled to observed
#     fix legend 







