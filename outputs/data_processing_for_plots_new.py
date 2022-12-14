# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:05:44 2022

@author: Alessandro Pieruzzi
"""

"TODO:"
# Import cabauw data and calculate
    # height from 12,5 to 5000, steps of 25
    # Calculate: thl, qt, u, v, tke and interpolate them 
# Retrieve data from radiosondes for the same variables (interpolated)
# Plot to see the difference and match them 
# Print them in the file 
# GIve the file to motherfuckter NIckolaos
#

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
from scipy.interpolate import interp1d
from netCDF4 import Dataset

R = 287 #[J/(kg*K)] dry air
Rv = 461 #[J/(kg*K)] vapour
Lv = 2.5*10**6 #J/kg latent heat of vaporization
e0 = 6.11 #hPa
g = 9.8
cp = 1005 #J/kg*K
x = 13 #to cut outliers
p0 = 1000
z = np.arange(12.5,5000,25)

#Import files
cabauw4= nc.Dataset('cesar_tower_meteo_lb1_t10_v1.2_200905.nc')
cabauw1= nc.Dataset('cesar_tower_meteo_lb1_t10_v1.2_200805.nc')
profiles = nc.Dataset('raob_soundings13628.cdf')

#Import relevant variables
temp_c1 = np.array(cabauw1['TA'])
wd_c1 = np.array(cabauw1['D'])
ws_c1 = np.array(cabauw1['F'])
q_c1 = np.array(cabauw1['Q'])/1000 #g/kg!!!
date_c1 = np.array(cabauw1['date'])
height_c1 = np.array(cabauw1['z'])

temp_c4 = np.array(cabauw4['TA'])
wd_c4 = np.array(cabauw4['D'])
ws_c4 = np.array(cabauw4['F'])
q_c4 = np.array(cabauw4['Q'])/1000 #kg/kg!!!
date_c4 = np.array(cabauw4['date'])
height_c4 = np.array(cabauw4['z'])

temp = np.array(profiles['tpMan'])
DPD = np.array(profiles['tdMan'])
P = np.array(profiles['prMan']) #hPa
height = np.array(profiles['htMan'])
wd = np.array(profiles['wdMan'])
ws = np.array(profiles['wsMan'])
time = np.array(profiles['synTime'])

#Select days we use
"FOR RADIOSONDES"
#Create list of dates (to translate the time in seconds to the time in date format)
time_day = time/86400
rel_time = (time_day-time_day[0]) #days from 1/1/2008
start_date = '1/1/08'
date_1 = datetime.datetime.strptime(start_date, "%d/%m/%y")
dateList = []
for i in range (len(rel_time)):
    if rel_time[i] > rel_time[-1]:
        rel_time[i] = (rel_time[i-1]+rel_time[i+1])/2 #fix outliers
    dateList.append(date_1 + datetime.timedelta(days=rel_time[i]))

#Selected days for initial observations crossing with CCN plots:
day_min = '13/05/08 00'
day_max = '31/05/09 00'
date_min = datetime.datetime.strptime(day_min, "%d/%m/%y %H")
date_max = datetime.datetime.strptime(day_max, "%d/%m/%y %H")

ind_min = np.where(np.array(dateList) == date_min)[0][0]
ind_max = np.where(np.array(dateList) == date_max)[0][0]
ind = np.array([ind_min, ind_max])

temp = temp[ind,0:x]
DPD = DPD[ind,0:x]
P = P[ind,0:x]
height = height[ind,0:x]
wd = wd[ind,0:x]
ws = ws[ind,0:x]


"FOR CABAUW"
daymin = 20080513
daymax = 20090531
ind_c1 = np.where((date_c1 == daymin) | (date_c1 == daymin+1))[0][0:73] #to get from 00 to 12
ind_c4 = np.where((date_c4 == daymax) | (date_c4 == daymax+1))[0][0:73]


temp_c1 = temp_c1[ind_c1,:]
wd_c1 = wd_c1[ind_c1,:]
ws_c1 = ws_c1[ind_c1,:]
q_c1 = q_c1[ind_c1,:]

temp_c4 = temp_c4[ind_c4,:]
wd_c4 = wd_c4[ind_c4,:]
ws_c4 = ws_c4[ind_c4,:] 
q_c4 = q_c4[ind_c4,:] 

#Fixing wind speed and direction outliers
for i in range(len(ws_c1)):
    for j in range(ws_c1.shape[1]):
        if abs(ws_c1[i,j]) > 100:
            ws_c1[i,j]=ws_c1[i,j-1]
        if abs(ws_c4[i,j]) > 100:
            ws_c4[i,j]=ws_c4[i,j-1]
        if abs(wd_c1[i,j])>360:
            wd_c1[i,j]=wd_c1[i,j-1]
        if abs(wd_c4[i,j])>360:
            wd_c4[i,j]=wd_c4[i,j-1]

dim = ws.shape
for i in range(dim[0]):
    for j in range(dim[1]):
        if ws[i,j]>100:
            ws[i,j] = ws[i,j-1]

    


#Calculate wind speed in x and y direction (U and V)

dim = wd_c1.shape
for i in range(dim[0]):
    for j in range(dim[1]):
        if wd_c1[i,j]<180:
            wd_c1[i,j] = wd_c1[i,j] +180
        elif wd_c1[i,j]>180:
            wd_c1[i,j]=wd_c1[i,j]-180
        if wd_c4[i,j]<180:
            wd_c4[i,j] = wd_c4[i,j] +180
        elif wd_c4[i,j]>180:
            wd_c4[i,j]=wd_c4[i,j]-180


dim = wd.shape
for i in range(dim[0]):
    for j in range(dim[1]):
        if wd[i,j]<180:
            wd[i,j] = wd[i,j] +180
        elif wd[i,j]>180:
            wd[i,j]=wd[i,j]-180
    
#Angle assumed to be from the north
U = ws*np.sin(np.deg2rad(wd))
V = ws*np.cos(np.deg2rad(wd))
    
U_c1 = ws_c1*np.cos(np.deg2rad(wd_c1))
V_c1 = ws_c1*np.sin(np.deg2rad(wd_c1))

U_c4 = ws_c4*np.cos(np.deg2rad(wd_c4))
V_c4 = ws_c4*np.sin(np.deg2rad(wd_c4))


"INTERPOLATION"
# Interpolation function with plotted example. linspacetointerp is essentially the new x-space.

def interpolate(xdataset, ydataset, linspacetointerp, linear=True):
    kindtoUse = ''
    if linear:
        kindtoUse = 'linear'
    else:
        kindtoUse = 'cubic'

    f = interp1d(xdataset, ydataset, kind = kindtoUse, fill_value='extrapolate')
    return f(linspacetointerp)


#Obtain interpolated profiles for each day:
dim = [len(ind), len(z)]
dim_c = [len(ind_c1),9]

temp_interp = np.zeros(dim)
DPD_interp = np.zeros(dim)
P_interp = np.zeros(dim)
U_interp = np.zeros(dim)
V_interp = np.zeros(dim)

tempc1_interp = np.zeros(dim_c)
U_c1_interp = np.zeros(dim_c)
V_c1_interp = np.zeros(dim_c)
qc1_interp = np.zeros(dim_c)

tempc4_interp = np.zeros(dim_c)
U_c4_interp = np.zeros(dim_c)
V_c4_interp = np.zeros(dim_c)
qc4_interp = np.zeros(dim_c)

for i in range(len(ind)):
    temp_interp[i] = interpolate(height[i], temp[i], z, linear=False)
    DPD_interp[i] = interpolate(height[i], DPD[i], z, linear=False)
    P_interp[i] = interpolate(height[i], P[i], z, linear=False)
    U_interp[i] = interpolate(height[i], U[i], z, linear=False)
    V_interp[i] = interpolate(height[i], V[i], z, linear=False)

for i in range(len(ind_c1)):
    tempc1_interp[i] = interpolate(height_c1, temp_c1[i], z[0:9], linear=False)
    U_c1_interp[i] = interpolate(height_c1, U_c1[i], z[0:9], linear=False)
    V_c1_interp[i] = interpolate(height_c1, V_c1[i], z[0:9], linear=False)
    qc1_interp[i] = interpolate(height_c1, q_c1[i], z[0:9], linear=False)

    tempc4_interp[i] = interpolate(height_c4, temp_c4[i], z[0:9], linear=False)
    U_c4_interp[i] = interpolate(height_c4, U_c4[i], z[0:9], linear=False)
    V_c4_interp[i] = interpolate(height_c4, V_c4[i], z[0:9], linear=False)
    qc4_interp[i] = interpolate(height_c1, q_c4[i], z[0:9], linear=False)


#Save original data (to plot later) and change names of interpolated ones (to use for calculations)
temp_o = temp
DPD_o = DPD
P_o = P
U_o = U
V_o = V


temp = temp_interp
DPD = DPD_interp
P = P_interp
U = U_interp
V = V_interp

tempc1_o = temp_c1
qc1_o = q_c1
Uc1_o = U_c1
Vc1_o = V_c1

temp_c1 = tempc1_interp
q_c1 = qc1_interp
U_c1 = U_c1_interp
V_c1 = V_c1_interp

tempc4_o = temp_c4
qc4_o = q_c4
Uc4_o = U_c4
Vc4_o = V_c4

temp_c4 = tempc4_interp
q_c4 = qc4_interp
U_c4 = U_c4_interp
V_c4 = V_c4_interp

#Fixing DPD outliers
for i in range(DPD.shape[0]):
    for j in range(DPD.shape[1]):
        if DPD[i,j] < 0:
            DPD[i,j]=0


"CALCULATIONS"

#For radiosondes:
#translate dew point depression to specific and relative humidity
T0 = 273.15
Td = temp - DPD
e = e0 *np.exp(Lv/Rv*(1/T0-1/Td))
es = e0 *np.exp(Lv/Rv*(1/T0-1/temp)) #kPa
RH = e/es
qs = (R/Rv * es/P)#[kg/kg]
q = RH*qs

# #calculate the liquid humidity (WRONG)
# ql = np.zeros(q.shape)

# for i in range(len(DPD)):
#     for j in range(0,22,1):
#         if DPD[i,j]==0:
#             ql[i,j] = q[i,j]

# #translate temperature to potential liquid temperature
# thl = temp + g*height/cp - Lv*ql #chec if its the same with other version on the manual
# thl_man = (p0/P)**(R/cp) * (temp - Lv/cp *ql)

#Calculate potential temperature (clear case -> th = thl)
thl = temp * (p0/P)**(R/cp)



#Calculating TKE

Cd = 0.005 #drag coefficient for praire
h_bl = 1000 #change manually depending on where the inversion jump is
w10 = U[:,1] #wind speed at 7.5m (should be at 10m, approximation)
u_st = np.sqrt(Cd * w10**2)
Up = np.zeros(dim)
Vp = np.zeros(dim)

for i in range(len(u_st)):
    for j in range(len(z)):
        if z[j]<=h_bl:
            Up[i,j] = 2*u_st[i]*(1-(z[j]/h_bl))**(3/4)
            Vp[i,j] = 2.2*u_st[i]*(1-(z[j]/h_bl))**(3/4)
        else:
            Up[i,j] = 0
            Vp[i,j] = 0

tke = 1/2 * (Up**2 + Vp**2)




#For Cabauw

thl_c1 = temp_c1 * (p0/P[0,0:9])**(R/cp)
thl_c4 = temp_c4 * (p0/P[1,0:9])**(R/cp)



"PLOTTING PROFILES"

#First 200m
# plt.figure()
# plt.plot(temp[0,0:9],z[0:9], label="day_min")
# plt.plot(temp[1,0:9],z[0:9], label="day_max")
# plt.plot(temp_o[0,0:2],height[0,0:2], 'o', label="day_min original")
# plt.plot(temp_o[1,0:2],height[1,0:2], 'o', label="day_max original")
# plt.plot(temp_c1[0,:],z[0:9],'--', label="day_min_c")
# plt.plot(temp_c4[0,:],z[0:9],'--', label="day_max_c")
# plt.plot(tempc1_o[0,:],height_c1,'o', label="day_min_c")
# plt.plot(tempc4_o[0,:],height_c4,'o', label="day_max_c")
# plt.xlabel('Temperature [K]')
# plt.ylabel('Height [m]')
# plt.legend()
# plt.title("Temperature")

# plt.figure()
# plt.plot(thl[0,0:9],z[0:9], label="day_min")
# plt.plot(thl[1,0:9],z[0:9], label="day_max")
# plt.plot(thl_c1[0,:],z[0:9],'--', label="day_min_c")
# plt.plot(thl_c4[0,:],z[0:9],'--', label="day_max_c")
# plt.xlabel('thl [K]')
# plt.ylabel('Height [m]')
# plt.title("Liquid potential temperature")
# plt.legend()

# plt.figure()
# plt.plot(q[0,0:9],z[0:9], label="day_min")
# plt.plot(q[1,0:9],z[0:9], label="day_max")
# plt.plot(q_c1[0,:],z[0:9],'--', label="day_min_c")
# plt.plot(q_c4[0,:],z[0:9],'--', label="day_max_c")
# plt.plot(qc1_o[0,:],height_c1,'o', label="day_min_c")
# plt.plot(qc4_o[0,:],height_c4,'o', label="day_max_c")
# plt.xlabel('qt [kg/kg]')
# plt.ylabel('Height [m]')
# plt.legend()
# plt.title("Total specific humidity")

plt.figure()
plt.plot(U[0,0:9],z[0:9], label="day_min")
plt.plot(U[1,0:9],z[0:9], label="day_max")
plt.plot(Uc1_o[0,:],z[0:7],'o', label="day_min_c")
plt.plot(Uc4_o[0,:],z[0:7],'o', label="day_min_c")
plt.plot(U_c1[0,:],z[0:9],'--', label="day_min_c")
plt.plot(U_c4[0,:],z[0:9],'--', label="day_max_c")
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Height [m]')
plt.legend()
plt.title("Wind speed (U)")

plt.figure()
plt.plot(V[0,0:9],z[0:9], label="day_min")
plt.plot(V[1,0:9],z[0:9], label="day_max")
plt.plot(V_c1[0,:],z[0:9],'--', label="day_min_c")
plt.plot(Vc1_o[0,:],z[0:7],'o', label="day_min_c")
plt.plot(Vc4_o[0,:],z[0:7],'o', label="day_min_c")
plt.plot(V_c4[0,:],z[0:9],'--', label="day_max_c")
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Height [m]')
plt.legend()
plt.title("Wind speed (V)")

#Whole profile
# plt.figure()
# plt.plot(temp[0,:],z, label="day_min")
# plt.plot(temp[1,:],z, label="day_max")
# plt.plot(temp_o[0,0:6],height[0,0:6], 'o', label="day_min original")
# plt.plot(temp_o[1,0:6],height[1,0:6], 'o', label="day_max original")
# plt.xlabel('Temperature [K]')
# plt.ylabel('Height [m]')
# plt.legend()
# plt.title("Temperature")

# plt.figure()
# plt.plot(thl[0,:],z, label="day_min")
# plt.plot(thl[1,:],z, label="day_max")
# plt.xlabel('Temperature [K]')
# plt.ylabel('Height [m]')
# plt.title("Liquid potential temperature")
# plt.legend()

# plt.figure()
# plt.plot(q[0,:],z, label="day_min")
# plt.plot(q[1,:],z, label="day_max")
# plt.legend()
# plt.title("Total specific humidity")

plt.figure()
plt.plot(U[0,:],z, label="day_min")
plt.plot(U[1,:],z, label="day_max")
plt.legend()
plt.title("Wind speed (U)")

plt.figure()
plt.plot(V[0,:],z, label="day_min")
plt.plot(V[1,:],z, label="day_max")
plt.legend()
plt.title("Wind speed (V)")

#Other variables

# plt.figure()
# plt.plot(tke[0,:],z, label="day_min")
# plt.plot(tke[1,:],z, label="day_max")
# plt.legend()
# plt.title("TKE")

# plt.figure()
# plt.plot(RH[0,:],z, label="day_min")
# plt.plot(RH[1,:],z, label="day_max")
# plt.title("Relative humidity")
# plt.legend()

# plt.figure()
# plt.plot(P[0,:],z, label="day_min")
# plt.plot(P[1,:],z, label="day_max")
# plt.plot(P_o[0,0:6],height[0,0:6], 'o', label="day_min original")
# plt.plot(P_o[1,0:6],height[1,0:6], 'o', label="day_max original")
# plt.legend()
# plt.title("Pressure")

print("Vgeo_min = ", np.mean(V_o[0,:]), '\n')
print("Ugeo_min = ", np.mean(U_o[0,:]), '\n')
print("Vgeo_max = ", np.mean(V_o[1,:]), '\n')
print("Ugeo_max = ", np.mean(U_o[1,:]), '\n')



"CREATING INPUT FILES"


# Profile for day_min
profile = open("obs_min.txt", "w")
profile.write("#ASTEX case using prescribed vertical grid, Nlev = 427 1 date" + day_min +'\n' \
      "height(m)   T(K)     qt(kg/kg)       u(m/s)     v(m/s)     tke(m2/s2)     pres\n")


def format_num(n):
    return '{:.7s}'.format('{:0.4f}'.format(n))

def format_q(q):
    return "{:.3E}".format(q)

def add_line(z, temp, qt, u, v, tke, P):

    return "      " + format_num(z) + "      " \
    + format_num(temp) + "      " + format_q(qt) + "      " \
    + format_num(u) + "      " + format_num(v) + "      " \
    + format_num(tke) + "      " + format_num(P) + "\n"

profile = open("obs_min.txt", "a")



#save to file
for i in range(0,len(z),1):
    if i < 8:
        profile.write(add_line(z[i], temp_c1[0,i], q_c1[0,i], U_c1[0,i], V_c1[0,i], tke[0,i], P[0,i]))
    else:
        profile.write(add_line(z[i], temp[0,i], q[0,i], U[0,i], V[0,i], tke[0,i], P[0,i]))



# Profile for day_max
profile = open("obs_max.txt", "w")
profile.write("#ASTEX case using prescribed vertical grid, Nlev = 427 1 date" + day_min +'\n'  \
      "height(m)   T(K)     qt(kg/kg)       u(m/s)     v(m/s)     tke(m2/s2)     pres\n")

profile = open("obs_max.txt", "a")

#save to file
for i in range(0,len(z),1):
    if i < 8:
        profile.write(add_line(z[i], temp_c4[0,i], q_c4[0,i], U_c4[0,i], V_c4[0,i], tke[0,i], P[0,i]))
    else:
        profile.write(add_line(z[i], temp[1,i], q[1,i], U[1,i], V[1,i], tke[1,i], P[1,i]))

profile.close()


"SAVING CABAUW DATA EVERY HOUR"


#Creating file
ncfile = Dataset('cabauw_hourly.nc', mode='w', format='NETCDF4_CLASSIC')

#Creating dimensions
time_dim = ncfile.createDimension('time', 73)
height_dim = ncfile.createDimension('height', 9)

#Creating variables
#Variables: T_c1, T_c4, q_c1, q_c4

temp_jan = ncfile.createVariable('temp_jan', np.float64,('time', 'height'))
temp_apr = ncfile.createVariable('temp_apr', np.float64,('time', 'height'))
q_jan = ncfile.createVariable('q_jan', np.float64,('time', 'height'))
q_apr = ncfile.createVariable('q_apr', np.float64,('time', 'height'))

temp_jan[:] = temp_c1
temp_apr[:] = temp_c4
q_jan[:] = q_c1
q_apr[:] = q_c4

ncfile.close()











