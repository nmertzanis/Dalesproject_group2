# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:40:59 2022

@author: Alessandro Pieruzzi
"""

"TODO"
# - what is ASTEX?
# - Check hbl for TKE
# - Run simulation

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
from scipy.interpolate import interp1d
import seaborn as sns

sns.set()

R = 287 #[J/(kg*K)] dry air
Rv = 461 #[J/(kg*K)] vapour
Lv = 2.5*10**6 #J/kg latent heat of vaporization
e0 = 0.611 #kPa
g = 9.8
cp = 1005 #J/kg*K
x = 15 #to cut outliers
p0 = 1000

"DATA PREPROCESSING"

# Import data
profiles = nc.Dataset('raob_soundings13628.cdf')
temp = np.array(profiles['tpMan'])
DPD = np.array(profiles['tdMan'])
P = np.array(profiles['prMan']) #hPa
height = np.array(profiles['htMan'])
wd = np.array(profiles['wdMan'])
ws = np.array(profiles['wsMan'])
time = np.array(profiles['synTime'])

#Select days we use

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
day_min = '06/01/09 12'
day_max = '03/04/09 12'
date_min = datetime.datetime.strptime(day_min, "%d/%m/%y %H")
date_max = datetime.datetime.strptime(day_max, "%d/%m/%y %H")

ind_min = np.where(np.array(dateList) == date_min)[0][0]
ind_max = np.where(np.array(dateList) == date_max)[0][0]
ind = np.array([ind_min, ind_max])

#Select respective days in the data and remove the outliers
temp = temp[ind,0:x]
DPD = DPD[ind,0:x]
P = P[ind,0:x]
height = height[ind,0:x]
wd = wd[ind,0:x]
ws = ws[ind,0:x]


# #Find day with clouds (not used):
# ind = np.where(DPD==0)[0]
# time = np.array(profiles['synTime'])
# time_day = time/86400
# rel_time = (time_day[ind]-time_day[0]) #days from 1/1/2008

# start_date = '1/1/08'
# date_1 = datetime.datetime.strptime(start_date, "%d/%m/%y")
# dateList = []
# for i in range (len(ind)):
#     dateList.append(date_1 + datetime.timedelta(days=rel_time[i]))
# # print(dateList)

# #Select days with clouds only (not used)
# temp = temp[ind]
# height = height[ind]
# DPD = DPD[ind]
# P = P[ind]
# wd = wd[ind]
# ws = ws[ind]

"CREATING HEIGHTS AND INTERPOLATING the variables"

#Using heights from example (radtransf, ASTEX)
# prof_ex = np.genfromtxt('prof_ex_radtransf.txt')
# z = prof_ex[:,0]
z = np.linspace(50, 19950, 200)


#interpolation
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
temp_interp = np.zeros(dim)
DPD_interp = np.zeros(dim)
P_interp = np.zeros(dim)
ws_interp = np.zeros(dim)
wd_interp = np.zeros(dim)

for i in range(len(ind)):
    temp_interp[i] = interpolate(height[i], temp[i], z, linear=False)
    DPD_interp[i] = interpolate(height[i], DPD[i], z, linear=False)
    P_interp[i] = interpolate(height[i], P[i], z, linear=False)
    wd_interp[i] = interpolate(height[i], wd[i], z, linear=False)
    ws_interp[i] = interpolate(height[i], ws[i], z, linear=False)

#Save original data (to plot later) and change names of interpolated ones (to use for calculations)
temp_o = temp
DPD_o = DPD
P_o = P
ws_o = ws
wd_o = wd


temp = temp_interp
DPD = DPD_interp
P = P_interp
ws = ws_interp
wd = wd_interp

"CALCULATIONS"

#translate dew point depression to specific and relative humidity
T0 = 273.15
Td = temp - DPD
e = e0 *np.exp(Lv/Rv*(1/T0-1/Td))
es = e0 *np.exp(Lv/Rv*(1/T0-1/temp)) #kPa
RH = e/es
qs = (R/Rv * es*10/P)#[kg/kg]
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

#Calculate wind speed in x and y direction (U and V)

#Angle assumed to be from the north
U = ws*np.sin(np.deg2rad(wd))
V = ws*np.cos(np.deg2rad(wd))

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


"PLOTTING PROFILES"


#plots

plt.figure(figsize = (8,6))
plt.plot(DPD[0,:],z, label="6/01/2009")
plt.plot(DPD[1,:],z, label="03/04/2009")
plt.plot(DPD_o[0,0:9],height[0,0:9], 'o', label="6/01/2009 original")
plt.plot(DPD_o[1,0:9],height[1,0:9], 'o', label="03/04/2009 original")
plt.legend()
plt.xlabel("T-Td [K]")
plt.ylabel("Height [m]")
plt.title("Dew point depression")

plt.figure(figsize = (8,6))
plt.plot(P[0,:],z, label="6/01/2009")
plt.plot(P[1,:],z, label="03/04/2009")
plt.plot(P_o[0,0:9],height[0,0:9], 'o', label="6/01/2009 original")
plt.plot(P_o[1,0:9],height[1,0:9], 'o', label="03/04/2009 original")
plt.xlabel("P [hPa]")
plt.ylabel("Height [m]")
plt.legend()
plt.title("Pressure")

plt.figure(figsize = (8,6))
plt.plot(temp[0,:],z, label="6/01/2009")
plt.plot(temp[1,:],z, label="03/04/2009")
plt.plot(temp_o[0,0:9],height[0,0:9], 'o', label="6/01/2009 original")
plt.plot(temp_o[1,0:9],height[1,0:9], 'o', label="03/04/2009 original")
plt.xlabel("T [K]")
plt.ylabel("Height [m]")
plt.legend()
plt.title("Temperature")

plt.figure(figsize = (8,6))
plt.plot(thl[0,:],z, label="6/01/2009")
plt.plot(thl[1,:],z, label="03/04/2009")
plt.xlabel("Thl [K]")
plt.ylabel("Height [m]")
plt.title("Liquid potential temperature")
plt.legend()

plt.figure(figsize = (8,6))
plt.plot(RH[0,:],z, label="6/01/2009")
plt.plot(RH[1,:],z, label="03/04/2009")
plt.title("Relative humidity")
plt.legend()

plt.figure(figsize = (8,6))
plt.plot(q[0,:],z, label="6/01/2009")
plt.plot(q[1,:],z, label="03/04/2009")
plt.legend()
plt.xlabel("qt [kg/kg]")
plt.ylabel("Height [m]")
plt.title("Total specific humidity")

plt.figure(figsize = (8,6))
plt.plot(U[0,:],z, label="6/01/2009")
plt.plot(U[1,:],z, label="03/04/2009")
plt.legend()
plt.xlabel("U [m/s]")
plt.ylabel("Height [m]")
plt.title("Wind speed (U)")

plt.figure(figsize = (8,6))
plt.plot(V[0,:],z, label="6/01/2009")
plt.plot(V[1,:],z, label="03/04/2009")
plt.legend()
plt.xlabel("V [m/s]")
plt.ylabel("Height [m]")
plt.title("Wind speed (V)")

plt.figure(figsize = (8,6))
plt.plot(tke[0,:],z, label="6/01/2009")
plt.plot(tke[1,:],z, label="03/04/2009")
plt.legend()
plt.title("TKE")




# "CREATING INPUT FILES"

# def format_brnum(n):
#     return '{:.10s}'.format('{:0.10f}'.format(n))

# def get_backrad_input(pressure, temperature, humidity):
#     # height = np.array(radiosonde['htMan'][daytime][0:12])
#     # pressure = np.array(radiosonde['prMan'][daytime][0:15])
#     # temperature = np.array(radiosonde['tpMan'][daytime][0:15])
#     # print(pressure)
#     # humidity = get_humidity(daytime)

#     # height = np.flip(height)
#     pressure = np.flip(pressure) * 100
#     temperature = np.flip(temperature)
#     humidity = np.flip(humidity)
#     ozone = np.array([1.23 * (10 ** (-6)), 0.51 * (10 ** (-6)), 2.07 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
#     1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
#     1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7))])
#     water = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#     return pressure, temperature, humidity, ozone, water
#     # humidity = np.flip(humidity)

# def add_brline(pressure, temperature, humidity, ozone, water):

#     return format_brnum(pressure) + "      " +  format_brnum(temperature) + "      " +  format_brnum(humidity)\
#     + "      " +  format_brnum(ozone) + "      " + str(water) + "\n"


# # Profile for day_min
# profile = open("prof.inp.001.txt", "w")
# profile.write("#ASTEX case using prescribed vertical grid, Nlev = 427 1 date" + day_min +'\n' \
#       "height(m)   thl(K)     qt(kg/kg)       u(m/s)     v(m/s)     tke(m2/s2)\n")


# def format_num(n):
#     return '{:.7s}'.format('{:0.4f}'.format(n))

# def format_q(q):
#     return "{:.3E}".format(q)

# def add_line(z, thl, qt, u, v, tke):

#     return "      " + format_num(z) + "      " \
#     + format_num(thl) + "      " + format_q(qt) + "      " \
#     + format_num(u) + "      " + format_num(v) + "      " \
#     + format_num(tke) + "\n"

# profile = open("prof.inp.001.txt", "a")


# #save to file
# for i in range(0,len(z),1):
#     profile.write(add_line(z[i], thl[0,i], q[0,i], U[0,i], V[0,i], tke[0,i]))



# # Profile for day_max
# profile = open("prof.inp.002.txt", "w")
# profile.write("#ASTEX case using prescribed vertical grid, Nlev = 427 1 date=" + day_max + '\n' \
#       "height(m)   thl(K)     qt(kg/kg)       u(m/s)     v(m/s)     tke(m2/s2)\n")

# profile = open("prof.inp.002.txt", "a")

# #save to file
# for i in range(0,len(z),1):
#     profile.write(add_line(z[i], thl[1,i], q[1,i], U[1,i], V[1,i], tke[1,i]))
# profile.close()

# brprofile = open("backradmax.inp.txt", "w")


# currnum = 1

# temp = np.array(profiles['tpMan'])
# P = np.array(profiles['prMan']) #hPa
# DPD = np.array(profiles['tdMan'])
# temp = temp[ind,0:15]
# P = P[ind,0:15]
# DPD = DPD[ind, 0:15]

# T0 = 273.15
# Td = temp - DPD
# e = e0 *np.exp(Lv/Rv*(1/T0-1/Td))
# es = e0 *np.exp(Lv/Rv*(1/T0-1/temp)) #kPa
# RH = e/es
# q = (R/Rv * es*10/P)/1000 #[kg/kg]

# pressure, temperature, humidity, ozone, water = get_backrad_input(P[currnum][0:15], temp[currnum][0:15], q[currnum][0:15])

# brprofile.write(format_num(temperature[-1]) +  "      15 \n")


# for i in range(0, 15):
#     brprofile.write(add_brline(pressure[i], temperature[i], humidity[i], ozone[i], water[i]))

# brprofile.close()
