# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:40:59 2022

@author: Alessandro Pieruzzi
"""


import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import datetime

R = 287 #[J/(kg*K)] dry air
Rv = 461 #[J/(kg*K)] vapour
Lv = 2.5*10**6 #J/kg latent heat of vaporization
e0 = 0.611 #kPa
g = 9.8
cp = 1005 #J/kg*K
x = 5 #to cut outliers
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


#Find day with clouds:
ind = np.where(DPD==0)[0]
time = np.array(profiles['synTime'])
time_day = time/86400
rel_time = (time_day[ind]-time_day[0]) #days from 1/1/2008

start_date = '1/1/08'
date_1 = datetime.datetime.strptime(start_date, "%d/%m/%y")
dateList = []
for i in range (len(ind)):
    dateList.append(date_1 + datetime.timedelta(days=rel_time[i]))
# print(dateList)

#Select days with clouds only 
temp = temp[ind]
height = height[ind]
DPD = DPD[ind]
P = P[ind]
wd = wd[ind]
ws = ws[ind]



"CALCULATIONS"

#translate dew point depression to specific and relative humidity 
T0 = 273.15
Td = temp - DPD
e = e0 *np.exp(Lv/Rv*(1/T0-1/Td))
es = e0 *np.exp(Lv/Rv*(1/T0-1/temp)) #kPa
RH = e/es 
# q = 621.97/P * np.exp((Lv/Rv)*(Td-T0)/(Td*T0))
# q = (0.622*e0)/P * np.exp((Lv/Rv)*((1/T0)-(1/Td)))

q = R/Rv * es*10/P
q = q/1000

#calculate the liquid humidity 
ql = np.zeros(q.shape)

for i in range(len(DPD)):
    for j in range(0,22,1):
        if DPD[i,j]==0:
            ql[i,j] = q[i,j]

#translate temperature to potential liquid temperature
thl = temp + g*height/cp - Lv*ql #chec if its the same with other version on the manual
thl_man = (p0/P)**(R/cp) * (temp - Lv/cp *ql)             
#Calculate wind speed in x and y direction (U and V)

#Probable issue: we don't know how the angle is defined (I assumed angle from north, usually it is the case)
U = ws*np.sin(np.deg2rad(wd))
V = ws*np.cos(np.deg2rad(wd))

#Creating TKE array 

Cd = 0.005 #drag coefficient for praire
h_bl = 1000 #change manually depending on where the inversion jump is
w10 = U[:,0] #wind speed at 2m (should be at 10m, use interp once available)
u_st = np.sqrt(Cd * w10**2)
Up = np.zeros(height.shape)
Vp = np.zeros(height.shape)

for i in range(len(u_st)):
    for j in range(0,x,1):
        if height[i,j]<=h_bl:
            Up[i,j] = 2*u_st[i]*(1-(height[i,j]/h_bl))**(3/4)
            Vp[i,j] = 2.2*u_st[i]*(1-(height[i,j]/h_bl))**(3/4)
        else: 
            Up[i,j] = 0
            Vp[i,j] = 0
        
tke = 1/2 * (Up**2 + Vp**2)

"CREATING INPUT FILES"

#interpolation

#save to file 





"PLOTTING PROFILES"

#Selected days for initial observations crossing with CCN plots:
day_min = '28/02/09 12'
day_max = '28/05/09'
date_min = datetime.datetime.strptime(day_min, "%d/%m/%y %H")
date_max = datetime.datetime.strptime(day_max, "%d/%m/%y")

ind_min = np.where(np.array(dateList) == date_min)[0][0]
ind_max = np.where(np.array(dateList) == date_max)[0][0]

#plots

plt.figure()
plt.plot(DPD[ind_min,0:x],height[ind_min,0:x])
plt.title("Dew point depression on day_min")

plt.figure()
plt.plot(temp[ind_min,0:x],height[ind_min,0:x])
plt.title("Temperature on day_min")
plt.show()

plt.figure()
plt.plot(thl[ind_min,0:x],height[ind_min,0:x])
plt.title("Liquid potential emperature on day_min")
plt.show()

plt.figure()
plt.plot(RH[ind_min,0:x],height[ind_min,0:x])
plt.title("Relative humidity on day_min")
plt.show()

plt.figure()
plt.plot(q[ind_min,0:x],height[ind_min,0:x])
plt.title("Total specific humidity on day_min")

plt.figure()
plt.plot(ql[ind_min,0:x],height[ind_min,0:x])
plt.title("Liquid specific humidity on day_min")

plt.figure()
plt.plot(U[ind_min,0:x],height[ind_min,0:x])
plt.title("Wind speed (U) on day_min")

plt.figure()
plt.plot(V[ind_min,0:x],height[ind_min,0:x])
plt.title("Wind speed (U) on day_min")

plt.figure()
plt.plot(thl[ind_min,0:x],height[ind_min,0:x], label='thl')
plt.plot(thl_man[ind_min,0:x],height[ind_min,0:x], label='thl_man')
plt.title("Liquid potential emperature on day_min")
plt.legend()
plt.show()

plt.figure()
plt.plot(tke[ind_min,0:x],height[ind_min,0:x])
plt.title("TKE on day_min")

# plt.figure()
# plt.plot(temp[ind_max,0:x],height[ind_max,0:x])
# plt.title("Temperature on day_max")
# plt.show()

# plt.figure()
# plt.plot(thl[ind_max,0:x],height[ind_max,0:x])
# plt.title("Liquid potential emperature on day_max")
# plt.show()

# plt.figure()
# plt.plot(RH[ind_max,0:x],height[ind_max,0:x])
# plt.title("Relative humidity on day_max")
# plt.show()

# plt.figure()
# plt.plot(q[ind_max,0:x],height[ind_max,0:x])
# plt.title("Total specific humidity on day_max")

# plt.figure()
# plt.plot(DPD[ind_max,0:x],height[ind_max,0:x])
# plt.title("Dew point depression on day_max")

# plt.figure()
# plt.plot(ql[ind_max,0:x],height[ind_max,0:x])
# plt.title("Liquid specific humidity on day_max")


