    #######################################################################
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
#import metpy.calc
#from metpy.units import units
#plt.rcParams["figure.figsize"] = 8, 8
from datetime import datetime
from scipy.interpolate import interp1d
########################################################################


########################################################################
#Constants
Rd = 287.04 #J kg−1 K−1
cp = 1004   #J kg-1 K-1
p0 = 100000 #Pa
########################################################################




########################################################################
#Inmport Radiosonde Data - all relevant variables
#soundigns from Jan 1 2008 thorugh Jan 1 2010
rs_profiles = nc.Dataset('raob_soundings43790.cdf')
variables = rs_profiles.variables
#Variables from .nc files profiles
lvls = rs_profiles['numMand']
p = rs_profiles['prMan'][0][0:10]
t = rs_profiles['tpMan'][0][0:10]
td = rs_profiles['tdMan'][:]
ht = rs_profiles['htMan'][0][0:10]
ts = rs_profiles['synTime'][:]
##########################################################################


##########################################################################
# Input date times for the 10 selected days to study:
# max at time 00: 22947 cm-3, day 28 of february
# hig at time 00:
# avg at time 00:
# low at time 00:
# min at time 00: 512 cm-3, day 31 of may 2009
# max at time 12: 14284 cm-3, day 28 of february 2009
# hig at time 12:
# avg at time 12:
# low at time 12:
# min at time 12: 420 cm-3, day 28 of may 2009
t00max = '2009-02-28 00:00:00'
t00max_index = 366 * 2 + (31 + 27) * 2 - 3 #adjusted
t00max_rec = (datetime.utcfromtimestamp(ts[t00max_index]).strftime('%Y-%m-%d %H:%M:%S'))
t00min = '2009-05-31 00:00:00'
t00min_index = 366 * 2 + (31 + 28 + 31 + 30 + 30) * 2 - 4 #adjusted
t00min_rec = (datetime.utcfromtimestamp(ts[t00min_index]).strftime('%Y-%m-%d %H:%M:%S'))
t12max = '2009-02-28 12:00:00'
t12max_index = 366 * 2 + (31 + 27) * 2 - 2 #adjusted
t12max_rec = (datetime.utcfromtimestamp(ts[t12max_index]).strftime('%Y-%m-%d %H:%M:%S'))
t12min = '2009-05-28 12:00:00'
t12min_index = 366 * 2 + (31 + 28 + 31 + 30 + 27) * 2 - 3 #adjusted
t12min_rec = (datetime.utcfromtimestamp(ts[t12min_index]).strftime('%Y-%m-%d %H:%M:%S'))
select_indices = [t00max_index, t00min_index, t12max_index, t12min_index]
##########################################################################




##########################################################################
#Check whether stable or unstable
def thl_vs_z(i, select_indices):
        th = t[select_indices[i], :] * (p0/p[select_indices[i], :]) ** (Rd/cp)
        plt.plot(th, ht[select_indices[i], :], label=str(datetime.utcfromtimestamp(ts[select_indices[i]]).strftime('%Y-%m-%d %H:%M:%S')))
        plt.xlabel('Potential Temperature [K]')
        plt.ylabel('Geopotential height [m]')
        return th

# for i in range(len(select_indices)):
#     th1 = thl_vs_z(i, select_indices)
#     plt.legend()
##########################################################################





##########################################################################
#Build lscale.inp files
#Based on the cases run previously, CBL uses ugeo = 1, and SBL uses ugeo=8, and all other values= 0 in both cases

heights = np.linspace(0, 20000, 101)


def format_num(n):
        return '{:.6s}'.format('{:0.3f}'.format(n))

def add_line(height):
        return "          " + format_num(height) + "      " + format_num(8) + "      " + format_num(0) + "      " + format_num(0) + "         " + format_num(0) + "              " + format_num(0)+ "      " + format_num(0)+ "      " + format_num(0)  +"\n"

def build_inp():
        profile = open("inputfiles/lscale.inp.txt", "w")
        profile.write(" Midnight with high concentrations 1\n \
         height(m)   ugeo(m/s) vgeo(m/s)  wfls(m/s)    not_used   not_used   dqtdtls(kg/kg/s)    dthldt(K/s)\n")
    #profile.write(format_num(temperature[-1]) +  "      13 \n")
        for i in range(0, len(heights)):
            profile.write(add_line(heights[i]))
            profile.close()
            return profile

# build_inp()

#########################################################################

# Interpolation function with plotted example. linspacetointerp is essentially the new x-space.

def interpolate(xdataset, ydataset, linspacetointerp, linear=True):
        kindtoUse = ''
        if linear:
            kindtoUse = 'linear'
        else:
            kindtoUse = 'cubic'
    
        f = interp1d(xdataset, ydataset, kind = kindtoUse, fill_value='extrapolate')
        plt.plot(xdataset, ydataset, 'o', linspacetointerp, f(linspacetointerp), '-')
        plt.legend(['data', kindtoUse], loc='best')
        plt.show()
        return f(linspacetointerp)

interpolate(ht, t, np.linspace(0, 12000, 100), linear=False)





#########################################################################
#geenrate prof.inp file
#formual for liquid potential temperature
#get qt and when it reaches qs, say it is ql --> Ale
