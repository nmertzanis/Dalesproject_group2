#######################################################################
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
#import metpy.calc
#from metpy.units import units
#plt.rcParams["figure.figsize"] = 8, 8
from datetime import datetime
########################################################################


prof = np.loadtxt('prof.inp.001.txt', skiprows=2)
heights = prof[:, 0]

##########################################################################
#Build lscale.inp files
#Based on the cases run previously, CBL uses ugeo = 1, and SBL uses ugeo=8, and all other values= 0 in both cases

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

build_inp()

#########################################################################

