import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import metpy.calc
from metpy.units import units

radiosonde = nc.Dataset('raob_soundings40363.cdf')

def format_num(n):
    return '{:.7s}'.format('{:0.4f}'.format(n))


# variable daytime corresponds to day and time requested.
# For example, 0 is the first day at 00:00 hour while 1 is the first day at 12:00 and so on
def get_backrad_input(daytime = 0):
    # height = np.array(radiosonde['htMan'][daytime][0:12])
    pressure = np.array(radiosonde['prMan'][daytime][0:12])
    temperature = np.array(radiosonde['tpMan'][daytime][0:12])
    print(pressure)
    # humidity = get_humidity(daytime)

    # height = np.flip(height)
    pressure = np.flip(pressure) * 100
    temperature = np.flip(temperature)
    ozone = np.array([1.23 * (10 ** (-6)), 0.51 * (10 ** (-6)), 2.07 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
    1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
    1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7))])
    water = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return pressure, temperature, water, ozone, water
    # humidity = np.flip(humidity)

def add_line(pressure, temperature, humidity, ozone, water):

    return format_num(pressure) + "      " + format_num(temperature) + "      " + format_num(humidity)\
    + "      " + format_num(ozone) + "      " + format_num(water) + "\n"


pressure, temperature, humidity, ozone, water = get_backrad_input()

profile = open("inputfiles/backrad.inp.txt", "w")

profile.write(format_num(temperature[-1]) +  "      13 \n")


for i in range(0, 12):
    profile.write(add_line(pressure[i], temperature[i], humidity[i], ozone[i], water[i]))

profile.close()
