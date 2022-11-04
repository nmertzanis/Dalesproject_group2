import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime

radiosonde = nc.Dataset('raob_soundings13628.cdf')

def format_num(n):
    return '{:.7s}'.format('{:0.4f}'.format(n))

# variable daytime corresponds to day and time requested.
# For example, 0 is the first day at 00:00 hour while 1 is the first day at 12:00 and so on
def get_backrad_input(daytime = 0):
    # height = np.array(radiosonde['htMan'][daytime][0:12])
    pressure = np.array(radiosonde['prMan'][daytime][0:15])
    temperature = np.array(radiosonde['tpMan'][daytime][0:15])
    print(pressure)
    # humidity = get_humidity(daytime)

    # height = np.flip(height)
    pressure = np.flip(pressure) * 100
    temperature = np.flip(temperature)
    ozone = np.array([1.23 * (10 ** (-6)), 0.51 * (10 ** (-6)), 2.07 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
    1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
    1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7))])
    water = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    return pressure, temperature, water, ozone, water
    # humidity = np.flip(humidity)

def add_line(pressure, temperature, humidity, ozone, water):

    return str(pressure) + "      " + str(temperature) + "      " + str(humidity)\
    + "      " + str(ozone) + "      " + str(water) + "\n"

#Create list of dates (to translate the time in seconds to the time in date format)
time = np.array(radiosonde['synTime'])
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

pressure, temperature, humidity, ozone, water = get_backrad_input(ind[1])

profile = open("inputfiles/backradmax.inp.txt", "w")

profile.write(format_num(temperature[-1]) +  "      15 \n")


for i in range(0, 15):
    profile.write(add_line(pressure[i], temperature[i], humidity[i], ozone[i], water[i]))

profile.close()
