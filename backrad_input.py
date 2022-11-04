import netCDF4 as nc
import numpy as np
import datetime


# Import data
profiles = nc.Dataset('raob_soundings13628.cdf')
time = np.array(profiles['synTime'])

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

x = 15 #remove outliers



def format_num(n):
    return '{:.7s}'.format('{:0.4f}'.format(n))



def get_backrad_input(day_index):
    temperature = np.array(profiles['tpMan'])[day_index,0:x] 
    pressure = np.array(profiles['prMan'])[day_index,0:x] #hPa
    
    pressure = np.flip(pressure) * 100
    temperature = np.flip(temperature)
    ozone = np.array([1.23 * (10 ** (-6)), 0.51 * (10 ** (-6)), 2.07 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
    1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), \
    1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7)) , 1.52 * (10 ** (-7)), 1.52 * (10 ** (-7))])
    water = np.zeros(15)

    return pressure, temperature, water, ozone, water
    # humidity = np.flip(humidity)

def add_line(pressure, temperature, humidity, ozone, water):

    return format_num(pressure) + "      " + format_num(temperature) + "      " + format_num(humidity)\
    + "      " + format_num(ozone) + "      " + format_num(water) + "\n"


"PROFILE FOR DAY_MIN"
pressure, temperature, humidity, ozone, water = get_backrad_input(ind_min)

profile = open("inputfiles/backrad.inp.001.txt", "w")

profile.write(format_num(temperature[-1]) +  "      13 \n")


for i in range(0,x,1):
    profile.write(add_line(pressure[i], temperature[i], humidity[i], ozone[i], water[i]))

profile.close()



"PROFILE FOR DAY_MAX"

pressure, temperature, humidity, ozone, water = get_backrad_input(ind_max)

profile = open("inputfiles/backrad.inp.002.txt", "w")

profile.write(format_num(temperature[-1]) +  "      13 \n")


for i in range(0,x,1):
    profile.write(add_line(pressure[i], temperature[i], humidity[i], ozone[i], water[i]))

profile.close()










