from datetime import datetime, timedelta
import math
import numpy as np


import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u

from misc import lat_lon_to_meters


def scale_solar_angle(sun_el_and_az, original_bounds_LLA, new_bounds = np.array([[-1.,1], [-1,1], [-1,1]])):
    original_bounds = original_bounds_LLA
    r = new_bounds[:, 1] - new_bounds[:, 0]
    delta_LLA = original_bounds[:, 1] - original_bounds[:, 0]
    S = np.array([[r[0] / delta_LLA[0], 0, 0, -r[0] * original_bounds[0, 0] / delta_LLA[0] + new_bounds[0, 0]],
                       [0, r[1] / delta_LLA[1], 0, -r[1] * original_bounds[1, 0] / delta_LLA[1] + new_bounds[1, 0]],
                       [0, 0, r[2] / delta_LLA[2], -r[2] * original_bounds[2, 0] / delta_LLA[2] + new_bounds[2, 0]],
                       [0, 0, 0, 1]])
    # S_inv = np.linalg.inv(S)

    # self.P = self.P @ self.S_inv
    # self.norm_P()

    area_center = np.mean(original_bounds, 1)
    direction_vec = area_center[0:2] + np.array([np.cos(sun_el_and_az[1] * np.pi / 180), np.sin(sun_el_and_az[1] * np.pi / 180)]) * .01
    dist_meters = lat_lon_to_meters(area_center[0], area_center[1], direction_vec[0], direction_vec[1])
    h_meters = np.tan(sun_el_and_az[0] * np.pi / 180) * dist_meters + area_center[2]
    sun_loc = np.array([direction_vec[0], direction_vec[1], h_meters, 1])
    area_center = np.array([area_center[0], area_center[1], area_center[2], 1])
    sun_el_and_az_vec = (S @ sun_loc)
    sun_el_and_az_vec = sun_el_and_az_vec[0:3] / sun_el_and_az_vec[3]
    area_center = S @ area_center
    area_center = area_center[0:3] / area_center[3]
    sun_el_and_az_vec = sun_el_and_az_vec - area_center
    sun_el_and_az_vec = sun_el_and_az_vec / np.sqrt(np.sum(sun_el_and_az_vec ** 2))
    return sun_el_and_az_vec

#time_array should be array of ints: year, month, day, hour, minute, second time zone is UTC Time
#returns elevation and azmuth in degrees
def get_solar_angle(bounds_LLA_center, time_array):
    loc = coord.EarthLocation(lon=bounds_LLA_center[1] * u.deg, lat=bounds_LLA_center[0] * u.deg)
    #        {'Year': 2014, 'Month': 9, 'Day': 8, 'Hour': 17, 'Minute': 10, 'Second': 24.797575}

    year, month, day, hour, minute, second = time_array
    Time_string = str(year) + "-" + str(month) + "-" + str(day) + " " + str(hour) + ":" + str(minute) + ":" + str(
        second)
    img_Time = Time(Time_string)  # '2014-9-8 17:10:24'
    altaz = coord.AltAz(location=loc, obstime=img_Time)
    sun = coord.get_sun(Time.now())
    store = sun.transform_to(altaz)
    return store.alt.degree, store.az.degree

def time_encode(month_day_frac_year, min_sec_hour_frac_day):
    return np.array([np.cos(month_day_frac_year * 2 * math.pi), np.sin(month_day_frac_year * 2 * math.pi),
           np.cos(min_sec_hour_frac_day * 2 * math.pi), np.sin(min_sec_hour_frac_day * 2 * math.pi)])


def time_encode_month_day_only(month_day_frac_year):
    return np.array([np.cos(month_day_frac_year * 2 * math.pi), np.sin(month_day_frac_year * 2 * math.pi)])

# format of UTC_time_str: YYYY-MM-DDThh:mm:ss.ddddddZ
class mg_time():
    def __init__(self, UTC_time_str):
        self.raw_str = UTC_time_str
        self._split_str()

    def _split_str(self):
        year, month, rest = self.raw_str.split("-")
        day, rest = rest.split("T")
        hour, minute, sec = rest.split(":")
        sec = sec[0:-2]

        self.month = int(month)
        self.day = int(day)
        self.year = int(year)
        self.hour = int(hour)
        self.minute = int(minute)
        self.sec = float(sec)

        self.Time = {"Year":self.year, "Month":self.month, "Day":self.day, "Hour":self.hour, "Minute":self.minute, "Second":self.sec}
        self.month_day_frac_year = datetime(self.year, self.month, self.day).timetuple().tm_yday / datetime(self.year, 12, 31).timetuple().tm_yday
        self.d_time = datetime(self.year, self.month, self.day, self.hour, self.minute, int(self.sec), int((self.sec-int(self.sec))*1000**2))

        self.min_sec_hour_frac_day = ((self.hour*60 + self.minute)*60 + self.sec) / (24*60*60)

        self.encode_frac = self.year, math.cos(self.month_day_frac_year * 2 * math.pi), math.sin(self.month_day_frac_year * 2 * math.pi), \
                           math.cos(self.min_sec_hour_frac_day * 2 * math.pi), math.sin(self.min_sec_hour_frac_day * 2 * math.pi),


        # time_diff_start = (datetime(self.year, self.month, self.day, self.hour, self.minute, int(self.sec)) - datetime(self.year, 1,1))
        # time_diff_end = (datetime(self.year+1, 1, 1) - datetime(self.year, 1,1))
        # print(time_diff_start)
        # print(time_diff_end)


    #Returns Year, Month, Day, Hour, Minute, Second
    def get_time(self):
        return self.year, self.month, self.day, self.hour, self.minute, self.sec

    #Returns year, decimal representing amount of year complete, decimal representing amount of day complete
    def get_time_frac(self):
        return self.year, self.month_day_frac_year, self.min_sec_hour_frac_day

    def get_time_encode(self):
        return self.encode_frac