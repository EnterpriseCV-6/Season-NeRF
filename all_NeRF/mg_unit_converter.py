import math
import numpy as np


def world_angle_2_local_vec(world_el, world_az, world_center, World2Local_H):
    ans = LLA_get_vec(world_center, world_az, world_el)
    temp = (World2Local_H @ np.array([[ans[0], ans[1], ans[2], 1]]).T)[0:-1]
    sun_el_and_az_vec = ((temp / np.sqrt(np.sum(temp ** 2)))).T[0]
    return sun_el_and_az_vec

def az_el_2_vec(az, el):
    pass

def vec_2_az_el(vec):
    pass

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000 # meters

#Given lat, lon, and shift in lat direction (North) and lon direction (East), return lat and lon shifted by that many meters
def lat_lon_shift(lat, lon, d_lat_m, d_lon_m):
    R = 6378.137 # Radius of earth in KM
    dLat = d_lat_m / (1000. * R)
    dLon = d_lon_m / (1000. * R * np.cos(np.deg2rad(lat)))

    return lat + np.rad2deg(dLat), lon + np.rad2deg(dLon)

#Given two LLA cordinates, find the azmuth and elevation angle between them
#Warning: Inaccuarte for far coordinates
def LLAs_get_angle_vec(LLA1, LLA2):
    X = lat_lon_to_meters(LLA1[0], LLA1[1], LLA1[0], LLA2[1]) * np.sign(LLA2[1] - LLA1[1])
    Y = lat_lon_to_meters(LLA1[0], LLA1[1], LLA2[0], LLA1[1]) * np.sign(LLA2[0] - LLA1[0])
    H = lat_lon_to_meters(LLA1[0], LLA1[1], LLA2[0], LLA2[1])
    A = LLA2[2] - LLA1[2]

    theta = np.rad2deg(np.arctan2(X,Y))
    theta2 = np.rad2deg(np.arccos(X/H)) if H != 0 else np.NaN
    theta3 = np.rad2deg(np.arcsin(Y/H)) if H != 0 else np.NaN
    # print(theta1, theta2, theta3)

    H_error = H - np.sqrt(X**2 + Y**2)
    # print(X, Y, H)
    # print(H_error)
    rho = np.rad2deg(np.arctan2(A, H))

    other_args = {"dx":X, "dy":Y, "H":H, "Error":(H_error, theta2, theta3)}

    return theta, rho, other_args

#Given an LLA coordinate, an azmuth, and an elevation angle return a vector pointing away from the LLA coordinate in the desired direction
def LLA_get_vec(LLA_Center, theta_deg, rho_deg):
    Y = np.cos(np.deg2rad(theta_deg))
    X = np.sin(np.deg2rad(theta_deg))
    Z = np.tan(np.deg2rad(rho_deg)) * np.sqrt(X**2 + Y**2)
    norm = np.sqrt(X**2 + Y**2 + Z**2)/1000 #divide by 1000 useful for numical stability
    X, Y, Z = X/norm, Y/norm, Z/norm
    # print(X, Y, Z)
    new_lat, new_lon = lat_lon_shift(LLA_Center[0], LLA_Center[1], Y, X)
    ans = np.array([new_lat, new_lon, LLA_Center[2] + Z])
    return ans


def lat_lon_to_meters_array(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + \
        np.cos(lat1 * math.pi / 180) * np.cos(lat2 * math.pi / 180) * \
        np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d * 1000 # meters

def LLA_Cube_to_meter_Cube(LLA_Start, LLA_End, Cube_Size):
    num_x = int(math.ceil(abs(lat_lon_to_meters(LLA_Start[0], LLA_Start[1], LLA_End[0], LLA_Start[1]) / Cube_Size[1])))
    num_y = int(math.ceil(abs(lat_lon_to_meters(LLA_Start[0], LLA_Start[1], LLA_Start[0], LLA_End[1]) / Cube_Size[0])))
    num_z = int(math.ceil(abs((LLA_End[2] - LLA_Start[2]) / Cube_Size[2])))

    return num_x, num_y, num_z


# Convert scalar WGS84 to UTM
# Code From https://github.com/pubgeo/dfc2019/blob/c59af36ad984da4aa36edea5c8912c2096db5394/track3/mvs/utm.py#L190
def wgs84_to_utm(latitude, longitude, force_zone_number=None):
    """This function convert Latitude and Longitude to UTM coordinate
        Parameters
        ----------
        latitude: float
            Latitude between 80 deg S and 84 deg N, e.g. (-80.0 to 84.0)
        longitude: float
            Longitude between 180 deg W and 180 deg E, e.g. (-180.0 to 180.0).
        force_zone number: int
            Zone Number is represented with global map numbers of an UTM Zone
            Numbers Map. You may force conversion including one UTM Zone Number.
            More information see utmzones [1]_
       .. _[1]: http://www.jaworski.ca/utmzones.htm
    """

    # Define constants for conversions
    K0 = 0.9996

    E = 0.00669438
    E2 = E * E
    E3 = E2 * E
    E_P2 = E / (1.0 - E)

    SQRT_E = math.sqrt(1 - E)
    _E = (1 - SQRT_E) / (1 + SQRT_E)
    _E2 = _E * _E
    _E3 = _E2 * _E
    _E4 = _E3 * _E
    _E5 = _E4 * _E

    M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
    M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
    M3 = (15 * E2 / 256 + 45 * E3 / 1024)
    M4 = (35 * E3 / 3072)

    P2 = (3. / 2 * _E - 27. / 32 * _E3 + 269. / 512 * _E5)
    P3 = (21. / 16 * _E2 - 55. / 32 * _E4)
    P4 = (151. / 96 * _E3 - 417. / 128 * _E5)
    P5 = (1097. / 512 * _E4)

    R = 6378137

    if not -80.0 <= latitude <= 84.0:
        raise OutOfRangeError('latitude out of range (must be between 80 deg S and 84 deg N)')
    if not -180.0 <= longitude <= 180.0:
        raise OutOfRangeError('longitude out of range (must be between 180 deg W and 180 deg E)')

    lat_rad = math.radians(latitude)
    lat_sin = math.sin(lat_rad)
    lat_cos = math.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    if force_zone_number is None:
        zone_number = latlon_to_zone_number(latitude, longitude)
    else:
        zone_number = force_zone_number

    zone_letter = latitude_to_zone_letter(latitude)

    lon_rad = math.radians(longitude)
    central_lon = zone_number_to_central_longitude(zone_number)
    central_lon_rad = math.radians(central_lon)

    n = R / math.sqrt(1 - E * lat_sin**2)
    c = E_P2 * lat_cos**2

    a = lat_cos * (lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = R * (M1 * lat_rad -
             M2 * math.sin(2 * lat_rad) +
             M3 * math.sin(4 * lat_rad) -
             M4 * math.sin(6 * lat_rad))

    easting = K0 * n * (a +
                        a3 / 6 * (1 - lat_tan2 + c) +
                        a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000

    northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                        a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                        a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))

    if latitude < 0:
        northing += 10000000

    return easting, northing, zone_number, zone_letter

# Define out of range exception
class OutOfRangeError(ValueError):
    pass


def latitude_to_zone_letter(latitude):
    ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"
    if -80 <= latitude <= 84:
        return ZONE_LETTERS[int(latitude + 80) >> 3]
    else:
        return None

def latlon_to_zone_number(latitude, longitude):
    if 56 <= latitude < 64 and 3 <= longitude < 12:
        return 32

    if 72 <= latitude <= 84 and longitude >= 0:
        if longitude <= 9:
            return 31
        elif longitude <= 21:
            return 33
        elif longitude <= 33:
            return 35
        elif longitude <= 42:
            return 37

    return int((longitude + 180) / 6) + 1

def zone_number_to_central_longitude(zone_number):
    return (zone_number - 1) * 6 - 180 + 3

#End of Code from https://github.com/pubgeo/dfc2019/blob/c59af36ad984da4aa36edea5c8912c2096db5394/track3/mvs/utm.py#L190