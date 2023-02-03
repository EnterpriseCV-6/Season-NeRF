import numpy as np
from matplotlib import pyplot as plt
import gdal
# from lego_rvl.rpc.rpc import RPC_py as RPC
from .mg_time import mg_time
import rpcm

class sat_img:
    def __init__(self, img_loc, rpc_loc, file_name, has_rpc = False, end ='tif', load_IMD = False, IMD_loc = None, IMD_loc_full = None):

        self.sun_El = None
        self.sun_Az = None
        self.off_Nadir = None
        self.view_azmuth = None
        self.time_obj = None
        found_Az = ""
        found_El = ""
        found_offNadirViewAngle = ""
        found_time = ""
        found_viewAzmuth = ""
        if load_IMD:
            if IMD_loc is None:
                fin = open(rpc_loc + "/" + file_name + ".IMD", "r")
            else:
                if IMD_loc_full is None:
                    fin = open(IMD_loc + "/" + file_name + ".IMD", "r")
                else:
                    fin = open(IMD_loc_full, "r")
            the_lines = fin.readlines()
            for a_line in the_lines:
                a_line = a_line.strip().split(" ")
                if a_line[0] == "meanSunAz":
                    found_Az = a_line[-1]
                if a_line[0] == "meanSunEl":
                    found_El = a_line[-1]
                if a_line[0] == "meanOffNadirViewAngle":
                    found_offNadirViewAngle = a_line[-1]
                if a_line[0] == "firstLineTime":
                    found_time = a_line[-1]
                if a_line[0] == "meanSatAz":
                    found_viewAzmuth = a_line[-1]
                if len(found_El) > 0 and len(found_Az) > 0 and len(found_offNadirViewAngle) > 0 and len(found_time) > 0 and len(found_viewAzmuth) > 0:
                    break

            fin.close()
            if len(found_El) > 0 and len(found_Az) > 0:
                found_El = float(found_El[0:-1])
                found_Az = float(found_Az[0:-1])
                if len(found_offNadirViewAngle) > 0:
                    found_OffNadir = float(found_offNadirViewAngle[0:-1])
                    self.off_Nadir = found_OffNadir
                if len(found_viewAzmuth) > 0:
                    found_viewAzmuth = float(found_viewAzmuth[0:-1])
                    self.view_azmuth = found_viewAzmuth
                self.sun_El = found_El
                self.sun_Az = found_Az #substract 90 as Azmuth 0 is North but unit conversion zero is East
            else:
                print("Error: Sun angle not in IMD file!")
                print(file_name)
                exit()
            if len(found_time) > 0:
                self.time_obj = mg_time(found_time)
            else:
                print("Error: firstLineTime not in IMD file!")
                print(file_name)
                exit()

        if end == 'tif':
            self.img = gdal.Open(img_loc + "/" + file_name + ".tif").ReadAsArray()
            self.img = np.array(self.img, dtype=np.float32)
            if len(self.img.shape) != 3:
                self.img.resize([self.img.shape[0], self.img.shape[1], 1])
            else:
                self.img = np.moveaxis(self.img, 0, -1)
        elif end == 'png':
            self.img = plt.imread(img_loc + "/" + file_name + ".png")
            self.img = np.array(self.img, dtype=np.float32)
        self.img_name = file_name
        self.vis_img = np.ones(self.img.shape, dtype=bool)
        self.vis_img_h = np.zeros(self.img.shape)
        if has_rpc == False:
            self.rpc = None
            self.has_rpc = False
            self.rpcm_like = False
        else:
            self.rpc = RPC.from_file(rpc_loc + "/" + file_name + ".RPB")
            self.has_rpc = True
            self.rpcm_like = True

        self.shape = self.img.shape
        self.P_ready = False

    def load_rpc_from_tif(self, img_loc, use_rpcm = True):
        # self.rpc = my_RPC.RPC_py.from_file(img_loc)
        if use_rpcm:
            self.rpc = rpcm.rpc_from_geotiff(img_loc)
            self.rpcm_like = True
        else:
            self.rpc = RPC.from_file(img_loc)
            self.rpcm_like = False
        self.has_rpc = True

    def get_vis_at_h(self, h):
        # print(h, h > self.vis_img_h)
        vis = self.vis_img * (h >= self.vis_img_h)
        return vis

    def apply_rpc(self, lat, lon, h):
        if self.has_rpc == False:
            print("Error, no rpc in object!")
            exit()
            return -1, -1
        else:
            if self.rpcm_like:
                X, Y = self.rpc.projection(lon, lat, h)
                return Y, X
            else:
                return self.rpc.rpc(lat, lon, h)

    def invert_rpc(self, row, col, h=0):
        if self.has_rpc == False:
            return -1, -1, -1
        else:
            if self.rpcm_like:
                lon, lat = self.rpc.localization(col, row, h)
                return lat, lon, h
            else:
                return self.rpc.inv_rpc(row, col, h)

    def apply_approx_RPC(self, lat, lon, h):
        if self.P_ready == True:
            # lat_n = (lat - self.lat_n[0]) / self.lat_n[1] * 1000
            # lon_n = (lon - self.lon_n[0]) / self.lon_n[1] * 1000
            # h_n = (h - self.h_n[0]) / self.h_n[1] * 1000
            X = np.stack([lat, lon, h, np.ones(lat.shape[0])], 1).T
            xy = self.coes @ X
            xy = xy[0:2] / xy[2]

            return xy[0], xy[1]
        else:
            return -1, -1

    def invert_approximate_rpc(self, row, col, h=0):
        if self.P_ready == False:
            return -1, -1, -1
        else:

            # h = (h - self.h_n[0]) / self.h_n[1] * 1000
            # print(self.coes)
            # print(self.coes.shape)

            # P12mP32x = self.coes[0,1] - self.coes[2,1] * row
            P23ZpP24mP33ZymP34y = self.coes[1, 2] * h + self.coes[1, 3] - self.coes[2, 2] * h * col - self.coes[
                2, 3] * col
            # P22mP32y = self.coes[1,1] - self.coes[2,1] * col
            P13ZpP14mP33ZxmP34x = self.coes[0, 2] * h + self.coes[0, 3] - self.coes[2, 2] * h * row - self.coes[
                2, 3] * row
            P11mP31x = self.coes[0, 0] - self.coes[2, 0] * row
            P22mP32y = self.coes[1, 1] - self.coes[2, 1] * col
            P12mP32x = self.coes[0, 1] - self.coes[2, 1] * row
            P21mP31y = self.coes[1, 0] - self.coes[2, 0] * col

            x = (P12mP32x * P23ZpP24mP33ZymP34y - P22mP32y * P13ZpP14mP33ZxmP34x) / (
                        P11mP31x * P22mP32y - P12mP32x * P21mP31y)
            y = (-P11mP31x * P23ZpP24mP33ZymP34y + P21mP31y * P13ZpP14mP33ZxmP34x) / (
                        P11mP31x * P22mP32y - P12mP32x * P21mP31y)

            # x = x / 1000 * self.lat_n[1] + self.lat_n[0]
            # y = y / 1000 * self.lon_n[1] + self.lon_n[0]
            # h = h / 1000 * self.h_n[1] + self.h_n[0]

            return x, y, h

    def compute_Approx_RPC(self, H_min, H_max, num_train_points=10, num_test_points=50, show_details=True, method='Chebyshev'):
        self.P_ready = True
        # H_range = np.arange(self.bounds[2][0], self.bounds[2][1], h_step)

        self.sub_section = np.array([[0, self.img.shape[0]],
                                     [0, self.img.shape[1]]])

        # exit()
        H_range = [H_min, H_max]
        if method == 'Chebyshev':
            n_pts_per_axis = num_train_points  # int(np.round(num_train_points ** (1 / 3)))-1
            c_pts = np.cos((2 * np.arange(0, n_pts_per_axis + 1) + 1) / (2 * (n_pts_per_axis + 1)) * np.pi)
            # print(c_pts)

            x_pts = (self.sub_section[0, 1] - self.sub_section[0, 0]) / 2 * (c_pts + 1) + self.sub_section[0, 0]
            y_pts = (self.sub_section[1, 1] - self.sub_section[1, 0]) / 2 * (c_pts + 1) + self.sub_section[1, 0]
            z_pts = (H_range[-1] - H_range[0]) / 2 * (c_pts + 1) + H_range[0]

            x, y, z = np.meshgrid(x_pts, y_pts, z_pts)
            x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
            n_pts = x.shape[0]

        elif method == 'Uniform':
            step_size = num_train_points
            x_step = (self.sub_section[0, 1] - self.sub_section[0, 0]) / step_size
            y_step = (self.sub_section[1, 1] - self.sub_section[1, 0]) / step_size
            h_step = (H_range[-1] - H_range[0]) / step_size

            x, y, z = np.meshgrid(np.arange(self.sub_section[0, 0], self.sub_section[0, 1] + x_step, x_step),
                                  np.arange(self.sub_section[1, 0], self.sub_section[1, 1] + y_step, y_step),
                                  np.arange(H_range[0], H_range[-1] + h_step, h_step))
            x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
            n_pts = x.shape[0]

        elif method == 'Random':
            n_pts = (num_train_points + 1) ** 3

            x = np.random.rand(n_pts) * (self.sub_section[0, 1] - self.sub_section[0, 0]) + self.sub_section[0, 0]
            y = np.random.rand(n_pts) * (self.sub_section[1, 1] - self.sub_section[1, 0]) + self.sub_section[1, 0]
            z = np.random.rand(n_pts) * (H_range[-1] - H_range[0]) + H_range[0]

        else:
            x, y, z = [0], [0], [0]
            n_pts = 1
            print("Error: method", method, "is unknown!")
            print("Valid Methods:")
            print("Chebyshev   -- Use Chebyshev points")
            print("Uniform     -- Use points uniformly distributed")
            print("Random      -- Randomly select points via uniform distribution")
            exit()

        lat, lon, h = self.invert_rpc(x[0:n_pts], y[0:n_pts], z[0:n_pts])
        lat_c, lon_c, h_c = lat, lon, h
        # print(lat, lon, h)
        # exit()

        self.lat_n = (np.min(lat), np.max(lat - np.min(lat)))
        self.lon_n = (np.min(lon), np.max(lon - np.min(lon)))
        self.h_n = (np.min(h), np.max(h - np.min(h)))
        lat = (lat - self.lat_n[0]) / self.lat_n[1] * 1000
        lon = (lon - self.lon_n[0]) / self.lon_n[1] * 1000
        h = (h - self.h_n[0]) / self.h_n[1] * 1000

        X = np.zeros([2 * n_pts, 11])
        Y = np.zeros([2 * n_pts, 1])
        # print(x, y, z)
        # print(lat, lon, h)

        for i in range(n_pts):
            X[2 * i, :] = [lat[i], lon[i], h[i], 1, 0, 0, 0, 0, -x[i] * lat[i], -x[i] * lon[i], -x[i] * h[i]]
            Y[2 * i, 0] = x[i]
            X[2 * i + 1, :] = [0, 0, 0, 0, lat[i], lon[i], h[i], 1, -y[i] * lat[i], -y[i] * lon[i], -y[i] * h[i]]
            Y[2 * i + 1, 0] = y[i]

        coes = np.linalg.inv(X.T @ X) @ X.T @ Y
        self.coes = np.ones([3, 4])
        self.coes[0, :] = coes[0:4, 0]
        self.coes[1, :] = coes[4:8, 0]
        self.coes[2, 0:3] = coes[8:, 0]

        A = np.array([[1000 / self.lat_n[1], 0, 0, -1000 * self.lat_n[0] / self.lat_n[1]],
                      [0, 1000 / self.lon_n[1], 0, -1000 * self.lon_n[0] / self.lon_n[1]],
                      [0, 0, 1000 / self.h_n[1], -1000 * self.h_n[0] / self.h_n[1]],
                      [0, 0, 0, 1]])

        self.coes = self.coes @ A


        # if np.max(np.abs(self.coes[2,0:-1])) <= 10e-7 and np.abs(1-self.coes[2,-1]) < 10e-7:
        #     self.Jac_F = 1/(self.coes[0,0]*self.coes[1,1] - self.coes[0,1] * self.coes[1,0])
        # else:
        #     print("Assumption that last row of P is [0,0,0,1] does not Hold!")
        #     exit()


        # out_func, Jac_F = get_Jac_with_P(np.round(self.coes, 5))
        # print(self.coes)
        # print(np.round(self.coes, 5))
        # print(out_func)
        # self.Jac_F = Jac_F

        if show_details:
            x_g, y_g = self.apply_approx_RPC(lat_c, lon_c, h_c)
            lat_g, lon_g, h_g = self.invert_approximate_rpc(x, y, z)

            print("Training Error")
            print("Cat.  Min Max Mean")
            l1_error = np.abs(x_g - x[0:n_pts])
            print("x dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l1_error = np.abs(y_g - y[0:n_pts])
            print("y dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l2_error = np.sqrt((x_g - x[0:n_pts]) ** 2 + (y_g - y[0:n_pts]) ** 2)
            print("Dist", np.min(l2_error), np.max(l2_error), np.mean(l2_error))
            print()
            l1_error = np.abs(lat_c - lat_g)
            print("lat dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l1_error = np.abs(lon_c - lon_g)
            print("lon dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l1_error = np.abs(z - h_g)
            print("h dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            dist = mg_unit_converter.lat_lon_to_meters_array(lat_c, lon_c, lat_g, lon_g)
            # print(dist)
            print("Dist (m)", np.min(dist), np.max(dist), np.mean(dist))
            # print("lat dist m", mg_unit_converter.lat_lon_to_meters(lat[0], lon[0], lat[0] + np.min(l1_error), lon[0]),
            #       mg_unit_converter.lat_lon_to_meters(lat[0], lon[0], lat[0] + np.max(l1_error), lon[0]),
            #       mg_unit_converter.lat_lon_to_meters(lat[0], lon[0], lat[0] + np.mean(l1_error), lon[0]))
            # l1_error = np.abs(lon_c - lon_g)
            # print("lon dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            # print("lon dist m", mg_unit_converter.lat_lon_to_meters(lat[0], lon[0], lat[0], lon[0] + np.min(l1_error)),
            #       mg_unit_converter.lat_lon_to_meters(lat[0], lon[0], lat[0], lon[0] + np.max(l1_error)),
            #       mg_unit_converter.lat_lon_to_meters(lat[0], lon[0], lat[0], lon[0] + np.mean(l1_error)))
            # exit()

            step_size = num_test_points
            x_step = (self.sub_section[0, 1] - self.sub_section[0, 0]) / step_size
            y_step = (self.sub_section[1, 1] - self.sub_section[1, 0]) / step_size
            h_step = (H_range[-1] - H_range[0]) / step_size
            x, y, z = np.meshgrid(np.arange(self.sub_section[0, 0], self.sub_section[0, 1], x_step),
                                  np.arange(self.sub_section[1, 0], self.sub_section[1, 1], y_step),
                                  np.arange(H_range[0], H_range[-1], h_step))
            x, y, z = np.ravel(x) + x_step / 2, np.ravel(y) + y_step / 2, np.ravel(z) + h_step / 2

            lat, lon, h = self.invert_rpc(x, y, z)
            lat_c, lon_c, h_c = lat, lon, h
            x_g, y_g = self.apply_approx_RPC(lat_c, lon_c, h_c)
            lat_g, lon_g, h_g = self.invert_approximate_rpc(x, y, z)

            print("Testing Error")
            print("Cat.  Min Max Mean")
            l1_error = np.abs(x_g - x)
            print("x dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l1_error = np.abs(y_g - y)
            print("y dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l2_error = np.sqrt((x_g - x) ** 2 + (y_g - y) ** 2)
            print("Dist", np.min(l2_error), np.max(l2_error), np.mean(l2_error))
            print()
            l1_error = np.abs(lat_c - lat_g)
            print("lat dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l1_error = np.abs(lon_c - lon_g)
            print("lon dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            l1_error = np.abs(h - h_g)
            print("h dist", np.min(l1_error), np.max(l1_error), np.mean(l1_error))
            dist = mg_unit_converter.lat_lon_to_meters_array(lat_c, lon_c, lat_g, lon_g)
            # print(dist)
            print("Dist (m)", np.min(dist), np.max(dist), np.mean(dist))
            # print("|Jacobian| of Projection:", self.Jac_F)

            return x, y, x_g, y_g

        return -1, -1, -1, -1

def find_bounds_sat_img(sat_img_list, h_range):
    # lats, lons = np.meshgrid(np.linspace(-180,180,360), np.linspace(-180,180,360))
    # lats, lons = np.ravel(lats), np.ravel(lons)
    lat_0, lon_0, lat_1, lon_1, d_lat, d_lon = -1,-1,-1,-1,-1,-1
    for i in range(len(sat_img_list)):
        c1s, c2s, hs = [0,sat_img_list[i].img.shape[0],0,sat_img_list[i].img.shape[0],0,sat_img_list[i].img.shape[0],0,sat_img_list[i].img.shape[0]], [0,0, sat_img_list[i].img.shape[1], sat_img_list[i].img.shape[1],0,0, sat_img_list[i].img.shape[1], sat_img_list[i].img.shape[1]], [h_range[0], h_range[0], h_range[0], h_range[0], h_range[1], h_range[1], h_range[1], h_range[1]]
        # print(c1s, c2s, hs)
        lat, lon, h = sat_img_list[i].invert_rpc(c1s, c2s, hs)
        # print(lat, lon, h)
        if i == 0:
            lat_0, lat_1, lon_0, lon_1 = np.min(lat),np.max(lat),np.min(lon),np.max(lon)
            # lat_0, lon_0 = np.min(lat), np.min(lon)
            # d_lat, d_lon = np.max(lat) - lat_0, np.max(lon) - lon_0
        else:
            lat_0, lat_1, lon_0, lon_1 = max(lat_0, np.min(lat)), min(lat_1, np.max(lat)), max(lon_0, np.min(lon)), min(lon_1, np.max(lon))
    # print(lat_0, lat_1, lon_0, lon_1)
    # lat_1, lon_1 = lat_0 + d_lat, lon_0 + d_lon
    i = 0
    n = len(sat_img_list)
    xt = 1.
    c = 0
    while i <  n:
        c1s, c2s, hs = [lat_0, lat_1, lat_0, lat_1, lat_0, lat_1, lat_0, lat_1], [lon_0, lon_0, lon_1, lon_1, lon_0, lon_0, lon_1, lon_1], [h_range[0], h_range[0], h_range[0], h_range[0], h_range[1], h_range[1], h_range[1], h_range[1]]
        X, Y = sat_img_list[i].apply_rpc(np.array(c1s), np.array(c2s), np.array(hs))
        if np.min(X) < 0 or np.max(X) > sat_img_list[i].img.shape[0] or np.min(Y) < 0 or np.max(Y) > sat_img_list[i].img.shape[1]:
            d_lat = (lat_1 - lat_0)
            d_lon = (lon_1 - lon_0)
            lat_0 = (lat_0 + d_lat * .05)
            lon_0 = (lon_0 + d_lon * .05)
            lat_1 = (lat_1 - d_lat * .05)
            lon_1 = (lon_1 - d_lon * .05)
            c += 1
        else:
            i += 1
            c = 0
        if c > 100000:
            print("Error, unable to find bounds!")
            exit()

    bounds = np.array([[lat_0, lat_1],
                      [lon_0, lon_1],
                      [h_range[0], h_range[1]]]).T
    return bounds