import numpy as np
import math
from misc import lat_lon_to_meters
from scipy import linalg
from all_NeRF.mg_unit_converter import LLA_get_vec, world_angle_2_local_vec

class P_img():
    def __init__(self, sat_img):
        self.img_name = sat_img.img_name
        self.img = sat_img.img * 1. / 255
        self.scaled = False

        self._cam_center = None
        self.has_sun_angle = True
        self.sun_el_and_az_vec = self._elevation_azmuth_to_vec(sat_img.sun_El, sat_img.sun_Az)
        self.sun_el_and_az = [sat_img.sun_El, sat_img.sun_Az]
        self.off_Nadir_from_IMD = sat_img.off_Nadir
        self.Azmuth_from_IMD = sat_img.view_azmuth
        self.P = sat_img.rpc
        self.S = np.eye(4)
        self.S_inv = np.eye(4)
        self.time_obj = sat_img.time_obj

    def get_mask(self):
        XY = np.stack(np.meshgrid(np.linspace(-1, 1, self.img.shape[0]), np.linspace(-1,1, self.img.shape[1]), indexing="ij"), -1).reshape([-1,2])
        xy = np.stack(self.apply_P(XY[:,0], XY[:,1], -1), -1)
        xy2 = np.stack(self.apply_P(XY[:, 0], XY[:, 1], -1), 1)
        xy = xy.astype(int)
        xy2 = xy2.astype(int)
        good = (xy[:,0] >= 0) * (xy[:,0] < self.img.shape[0]) * (xy[:,1] >= 0) * (xy[:,1] < self.img.shape[1])
        good = good * (xy2[:, 0] >= 0) * (xy2[:, 0] < self.img.shape[0]) * (xy2[:, 1] >= 0) * (xy2[:, 1] < self.img.shape[1])

        mask = np.zeros([self.img.shape[0], self.img.shape[1]], dtype=bool)
        mask[xy[good, 0], xy[good, 1]] = True
        return mask


    def get_meta_data(self):
        name = self.img_name
        sat_el_and_az = (90 - self.off_Nadir_from_IMD, self.Azmuth_from_IMD)
        sun_el_and_az = self.sun_el_and_az
        year_frac = self.get_year_frac()
        return name, sat_el_and_az, sun_el_and_az, year_frac

    def get_year_frac(self):
        return self.time_obj.get_time_frac()[1]

    def get_world_center(self):
        area_center = self.S_inv @ np.array([[0, 0, 0, 1.]]).T
        area_center = ((area_center[0:-1] / area_center[-1::]).T)[0]
        return area_center


    def apply_P(self, lat, lon, h):
        X, Y = self.P.rpc(lat * self.S_inv[0,0] + self.S_inv[0,3], lon  * self.S_inv[1,1] + self.S_inv[1,3], h * self.S_inv[2,2] + self.S_inv[2,3])
        return X, Y

    def invert_P(self, row, col, h):
        lat, lon, h = self.P.inv_rpc(row, col, h * self.S_inv[2,2] + self.S_inv[2,3])
        return lat * self.S[0,0] + self.S[0,3], lon  * self.S[1,1] + self.S[1,3], h * self.S[2,2] + self.S[2,3]

    def _elevation_azmuth_to_vec(self, solar_elevation, solar_azmuth):
        sun_el_and_az_vec = np.array([np.cos(solar_azmuth / 180 * np.pi), np.sin(solar_azmuth / 180 * np.pi),
                                      np.tan(solar_elevation / 180 * np.pi)])
        sun_el_and_az_vec = sun_el_and_az_vec / np.sqrt(np.sum(sun_el_and_az_vec ** 2))

        return sun_el_and_az_vec

    def scale_P(self, original_bounds, new_bounds):
        self.scaled = True
        r = new_bounds[:, 1] - new_bounds[:, 0]
        delta_LLA = original_bounds[:, 1] - original_bounds[:, 0]
        self.S = np.array([[r[0] / delta_LLA[0], 0, 0, -r[0] * original_bounds[0, 0] / delta_LLA[0] + new_bounds[0, 0]],
                           [0, r[1] / delta_LLA[1], 0, -r[1] * original_bounds[1, 0] / delta_LLA[1] + new_bounds[1, 0]],
                           [0, 0, r[2] / delta_LLA[2], -r[2] * original_bounds[2, 0] / delta_LLA[2] + new_bounds[2, 0]],
                           [0, 0, 0, 1]])
        self.S_inv = np.linalg.inv(self.S)

        # self.P = self.P @ self.S_inv
        # self.norm_P()

        if self.has_sun_angle:
            area_center = np.mean(original_bounds, 1)
            ans = LLA_get_vec(area_center, self.sun_el_and_az[1], self.sun_el_and_az[0])
            temp = (self.S @ np.array([[ans[0], ans[1], ans[2], 1]]).T)[0:-1]
            self.sun_el_and_az_vec = ((temp / np.sqrt(np.sum(temp ** 2)))).T[0]
            # direction_vec = area_center[0:2] + np.array(
            #     [np.cos(self.sun_el_and_az[1] * np.pi / 180), np.sin(self.sun_el_and_az[1] * np.pi / 180)]) * .01
            # dist_meters = lat_lon_to_meters(area_center[0], area_center[1], direction_vec[0], direction_vec[1])
            # h_meters = np.tan(self.sun_el_and_az[0] * np.pi / 180) * dist_meters + area_center[2]
            # sun_loc = np.array([direction_vec[0], direction_vec[1], h_meters, 1])
            # area_center = np.array([area_center[0], area_center[1], area_center[2], 1])
            # self.sun_el_and_az_vec = (self.S @ sun_loc)
            # self.sun_el_and_az_vec = self.sun_el_and_az_vec[0:3] / self.sun_el_and_az_vec[3]
            # area_center = self.S @ area_center
            # area_center = area_center[0:3] / area_center[3]
            # self.sun_el_and_az_vec = self.sun_el_and_az_vec - area_center
            # self.sun_el_and_az_vec = self.sun_el_and_az_vec / np.sqrt(np.sum(self.sun_el_and_az_vec ** 2))


        self._cam_center = None
        # self.P_pinv = self.P.T @ np.linalg.inv(self.P @ self.P.T)

    def world_angle_2_local_vec(self, world_el, world_az):
        # area_center = self.get_world_center()
        # ans = LLA_get_vec(area_center, world_az, world_el)
        # temp = (self.S @ np.array([[ans[0], ans[1], ans[2], 1]]).T)[0:-1]
        # sun_el_and_az_vec = ((temp / np.sqrt(np.sum(temp ** 2)))).T[0]
        sun_el_and_az_vec = world_angle_2_local_vec(world_el, world_az, self.get_world_center(), self.S)
        return sun_el_and_az_vec

class P_img_Pinhole(P_img):
    def __init__(self, sat_img, num_train_points, min_H, max_H):
        super(P_img_Pinhole, self).__init__(sat_img)

        self.P = compute_Approx_RPC(sat_img, min_H, max_H, num_train_points=num_train_points)
        self.norm_P()

    def get_camera_center(self):
        if self._cam_center is None:
            cam_center_LLA = linalg.null_space(self.P)
            cam_center_LLA = np.squeeze(cam_center_LLA)
            cam_center_LLA = cam_center_LLA[0:-1] / cam_center_LLA[-1]
            self._cam_center = cam_center_LLA

        return self._cam_center

    def norm_P(self):
        self.P = self.P / self.P[-1, -1]
        self.P_pinv = self.P.T @ np.linalg.inv(self.P @ self.P.T)
        self._cam_center = None

    def invert_P(self, row, col, h=0):
        P23ZpP24mP33ZymP34y = self.P[1, 2] * h + self.P[1, 3] - self.P[2, 2] * h * col - self.P[2, 3] * col
        # P22mP32y = self.coes[1,1] - self.coes[2,1] * col
        P13ZpP14mP33ZxmP34x = self.P[0, 2] * h + self.P[0, 3] - self.P[2, 2] * h * row - self.P[2, 3] * row
        P11mP31x = self.P[0, 0] - self.P[2, 0] * row
        P22mP32y = self.P[1, 1] - self.P[2, 1] * col
        P12mP32x = self.P[0, 1] - self.P[2, 1] * row
        P21mP31y = self.P[1, 0] - self.P[2, 0] * col

        x = (P12mP32x * P23ZpP24mP33ZymP34y - P22mP32y * P13ZpP14mP33ZxmP34x) / (
                P11mP31x * P22mP32y - P12mP32x * P21mP31y)
        y = (-P11mP31x * P23ZpP24mP33ZymP34y + P21mP31y * P13ZpP14mP33ZxmP34x) / (
                P11mP31x * P22mP32y - P12mP32x * P21mP31y)

        return x, y, h

    def apply_P(self, X, Y, Z):
        P_11, P_12, P_13, P_14, P_21, P_22, P_23, P_24, P_31, P_32, P_33, P_34 = self.P[0, 0], \
                                                                                 self.P[0, 1], \
                                                                                 self.P[0, 2], \
                                                                                 self.P[0, 3], \
                                                                                 self.P[1, 0], \
                                                                                 self.P[1, 1], \
                                                                                 self.P[1, 2], \
                                                                                 self.P[1, 3], \
                                                                                 self.P[2, 0], \
                                                                                 self.P[2, 1], \
                                                                                 self.P[2, 2], \
                                                                                 self.P[2, 3]
        x = P_11 * X + P_12 * Y + P_13 * Z + P_14
        y = P_21 * X + P_22 * Y + P_23 * Z + P_24
        c = P_31 * X + P_32 * Y + P_33 * Z + P_34

        return x / c, y / c

    def scale_P(self, original_bounds, new_bounds):
        self.scaled = True
        r = new_bounds[:, 1] - new_bounds[:, 0]
        delta_LLA = original_bounds[:, 1] - original_bounds[:, 0]
        self.S = np.array([[r[0] / delta_LLA[0], 0, 0, -r[0] * original_bounds[0, 0] / delta_LLA[0] + new_bounds[0, 0]],
                           [0, r[1] / delta_LLA[1], 0, -r[1] * original_bounds[1, 0] / delta_LLA[1] + new_bounds[1, 0]],
                           [0, 0, r[2] / delta_LLA[2], -r[2] * original_bounds[2, 0] / delta_LLA[2] + new_bounds[2, 0]],
                           [0, 0, 0, 1]])
        self.S_inv = np.linalg.inv(self.S)

        self.P = self.P @ self.S_inv
        self.norm_P()
        if self.has_sun_angle:
            area_center = np.mean(original_bounds, 1)
            ans = LLA_get_vec(area_center, self.sun_el_and_az[1], self.sun_el_and_az[0])
            temp = (self.S @ np.array([[ans[0], ans[1], ans[2], 1]]).T)[0:-1]
            self.sun_el_and_az_vec = ((temp / np.sqrt(np.sum(temp ** 2)))).T[0]
            # area_center = np.mean(original_bounds, 1)
            # direction_vec = area_center[0:2] + np.array(
            #     [np.cos(self.sun_el_and_az[1] * np.pi / 180), np.sin(self.sun_el_and_az[1] * np.pi / 180)]) * .01
            # dist_meters = lat_lon_to_meters(area_center[0], area_center[1], direction_vec[0], direction_vec[1])
            # h_meters = np.tan(self.sun_el_and_az[0] * np.pi / 180) * dist_meters + area_center[2]
            # sun_loc = np.array([direction_vec[0], direction_vec[1], h_meters, 1])
            # area_center = np.array([area_center[0], area_center[1], area_center[2], 1])
            # self.sun_el_and_az_vec = (self.S @ sun_loc)
            # self.sun_el_and_az_vec = self.sun_el_and_az_vec[0:3] / self.sun_el_and_az_vec[3]
            # area_center = self.S @ area_center
            # area_center = area_center[0:3] / area_center[3]
            # self.sun_el_and_az_vec = self.sun_el_and_az_vec - area_center
            #
            # self.sun_el_and_az_vec = self.sun_el_and_az_vec / np.sqrt(np.sum(self.sun_el_and_az_vec ** 2))

        self._cam_center = None
        self.P_pinv = self.P.T @ np.linalg.inv(self.P @ self.P.T)

class P_img_Parallel(P_img):
    def __init__(self, sat_img, num_train_points, min_H, max_H):
        super(P_img_Parallel, self).__init__(sat_img)

        self.P = compute_Approx_Par(sat_img, min_H, max_H, num_train_points=num_train_points)
        self.norm_P()




    def norm_P(self):
        self.P = self.P / self.P[-1, -1]
        self.P_pinv = self.P.T @ np.linalg.inv(self.P @ self.P.T)
        self._cam_center = None

    def invert_P(self, row, col, h=0):
        P23ZpP24mP33ZymP34y = self.P[1, 2] * h + self.P[1, 3] - self.P[2, 2] * h * col - self.P[2, 3] * col
        # P22mP32y = self.coes[1,1] - self.coes[2,1] * col
        P13ZpP14mP33ZxmP34x = self.P[0, 2] * h + self.P[0, 3] - self.P[2, 2] * h * row - self.P[2, 3] * row
        P11mP31x = self.P[0, 0] - self.P[2, 0] * row
        P22mP32y = self.P[1, 1] - self.P[2, 1] * col
        P12mP32x = self.P[0, 1] - self.P[2, 1] * row
        P21mP31y = self.P[1, 0] - self.P[2, 0] * col

        x = (P12mP32x * P23ZpP24mP33ZymP34y - P22mP32y * P13ZpP14mP33ZxmP34x) / (
                P11mP31x * P22mP32y - P12mP32x * P21mP31y)
        y = (-P11mP31x * P23ZpP24mP33ZymP34y + P21mP31y * P13ZpP14mP33ZxmP34x) / (
                P11mP31x * P22mP32y - P12mP32x * P21mP31y)

        return x, y, h

    def apply_P(self, X, Y, Z):
        P_11, P_12, P_13, P_14, P_21, P_22, P_23, P_24, P_31, P_32, P_33, P_34 = self.P[0, 0], \
                                                                                 self.P[0, 1], \
                                                                                 self.P[0, 2], \
                                                                                 self.P[0, 3], \
                                                                                 self.P[1, 0], \
                                                                                 self.P[1, 1], \
                                                                                 self.P[1, 2], \
                                                                                 self.P[1, 3], \
                                                                                 self.P[2, 0], \
                                                                                 self.P[2, 1], \
                                                                                 self.P[2, 2], \
                                                                                 self.P[2, 3]
        x = P_11 * X + P_12 * Y + P_13 * Z + P_14
        y = P_21 * X + P_22 * Y + P_23 * Z + P_24
        c = P_31 * X + P_32 * Y + P_33 * Z + P_34

        return x / c, y / c

    def scale_P(self, original_bounds, new_bounds):
        self.scaled = True
        r = new_bounds[:, 1] - new_bounds[:, 0]
        delta_LLA = original_bounds[:, 1] - original_bounds[:, 0]
        self.S = np.array([[r[0] / delta_LLA[0], 0, 0, -r[0] * original_bounds[0, 0] / delta_LLA[0] + new_bounds[0, 0]],
                           [0, r[1] / delta_LLA[1], 0, -r[1] * original_bounds[1, 0] / delta_LLA[1] + new_bounds[1, 0]],
                           [0, 0, r[2] / delta_LLA[2], -r[2] * original_bounds[2, 0] / delta_LLA[2] + new_bounds[2, 0]],
                           [0, 0, 0, 1]])
        self.S_inv = np.linalg.inv(self.S)

        self.P = self.P @ self.S_inv
        self.norm_P()
        if self.has_sun_angle:
            area_center = np.mean(original_bounds, 1)
            direction_vec = area_center[0:2] + np.array(
                [np.cos(self.sun_el_and_az[1] * np.pi / 180), np.sin(self.sun_el_and_az[1] * np.pi / 180)]) * .01
            dist_meters = lat_lon_to_meters(area_center[0], area_center[1], direction_vec[0], direction_vec[1])
            h_meters = np.tan(self.sun_el_and_az[0] * np.pi / 180) * dist_meters + area_center[2]
            sun_loc = np.array([direction_vec[0], direction_vec[1], h_meters, 1])
            area_center = np.array([area_center[0], area_center[1], area_center[2], 1])
            self.sun_el_and_az_vec = (self.S @ sun_loc)
            self.sun_el_and_az_vec = self.sun_el_and_az_vec[0:3] / self.sun_el_and_az_vec[3]
            area_center = self.S @ area_center
            area_center = area_center[0:3] / area_center[3]
            self.sun_el_and_az_vec = self.sun_el_and_az_vec - area_center

            self.sun_el_and_az_vec = self.sun_el_and_az_vec / np.sqrt(np.sum(self.sun_el_and_az_vec ** 2))

        self._cam_center = None
        self.P_pinv = self.P.T @ np.linalg.inv(self.P @ self.P.T)

def sample_pts(img_shape, H_min, H_max, method, num_train_points):
    sub_section = np.array([[0, img_shape[0]],
                            [0, img_shape[1]]])
    H_range = [H_min, H_max]
    if method == 'Chebyshev':
        n_pts_per_axis = num_train_points  # int(np.round(num_train_points ** (1 / 3)))-1
        c_pts = np.cos((2 * np.arange(0, n_pts_per_axis + 1) + 1) / (2 * (n_pts_per_axis + 1)) * np.pi)
        # print(c_pts)

        x_pts = (sub_section[0, 1] - sub_section[0, 0]) / 2 * (c_pts + 1) + sub_section[0, 0]
        y_pts = (sub_section[1, 1] - sub_section[1, 0]) / 2 * (c_pts + 1) + sub_section[1, 0]
        z_pts = (H_range[-1] - H_range[0]) / 2 * (c_pts + 1) + H_range[0]

        x, y, z = np.meshgrid(x_pts, y_pts, z_pts)
        x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
        n_pts = x.shape[0]

    elif method == 'Uniform':
        step_size = num_train_points
        x_step = (sub_section[0, 1] - sub_section[0, 0]) / step_size
        y_step = (sub_section[1, 1] - sub_section[1, 0]) / step_size
        h_step = (H_range[-1] - H_range[0]) / step_size

        x, y, z = np.meshgrid(np.arange(sub_section[0, 0], sub_section[0, 1] + x_step, x_step),
                              np.arange(sub_section[1, 0], sub_section[1, 1] + y_step, y_step),
                              np.arange(H_range[0], H_range[-1] + h_step, h_step))
        x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
        n_pts = x.shape[0]

    elif method == 'Random':
        n_pts = (num_train_points + 1) ** 3

        x = np.random.rand(n_pts) * (sub_section[0, 1] - sub_section[0, 0]) + sub_section[0, 0]
        y = np.random.rand(n_pts) * (sub_section[1, 1] - sub_section[1, 0]) + sub_section[1, 0]
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
    return x, y, z, n_pts

def compute_Approx_RPC(sat_img, H_min, H_max, num_train_points=10, num_test_points=50, show_details=True,
                       method='Chebyshev'):


    # exit()
    x,y,z,n_pts = sample_pts(sat_img.img.shape, H_min, H_max, method, num_train_points)

    lat, lon, h = sat_img.invert_rpc(x[0:n_pts], y[0:n_pts], z[0:n_pts])


    lat_n = (np.min(lat), np.max(lat - np.min(lat)))
    lon_n = (np.min(lon), np.max(lon - np.min(lon)))
    h_n = (np.min(h), np.max(h - np.min(h)))
    lat = (lat - lat_n[0]) / lat_n[1] * 1000
    lon = (lon - lon_n[0]) / lon_n[1] * 1000
    h = (h - h_n[0]) / h_n[1] * 1000

    X = np.zeros([2 * n_pts, 11])
    Y = np.zeros([2 * n_pts, 1])


    for i in range(n_pts):
        X[2 * i, :] = [lat[i], lon[i], h[i], 1, 0, 0, 0, 0, -x[i] * lat[i], -x[i] * lon[i], -x[i] * h[i]]
        Y[2 * i, 0] = x[i]
        X[2 * i + 1, :] = [0, 0, 0, 0, lat[i], lon[i], h[i], 1, -y[i] * lat[i], -y[i] * lon[i], -y[i] * h[i]]
        Y[2 * i + 1, 0] = y[i]

    coes_temp = np.linalg.inv(X.T @ X) @ X.T @ Y
    coes = np.ones([3, 4])
    coes[0, :] = coes_temp[0:4, 0]
    coes[1, :] = coes_temp[4:8, 0]
    coes[2, 0:3] = coes_temp[8:, 0]

    A = np.array([[1000 / lat_n[1], 0, 0, -1000 * lat_n[0] / lat_n[1]],
                  [0, 1000 / lon_n[1], 0, -1000 * lon_n[0] / lon_n[1]],
                  [0, 0, 1000 / h_n[1], -1000 * h_n[0] / h_n[1]],
                  [0, 0, 0, 1]])

    coes = coes @ A

    return coes

def test_accuracy(sat_img, P_img, num_test_pts, H_min, H_max):
    x,y,z, n_pts = sample_pts(sat_img.img.shape, H_min, H_max, method="Uniform", num_train_points=num_test_pts)
    lat, lon, h = sat_img.invert_rpc(x[0:n_pts], y[0:n_pts], z[0:n_pts])
    X_gt, Y_gt = sat_img.apply_rpc(lat,lon,h)
    X_est, Y_est = P_img.apply_P(lat,lon,h)
    error_X = X_est - X_gt
    error_Y = Y_est - Y_gt
    dist_error = np.sqrt(error_X**2+error_Y**2)

    # print("Mean Dist Error:", np.mean(dist_error))
    # print("Std Dist Error", np.std(dist_error))
    # print("Min Dist Error:", np.min(dist_error))
    # print("Max Dist Error:", np.max(dist_error))
    return  np.mean(dist_error), np.std(dist_error), np.min(dist_error), np.max(dist_error)


