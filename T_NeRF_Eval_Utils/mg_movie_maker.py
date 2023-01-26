from tqdm import tqdm
from matplotlib import pyplot as plt
import torch as t
import numpy as np
from T_NeRF_Eval_Utils import load_t_nerf
from opt import get_opts
from pre_NeRF.mg_time import time_encode_month_day_only
from all_NeRF.mg_spline import spline_3
from scipy.integrate import quad
from scipy.optimize import root_scalar
from copy import deepcopy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# def sample_rays(X0, X1, phi_deg, theta_deg, dist_to_center, vertical_sample_pts, img_size, h_bounds = (-1., 1)):
#     # XY = np.stack(np.meshgrid(np.linspace(X0[0],X1[0], img_size[0]), np.linspace(X0[1],X1[1], img_size[1]), np.linspace(h_bounds[0], h_bounds[1], vertical_sample_pts)), 0)
#     # print(XY)
#     # print(XY.shape)
#
#     Camera_Center = np.array([0,0,dist_to_center])
#     pitch = phi_deg * np.pi / 180
#     rot_phi = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                   [0,1,0],
#                   [-np.sin(pitch), 0, np.cos(pitch)]])
#     yaw = theta_deg * np.pi / 180
#     rot_theta = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                           [np.sin(yaw), np.cos(yaw), 0],
#                           [0,0,1]])
#     Camera_Center = rot_theta @ rot_phi @ Camera_Center
#     print(Camera_Center)
#
# def sample_rays(camera_cent_location, focal_direction, focal_length):
#     #camera cent location: (dist from origin, phi, theta)
#     #focal point: (X, Y, Z) in world coordinates such that the focal line travels through the focal point and camera center
#     #focal length: distance between the focal point and camera center, positive indicates closer to focal direction point
#
#     dist_to_center, phi_deg, theta_deg = camera_cent_location
#     Camera_Center = np.array([0,0,dist_to_center])
#     pitch = phi_deg * np.pi / 180
#     rot_phi = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                   [0,1,0],
#                   [-np.sin(pitch), 0, np.cos(pitch)]])
#     yaw = theta_deg * np.pi / 180
#     rot_theta = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                           [np.sin(yaw), np.cos(yaw), 0],
#                           [0,0,1]])
#     Camera_Center = rot_theta @ rot_phi @ Camera_Center
#     focal_point = Camera_Center - focal_direction
#     focal_point = -focal_point / np.sqrt(np.sum(focal_point**2)) * focal_length + Camera_Center
#     print(Camera_Center)
#     print(focal_point)

def sample_rays_projective(Img_Center, Img_length, phi_deg, theta_deg, Img_Size):
    XYZ = np.expand_dims(np.stack(np.meshgrid(np.linspace(-Img_length[1],Img_length[1], Img_Size[1]), np.linspace(-Img_length[0],Img_length[0], Img_Size[0]), np.linspace(Img_length[2], -Img_length[2], Img_Size[2])), -1), -1)
    pitch = phi_deg * np.pi / 180
    rot_phi = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                  [0,1,0],
                  [-np.sin(pitch), 0, np.cos(pitch)]])
    yaw = theta_deg * np.pi / 180
    rot_theta = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0,0,1]])

    Rays = (rot_theta @ rot_phi).reshape([1,1,1,3,3]) @ XYZ + Img_Center.reshape(1,1,1,3,1)
    Rays = Rays[:,:,:,:,0]
    deltas = np.sqrt(np.sum((Rays[:,:,1::] - Rays[:,:,0:-1])**2, -1))
    # print(deltas)
    # print(deltas.shape)


    return Rays, deltas[0,0,0]

class get_Img():
    def __init__(self, network, device, valid_range = np.array([[-1,1.], [-1,1], [-1,1]]), max_batch_size = 100, per_img_tqdm = True):
        self.network = network
        self.valid_range = valid_range
        self.device = device
        self.max_batch_size = max_batch_size
        self.per_img_tqdm = per_img_tqdm

    def capture_frame(self, Img_Center, Img_length, phi_deg, theta_deg, Img_Size, Solar_Angle, Time, use_Time = True, Sky_Color = None):
        Rays, delta = sample_rays_projective(Img_Center, Img_length, phi_deg, theta_deg, Img_Size)
        Img, Point_Map = self.eval_rays(Rays, Solar_Angle, Time, use_Time, Sky_Color, delta)

        return Img

    def capture_frame_advanced(self, Img_Center, Img_length, phi_deg, theta_deg, Img_Size, Solar_Angle, Time, use_Time=True,Sky_Color=None):
        fig = plt.figure()
        plt.scatter(phi_deg, theta_deg)
        plt.xlim([-.5, 20])
        plt.ylim([-180, 180])
        plt.xlabel("Azimuth Angle (Deg)")
        plt.ylabel("Rotation Angle (Deg)")
        plt.title("Camera Angle")
        canvas = FigureCanvas(fig)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()
        Cam_Angle_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), -1, 3)
        plt.close()


        Rays, delta = sample_rays_projective(Img_Center, Img_length, phi_deg, theta_deg, Img_Size)
        Img, HM = self.eval_rays_advanced(Rays, Solar_Angle, Time, use_Time, Sky_Color, delta)
        # Img = None
        # HM = None
        return Img, HM, Cam_Angle_img


    def eval_rays(self, Rays, Solar_Angle, Time, use_Time = True, Sky_Color = None, delta = None):
        Rays_Reshape = t.tensor(Rays.reshape([-1,3]), dtype=t.float)
        SA = t.ones_like(Rays_Reshape) * t.unsqueeze(t.tensor(Solar_Angle, dtype=t.float), 0)
        T = time_encode_month_day_only(Time)
        T = t.ones([Rays_Reshape.shape[0], 2], dtype=t.float) * t.unsqueeze(t.tensor(T[0:2], dtype=t.float), 0)

        N = Rays_Reshape.shape[0]
        Rhos_out, Cols_out, Solar_Vis_out, Sky_Col_out = np.ones([N, 1]), np.ones([N, 3]), np.ones([N, 1]), np.ones([N, 3])

        with t.no_grad():
            if self.per_img_tqdm:
                for i in tqdm(range(0, Rays_Reshape.shape[0], self.max_batch_size)):
                    i_end = min(i+self.max_batch_size, Rays_Reshape.shape[0])
                    X = Rays_Reshape[i:i_end].to(self.device)
                    Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col = self.network(X, SA[i:i_end].to(self.device), T[i:i_end].to(self.device))
                    Rhos_out[i:i_end] = Rho.cpu().numpy()
                    Cols_out[i:i_end] = Col.cpu().numpy()
                    Solar_Vis_out[i:i_end] = Solar_Vis.cpu().numpy()
                    Sky_Col_out[i:i_end] = Sky_Col.cpu().numpy()
            else:
                for i in range(0, Rays_Reshape.shape[0], self.max_batch_size):
                    i_end = min(i + self.max_batch_size, Rays_Reshape.shape[0])
                    X = Rays_Reshape[i:i_end].to(self.device)
                    Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col = self.network(X,
                                                                                          SA[i:i_end].to(self.device),
                                                                                          T[i:i_end].to(self.device))
                    Rhos_out[i:i_end] = Rho.cpu().numpy()
                    Cols_out[i:i_end] = Col.cpu().numpy()
                    Solar_Vis_out[i:i_end] = Solar_Vis.cpu().numpy()
                    Sky_Col_out[i:i_end] = Sky_Col.cpu().numpy()
        if delta is None:
            print("CASE NOT IMPLEMENTED PLEASE PROVIDE DELTA!!!")
            exit()
        Rays_Reshape_bad =   (Rays_Reshape[:, 0] < self.valid_range[0,0]) + (self.valid_range[0, 1] < Rays_Reshape[:, 0]) + \
                             (Rays_Reshape[:, 1] < self.valid_range[1, 0]) + (self.valid_range[1, 1] < Rays_Reshape[:, 1]) + \
                             (Rays_Reshape[:, 2] < self.valid_range[2, 0]) + (self.valid_range[2, 1] < Rays_Reshape[:, 2])
        # print(Rays_Reshape_bad)
        # exit()

        Rhos_out[Rays_Reshape_bad] = 0
        Rhos_out = Rhos_out.reshape([Rays.shape[0], Rays.shape[1], Rays.shape[2], 1])
        Cols_out = Cols_out.reshape([Rays.shape[0], Rays.shape[1], Rays.shape[2], 3])
        Solar_Vis_out = Solar_Vis_out.reshape([Rays.shape[0], Rays.shape[1], Rays.shape[2], 1])
        Sky_Col_out = Sky_Col_out.reshape([Rays.shape[0], Rays.shape[1], Rays.shape[2], 3])

        PE = 1-np.exp(-Rhos_out*delta)
        PV = np.exp(-np.cumsum(np.concatenate([np.zeros([Rhos_out.shape[0],Rhos_out.shape[1],1,1]), Rhos_out*delta], 2), 2)[:,:,0:-1])
        PS = PE * PV

        Final_Cols = (Solar_Vis_out + (1-Solar_Vis_out) * Sky_Col_out) * Cols_out
        # Surf_Loc = np.sum(PS * Rays, 2)
        # Depth_Map = np.sum(PS * np.arange(0, PS.shape[2]).reshape([1,1,-1,1]) * delta, 2)[:,:,0]

        Out_Img = np.sum(PS * Final_Cols, 2)
        # plt.subplot(1,2,1)
        # plt.imshow(Out_Img)
        # plt.subplot(1,2,2)
        # plt.imshow(Depth_Map)
        # plt.show()
        #
        # # dist_map = np.sum(PS * Rays, 2) / np.sum(PS, 2)
        #
        # # plt.imshow(dist_map[:,:,2])
        # # plt.show()
        # # print(Rhos_out.shape)
        #
        # exit()
        # Out_Img = np.moveaxis(Out_Img, 0, 1)
        # PS = np.moveaxis(PS, 0, 1)
        return Out_Img, PS

    def eval_rays_advanced(self, Rays, Solar_Angle, Time, use_Time=True, Sky_Color=None, delta=None):
        PS = np.zeros([1,1,1])
        Out_Imgs = []
        for i in range(Time.shape[0]):
            Out_Img, PS = self.eval_rays(Rays, Solar_Angle, Time[i], use_Time, Sky_Color, delta)
            Out_Imgs.append(Out_Img)
        HM = PS * np.linspace(0,2,PS.shape[2]).reshape([1,1,-1,1])
        HM = np.sum(HM, 2)[:, :, 0]
        return Out_Imgs, HM

def scout_Time(Camera:get_Img, ts, outloc = "./Movie_Imgs/Time_Shots"):
    Sun_Angle = np.array([0,0,1])
    i = 0
    for a_t in tqdm(ts):
        Img = Camera.capture_frame(Img_Center=np.array([0, 0, 0]), Img_length=(1, 1, 1), phi_deg=0, theta_deg=0, Img_Size=(128, 128, 64), Solar_Angle=Sun_Angle, Time=a_t)
        plt.imsave(outloc + "/" + str(i) + ".png", Img)
        i += 1

def scout_Angle(Camera:get_Img, phis, thetas, outloc = "./Movie_Imgs/Angle_Shots"):
    Sun_Angle = np.array([0,0,1])
    i = 0


    Phi_Theta = np.stack([np.repeat(phis, thetas.shape[0]), np.tile(thetas, phis.shape)], 1)

    for a_t in tqdm(range(Phi_Theta.shape[0])):
        Img = Camera.capture_frame(Img_Center=np.array([0, 0, 0]), Img_length=(1, 1, np.sqrt(3)), phi_deg=Phi_Theta[i,0], theta_deg=Phi_Theta[i,1], Img_Size=(128, 128, 64), Solar_Angle=Sun_Angle, Time=a_t)
        plt.imsave(outloc + "/" + str(Phi_Theta[i,0]) + "_" + str(Phi_Theta[i,1]) + ".png", Img)
        i += 1

class script():
    def __init__(self):
        self.fixed_points = []
        self.splines_ready = False

    def add_fixed_point(self, Img_Center, Img_length, phi_deg, theta_deg, Img_Size, Solar_Angle, Time):
        a_scene = {"Center":Img_Center, "Length":Img_length, "Phi_Deg":phi_deg, "Theta_Deg":theta_deg, "Img_Size":Img_Size, "Solar_Angle":Solar_Angle, "Time":Time}
        self.fixed_points.append(a_scene)
        self.splines_ready = False

    def eval_fixed_scene(self, Camera:get_Img):
        imgs = [Camera.capture_frame(i["Center"], i["Length"], i["Phi_Deg"], i["Theta_Deg"], i["Img_Size"], i["Solar_Angle"], i["Time"]) for i in tqdm(self.fixed_points)]
        return imgs

    def _r_cent(self, phi_deg, theta_deg):
        pitch = phi_deg * np.pi / 180
        rot_phi = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])
        yaw = theta_deg * np.pi / 180
        rot_theta = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                              [np.sin(yaw), np.cos(yaw), 0],
                              [0, 0, 1]])

        return rot_theta @ rot_phi @ np.array([[0], [0], [1.]])

    def get_Splines(self, safe_mode = True):
        if safe_mode == True or self.splines_ready == False:
            self.splines_ready = True
            C = []
            Phis, Theta, Cs, Ws, time, SA = [], [], [], [], [], []
            for i in self.fixed_points:
                C.append(np.squeeze(self._r_cent(i["Phi_Deg"], i["Theta_Deg"]) + i["Center"].reshape([3, 1])))
                Phis.append(i["Phi_Deg"])
                Theta.append(i["Theta_Deg"])
                Cs.append(i["Center"])
                Ws.append(i["Length"])
                time.append(i["Time"])
                SA.append(i["Solar_Angle"])


            self.C = np.array(C)
            self.Phis = np.array(Phis)
            self.Theta = np.array(Theta)
            self.Cs = np.array(Cs)
            self.Ws = np.array(Ws)
            self.times = np.array(time)
            self.SA = np.array(SA)

            self.T = np.arange(len(self.fixed_points))

            # T = np.zeros([len(self.fixed_points)])
            # for i in range(1, len(self.fixed_points)):
            #     T[i] = np.sqrt(np.sum(((C[i]*IL[i] - C[i-1]*IL[i-1]))**2)) + np.sqrt(np.sum(((Cs[i] - Cs[i-1]))**2)) + T[i-1] + 1e-8
            # T = T/ T[-1] * len(self.fixed_points)
            # print(T)
            # plt.plot(T, "x")
            # exit()
            self.Xs_spline = spline_3(self.T, self.C[:, 0])
            self.Ys_spline = spline_3(self.T, self.C[:, 1])
            self.Zs_spline = spline_3(self.T, self.C[:, 2])

            self.Centers_X_spline = spline_3(self.T, self.Cs[:, 0])
            self.Centers_Y_spline = spline_3(self.T, self.Cs[:, 1])
            self.Centers_Z_spline = spline_3(self.T, self.Cs[:, 2])

            self.Width_X_spline = spline_3(self.T, self.Ws[:, 0])
            self.Width_Y_spline = spline_3(self.T, self.Ws[:, 1])
            self.Width_Z_spline = spline_3(self.T, self.Ws[:, 2])

            self.Time_Spline = [spline_3(self.T, self.times[:,i]) for i in range(self.times.shape[1])]

            self.SA_X_spline = spline_3(self.T, self.SA[:, 0])
            self.SA_Y_spline = spline_3(self.T, self.SA[:, 1])
            self.SA_Z_spline = spline_3(self.T, self.SA[:, 2])

        return [self.Xs_spline, self.Ys_spline, self.Zs_spline], [self.Centers_X_spline, self.Centers_Y_spline, self.Centers_Z_spline], [self.Width_X_spline, self.Width_Y_spline, self.Width_Z_spline]


    def _pose_score(self, X):
        Y = np.sqrt(self.Xs_spline(X)**2 + self.Ys_spline(X)**2 + self.Zs_spline(X)**2) + \
               np.sqrt(self.Centers_X_spline(X)**2 + self.Centers_Y_spline(X)**2 + self.Centers_Z_spline(X)**2) + \
               np.abs(self.SA_X_spline(X))
        return Y

    def _pose(self, X):
        Y = np.array([self.Xs_spline(X), self.Ys_spline(X), self.Zs_spline(X),
            self.Centers_X_spline(X), self.Centers_Y_spline(X), self.Centers_Z_spline(X),
            self.SA_X_spline(X)])
        return Y

    def _pose_score_len(self, X):
        return np.sqrt(1+self._pose_score(X)**2)

    def _update_SA(self, SA):
        return SA

    def film_movie(self, Camera, n_frames, advanced_mode = True, output_loc = None):
        if self.splines_ready == False:
            self.get_Splines()

        T_min = 0
        T_max = self.T[-1]-.001
        full_Path_length = self.get_path_length(T_min, T_max+1)
        step_size = full_Path_length / n_frames

        current_T = T_min
        Last_T = T_min-1
        imgs = []
        # if advanced_mode:
        angle_plot_run = None
        _, _, _, _, _, SA, _  = self._extract_data(current_T)
        for i in tqdm(range(n_frames)):
            # print(current_T, i, X["Phi_Deg"], X["Theta_Deg"], self.fixed_points[i]["Phi_Deg"], self.fixed_points[i]["Theta_Deg"])
            if Last_T == current_T:
                break
            Last_T = current_T
            X = {}
            X["Center"], X["Length"], X["Phi_Deg"], X["Theta_Deg"], X["Img_Size"], X["Solar_Angle"], X["Time"] = self._extract_data(current_T)
            # try:
            if advanced_mode == False:
                a_img = Camera.capture_frame(X["Center"], X["Length"], X["Phi_Deg"], X["Theta_Deg"], X["Img_Size"], X["Solar_Angle"], X["Time"][0])
            else:
                a_img, h_map, angle_plot = Camera.capture_frame_advanced(X["Center"], X["Length"], X["Phi_Deg"], X["Theta_Deg"], X["Img_Size"],
                                             SA, X["Time"])
                if angle_plot_run is None:
                    angle_plot_run = angle_plot
                else:
                    diff_loc = angle_plot_run != angle_plot
                    angle_plot_run = 255 - angle_plot_run
                    angle_plot_run[diff_loc] = (angle_plot_run[diff_loc]*.9).astype(np.uint8)
                    angle_plot_run[diff_loc] += (255 - angle_plot[diff_loc])
                    angle_plot_run = 255 - angle_plot_run

                a_img = [a_img, h_map, angle_plot_run]
                # plt.imshow(angle_plot_run)
                # plt.show()

                SA = self._update_SA(SA)
                if output_loc is not None:
                    # idx = [1,2,4,5,3,6]
                    if len(a_img[0]) == 3:
                        plt.figure(figsize=(9,9))
                        for k in range(3):
                            plt.subplot(2,2,k+1)
                            plt.imshow(a_img[0][k])
                            plt.xticks([])
                            plt.yticks([])
                        plt.subplot(2,2, 4)
                    elif len(a_img[0]) == 1:
                        plt.figure(figsize=(9, 4.5))
                        plt.subplot(1, 2, 1)
                        plt.imshow(a_img[0][0])
                        plt.xticks([])
                        plt.yticks([])
                        plt.subplot(1, 2, 2)
                    plt.imshow(a_img[1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    # plt.subplot(2,3,idx[5])
                    # plt.imshow(angle_plot_run)

                    plt.savefig(output_loc + "/" + str(i) + ".png")
                    plt.close()
            # except:
            #     print("HERE")
            #     return imgs
            if output_loc is None:
                imgs.append(a_img)
            # current_T + (T_max - T_min) / n_frames
            current_T = min(self.get_T(current_T, step_size, T_max), T_max)
        return imgs

    def get_path_length(self, X0, X1):
        ans = quad(self._pose_score_len, X0, X1)
        return ans[0]

    def get_partial_path(self, X, X0, dist = 0):
        return self.get_path_length(X0, X) - dist


    def get_T(self, T0, dist, T_max):
        # temp = self._get_T_helper(X0, dist, self._pose)
        # print(T0, dist, T_max)
        ans = root_scalar(self.get_partial_path, bracket=(T0-1, T_max+1), args=(T0, dist), maxiter=100)
        # print(ans)
        # print(self.get_partial_path(ans.root, T0))
        # print()
        # exit()
        return ans.root

    def _extract_data(self, Current_T):
        time = np.array([T_spline.slow_eval(Current_T) for T_spline in self.Time_Spline])

        XYZ_Est = np.array([self.Xs_spline.slow_eval(Current_T), self.Ys_spline.slow_eval(Current_T), self.Zs_spline.slow_eval(Current_T)])
        XYZ_Cent_Est = np.array([self.Centers_X_spline.slow_eval(Current_T), self.Centers_Y_spline.slow_eval(Current_T), self.Centers_Z_spline.slow_eval(Current_T)])
        XYZ_Zoom_Est = np.array([self.Width_X_spline.slow_eval(Current_T), self.Width_Y_spline.slow_eval(Current_T), self.Width_Z_spline.slow_eval(Current_T)])
        XYZ_SA = np.array([self.SA_X_spline.slow_eval(Current_T), self.SA_Y_spline.slow_eval(Current_T), self.SA_Z_spline.slow_eval(Current_T)])
        XYZ_SA = XYZ_SA / np.sqrt(np.sum(XYZ_SA**2))

        end_point = XYZ_Est - XYZ_Cent_Est

        L = np.sqrt(np.sum(end_point**2))
        Phi = np.arccos(end_point[2] / L) * 180 / np.pi
        Theta = np.arctan2(end_point[1], end_point[0]) * 180 / np.pi

        return XYZ_Cent_Est, XYZ_Zoom_Est, Phi, Theta, self.fixed_points[0]["Img_Size"], XYZ_SA, time






    def show_path(self):

        if self.splines_ready == False:
            self.get_Splines()

        T_all = np.linspace(0, len(self.fixed_points)-1, 10000, endpoint=False)
        XYZ_Est = np.array([[self.Xs_spline.slow_eval(i), self.Ys_spline.slow_eval(i), self.Zs_spline.slow_eval(i)] for i in T_all])
        XYZ_Cent_Est = np.array([[self.Centers_X_spline.slow_eval(i), self.Centers_Y_spline.slow_eval(i), self.Centers_Z_spline.slow_eval(i)] for i in T_all])
        XYZ_Zoom_Est = np.array([[self.Width_X_spline.slow_eval(i), self.Width_Y_spline.slow_eval(i), self.Width_Z_spline.slow_eval(i)] for i in T_all])

        XYZ_Zoom_ell = np.sqrt(np.sum(XYZ_Zoom_Est**2, 1))
        # XYZ_Est_Diff = np.array([[Xs_spline.full_slow_eval(i)[1], Ys_spline.full_slow_eval(i)[1], Zs_spline.full_slow_eval(i)[1]] for i in T_all])

        plt.figure()
        plt.subplot(1,3,1)
        plt.title("Camera Center Position")
        plt.plot(self.C[:,0], self.C[:,1], "o-")
        plt.plot(XYZ_Est[:,0], XYZ_Est[:,1])

        plt.subplot(1, 3, 2)
        plt.title("Image Center Position")
        plt.plot(self.Cs[:, 0], self.Cs[:, 1], "o-")
        plt.plot(XYZ_Cent_Est[:, 0], XYZ_Cent_Est[:, 1])

        plt.subplot(1, 3, 3)
        plt.title("Zoom Level")
        plt.plot(self.T, self.Ws[:, 0], "o-")
        plt.plot(T_all, XYZ_Zoom_Est[:, 0])

        plt.figure()
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.plot(self.T, self.C[:,i], "o-")
            plt.plot(T_all, XYZ_Est[:,i])
        # plt.figure()
        # plt.plot(T_all, np.sqrt(np.sum((XYZ_Est_Diff)**2, 1)))

        plt.show()
        # exit()

def edit_film(dir = "", name = "test"):
    import os
    os.system("ffmpeg -f image2 -r 20 -i ./Movie_Imgs" + dir + "/%01d.png -vcodec mpeg4 -y  -b 5000k ./" + name + ".mp4")
    os.system("ffmpeg -i ./" + name + ".mp4" + " ./" + name + ".gif")