import numpy as np
import math
from math import ceil
import torch as t
from tqdm import tqdm
import cv2 as cv

from all_NeRF.mg_unit_converter import lat_lon_to_meters

def get_model_size(bounds, voxel_size_meters):
    z_steps = ceil((bounds[2, 1] - bounds[2, 0]) / voxel_size_meters[2])
    lat_meters = lat_lon_to_meters(bounds[0, 0], bounds[1, 0], bounds[0, 1], bounds[1, 0])
    y_steps = ceil(lat_meters / voxel_size_meters[1])
    lon_meters = lat_lon_to_meters(bounds[0, 0], bounds[1, 0], bounds[0, 0], bounds[1, 1])
    x_steps = ceil(lon_meters / voxel_size_meters[0])

    return np.array([y_steps, x_steps, z_steps])

def get_out_img_size(xy_list, P_imgs):
    x0, x1, y0, y1 = xy_list[0], xy_list[-1], xy_list[0], xy_list[-1]
    corners = np.array([[x0, y0, 0],
                        [x0, y1, 0],
                        [x1, y0, 0],
                        [x1, y1, 0]])
    largest_dist = 0
    for i in range(len(P_imgs)):
        proj_img_corners = np.array(P_imgs[i].apply_P(corners[:,0], corners[:,1], corners[:,2]))
        delta_x = np.max(proj_img_corners[0]) - np.min(proj_img_corners[0])
        delta_y = np.max(proj_img_corners[1]) - np.min(proj_img_corners[1])
        # print(proj_img_corners, delta_x, delta_y)
        largest_dist = np.max([largest_dist, delta_y, delta_x])
    largest_dist = np.int(largest_dist / (xy_list.shape[0] - 1))+1
    return largest_dist

class apply_H_in_p_tensor():
    def __init__(self, imgs, out_shape, num_zs, device, store_imgs_on_GPU = True):
        start_pts = np.ones([3, out_shape[0] * out_shape[1]], dtype=int)
        start_pts[0, :] = np.arange(0, out_shape[0]).repeat(out_shape[1])
        start_pts[1, :] = np.tile(np.arange(0, out_shape[1]), out_shape[0])
        if store_imgs_on_GPU:
            self.imgs = imgs.to(device)
            self._imgs_on_GPU = True
        else:
            self.imgs = imgs
            self._imgs_on_GPU = False

        self.start_pts1 = t.tensor(start_pts).double().to(device)

        start_pts = start_pts.T[:, 0:2]
        start_pts = np.stack([np.stack([start_pts] * num_zs, 0)] * imgs.shape[0])
        self.start_pts2 = t.tensor(start_pts.reshape([-1, 2])).to(device)

        z_idx, img_idx, pose_idx = np.meshgrid(np.arange(num_zs), np.arange(imgs.shape[0]),
                                               np.arange(out_shape[0] * out_shape[1]))
        self.img_idx, self.z_idx, self.pose_idx = t.tensor(np.ravel(img_idx)), t.tensor(np.ravel(z_idx)), t.tensor(np.ravel(pose_idx))
        self.img_idx, self.z_idx, self.pose_idx = self.img_idx.to(device), self.z_idx.to(device), self.pose_idx.to(device)
        self.out_shape = out_shape
        self.device = device


    def apply_H_in_parrellel(self, Hs):
        out_imgs = t.zeros([Hs.shape[0], Hs.shape[1], self.out_shape[0], self.out_shape[1], self.imgs.shape[3]]).to(self.device)
        H_inv = t.linalg.inv(Hs)
        end_pts = t.moveaxis(H_inv @ self.start_pts1, 2, 3)
        end_pts = (t.round(end_pts[:, :, :, 0:2] / end_pts[:, :, :, 2::]).long())

        good_idx = t.ravel(
            (end_pts[:, :, :, 1] >= 0) * (end_pts[:, :, :, 0] >= 0) * (end_pts[:, :, :, 0] < self.imgs.shape[1]) * (
                        end_pts[:, :, :, 1] < self.imgs.shape[2]))

        end_pts = end_pts.reshape([-1, 2])

        img_idx, z_idx, pose_idx = self.img_idx[good_idx], self.z_idx[good_idx], self.pose_idx[good_idx]
        start_pts = self.start_pts2[good_idx]
        end_pts = end_pts[good_idx]

        if self._imgs_on_GPU:
            out_imgs[img_idx, z_idx, start_pts[:, 0], start_pts[:, 1]] = self.imgs[img_idx, end_pts[:, 0], end_pts[:, 1]]
        else:
            out_imgs[img_idx, z_idx, start_pts[:, 0], start_pts[:, 1]] = self.imgs[
                img_idx, end_pts[:, 0], end_pts[:, 1]].to(self.device)

        return out_imgs

    def __call__(self, Hs):
        return self.apply_H_in_parrellel(Hs)

def find_Homography_multi(start_pts, end_pts):
    A = np.zeros([start_pts.shape[0], 8,8])
    A[:, 0:4, 0:2] = start_pts
    A[:, 0:4, 2] = 1
    A[:, 4::, 3:5] = start_pts
    A[:, 4::, 5] = 1
    A[:, 0:4, 6] = -(start_pts[:,:,0] * end_pts[:,:,0])
    A[:, 4::, 6] = -(start_pts[:,:, 0] * end_pts[:,:, 1])
    A[:, 0:4, 7] = -(start_pts[:,:, 1] * end_pts[:,:, 0])
    A[:, 4::, 7] = -(start_pts[:,:, 1] * end_pts[:,:, 1])
    b = np.reshape(end_pts, [start_pts.shape[0], 8, 1], order='F')
    H_streched = (np.linalg.inv(A) @ b)
    H = np.ones([H_streched.shape[0], 3, 3])
    H[:, 0] = H_streched[:, 0:3, 0]
    H[:, 1] = H_streched[:, 3:6, 0]
    H[:, 2, 0] = H_streched[:, 6, 0]
    H[:, 2, 1] = H_streched[:, 7, 0]


    return H

def get_H_in_parrellel(inputs, targets):
    inputs_adj = inputs.reshape([inputs.shape[0] * inputs.shape[1], 4, 2])
    targets_adj = np.repeat(np.expand_dims(targets, 0), inputs_adj.shape[0], 0)
    Hs = find_Homography_multi(inputs_adj, targets_adj)
    Hs = Hs.reshape([inputs.shape[0], inputs.shape[1], 3, 3])
    return Hs

def get_sub_imgs(device, x0, y0, y1, x1, P_imgs, z_steps, out_img_size, apply_H_tool_tensor):
    if device is None:
        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    corners = np.array([[x0, y0, -1],
                        [x0, y1, -1],
                        [x1, y0, -1],
                        [x1, y1, -1]])

    Zps = np.linspace(-1, 1, num=z_steps, endpoint=True)
    all_corners = np.stack([np.tile(corners[:, 0], z_steps), np.tile(corners[:, 1], z_steps), np.repeat(Zps, 4)], 1)
    # print(all_corners.shape)
    # exit()
    a_pt_in_img = np.array(
        [a_P_img.apply_P(all_corners[:, 0], all_corners[:, 1], all_corners[:, 2]) for a_P_img in P_imgs])
    a_pt_in_img = np.moveaxis(np.moveaxis(a_pt_in_img.reshape([a_pt_in_img.shape[0], 2, z_steps, 4]), 1, 2), 2, 3)

    out_corners = np.array([[0, 0],
                            [0, out_img_size],
                            [out_img_size, 0],
                            [out_img_size, out_img_size]])

    # print(a_pt_in_img.shape, out_corners.shape)
    # exit()
    all_Hs = get_H_in_parrellel(a_pt_in_img, out_corners)
    all_Hs = t.tensor(all_Hs).to(device)
    all_sub_imgs = apply_H_tool_tensor.apply_H_in_parrellel(all_Hs)
    return all_sub_imgs

class SC_builder():
    def __init__(self, loss_gen):
        self.loss_f = loss_gen
        test_img = t.rand([1,1,25,25,1])
        test_img = t.cat([test_img, 1-test_img], 1)
        output = self.loss_f.apply_tensor(test_img)
        self.ideal_val = output[0,0,0].item()
        self.k = -1 if output[0,0,0].item() < output[0,0,1].item() else 1
        print(f"Ideal Value: {self.ideal_val}")
        print(f"K Value: {self.k}")

    def _get_Scores(self, P_imgs, model_size, out_img_size, show_process=True, device = None):
        if device is None:
            device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        all_set_scores = np.zeros([model_size[0], model_size[1], model_size[2]])
        z_steps = model_size[2]
        x_list, y_list = np.linspace(-1, 1, model_size[0] + 1), np.linspace(-1, 1, model_size[1] + 1)
        imgs = t.zeros([len(P_imgs), P_imgs[0].img.shape[0], P_imgs[0].img.shape[1], P_imgs[0].img.shape[2]],
                       requires_grad=False)
        for i in range(len(P_imgs)):
            imgs[i] = t.tensor(P_imgs[i].img)
        # # print("Image Size", out_img_size)
        apply_H_tool = apply_H_in_p_tensor(imgs, [out_img_size, out_img_size], z_steps, device, store_imgs_on_GPU=False)
        for x_idx in tqdm(range(x_list.shape[0] - 1)):
            x0, x1 = x_list[x_idx], x_list[x_idx + 1]
            for y_idx in range(y_list.shape[0] - 1):
                y0, y1 = y_list[y_idx], y_list[y_idx + 1]
                # all_scores[x_idx, y_idx] = get_height_scores_torch(P_imgs, z_steps, x0, x1, y0,
                #                                                    y1, out_img_size, apply_H_tool,
                #                                                    use_shadow_glare=shadow_glare_prediction,
                #                                                    discard=0, device=device)
                all_sub_imgs = t.transpose(get_sub_imgs(device, x0, y0, y1, x1, P_imgs, z_steps, out_img_size, apply_H_tool), 0, 1)
                all_scores = self.loss_f.apply_tensor(all_sub_imgs)
                # print(all_scores.shape)
                set_score = (t.sum(all_scores, (1,2)) - all_scores.shape[1]*self.ideal_val)/(all_scores.shape[1] * (all_scores.shape[1]-1))
                # print(set_score)
                #
                # all_scores = t.sum(all_scores, 2)
                # all_scores -= 1
                # discard = 0
                # all_scores_adj = t.sum(all_scores, 1) / (
                #         (all_scores.shape[1]) * (all_scores.shape[1]  - 1))
                # print(all_scores_adj)
                # exit()
                all_set_scores[x_idx, y_idx] = self.k * set_score.cpu()

            if show_process:
                summary_img = np.array(np.argmax(all_set_scores, 2) / z_steps * 255, dtype=np.uint8)
                cv.imshow("TEST", summary_img)
                cv.waitKey(1)
            # if x_idx == 0:
            #     print("Image Size", out_img_size)
        if show_process:
            cv.imshow("TEST", summary_img)
            cv.waitKey(-1)
            cv.destroyAllWindows()
        return all_set_scores

    def run_SC(self, P_imgs, bounds_LLA, device, voxel_size, validation_size=3, return_verbose = False):
        if validation_size == 0:
            validation_size = -len(P_imgs)
        model_size = get_model_size(bounds_LLA, voxel_size)
        print(len(P_imgs[0:-validation_size]), len(P_imgs))
        print(model_size)

        xy_list = np.linspace(-1., 1., max(model_size[0:2]) + 1)
        largest_dist = get_out_img_size(xy_list, P_imgs)
        print("Sub Image Size:", largest_dist)
        all_Scores = self._get_Scores(P_imgs[0:-validation_size], model_size, largest_dist, show_process=False, device=device)

        if return_verbose:
            return all_Scores, model_size, largest_dist, len(P_imgs[0:-validation_size])
        else:
            return all_Scores