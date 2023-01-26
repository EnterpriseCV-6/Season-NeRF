import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import exists
import pickle
from pre_NeRF.mg_time import mg_time
import cv2 as cv


class basic_NeRF_info():
    def __init__(self, img_name, camera_name, valid_img_pts, valid_img_pts_col, valid_world_pts_top, valid_world_pts_bot,
                 solar_vec, img_size, time_info, img_weight = 1.0):
        self.img_name = img_name
        self.camera_name = camera_name
        self.valid_img_pts = valid_img_pts
        self.valid_img_pts_col = valid_img_pts_col
        self.valid_world_pts_top = valid_world_pts_top
        self.valid_world_pts_bot = valid_world_pts_bot
        self.view_angles_vec = self._get_view_angles()
        self.solar_vec = solar_vec
        self.img_size = img_size
        self.time_info = time_info
        self.img_weight = img_weight
        # self.clear_DSM_info()

    def _get_view_angles(self):
        v_angle = self.valid_world_pts_bot - self.valid_world_pts_top
        v_angle = v_angle / np.sqrt(np.sum(v_angle**2,1, keepdims=True))
        return v_angle

    # def clear_DSM_info(self):
    #     self.DSM_info_GT = -1*np.ones(self.valid_world_pts_top.shape[0])
    #     self.DSM_info_Training = -1 * np.ones(self.valid_world_pts_top.shape[0])

class basic_Ortho_info(basic_NeRF_info):
    def __init__(self, img_shape):
        print(img_shape)
        img_name = "Ortho"
        camera_name = "Ortho"
        XY = np.stack([np.repeat(np.linspace(-1, 1, img_shape[0]), (img_shape[1])),
                       np.tile(np.linspace(-1, 1, img_shape[1]), img_shape[0])], -1)
        XY_loc = np.stack([np.repeat(np.arange(img_shape[0]), (img_shape[1])),
                       np.tile(np.arange(img_shape[1]), img_shape[0])], -1)
        # XY_loc = (XY + 1)/2
        # XY_loc[:,0] *= (img_shape[0]-1)
        # XY_loc[:,1] *= (img_shape[1]-1)
        # XY_loc = XY_loc.astype(int)
        Z_top = np.ones([XY.shape[0],1])
        Z_bot = -np.ones([XY.shape[0],1])
        valid_world_pts_top = np.concatenate([XY, Z_top],1)
        valid_world_pts_bot = np.concatenate([XY, Z_bot],1)
        solar_vec = np.array([0,0,1.])
        print("WARNING: RANDOM TIME BEING USED FOR ORTHO IMAGES! 4981")
        super(basic_Ortho_info, self).__init__(img_name, camera_name, XY_loc, np.ones([XY.shape[0], img_shape[2]]),
                                               valid_world_pts_top, valid_world_pts_bot, solar_vec, img_shape,
                                               mg_time("2014-10-05T16:01:38.873575Z"), 1.0)
        # exit()

def weight_Xs(pts, starts, ends, circular):
    all_scores = np.zeros_like(pts)
    for i in range(pts.shape[1]):
        sorted_ord = np.argsort(pts[:,i])
        # if np.unique(pts[:,i]).shape[0] != pts[:,i].shape[0]:
        #     print("Warning: Points overlap!")
        dist = np.zeros(pts.shape[0])
        for j in range(1, pts.shape[0]-1):
            dist[sorted_ord[j]] = ((pts[sorted_ord[j],i] - pts[sorted_ord[j-1],i]) + (pts[sorted_ord[j+1],i] - pts[sorted_ord[j],i])) /2
        if circular[i]:
            circ_sep = (pts[sorted_ord[0], i] - starts[i]) + (ends[i] - pts[sorted_ord[-1], i])
            dist[sorted_ord[0]] = circ_sep/2 + ((pts[sorted_ord[1], i] - pts[sorted_ord[0], i])) / 2
            dist[sorted_ord[-1]] = circ_sep / 2 + ((pts[sorted_ord[-1], i] - pts[sorted_ord[-2], i])) / 2
        else:
            dist[sorted_ord[0]] = (pts[sorted_ord[0],i] - starts[i]) + ((pts[sorted_ord[1],i] - pts[sorted_ord[0],i])) /2
            dist[sorted_ord[-1]] = (ends[i] - pts[sorted_ord[-1], i]) + ((pts[sorted_ord[-1], i] - pts[sorted_ord[-2], i])) / 2
        for j in np.unique(pts[:,i]):
            dist[pts[:,i] == j] = np.mean(dist[pts[:,i] == j])
        scores = dist / np.sum(dist) * pts.shape[0]
        all_scores[:,i] = scores
    return np.mean(all_scores, 1), all_scores

def weight_Xs_v2(X, starts, ends, circle, sigma = None):
    pair_dist = np.zeros([X.shape[0], X.shape[0], X.shape[1]])
    for j in range(X.shape[1]):
        if circle[j] == False:
            for i in range(X.shape[0]):
                for k in range(X.shape[0]):
                    pair_dist[i,k,j] = np.abs(X[i,j]-X[k,j])
        else:
            for i in range(X.shape[0]):
                for k in range(X.shape[0]):
                    D1 = np.abs(X[i, j] - X[k, j])
                    D0 = np.abs(X[i,j] - starts[j]) + np.abs(X[k,j] - ends[j])
                    D2 = np.abs(X[i,j] - ends[j]) + np.abs(X[k,j] - starts[j])
                    pair_dist[i, k, j] = np.min([D0, D1, D2])
        if sigma is None:
            est_sigma = np.std(pair_dist[:,:,j])
        else:
            est_sigma = sigma[j]
        pair_dist[:,:,j] /= est_sigma

    dists = np.sum(pair_dist ** 2, -1)
    X_star = 1/np.sum(np.exp(-dists), 1)
    X_star = X_star / np.max(X_star)
    X_star = X_star / np.sum(X_star) * X_star.shape[0]
    return X_star



def setup_quick_loader(args, bounds_model = np.array([[-1.,1], [-1,1], [-1,1]])):
    camera_name = args.camera_model
    cache_path = args.cache_dir
    if args.use_Bundle_Adjust == True:
        camera_name += "_Refined"

    fin = open(args.cache_dir + "/P_imgs_" + camera_name + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()


    if args.weight_training_samples == False:
        img_weights = np.ones(len(P_imgs))
    else:
        X = np.array(
            [[a_P_img.off_Nadir_from_IMD, a_P_img.Azmuth_from_IMD, a_P_img.time_obj.get_time_frac()[1]] for a_P_img in
             P_imgs])
        starts = np.array([0, 0, 0])
        ends = np.array([min(np.max(X[:, 0]) + 5, 180), 360, 1])
        circle = np.array([False, True, True])
        img_weights = weight_Xs_v2(X, starts, ends, circle, sigma=None) #sigma_man: np.array([2, 30, .075])


    training_names = []
    validation_names = []

    if args.testing_image_names is None:
        val_idx = np.linspace(0,len(P_imgs)-1, args.testing_size, dtype=int)
    else:
        print("Using file", args.testing_image_names, "to get testing images.")
        fin = open(args.testing_image_names, "r")
        testing_names = fin.readlines()
        fin.close()
        testing_names = np.array([a_name.strip() for a_name in testing_names])
        print("Testing Region Names:")
        print(testing_names)
        val_idx = []
        valid_names = np.array([a_P_img.img_name for a_P_img in P_imgs])
        all_good = True
        for a_testing_name in testing_names:
            matches = np.where(a_testing_name == valid_names)[0]
            if matches.shape[0] == 0:
                all_good = False
                print("Unable to find match for testing name", a_testing_name)
            elif matches.shape[0] > 1:
                all_good = False
                print("Multiple matches for testing name", a_testing_name)
            else:
                val_idx.append(matches[0])
        val_idx = np.array(val_idx)
        if all_good == False:
            print("Problem with loading testing file!")
            print("Please check arg testing_image_names.")
            exit()

    idx = 0

    train_file = open(args.logs_dir + "/Training_Imgs.txt", "w")
    test_file = open(args.logs_dir + "/Testing_Imgs.txt", "w")

    # DS = args.img_downscale
    DS = 1

    print("Setting Up Info.")
    for a_P_img in tqdm(P_imgs):
        DS = args.img_validation_downscale if idx in val_idx else args.img_training_downscale
        cache_file_name = a_P_img.img_name + "_" + camera_name + "_Basic_Info_DS_" + str(DS) + ".pickle"
        if exists(cache_path + "/" + cache_file_name) == False:
            a_P_img.img = a_P_img.img
            img_shape = np.array(a_P_img.img.shape) // DS
            img_shape[2] = a_P_img.img.shape[2]
            XY = np.stack([np.repeat(np.arange(0, img_shape[0]), (img_shape[1])), np.tile(np.arange(0, img_shape[1]), img_shape[0])], -1)
            Z_top = np.ones([XY.shape[0]]) * bounds_model[2, 1]
            Z_bot = np.ones([XY.shape[0]]) * bounds_model[2, 0]
            tops = np.stack(a_P_img.invert_P(XY[:,0] * DS, XY[:,1] * DS, Z_top))
            bots = np.stack(a_P_img.invert_P(XY[:,0] * DS, XY[:,1] * DS, Z_bot))

            good = (tops[0] <= bounds_model[0, 1]) * (bounds_model[0, 0] <= tops[0]) * \
                   (tops[1] <= bounds_model[1, 1]) * (bounds_model[1, 0] <= tops[1]) * \
                   (bots[0] <= bounds_model[0, 1]) * (bounds_model[0, 0] <= bots[0]) * \
                   (bots[1] <= bounds_model[1, 1]) * (bounds_model[1, 0] <= bots[1])

            valid_img_pts = XY[good]
            valid_img_pts_cols = a_P_img.img[valid_img_pts[:,0] * DS, valid_img_pts[:,1] * DS]
            valid_world_pts_top = tops.T[good]
            valid_world_pts_bot = bots.T[good]

            img_info = basic_NeRF_info(a_P_img.img_name, camera_name, valid_img_pts, valid_img_pts_cols, valid_world_pts_top,
                            valid_world_pts_bot, a_P_img.sun_el_and_az_vec, img_shape, a_P_img.time_obj, img_weights[idx])
            fout = open(cache_path + "/" + cache_file_name, "wb")
            pickle.dump(img_info, fout)
            fout.close()
        if idx in val_idx:
            test_file.write(a_P_img.img_name + "\n")
        else:
            train_file.write(a_P_img.img_name + "\n")
        idx += 1
    train_file.close()
    test_file.close()


