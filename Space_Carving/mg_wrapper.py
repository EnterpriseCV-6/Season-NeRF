from .mg_Img_metric import mg_ssim_v0
from .mg_SC_multi import SC_builder
import numpy as np
from os.path import exists
import pickle
from .mg_3d_to_dist import energy_min_H_map

def SC(P_imgs, bounds, device):
    SC_tool = SC_builder(mg_ssim_v0())
    scores = SC_tool.run_SC(P_imgs, bounds, device, np.array([2.0,2.0,.25]), validation_size=0)
    return scores

def get_DSM_SC(args, device, force_build = False):
    print("Checking Cache for Space Carved DSM...")
    file_name = args.cache_dir + "/SC_" + args.site_name + "_hm" + ".npy"
    if exists(file_name) and force_build == False:
        hm = np.load(file_name)
    else:
        filler = "_Refined" if args.use_Bundle_Adjust else ""
        fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + filler + ".pickle", "rb")
        all_P_imgs = pickle.load(fin)
        fin.close()
        training_P_imgs = []

        fin = open(args.logs_dir + "/Training_Imgs.txt", "r")
        name_list = fin.read().split("\n")
        fin.close()

        bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + filler + ".npy")

        for a_P_img in all_P_imgs:
            if a_P_img.img_name in name_list:
                training_P_imgs.append(a_P_img)
        score_map = SC(training_P_imgs, bounds_LLA, device)
        np.save(args.cache_dir + "/Full_Scores_" + args.camera_model + filler + ".npy", score_map)
        hm = energy_min_H_map(score_map, start=0, end=-1, h=1 / 3) * 2 - 1
        np.save(file_name, hm)

    return hm