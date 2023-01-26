# coding=utf-8

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib import gridspec
import datetime

import Generate_Summary_Images
from .mg_Img_Eval import component_render_by_dir, get_imgs_from_Img_Dict, show_dict_struc
from all_NeRF import mg_EM_Imgs, CV_reshape

from tabulate import tabulate
import pickle

def _show_Season_Output(Season_dict, output_path = None):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    Sat_el_az = Season_dict["Input_Vals"]["Idx_1_sat_angle"]
    Sun_el_az = Season_dict["Input_Vals"]["Idx_2_sun_angle"]
    times = Season_dict["Input_Vals"]["Idx_3_Time_Frac"]

    ncol = times.shape[0] + 1
    nrow = Sun_el_az.shape[0]
    # print(Sat_el_az)
    # print(Sun_el_az)
    # print(times)
    # print(Season_dict["Array_of_Img_dict"].shape)

    # fig, ax = plt.subplots(ncols=walking_times.shape[0], nrows=nrow, figsize=(ncol+1,nrow+1))
    date_1 = datetime.datetime.strptime("01/01/21", "%m/%d/%y")
    for i in range(Sat_el_az.shape[0]):
        r = 0
        fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max(100, Season_dict["Array_of_Img_dict"][0, 0, 0]["Season_Adj_Img"].shape[0]))
        gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                               top=1. - 0.5 / (nrow + 1),
                               bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1),
                               right=1 - 0.5 / (ncol + 1))
        fig.suptitle("View El. and Az: " + str(Sat_el_az[i]))

        for j in range(Sun_el_az.shape[0]):
            c = 0
            for k in range(times.shape[0]):
                # print(i,j,k)
                # print(walking_views[r], a_walking_shadow, a_waking_time)
                ax = plt.subplot(gs[r, c])
                out_img = Season_dict["Array_of_Img_dict"][i, j, k]["Season_Adj_Img"] * Season_dict["Array_of_Img_dict"][i, j, k]["Shadow_Adjust"]
                ax.imshow(out_img)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    end_date = date_1 + datetime.timedelta(days=365.24 * times[k])
                    ax.set_title(months[end_date.month - 1] + ". " + str(end_date.day))
                if c == 0:
                    # the_lab = "(" + str(np.round(90-walking_views[r][0])) + ", " + str(np.round(walking_views[r][1])) + ")\n(" + str(np.round(a_walking_shadow[0])) + ", " + str(np.round(a_walking_shadow[1])) + ")"
                    the_lab = str(r + 1)
                    ax.set_ylabel(the_lab, rotation="horizontal", labelpad=5)  # , rotation=0, labelpad=23)
                c += 1
            ax = plt.subplot(gs[r, c])
            ax.imshow(Season_dict["Array_of_Img_dict"][i, j, 0]["Shadow_Mask"], vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("Shadow Mask")
            r += 1
        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path + "/Walking_" + str(i+1) + ".png")
            plt.close("all")

def Full_Eval_Seasons(a_t_nerf, P_imgs, device, out_img_size, n_walking_sun_angles, n_walking_view_angles, n_walking_times, use_exact_solar = False, use_classic_shadows = False):
    walk_view, walk_sun, walking_times = Generate_Summary_Images.get_walking_Points(P_imgs, n_walking_view_angles, n_walking_sun_angles, n_walking_times, min_day_sep=20)

    n_walking_view_angles = walk_view.shape[0]
    n_walking_sun_angles = walk_sun.shape[0]
    n_walking_times = walking_times.shape[0]

    walk_view = np.array(walk_view)
    walk_sun = np.array(walk_sun)
    walking_times = np.array(walking_times)

    output = np.empty([n_walking_view_angles, n_walking_sun_angles, n_walking_times], dtype=dict)
    Input_vals = {"Idx_1_sat_angle":walk_view, "Idx_2_sun_angle":walk_sun, "Idx_3_Time_Frac":walking_times}

    with tqdm(total=n_walking_times*n_walking_sun_angles*n_walking_view_angles, leave=False, desc="Seasonal Walk") as pbar:
        for i in range(n_walking_view_angles):
            for j in range(n_walking_sun_angles):
                for k in range(n_walking_times):
                    raw_data = component_render_by_dir(a_t_nerf, walk_view[i], walk_sun[j], walking_times[k], out_img_size, W2C=P_imgs[0].get_world_center(), W2L_H=P_imgs[0].S, include_exact_solar = use_exact_solar, device=device)
                    imgs = get_imgs_from_Img_Dict(raw_data, out_img_size, use_classic_shadows)
                    output[i,j,k] = {"Season_Adj_Img":imgs["Season_Adj_Img"], "Shadow_Adjust":imgs["Shadow_Adjust"], "Shadow_Mask":imgs["Shadow_Mask"], "Time_Class":imgs["Time_Class"]}
                    pbar.update(1)

    ans = {"Input_Vals":Input_vals, "Array_of_Img_dict":output}
    return ans

def _show_Season_walk_overview(Season_Summary, P_imgs, training_idx, proto_idx, output_loc):

    testing_idx = []
    for i in range(len(P_imgs)):
        if i not in training_idx:
            testing_idx.append(i)
    walking_sat_angle = Season_Summary["Input_Vals"]["Idx_1_sat_angle"]
    walking_sun_angle = Season_Summary["Input_Vals"]["Idx_2_sun_angle"]
    walking_times = Season_Summary["Input_Vals"]["Idx_3_Time_Frac"]

    Generate_Summary_Images.gen_angle_images(P_imgs, testing_idx, walking_sat_angle, walking_sun_angle, annotate_pts=True, output_path=output_loc + "/Walking_Views.png")
    Generate_Summary_Images.show_proto_images(P_imgs, training_idx, testing_idx, proto_idx, walking_times, output_path=output_loc + "/Walking_Times.png")

def _Season_Stability_Tests(Season_Summary, P_imgs, proto_idx, outloc):
    base_img_size = Season_Summary["Array_of_Img_dict"][0,0,0]["Season_Adj_Img"].shape

    baseline_dists = np.zeros([proto_idx.shape[0], proto_idx.shape[0]]) + np.NaN
    plt.figure(figsize=((proto_idx.shape[0]+1)*4,5))
    for i in range(proto_idx.shape[0]):
        a_img = CV_reshape(P_imgs[proto_idx[i]].img, (base_img_size[0], base_img_size[1]))
        plt.subplot(1, proto_idx.shape[0]+1, i+1)
        plt.imshow(a_img)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(i))

        Img1 = mg_EM_Imgs.mg_EM(a_img)
        for j in range(i+1, proto_idx.shape[0]):
            Img2 = mg_EM_Imgs.mg_EM(CV_reshape(P_imgs[proto_idx[j]].img, (base_img_size[0], base_img_size[1])))
            dist, _, _ = mg_EM_Imgs.compare_EM_imgs(Img1, Img2)
            baseline_dists[i,j] = dist
            baseline_dists[j,i] = dist
    plt.subplot(1, proto_idx.shape[0]+1, proto_idx.shape[0]+1)
    plt.matshow(baseline_dists, fignum=0)
    plt.colorbar()
    plt.title("Image Distances")
    plt.xlabel("Most Similar Image Score: " + str(np.round(np.nanmin(baseline_dists), 3)))
    plt.tight_layout()
    plt.savefig(outloc + "/Prototypical_EM.png")
    plt.close("all")

    walking_sat_angle = Season_Summary["Input_Vals"]["Idx_1_sat_angle"]
    walking_sun_angle = Season_Summary["Input_Vals"]["Idx_2_sun_angle"]
    walking_times = Season_Summary["Input_Vals"]["Idx_3_Time_Frac"]

    all_EM_dists = []
    for i in tqdm(range(walking_times.shape[0]), desc="Stepping through times", leave=False):
        EM_img_array = np.empty([walking_sat_angle.shape[0], walking_sun_angle.shape[0]], dtype=mg_EM_Imgs.mg_EM)
        with tqdm(total=walking_sat_angle.shape[0] * walking_sun_angle.shape[0], leave=False,
                  desc="Getting Sigs") as pub:
            for j in range(walking_sat_angle.shape[0]):
                for k in range(walking_sun_angle.shape[0]):
                    EM_img_array[j,k] = mg_EM_Imgs.mg_EM(Season_Summary["Array_of_Img_dict"][j,k,i]["Season_Adj_Img"] * Season_Summary["Array_of_Img_dict"][j,k,i]["Shadow_Adjust"])
                    pub.update(1)
        EM_dists = np.zeros([walking_sat_angle.shape[0], walking_sun_angle.shape[0], walking_sat_angle.shape[0], walking_sun_angle.shape[0]]) + np.NaN
        with tqdm(total=walking_sat_angle.shape[0]**2 * walking_sun_angle.shape[0]**2, leave=False, desc="Computing EM") as pub:
            for j in range(walking_sat_angle.shape[0]):
                for k in range(walking_sun_angle.shape[0]):
                    for j2 in range(walking_sat_angle.shape[0]):
                        for k2 in range(walking_sun_angle.shape[0]):
                            a_dist,_,_ = mg_EM_Imgs.compare_EM_imgs(EM_img_array[j,k], EM_img_array[j2, k2])
                            EM_dists[j,k,j2,k2] = a_dist
                            pub.update(1)
        all_EM_dists.append(EM_dists)
        least_sim_imgs = np.unravel_index(np.argsort(np.ravel(EM_dists))[(np.prod(EM_dists.shape) - EM_dists.shape[0] * EM_dists.shape[1])//2], EM_dists.shape)
        plt.figure(figsize=(5,9))
        plt.subplot(3, 1, 2)
        bad1 = Season_Summary["Array_of_Img_dict"][least_sim_imgs[0], least_sim_imgs[1], i]["Season_Adj_Img"] * \
               Season_Summary["Array_of_Img_dict"][least_sim_imgs[0], least_sim_imgs[1], i]["Shadow_Adjust"]
        bad2 = Season_Summary["Array_of_Img_dict"][least_sim_imgs[2], least_sim_imgs[3], i]["Season_Adj_Img"] * \
               Season_Summary["Array_of_Img_dict"][least_sim_imgs[2], least_sim_imgs[3], i]["Shadow_Adjust"]
        bad = np.concatenate([bad1, np.ones([bad1.shape[0], bad1.shape[1] // 4, 3]), bad2], 1)
        plt.imshow(bad)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("EM Dist: " + str(
            np.round(EM_dists[least_sim_imgs[0], least_sim_imgs[1], least_sim_imgs[2], least_sim_imgs[3]], 3)))
        plt.ylabel("Median Similar Image Pair")

        least_sim_imgs = np.unravel_index(np.nanargmax(EM_dists), EM_dists.shape)
        plt.subplot(3,1,3)
        bad1 = Season_Summary["Array_of_Img_dict"][least_sim_imgs[0],least_sim_imgs[1],i]["Season_Adj_Img"] * Season_Summary["Array_of_Img_dict"][least_sim_imgs[0],least_sim_imgs[1],i]["Shadow_Adjust"]
        bad2 = Season_Summary["Array_of_Img_dict"][least_sim_imgs[2], least_sim_imgs[3], i]["Season_Adj_Img"] * \
               Season_Summary["Array_of_Img_dict"][least_sim_imgs[2], least_sim_imgs[3], i]["Shadow_Adjust"]
        bad = np.concatenate([bad1, np.ones([bad1.shape[0], bad1.shape[1]//4, 3]), bad2], 1)
        plt.imshow(bad)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("EM Dist: "+ str(np.round(EM_dists[least_sim_imgs[0], least_sim_imgs[1], least_sim_imgs[2], least_sim_imgs[3]], 3)))
        plt.ylabel("Least Similar Image Pair")

        plt.subplot(3,1,1)
        plt.hist(np.sort(np.ravel(EM_dists[EM_dists == EM_dists])))
        plt.xlabel("EM Dist Histogram")
        plt.title("Time: " + str(np.round(walking_times[i], 3)))
        plt.tight_layout()
        plt.savefig(outloc + "/EM_Dist_T" + str(i+1) + ".png")
        plt.close("all")
    all_EM_dists = np.array(all_EM_dists)
    return baseline_dists, all_EM_dists

def Orgainize_Output_Seasons(Season_Summary, P_imgs, training_idx, proto_idx, output_loc):
    _show_Season_Output(Season_Summary, output_loc)
    _show_Season_walk_overview(Season_Summary, P_imgs, training_idx, proto_idx, output_loc)
    baseline_dists, all_EM_dists = _Season_Stability_Tests(Season_Summary, P_imgs, proto_idx, output_loc)
    vals = all_EM_dists.reshape([all_EM_dists.shape[0],all_EM_dists.shape[1]*all_EM_dists.shape[2],all_EM_dists.shape[3] * all_EM_dists.shape[4]])
    vals = vals[vals == vals]

    Col_Labels = ["Case", "Mean EM", "Median EM", "95% Below EM", "Max EM"]
    Data = []
    Data.append(["Overall", np.round(np.mean(vals), 3), np.round(np.median(vals), 3), np.round(np.quantile(vals, .95), 3), np.round(np.max(vals), 3)])
    for i in range(all_EM_dists.shape[0]):
        vals = all_EM_dists[i]
        Data.append(
            [i+1, np.round(np.nanmean(vals), 3), np.round(np.nanmedian(vals), 3), np.round(np.nanquantile(vals, .95), 3),
             np.round(np.nanmax(vals), 3)])

    fout = open(output_loc + "/Stablility_sum.txt", "w")
    fout.write("Baseline:\n")
    [fout.write(str(np.round(baseline_dists[i], 3)) + "\n") for i in range(baseline_dists.shape[0])]
    fout.write("Min Baseline Score: " + str(np.round(np.nanmin(baseline_dists), 3)))
    fout.write("\nTime Data:\n")
    fout.write(tabulate(Data, Col_Labels))
    fout.close()

    fout = open(output_loc + "/Stability_sum_data.pickle", "wb")
    pickle.dump({"Baseline":baseline_dists, "Scores":all_EM_dists}, fout)
    fout.close()
