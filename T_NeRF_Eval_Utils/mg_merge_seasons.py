# coding=utf-8
import numpy as np
from tabulate import tabulate
import pickle
import os
from all_NeRF import show_dict_struc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import T_NeRF_Full_2
from .mg_Img_Eval import component_render_by_dir, get_imgs_from_Img_Dict_t_step
import torch as t
from .Eval_funcs import mask_PSNR


def merge_seasons(valid_areas, output_path, round_deg = 3):
    data = []
    for a_valid_area in valid_areas:
        if os.path.exists(a_valid_area + "/Stability_sum_data.pickle") and os.path.exists(a_valid_area + "/Season_Summary.pickle"):
            area_name = a_valid_area.split("/")[-2].replace("_", " ")

            fin = open(a_valid_area + "/Stability_sum_data.pickle", "rb")
            Season_Summary = pickle.load(fin)
            fin.close()

            SSS = Season_Summary["Scores"].reshape([Season_Summary["Scores"].shape[0],
                                                    Season_Summary["Scores"].shape[1] * Season_Summary["Scores"].shape[
                                                        2],
                                                    Season_Summary["Scores"].shape[3] * Season_Summary["Scores"].shape[
                                                        4]])
            for i in range(SSS.shape[1]):
                SSS[:, i, i] = np.NaN
            SSS = SSS.reshape(Season_Summary["Scores"].shape)

            # show_dict_struc(Season_Summary)
            Most_different_Image_at_same_time_idx = np.unravel_index(np.nanargmax(SSS), SSS.shape)
            Most_different_Image_at_same_time_score = SSS[Most_different_Image_at_same_time_idx[0], Most_different_Image_at_same_time_idx[1], Most_different_Image_at_same_time_idx[2], Most_different_Image_at_same_time_idx[3], Most_different_Image_at_same_time_idx[4]]

            mean_EM = np.nanmean(SSS)
            most_normal_img_idx = np.unravel_index(np.nanargmin(np.abs(SSS - mean_EM)), SSS.shape)

            unraveled_Scores = np.ravel(SSS[SSS == SSS])

            fin = open(a_valid_area + "/Season_Summary.pickle", "rb")
            Season_Imgs = pickle.load(fin)
            fin.close()
            a,b,c,d,e = Most_different_Image_at_same_time_idx
            img1 = Season_Imgs["Array_of_Img_dict"][b, c, a]["Season_Adj_Img"] * Season_Imgs["Array_of_Img_dict"][b, c, a]["Shadow_Adjust"]
            img2 = Season_Imgs["Array_of_Img_dict"][d, e, a]["Season_Adj_Img"] * Season_Imgs["Array_of_Img_dict"][d, e, a]["Shadow_Adjust"]
            most_different_imgs = [img1, img2]

            a, b, c, d, e = most_normal_img_idx
            img1 = Season_Imgs["Array_of_Img_dict"][b, c, a]["Season_Adj_Img"] * Season_Imgs["Array_of_Img_dict"][b, c, a]["Shadow_Adjust"]
            img2 = Season_Imgs["Array_of_Img_dict"][d, e, a]["Season_Adj_Img"] * Season_Imgs["Array_of_Img_dict"][d, e, a]["Shadow_Adjust"]
            avg_imgs = [img1, img2]

            data.append([area_name, Most_different_Image_at_same_time_score, mean_EM,
                         np.unique(Season_Summary["Baseline"][Season_Summary["Baseline"] == Season_Summary["Baseline"]]),
                         unraveled_Scores, most_different_imgs, avg_imgs])

    col_labels = ["Region", "Median", "95% Quantile", "Max", "Prototypical Min"]
    write_this = [["\\text{Average}", 0, 0, 0, 0]]
    write_this2 = [["Average", 0, 0, 0, 0]]
    for i in range(len(data)):
        # print(data[i][0:4])
        write_this.append(["\\text{" + data[i][0] + "}", np.median(data[i][4]), np.quantile(data[i][4], .95), np.max(data[i][4]), np.min(data[i][3])])
        write_this2.append([data[i][0], np.median(data[i][4]), np.quantile(data[i][4], .95), np.max(data[i][4]), np.min(data[i][3])])
        for j in range(4):
            write_this[0][1+j] += write_this[-1][1+j]/len(data)
            write_this2[-1][1+j] = np.round(write_this2[-1][1+j], round_deg)
    for i in range(4):
        write_this2[0][1 + i] = np.round(write_this[0][1 + i], round_deg)

    with open(output_path + "/Seasonal_Stability.txt", "w") as fout:
        fout.write(tabulate(write_this2, col_labels))
        fout.write("\n\n\n")
        np.savetxt(fout, write_this, delimiter=" & ", newline=" \\\\\n", fmt="%s")
        fout.close()

    fig = plt.figure(figsize=(4+2, 2+len(data)+4))
    all_data = []
    all_proto_data = []
    area_names = []
    worst = 0
    labels = ["Worst 1", "Worst 2", "Avg. 1", "Avg. 2"]
    for i in range(len(data)):
        imgs = [data[i][5][0], data[i][5][1], data[i][6][0], data[i][6][1]]
        all_data.append(data[i][4])
        all_proto_data += list(data[i][3])
        worst = max(worst, np.max(all_data[-1]))
        for j in range(4):
            an_ax = plt.subplot2grid(shape=(2+len(data)+1, 4), loc=(3+i, j))
            an_ax.imshow(imgs[j])
            an_ax.set_xticks([])
            an_ax.set_yticks([])
            if j == 0:
                an_ax.set_ylabel(data[i][0][0:7])
                area_names.append(data[i][0][0:7])
            if i == 0:
                an_ax.set_title(labels[j])


    bins = list(np.linspace(0, 2*worst, 30, endpoint=False)) + [max(np.max(all_proto_data), 2*worst)+1]
    counts, bin_vals = np.histogram(all_proto_data, bins)
    counts = counts / np.sum(counts)
    bin_vals[-1] = 2*worst
    ax1 = plt.subplot2grid(shape=(2 + len(data)+1, 4), loc=(0, 0), colspan=4, rowspan=2)
    ax1.hist(all_data, histtype="bar", stacked=True, density=True, bins=bin_vals, align="left")
    vals = ax1.get_xticks() #+ [2 * worst]
    vals = vals[1::]
    vals[-1] = 2*worst
    ax1.hist(bin_vals[:-1], bin_vals, weights=counts, align="right")
    ax1.legend(area_names + ["Proto Image"])
    VALS = [str(int(np.round(i))) for i in vals]
    VALS[-1] += "+"
    ax1.set_xticks(vals, VALS)
    ax1.set_xlabel("EM Dist")
    ax1.set_ylabel("Dist")
    plt.suptitle("Seasonal Stability")
    plt.savefig(output_path + "/Seasonal_Stability.png")
    plt.close("all")



    # for a_valid_area in valid_areas:
    #     area_name = a_valid_area.split("/")[-2].replace("_", " ")
    #     if os.path.exists(a_valid_area + "/Season_Summary.pickle"):
    #         fin = open(a_valid_area + "/Season_Summary.pickle", "rb")
    #         Season_Summary = pickle.load(fin)
    #         fin.close()
    #
    #         # show_dict_struc(Season_Summary)
    #         X = Season_Summary["Input_Vals"]["Idx_1_sat_angle"].shape[0]
    #         Y = Season_Summary["Input_Vals"]["Idx_2_sun_angle"].shape[0]
    #         Local_Imgs = Season_Summary["Array_of_Img_dict"][X//2, Y//2]
    #         # print(Local_Imgs.shape)
    #         print(Season_Summary["Input_Vals"]["Idx_3_Time_Frac"])
    #         # show_dict_struc(Local_Imgs[0])
    #         # for i in range(Local_Imgs.shape[0]):
    #         #     plt.imshow(Local_Imgs[i]["Season_Adj_Img"] * Local_Imgs[i]["Shadow_Adjust"])
    #         #     plt.show()
    #         # exit()
    # exit()

def merge_season_walk(valid_areas, output_path, out_img_size = (128,128,96), max_time_steps = 180, interest_steps = 9, full_out_img_size = (512,512,96)):
    row_labels = []
    GT_imgs = []

    all_SA = []
    all_view_angles = []
    all_times = []
    all_time_steps = np.linspace(0, 1, max_time_steps, endpoint=False)
    all_time_steps_encoded = t.tensor(T_NeRF_Full_2.encode_time(all_time_steps, np.zeros_like(all_time_steps)).T).float()
    area_data = []

    for a_valid_area in valid_areas:
        if os.path.exists(a_valid_area + "/Input_Dict.pickle"):
            area_name = a_valid_area.split("/")[-2].replace("_", " ")
            # try:
            fin = open(a_valid_area + "/Input_Dict.pickle", "rb")
            input_dict = pickle.load(fin)
            fin.close()
            P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device = T_NeRF_Full_2.load_from_input_dict(input_dict)

            GT_imgs.append([])
            row_labels.append(area_name[0:7])
            for a_P_img in P_imgs:
                if not (a_P_img.img_name in testing_imgs):
                    name, sat_el_and_az, sun_el_and_az, year_frac = a_P_img.get_meta_data()
                    all_SA.append(tuple(sun_el_and_az))
                    all_view_angles.append(tuple(sat_el_and_az))
                    all_times.append(year_frac)

            with t.no_grad():
                out_dict = component_render_by_dir(a_t_nerf, [80, 0], [30,90], 0.0, out_img_size=out_img_size, W2C=image_builder.world_center_LLA, W2L_H=image_builder.W2L_H, device=device, include_exact_solar=False)
                out_class = a_t_nerf.get_class_only(all_time_steps_encoded.to(device)).cpu().numpy()

            area_data.append(get_imgs_from_Img_Dict_t_step(out_dict, out_img_size, out_class))

            # except:
            #     print("Problem with ", area_name)
            # if len(row_labels) > 2:
            #     break
    area_data = np.array(area_data)
    print(area_data.shape)

    massive_img = np.zeros([area_data.shape[0] * area_data.shape[2], area_data.shape[1] * area_data.shape[3], 3])
    for i in range(area_data.shape[0]):
        for j in range(area_data.shape[1]):
            massive_img[i*area_data.shape[2]:(i+1)*area_data.shape[2], j*area_data.shape[2]:(j+1)*area_data.shape[2]] = area_data[i,j]
    plt.imsave(output_path + "/Full_Seasonal_Images.png", massive_img)

    # all_SA = np.array(list(set(all_SA)))
    # all_view_angles = np.array(list(set(all_view_angles)))
    all_times = np.unique(all_times)


    time_diff = np.abs(np.expand_dims(all_time_steps, 1) - np.expand_dims(all_times, 0))
    time_diff[time_diff > .5] = 1.0 - time_diff[time_diff > .5]
    time_diff = np.min(time_diff, 1)
    valid_time = time_diff < 10./365.24
    # print(all_time_steps)
    all_time_steps = all_time_steps[valid_time]
    # print(all_time_steps)


    area_data = area_data[:, valid_time]
    massive_img = np.zeros([area_data.shape[0] * area_data.shape[2], area_data.shape[1] * area_data.shape[3], 3])
    for i in range(area_data.shape[0]):
        for j in range(area_data.shape[1]):
            massive_img[i * area_data.shape[2]:(i + 1) * area_data.shape[2],
            j * area_data.shape[2]:(j + 1) * area_data.shape[2]] = area_data[i, j]
    plt.imsave(output_path + "/Near_Training_Seasonal_Images.png", massive_img)

    scores = np.zeros([area_data.shape[0], area_data.shape[1]])
    for i in range(area_data.shape[0]):
        mask = area_data[i,0] == area_data[i,0]
        left_score = mask_PSNR(area_data[i, -1], area_data[i, 0], mask)
        right_score = mask_PSNR(area_data[i, 1], area_data[i, 0], mask)
        scores[i,0] = left_score + right_score
        for j in range(1,area_data.shape[1]-1):
            left_score = mask_PSNR(area_data[i,j-1], area_data[i,j], mask)
            right_score = mask_PSNR(area_data[i,j+1], area_data[i,j], mask)
            scores[i, j] = left_score + right_score
        left_score = mask_PSNR(area_data[i, -2], area_data[i, -1], mask)
        right_score = mask_PSNR(area_data[i, 0], area_data[i, -1], mask)
        scores[i, -1] = left_score + right_score

    scores = scores / np.sum(scores, 1, keepdims=True) * scores.shape[1]
    while scores.shape[1] > interest_steps:
        remove_idx = np.argmax(np.mean(scores, 0))
        all_time_steps = np.delete(all_time_steps, remove_idx)
        area_data = np.delete(area_data, remove_idx, 1)
        scores = np.delete(scores, remove_idx, 1)

        for i in range(area_data.shape[0]):
            mask = area_data[i, 0] == area_data[i, 0]
            left_score = mask_PSNR(area_data[i, -1], area_data[i, 0], mask)
            right_score = mask_PSNR(area_data[i, 1], area_data[i, 0], mask)
            scores[i, 0] = left_score + right_score
            for j in range(1, area_data.shape[1] - 1):
                left_score = mask_PSNR(area_data[i, j - 1], area_data[i, j], mask)
                right_score = mask_PSNR(area_data[i, j + 1], area_data[i, j], mask)
                scores[i, j] = left_score + right_score
            left_score = mask_PSNR(area_data[i, -2], area_data[i, -1], mask)
            right_score = mask_PSNR(area_data[i, 0], area_data[i, -1], mask)
            scores[i, -1] = left_score + right_score
        scores = scores / np.sum(scores, 1, keepdims=True) * scores.shape[1]
    print(all_time_steps)
    massive_img = np.zeros([area_data.shape[0] * area_data.shape[2], area_data.shape[1] * area_data.shape[3], 3])
    for i in range(area_data.shape[0]):
        for j in range(area_data.shape[1]):
            massive_img[i * area_data.shape[2]:(i + 1) * area_data.shape[2],
            j * area_data.shape[2]:(j + 1) * area_data.shape[2]] = area_data[i, j]
    plt.imsave(output_path + "/Low_Res_Interesting_Seasonal_Images.png", massive_img)
    start_point = np.argmax(np.concatenate([[all_time_steps[0]], all_time_steps[1::] - all_time_steps[0:-1]]))

    all_time_steps_encoded = t.tensor(T_NeRF_Full_2.encode_time(all_time_steps, np.zeros_like(all_time_steps)).T).float()
    area_data = []
    for a_valid_area in valid_areas:
        if os.path.exists(a_valid_area + "/Input_Dict.pickle"):
            # try:
            fin = open(a_valid_area + "/Input_Dict.pickle", "rb")
            input_dict = pickle.load(fin)
            fin.close()
            P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device = T_NeRF_Full_2.load_from_input_dict(
                input_dict)

            with t.no_grad():
                out_dict = component_render_by_dir(a_t_nerf, [80, 0], [30, 90], 0.0, out_img_size=full_out_img_size,
                                                   W2C=image_builder.world_center_LLA, W2L_H=image_builder.W2L_H,
                                                   device=device, include_exact_solar=False)
                out_class = a_t_nerf.get_class_only(all_time_steps_encoded.to(device)).cpu().numpy()

            area_data.append(get_imgs_from_Img_Dict_t_step(out_dict, full_out_img_size, out_class))
    area_data = np.array(area_data)

    massive_img = np.zeros([area_data.shape[0] * area_data.shape[2], area_data.shape[1] * area_data.shape[3], 3])
    for i in range(area_data.shape[0]):
        for j in range(area_data.shape[1]):
            J = (j + start_point) % area_data.shape[1]
            massive_img[i * area_data.shape[2]:(i + 1) * area_data.shape[2],j * area_data.shape[2]:(j + 1) * area_data.shape[2]] = area_data[i, J]

    plt.imsave(output_path + "/High_Res_Interesting_Seasonal_Images.png", massive_img)
    np.savetxt(output_path + "/season_times.txt", all_time_steps)


def merge_season_walk_split(valid_areas, output_path, out_img_size = (256,256,96), max_time_steps = 60):
    row_labels = []
    GT_imgs = []

    split_steps = max_time_steps // 4
    max_time_steps = split_steps * 4

    all_SA = []
    all_view_angles = []
    all_times = []
    all_time_steps = np.linspace(2./12., 2./12.+1, max_time_steps, endpoint=False)
    print(all_time_steps)
    all_time_steps_encoded = t.tensor(T_NeRF_Full_2.encode_time(all_time_steps, np.zeros_like(all_time_steps)).T).float()
    area_data = []

    for a_valid_area in valid_areas:
        if os.path.exists(a_valid_area + "/Input_Dict.pickle"):
            area_name = a_valid_area.split("/")[-2].replace("_", " ")
            # try:
            fin = open(a_valid_area + "/Input_Dict.pickle", "rb")
            input_dict = pickle.load(fin)
            fin.close()
            P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device = T_NeRF_Full_2.load_from_input_dict(input_dict)

            GT_imgs.append([])
            row_labels.append(area_name[0:7])
            for a_P_img in P_imgs:
                if not (a_P_img.img_name in testing_imgs):
                    name, sat_el_and_az, sun_el_and_az, year_frac = a_P_img.get_meta_data()
                    all_SA.append(tuple(sun_el_and_az))
                    all_view_angles.append(tuple(sat_el_and_az))
                    all_times.append(year_frac)

            with t.no_grad():
                out_dict = component_render_by_dir(a_t_nerf, [80, 0], [30,90], 0.0, out_img_size=out_img_size, W2C=image_builder.world_center_LLA, W2L_H=image_builder.W2L_H, device=device, include_exact_solar=False)
                out_class = a_t_nerf.get_class_only(all_time_steps_encoded.to(device)).cpu().numpy()

            area_data.append(get_imgs_from_Img_Dict_t_step(out_dict, out_img_size, out_class))

            # except:
            #     print("Problem with ", area_name)
            # if len(row_labels) > 2:
            #     break
    area_data = np.array(area_data)
    ncol = split_steps
    nrow = area_data.shape[0]

    region_types = ["Spring", "Summer", "Fall", "Winter"]
    region_range = ["Mar., Apr., May", "Jun. Jul. Aug.", "Sep., Oct. Nov.", "Dec. Jan. Feb."]


    for i in range(4):
        fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max(100, area_data.shape[2]))
        gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                               top=1. - 0.5 / (nrow + 1),
                               bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1),
                               right=1 - 0.5 / (ncol + 1))
        fig.suptitle(region_range[i] + " (" + region_types[i] + ")")
        for r in range(nrow):
            for c in range(ncol):
                ax = plt.subplot(gs[r, c])
                ax.imshow(area_data[r, i*split_steps + c])
                ax.set_xticks([])
                ax.set_yticks([])

                if c == 0:
                    ax.set_ylabel(row_labels[r].replace("_", " "))

        plt.savefig(output_path + "/" + region_types[i] + ".png")
        plt.close("all")