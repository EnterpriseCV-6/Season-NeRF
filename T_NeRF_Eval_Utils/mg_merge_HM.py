# coding=utf-8

import numpy as np
from tabulate import tabulate
import pickle
import os
from all_NeRF import show_dict_struc
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def merge_HMs(valid_areas, output_path, store_invalid_data = True, round_level = 2):
    # print(valid_areas)


    Consise_Area_Summary = []
    Consise_Area_Summary_no_round = []
    valid_names = []
    Consise_Area_Summary_Col_Labels = ["Region", "Prior MAE", "MAE", "RMSE", "% within 1 m", "Median"]
    Img_list = []
    Img_list_zoom = []
    full_min_max = []
    full_min_max_zoom = []
    running_scores = np.zeros(5)
    n_valid = 0
    valid_size = -1
    for a_valid_area in valid_areas:
        area_name = a_valid_area.split("/")[-2].replace("_", " ")
        if os.path.exists(a_valid_area + "/HM_Summary.pickle"):
            fin = open(a_valid_area + "/HM_Summary.pickle", "rb")
            HM_Summary = pickle.load(fin)
            fin.close()
            n_valid += 1

            new_entry = [HM_Summary["Prior_after_alignment"]["MAE"],
                                         HM_Summary["NeRF_after_alignment"]["MAE"],
                                         HM_Summary["NeRF_after_alignment"]["RMSE"],
                                         HM_Summary["NeRF_after_alignment"]["Percent_within_1_m"]*100,
                                         HM_Summary["NeRF_after_alignment"]["Median_Error"]]
            running_scores += np.array(new_entry)
            Consise_Area_Summary_no_round.append(["\\text{"+area_name+"}"] + new_entry)
            new_entry = [area_name] + [np.round(new_entry[i], round_level) for i in range(5)]
            Consise_Area_Summary.append(new_entry)
            valid_names.append(area_name)

            # show_dict_struc(HM_Summary)
            Img_list.append([HM_Summary["Ground_Truth"], HM_Summary["Prior_before_alginment"]["Image"], HM_Summary["NeRF_before_alignment"]["Image"]])

            sub_imgs = [HM_Summary["Ground_Truth"], HM_Summary["Prior_after_alignment"]["Image"], HM_Summary["NeRF_after_alignment"]["Image"]]

            while True:
                if np.any([np.all(sub_imgs[i][:,0] != sub_imgs[i][:,0]) for i in range(3)]):
                    sub_imgs = [sub_imgs[i][:,1::] for i in range(3)]
                elif  np.any([np.all(sub_imgs[i][0,:] != sub_imgs[i][0,:]) for i in range(3)]):
                    sub_imgs = [sub_imgs[i][1::] for i in range(3)]
                elif np.any([np.all(sub_imgs[i][-1, :] != sub_imgs[i][-1, :]) for i in range(3)]):
                    sub_imgs = [sub_imgs[i][0:-1] for i in range(3)]
                elif np.any([np.all(sub_imgs[i][:,-1] != sub_imgs[i][:,-1]) for i in range(3)]):
                    sub_imgs = [sub_imgs[i][:,0:-1] for i in range(3)]
                else:
                    break

            Img_list_zoom.append(sub_imgs)

            a_min = np.min([np.nanmin(Img_list[-1][0]), np.nanmin(Img_list[-1][1]), np.nanmin(Img_list[-1][2])])
            a_max = np.max([np.nanmax(Img_list[-1][0]), np.nanmax(Img_list[-1][1]), np.nanmax(Img_list[-1][2])])
            full_min_max.append([a_min, a_max])

            a_min = np.min([np.nanmin(Img_list_zoom[-1][0]), np.nanmin(Img_list_zoom[-1][1]), np.nanmin(Img_list_zoom[-1][2])])
            a_max = np.max([np.nanmax(Img_list_zoom[-1][0]), np.nanmax(Img_list_zoom[-1][1]), np.nanmax(Img_list_zoom[-1][2])])
            full_min_max_zoom.append([a_min, a_max])
            Img_list_zoom[-1].append(sub_imgs[2] - sub_imgs[0])

            valid_size = max(valid_size, HM_Summary["NeRF_after_alignment"]["Image"].shape[0])

        else:
            print("Unable to load HM_Summary.pickle for " + a_valid_area)
            if store_invalid_data:
                Consise_Area_Summary.append([area_name, np.nan,
                                             np.nan,
                                             np.nan,
                                             np.nan,
                                             np.nan])
                Consise_Area_Summary_no_round.append(["\\text{" + area_name + "}", np.nan,
                                             np.nan,
                                             np.nan,
                                             np.nan,
                                             np.nan])
    Consise_Area_Summary.append(["Average"] + list(running_scores/n_valid))
    Consise_Area_Summary_no_round.append(["\\text{Average}"] + list(running_scores / n_valid))

    a_tab = tabulate(Consise_Area_Summary, Consise_Area_Summary_Col_Labels)
    with open(output_path + "/HM_Scores.txt", "w") as fout:
        fout.write(a_tab)
        fout.write("\n\n\n")
        np.savetxt(fout, Consise_Area_Summary_no_round, delimiter=" & ", newline=" \\\\\n", fmt="%s")
        fout.close()
    print(a_tab)



    ncol = 4
    nrow = n_valid
    names = ["Lidar", "Prior", "NeRF", "Error"]
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max(100, valid_size))
    # gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
    #                        top=1. - 0.5 / (nrow + 1),
    #                        bottom=0.5 / (nrow + 1),
    #                        left=0.5 / (ncol + 1),
    #                        right=1 - 0.5 / (ncol + 1))
    fig.suptitle("Height Map, Zoom")
    for r in range(nrow):
        for c in range(ncol):
            ax = plt.subplot(nrow, ncol, ncol * r + c + 1)
            if c < ncol -1 :
                the_img = ax.imshow(Img_list_zoom[r][c], vmin=full_min_max_zoom[r][0], vmax=full_min_max_zoom[r][1])
            else:
                vr = np.nanmax(np.abs(Img_list_zoom[r][c]))
                the_img = ax.imshow(Img_list_zoom[r][c], cmap="bwr", vmin=-vr, vmax=vr)
            ax.set_yticks([])
            ax.set_xticks([])
            if c == 0:
                L = min(len(valid_names[r]), 10)
                ax.set_ylabel(valid_names[r][0:L])
            if c > 2:
                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(the_img, cax=cax1)
            if r == 0:
                ax.set_title(names[c])

    plt.subplots_adjust(top=1. - 0.5 / (nrow + 1),
                        bottom=0.5 / (nrow + 1),
                        left=0.5 / (ncol + 1),
                        right=1 - 0.5 / (ncol + 1), wspace=0, hspace=0.1)
    plt.savefig(output_path + "/Zoom_HM_imgs.png")
    plt.close("all")

    ncol = 3
    nrow = n_valid
    names = ["Lidar", "Prior", "NeRF"]
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max(100, valid_size))
    # gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
    #                        top=1. - 0.5 / (nrow + 1),
    #                        bottom=0.5 / (nrow + 1),
    #                        left=0.5 / (ncol + 1),
    #                        right=1 - 0.5 / (ncol + 1))
    fig.suptitle("Height Map, Full Area")
    for r in range(nrow):
        for c in range(ncol):
            ax = plt.subplot(nrow, ncol, ncol*r+c+1)
            the_img = ax.imshow(Img_list[r][c], vmin=full_min_max[r][0], vmax=full_min_max[r][1])
            ax.set_yticks([])
            ax.set_xticks([])
            if c == 0:
                L = min(len(valid_names[r]), 10)
                ax.set_ylabel(valid_names[r][0:L])
            if c == 2:
                divider = make_axes_locatable(ax)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(the_img, cax=cax1)
            if r == 0:
                ax.set_title(names[c])


    plt.subplots_adjust(top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1), wspace=0.0, hspace=0.1)
    plt.savefig(output_path + "/Full_HM_imgs.png")
    plt.close("all")