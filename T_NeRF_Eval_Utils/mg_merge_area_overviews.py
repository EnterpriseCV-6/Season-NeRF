# coding=utf-8
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import T_NeRF_Full_2
from all_NeRF import CV_reshape
from all_NeRF import show_dict_struc
from misc import load_args_from_json

def merge_area_overviews(valid_areas, output_path):
    col_labels = ["A", "B", "C"]
    row_labels = []
    imgs = []
    R_imgs = []
    Ra_imgs = []

    out_table = [["\\text{Area Index}", "\\text{Num. Imgs.}", "\\text{Alt. Bounds (m)}"]]
    max_dpi = 100
    for a_valid_area in valid_areas:
        print(a_valid_area)
        temp_area = ""
        for i in range(len(a_valid_area.split("/"))-1):
            temp_area = temp_area + a_valid_area.split("/")[i] + "/"
        temp_area += "opts.json"
        if os.path.exists(temp_area) and os.path.exists(a_valid_area + "/Img_Summary.pickle"):
            area_name = a_valid_area.split("/")[-2].replace("_", " ")
            try:

                args = load_args_from_json(temp_area)

                P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device = T_NeRF_Full_2.load_from_input_args(args)
                print(testing_imgs)
                out_table.append(["\\text{" + area_name + "}", str(len(P_imgs)), str(bounds_LLA[2,1] - bounds_LLA[2,0])])

                fin = open(a_valid_area + "/Img_Summary.pickle", "rb")
                Img_Sum = pickle.load(fin)
                fin.close()
                # show_dict_struc(Img_Sum)

                imgs.append([])
                R_imgs.append([])
                Ra_imgs.append([])
                row_labels.append(area_name[0:7])
                for i in range(3):
                    for a_P_img in P_imgs:
                        if a_P_img.img_name == testing_imgs[i]:
                            R_img = Img_Sum["Testing"][a_P_img.img_name]["Standard"]["Images"]["Season_Adj_Img"] * \
                                    Img_Sum["Testing"][a_P_img.img_name]["Standard"]["Images"]["Shadow_Adjust"]
                            Ra_img = Img_Sum["Testing"][a_P_img.img_name]["Standard"]["Seasonal_Aligned_Imgs"][
                                         "Season_Adj_Img"] * \
                                     Img_Sum["Testing"][a_P_img.img_name]["Standard"]["Seasonal_Aligned_Imgs"][
                                         "Shadow_Adjust"]

                            imgs[-1].append(CV_reshape(a_P_img.img, (R_img.shape[1],Ra_img.shape[0])))
                            max_dpi = max(max_dpi, Ra_img.shape[0])
                            imgs[-1][-1][R_img != R_img] = np.NaN
                            R_imgs[-1].append(R_img)
                            Ra_imgs[-1].append(Ra_img)

                            break
            except:
                print("Problem with ", area_name)

    ncol =11
    nrow = len(imgs)
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max_dpi)
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1))
    fig.suptitle("Image Overview")
    for r in range(nrow):
        for c in range(3):
            ax = plt.subplot(gs[r, c])
            ax.imshow(imgs[r][c])
            ax.set_yticks([])
            ax.set_xticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r])
            if r == nrow-1:
                ax.set_xlabel(col_labels[c])
            if r == 0 and c == 1:
                ax.set_title("Prototypical Ground Truth Images")

            ax = plt.subplot(gs[r, c+4])
            ax.imshow(R_imgs[r][c])
            ax.set_yticks([])
            ax.set_xticks([])
            if r == nrow - 1:
                ax.set_xlabel(col_labels[c])
            if r == 0 and c == 1:
                ax.set_title("Rendered Images")

            ax = plt.subplot(gs[r, c + 8])
            ax.imshow(Ra_imgs[r][c])
            ax.set_yticks([])
            ax.set_xticks([])
            if r == nrow - 1:
                ax.set_xlabel(col_labels[c])
            if r == 0 and c == 1:
                ax.set_title("Rendered Images after Seasonal Alignment")

    plt.savefig(output_path + "/Full_Testing_Images.png")
    plt.close("all")


    ncol = 3
    nrow = len(imgs)
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max_dpi)
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1))
    fig.suptitle("Prototypical Testing Images")
    for r in range(nrow):
        for c in range(ncol):
            ax = plt.subplot(gs[r,c])
            ax.imshow(imgs[r][c])
            ax.set_yticks([])
            ax.set_xticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r])
            if r == 0:
                ax.set_title(col_labels[c])
    plt.savefig(output_path + "/Testing_Images.png")
    plt.close("all")

    out_table = np.array(out_table)


    with open(output_path + "/Area_Overview.txt", "w") as fout:
        np.savetxt(fout, out_table, delimiter=" & ", newline=" \\\\ \\hline\n", fmt="%s")
        fout.close()