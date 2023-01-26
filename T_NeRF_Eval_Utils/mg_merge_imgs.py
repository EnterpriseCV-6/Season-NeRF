# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pickle
import os
from all_NeRF import show_dict_struc
from tabulate import tabulate

def merge_imgs(valid_areas, output_path, round_deg = 3):
    all_in_one_data = []
    all_in_one_imgs = []

    all_in_row_labels =[]
    all_in_one_col_labels_for_imgs = ["Ground Truth", "Rendered Image", "Rendered Image, aligned"]
    all_in_one_col_labels = ["Region", "PSNR", "SSIM", "EM", "PSNR, aligned", "SSIM,  aligned", "EM, aligned"]

    n_valid = 0
    valid_size = -1
    for a_valid_area in valid_areas:
        if os.path.exists(a_valid_area + "/Img_Summary.pickle"):
            fin = open(a_valid_area + "/Img_Summary.pickle", "rb")
            Img_Summary = pickle.load(fin)
            fin.close()
            n_valid += 1
            Img_Summary = Img_Summary["Testing"]
            # show_dict_struc(Img_Summary)

            all_in_one_data.append([])
            all_in_row_labels.append([])
            all_in_one_imgs.append([])
            for a_key in Img_Summary.keys():
                GT_img = Img_Summary[a_key]["Standard"]["Ground_Truth"]
                img = Img_Summary[a_key]["Standard"]["Images"]["Season_Adj_Img"] * Img_Summary[a_key]["Standard"]["Images"]["Shadow_Adjust"]
                img_aligned = Img_Summary[a_key]["Standard"]["Seasonal_Aligned_Imgs"]["Season_Adj_Img"] * Img_Summary[a_key]["Standard"]["Seasonal_Aligned_Imgs"]["Shadow_Adjust"]
                GT_img[img != img] = 0.

                valid_size = max(GT_img.shape[0], valid_size)
                scores = list(np.ravel(Img_Summary[a_key]["Standard"]["Scores"]["Table"][2:4]))
                scores = scores[1:4] + scores[5::]
                all_in_one_data[-1].append(scores)
                all_in_one_imgs[-1].append([GT_img, img, img_aligned])
                all_in_row_labels[-1].append(a_key.replace("_", " "))
    all_in_one_data = np.array(all_in_one_data)
    all_in_row_labels = np.array(all_in_row_labels)

    row_labels = [all_in_row_labels[i][0][0:7] for i in range(all_in_row_labels.shape[0])] + ["Average"]

    summarized_data = np.zeros([n_valid, 6])
    best_data = np.zeros([n_valid, 6])
    worst_data = np.zeros([n_valid, 6])
    sign_adj = np.array([1,1,-1,1,1,-1])
    for i in range(n_valid):
        summarized_data[i] = np.mean(all_in_one_data[i], 0)
        worst_data[i] = np.min(all_in_one_data[i] * np.expand_dims(sign_adj, 0), 0) * sign_adj
        best_data[i] = np.max(all_in_one_data[i] * np.expand_dims(sign_adj, 0), 0) * sign_adj

    full = np.stack([summarized_data, worst_data, best_data], 0)


    data_type = ["Average", "Worst", "Best"]

    with open(output_path + "/Img_Summary.txt", "w") as fout:
        for i in range(3):
            avg = np.round(np.mean(full[i], 0), round_deg)
            temp = list(np.round(full[i], round_deg))
            write_this = [[row_labels[j]] + list(temp[j]) for j in range(len(temp))]
            write_this += [[row_labels[-1]] + list(avg)]
            fout.write(data_type[i] + ":\n")
            fout.write(tabulate(write_this, all_in_one_col_labels))
            fout.write("\n\n")

        fout.write("\n\n\n\n\n\n")
        for i in range(3):
            avg = np.mean(full[i], 0)
            temp = list(full[i])
            write_this = [["\\text{" + row_labels[j] + "}"] + list(temp[j]) for j in range(len(temp))]
            write_this += [["\\text{" + row_labels[-1] + "}"] + list(avg)]
            fout.write(data_type[i] + ":\n")
            np.savetxt(fout, write_this, fmt="%s", delimiter=" & ", newline=" \\\\ \\hline\n")
            fout.write("\n\n")

        fout.close()

    nrow = len(all_in_one_imgs)
    ncol = 3
    restart = len(all_in_one_imgs[0])

    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max(100, valid_size))
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1))
    fig.suptitle("Example Rendered Images")
    IDX = 0
    for r in range(nrow):
        for c in range(ncol):
            ax = plt.subplot(gs[r, c])
            ax.imshow(all_in_one_imgs[r][IDX][c])
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(all_in_one_col_labels_for_imgs[c], fontsize="xx-small")
            if c == 0:
                ax.set_ylabel(row_labels[r].replace("_", " "), fontsize="xx-small")

        IDX = (IDX + 1) % restart
    plt.savefig(output_path + "/Rendered_Imgs.png")
    plt.close("all")

def merge_imgs_shadows(valid_areas, output_path, round_deg = 3):
    all_in_one_data = []
    all_in_one_imgs = []
    all_in_row_labels =[]
    all_in_worst_idx = []

    region_all_all_summary = []

    n_valid = 0
    valid_size = -1
    for a_valid_area in valid_areas:
        if os.path.exists(a_valid_area + "/Img_Summary.pickle") and os.path.exists(a_valid_area + "/Testing_Shadow_Scores.txt") and os.path.exists(a_valid_area + "/Shadow_Scores_Summary.pickle"):
            fin = open(a_valid_area + "/Shadow_Scores_Summary.pickle", "rb")
            all_summary = pickle.load(fin)
            fin.close()
            # show_dict_struc(all_summary)

            region_all_summary = [all_summary[a_key]["Acc"] for a_key in all_summary.keys()]
            region_all_all_summary.append(region_all_summary)


            fin = open(a_valid_area + "/Img_Summary.pickle", "rb")
            Img_Summary = pickle.load(fin)
            fin.close()

            fin = open(a_valid_area + "/Testing_Shadow_Scores.txt", "r")
            Shadow_Data = [i for i in fin.readlines()][2::]
            worst_acc = 1.
            worst_acc_idx = -1
            region = "ABCDEFG"
            scores = [0,0,0,0]
            for i in range(len(Shadow_Data)):
                temp = Shadow_Data[i].split()
                region, TP, TN, FP, FN = temp[0], int(temp[2]), int(temp[3]), int(temp[4]), int(temp[5])
                scores[0] += TP
                scores[1] += TN
                scores[2] += FP
                scores[3] += FN
                if (TP + TN) / (TP + TN + FP + FN) < worst_acc:
                    worst_acc =(TP + TN) / (TP + TN + FP + FN)
                    worst_acc_idx = i
            region = region[0:7]

            fin.close()

            n_valid += 1
            Img_Summary = Img_Summary["Testing"]

            all_in_one_data.append(scores)
            all_in_row_labels.append(region.replace("_", " "))
            all_in_one_imgs.append([])
            all_in_worst_idx.append(worst_acc_idx)
            for a_key in Img_Summary.keys():
                GT_img = Img_Summary[a_key]["Exact_Solar"]["Images"]["Shadow_Mask_Exact"]
                img = Img_Summary[a_key]["Exact_Solar"]["Images"]["Shadow_Mask"]
                GT_img[img != img] = 0.

                valid_size = max(GT_img.shape[0], valid_size)
                all_in_one_imgs[-1].append([GT_img, img])

    all_in_one_data = np.concatenate([all_in_one_data, np.sum(all_in_one_data, 0, keepdims=True)], 0)
    # all_in_one_data = np.concatenate([np.zeros([all_in_one_data.shape[0], 1], dtype=int), np.zeros_like(all_in_one_data), all_in_one_data], 1).astype(float)
    acc = (all_in_one_data[:,0] + all_in_one_data[:,1]) / (np.sum(all_in_one_data,1))
    Sun_Prec = all_in_one_data[:,0] / (all_in_one_data[:,0] + all_in_one_data[:,2])
    Sun_Recall = all_in_one_data[:, 0] / (all_in_one_data[:, 0] + all_in_one_data[:, 3])
    Shadow_Prec = all_in_one_data[:, 1] / (all_in_one_data[:, 1] + all_in_one_data[:, 3])
    Shadow_Recall = all_in_one_data[:, 1] / (all_in_one_data[:, 1] + all_in_one_data[:, 2])
    details = np.concatenate([np.stack([acc, Sun_Prec, Sun_Recall, Shadow_Prec, Shadow_Recall], 1), all_in_one_data], 1)

    data_type = ["Region", "Acc.", "Sun Prec.", "Sun Recall", "Shadow Prec.", "Shadow Recall", "TP", "TN", "FP", "FN"]
    data_type2 = ["Region", "Training Acc.", "Testing Acc.", "Near Acc.", "Full Acc."]
    all_in_row_labels += ["Overall"]



    with open(output_path + "/Shadow_Summary.txt", "w") as fout:
        table = []
        fout.write("Testing Region Stats:\n")
        for i in range(len(all_in_row_labels)):
            table.append([all_in_row_labels[i]] + list(np.round(details[i], round_deg)))
        fout.write(tabulate(table, data_type))
        fout.write("\n\n")

        table = []
        fout.write("Overall Region Stats:\n")
        for i in range(len(all_in_row_labels)-1):
            table.append([all_in_row_labels[i]] + list(np.round(region_all_all_summary[i], round_deg)))
        fout.write(tabulate(table, data_type2))


        fout.write("\n\n\n\n\n\n")
        fout.write("Testing Region Stats:\n")
        table = []
        for i in range(len(all_in_row_labels)):
            table.append(["\\text{" + all_in_row_labels[i] + "}"] + list(details[i]))
        np.savetxt(fout, table, fmt="%s", delimiter=" & ", newline=" \\\\\n")
        fout.write("\n\n")
        table = []
        fout.write("Overall Region Stats:\n")
        for i in range(len(all_in_row_labels) - 1):
            table.append(["\\text{" + all_in_row_labels[i] + "}"] + list(region_all_all_summary[i]))
        np.savetxt(fout, table, fmt="%s", delimiter=" & ", newline=" \\\\\n")

        fout.close()

    nrow = len(all_in_one_imgs)
    ncol = 2

    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)), dpi=max(100, valid_size))
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1))
    fig.suptitle("Example of Estimated Shadow Masks", fontsize="medium")
    col_labels = ["Exact", "Est"]
    for r in range(nrow):
        for c in range(ncol):
            ax = plt.subplot(gs[r, c])
            ax.imshow(all_in_one_imgs[r][all_in_worst_idx[r]][c])
            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title(col_labels[c], fontsize="x-small")
            if c == 0:
                ax.set_ylabel(all_in_row_labels[r].replace("_", " "), fontsize="x-small")

    plt.savefig(output_path + "/Shadow_Imgs.png")
    plt.close("all")