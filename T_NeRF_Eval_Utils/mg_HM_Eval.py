# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from all_NeRF import show_dict_struc
from tabulate import tabulate


def apply_H_metrics(Region1, Region2):
    MAE = np.nanmean(np.abs(Region1 - Region2))
    RMSE = np.sqrt(np.nanmean((Region1 - Region2) ** 2))
    Median_Error = np.nanmedian(np.abs(Region1 - Region2))
    percent_within_1_m = np.abs(Region1 - Region2)
    percent_within_1_m = percent_within_1_m[percent_within_1_m == percent_within_1_m]
    percent_within_1_m = np.sum(percent_within_1_m <= 1.0) / percent_within_1_m.shape[0]

    return MAE, RMSE, percent_within_1_m, Median_Error

def apply_T(Img, T:tuple):
    XY = np.stack(np.meshgrid(np.arange(Img.shape[0]), np.arange(Img.shape[1]), indexing="ij"), -1).reshape([-1,2])
    theta = T[0] * np.pi / 180
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    # new_XY = ((XY * 1.) * np.array([[T[1], T[3]]]) + np.array([[T[2], T[4]]])) @ R.T + np.array([[T[5], T[6]]])
    new_XY = ((XY - np.array([[T[5], T[6]]])) @ np.linalg.inv(R.T) - np.array([[T[2], T[4]]])) / np.array([[T[1], T[3]]])
    new_XY = np.round(new_XY).astype(int)

    new_img = np.zeros([Img.shape[0], Img.shape[1]], dtype=float)*np.nan
    good = (new_XY[:,0] >= 0) * (new_XY[:,0] < Img.shape[0]) * (new_XY[:,1] >= 0) * (new_XY[:,1] < Img.shape[1])
    new_img[XY[good,0], XY[good,1]] = Img[new_XY[good,0], new_XY[good,1]]

    return new_img

def step_T(T):
    new_Ts = np.stack(np.meshgrid([T[0] - 1, T[0], T[0] + 1],
                [T[1] * .99, T[1], T[1] * 1.01],
                [T[2] - 1, T[2], T[2] + 1],
                [T[3] * .99, T[3], T[3] * 1.01],
                [T[4] - 1, T[4], T[4] + 1],
                [T[5] -1, T[5], T[5] + 1],
                [T[6] -1, T[6], T[6] +1], indexing="ij"), -1).reshape([-1,7])
    return new_Ts

def Adjust_Region(Fixed_Region, Other_Region):
    x, y = np.ravel(Other_Region), np.ravel(Fixed_Region)
    good = (x == x) * (y == y)
    x, y = x[good], y[good]
    A, B = np.polyfit(x, y, deg=1)
    Other_Region_Adjust = Other_Region * A + B
    return Other_Region_Adjust, A, B

def simple_align(Fixed_Region, Other_Region, show_process = False, max_steps = 50):
    if show_process:
        MSE = np.nanmean((Fixed_Region - Other_Region) ** 2)
        print("Starting Error:", MSE)
    Other_Region_Adjust, A, B = Adjust_Region(Fixed_Region, Other_Region)
    MSE = np.nanmean((Fixed_Region - Other_Region_Adjust) ** 2)
    if show_process:
        print("Bias Adjust Error:", MSE, A, B)
    T = (0, 1, -(Other_Region.shape[0]//2), 1, -(Other_Region.shape[1]//2), Other_Region.shape[0]//2, Other_Region.shape[1]//2)
    best_T_idx = 0
    best_MSE = MSE
    new_Region = Other_Region_Adjust
    A_, B_ = 1, 0
    step = 0
    while best_T_idx != -1 and step < max_steps:
        last_error = "/" + str(max_steps) + " - " + str(np.round(best_MSE, 2))
        best_T_idx = -1
        Ts = step_T(T)
        c = 0
        for a_T in tqdm(Ts, desc="Step " + str(step+1) + last_error, leave=not show_process):
            temp_Region = apply_T(Other_Region, tuple(a_T))
            temp_Region, A, B = Adjust_Region(Fixed_Region, temp_Region)
            new_MSE = np.nanmean((Fixed_Region - temp_Region) ** 2)
            if new_MSE < best_MSE:
                best_MSE = new_MSE
                best_T_idx = c
                T = Ts[c]
                A_, B_ = A, B
                new_Region = np.copy(temp_Region)
            c += 1
        step += 1
        if show_process:
            print(T)
            print(best_MSE, A_, B_)
    if step == max_steps:
        print("Warning: Maximum steps reached!")
    if show_process == False:
        print("Final Error:", best_MSE)

    return new_Region, T, (A,B)

def Full_Eval_HM(image_builder, GT_DSM, prior_DSM, height_range):
    assert np.all(GT_DSM.shape == prior_DSM.shape)
    prior_DSM = (prior_DSM + 1) / 2 * (height_range[1] - height_range[0]) + height_range[0]
    GT_DSM = (GT_DSM + 1) / 2 * (height_range[1] - height_range[0]) + height_range[0]

    MAE, RMSE, percent_within_1_m, Median_Error = apply_H_metrics(GT_DSM, prior_DSM)
    Prior_before_sum = {"Image":prior_DSM, "MAE":MAE, "RMSE":RMSE, "Percent_within_1_m":percent_within_1_m, "Median_Error":Median_Error}
    print(MAE, RMSE, percent_within_1_m, Median_Error)

    print("Aligning prior...")
    aligned_prior_DSM, adjust, scale = simple_align(GT_DSM, prior_DSM)
    MAE, RMSE, percent_within_1_m, Median_Error = apply_H_metrics(GT_DSM, aligned_prior_DSM)
    Prior_after_sum = {"Image": aligned_prior_DSM, "MAE": MAE, "RMSE": RMSE, "Percent_within_1_m": percent_within_1_m, "Median_Error": Median_Error, "Alignment_shift_and_scale":(adjust, scale)}
    print(MAE, RMSE, percent_within_1_m, Median_Error)

    print("Building NeRF DSM...")
    NeRF_DSM = (image_builder.get_DSM(GT_DSM.shape) + 1) / 2 * (height_range[1] - height_range[0]) + height_range[0]
    MAE, RMSE, percent_within_1_m, Median_Error = apply_H_metrics(GT_DSM, NeRF_DSM)
    NeRF_before_sum = {"Image": NeRF_DSM, "MAE": MAE, "RMSE": RMSE, "Percent_within_1_m": percent_within_1_m,
                        "Median_Error": Median_Error}
    print(MAE, RMSE, percent_within_1_m, Median_Error)

    print("Aligning NeRF DSM...")
    NeRF_aligned_DSM, NeRF_adjust, NeRF_scale = simple_align(GT_DSM, NeRF_DSM)
    MAE, RMSE, percent_within_1_m, Median_Error = apply_H_metrics(GT_DSM, NeRF_aligned_DSM)
    NeRF_after_sum = {"Image": NeRF_aligned_DSM, "MAE": MAE, "RMSE": RMSE, "Percent_within_1_m": percent_within_1_m,
                             "Median_Error": Median_Error, "Alignment_shift_and_scale":(NeRF_adjust, NeRF_scale)}
    print(MAE, RMSE, percent_within_1_m, Median_Error)

    Summary = {"Ground_Truth":GT_DSM, "Prior_before_alginment":Prior_before_sum, "Prior_after_alignment":Prior_after_sum, "NeRF_before_alignment":NeRF_before_sum, "NeRF_after_alignment":NeRF_after_sum}

    return Summary

def Orgainize_Output_Imgs_HM(HM_Summary, output_loc):

    min_val = np.nanmin(HM_Summary["Ground_Truth"])
    max_val = np.nanmax(HM_Summary["Ground_Truth"])

    min_val = min(np.nanmin(HM_Summary["Prior_before_alginment"]["Image"]), min_val)
    min_val = min(np.nanmin(HM_Summary["Prior_after_alignment"]["Image"]), min_val)
    min_val = min(np.nanmin(HM_Summary["NeRF_before_alignment"]["Image"]), min_val)
    min_val = min(np.nanmin(HM_Summary["NeRF_after_alignment"]["Image"]), min_val)

    max_val = max(np.nanmax(HM_Summary["Prior_before_alginment"]["Image"]), max_val)
    max_val = max(np.nanmax(HM_Summary["Prior_after_alignment"]["Image"]), max_val)
    max_val = max(np.nanmax(HM_Summary["NeRF_before_alignment"]["Image"]), max_val)
    max_val = max(np.nanmax(HM_Summary["NeRF_after_alignment"]["Image"]), max_val)

    Results = np.array([[HM_Summary["Prior_before_alginment"]["MAE"], HM_Summary["Prior_before_alginment"]["RMSE"], HM_Summary["Prior_before_alginment"]["Percent_within_1_m"], HM_Summary["Prior_before_alginment"]["Median_Error"]],
               [HM_Summary["Prior_after_alignment"]["MAE"], HM_Summary["Prior_after_alignment"]["RMSE"],
                HM_Summary["Prior_after_alignment"]["Percent_within_1_m"], HM_Summary["Prior_after_alignment"]["Median_Error"]]])
    Results = np.round(Results, 3)

    fout = open(output_loc + "/HM_Stats.txt", "w")
    fout.write("Prior Stats\n")
    fout.write(tabulate(Results, ("MAE", "RMSE", "% in 1 m", "Median Error")))

    GT_Clipped = np.copy(HM_Summary["Ground_Truth"])
    BA_clipped = np.copy(HM_Summary["Prior_before_alginment"]["Image"])
    AA_clipped = np.copy(HM_Summary["Prior_after_alignment"]["Image"])
    while np.all(GT_Clipped[:, 0] != GT_Clipped[:, 0]):
        GT_Clipped = GT_Clipped[:, 1::]
        BA_clipped = BA_clipped[:, 1::]
        AA_clipped = AA_clipped[:, 1::]
    while np.all(GT_Clipped[0, :] != GT_Clipped[0, :]):
        GT_Clipped = GT_Clipped[1::, :]
        BA_clipped = BA_clipped[1::, :]
        AA_clipped = AA_clipped[1::, :]
    while np.all(GT_Clipped[:, -1] != GT_Clipped[:, -1]):
        GT_Clipped = GT_Clipped[:, 0:-1]
        BA_clipped = BA_clipped[:, 0:-1]
        AA_clipped = AA_clipped[:, 0:-1]
    while np.all(GT_Clipped[-1, :] != GT_Clipped[-1, :]):
        GT_Clipped = GT_Clipped[0:-1, :]
        BA_clipped = BA_clipped[0:-1, :]
        AA_clipped = AA_clipped[0:-1, :]

    plt.figure(figsize=(10,10), dpi=100)
    plt.subplot(3, 3, 1)
    plt.imshow(HM_Summary["Ground_Truth"], vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.title("Ground Truth")
    plt.subplot(3, 3, 2)
    plt.imshow(HM_Summary["Prior_before_alginment"]["Image"], vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("MAE: " + str(Results[0,0]))
    plt.title("Before Alignment")
    plt.subplot(3, 3, 3)
    plt.imshow(HM_Summary["Prior_after_alignment"]["Image"], vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("MAE: " + str(Results[1, 0]))
    plt.title("After Alignment")
    plt.colorbar()
    plt.subplot(3, 3, 4)
    plt.imshow(GT_Clipped, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 5)
    plt.imshow(BA_clipped, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 6)
    plt.imshow(AA_clipped, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,3,8)
    plt.ylabel("Error Images")
    E_img = HM_Summary["Prior_before_alginment"]["Image"] - HM_Summary["Ground_Truth"]
    while np.all(E_img[:,0] != E_img[:,0]):
        E_img = E_img[:, 1::]
    while np.all(E_img[0, :] != E_img[0, :]):
        E_img = E_img[1::, :]
    while np.all(E_img[:, -1] != E_img[:, -1]):
        E_img = E_img[:, 0:-1]
    while np.all(E_img[-1, :] != E_img[-1, :]):
        E_img = E_img[0:-1, :]

    max_range = np.nanmax(np.abs(E_img))
    plt.imshow(E_img, cmap="bwr", vmin=-max_range, vmax=max_range)
    plt.ylabel("Alt. Error Image")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(3, 3, 9)
    E_img = HM_Summary["Prior_after_alignment"]["Image"] - HM_Summary["Ground_Truth"]
    while np.all(E_img[:, 0] != E_img[:, 0]):
        E_img = E_img[:, 1::]
    while np.all(E_img[0, :] != E_img[0, :]):
        E_img = E_img[1::, :]
    while np.all(E_img[:, -1] != E_img[:, -1]):
        E_img = E_img[:, 0:-1]
    while np.all(E_img[-1, :] != E_img[-1, :]):
        E_img = E_img[0:-1, :]

    # max_range = np.nanmax(np.abs(E_img))
    plt.imshow(E_img, cmap="bwr", vmin=-max_range, vmax=max_range)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.savefig(output_loc + "/Prior_HM.png")
    plt.close("all")

    plt.figure(figsize=(10, 10), dpi=1000)
    Results = np.array([[HM_Summary["NeRF_before_alignment"]["MAE"], HM_Summary["NeRF_before_alignment"]["RMSE"],
                         HM_Summary["NeRF_before_alignment"]["Percent_within_1_m"],
                         HM_Summary["NeRF_before_alignment"]["Median_Error"]],
                        [HM_Summary["NeRF_after_alignment"]["MAE"], HM_Summary["NeRF_after_alignment"]["RMSE"],
                         HM_Summary["NeRF_after_alignment"]["Percent_within_1_m"],
                         HM_Summary["NeRF_after_alignment"]["Median_Error"]]])
    Results = np.round(Results, 3)

    fout.write("\nNeRF Stats\n")
    fout.write(tabulate(Results, ("MAE", "RMSE", "% in 1 m", "Median Error")))
    fout.close()

    GT_Clipped = np.copy(HM_Summary["Ground_Truth"])
    BA_clipped = np.copy(HM_Summary["NeRF_before_alignment"]["Image"])
    AA_clipped = np.copy(HM_Summary["NeRF_after_alignment"]["Image"])
    while np.all(GT_Clipped[:, 0] != GT_Clipped[:, 0]):
        GT_Clipped = GT_Clipped[:, 1::]
        BA_clipped = BA_clipped[:, 1::]
        AA_clipped = AA_clipped[:, 1::]
    while np.all(GT_Clipped[0, :] != GT_Clipped[0, :]):
        GT_Clipped = GT_Clipped[1::, :]
        BA_clipped = BA_clipped[1::, :]
        AA_clipped = AA_clipped[1::, :]
    while np.all(GT_Clipped[:, -1] != GT_Clipped[:, -1]):
        GT_Clipped = GT_Clipped[:, 0:-1]
        BA_clipped = BA_clipped[:, 0:-1]
        AA_clipped = AA_clipped[:, 0:-1]
    while np.all(GT_Clipped[-1, :] != GT_Clipped[-1, :]):
        GT_Clipped = GT_Clipped[0:-1, :]
        BA_clipped = BA_clipped[0:-1, :]
        AA_clipped = AA_clipped[0:-1, :]


    plt.subplot(3, 3, 1)
    plt.imshow(HM_Summary["Ground_Truth"], vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.title("Ground Truth")
    plt.subplot(3, 3, 2)
    plt.imshow(HM_Summary["NeRF_before_alignment"]["Image"], vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("MAE: " + str(Results[0, 0]))
    plt.title("Before Alignment")
    plt.subplot(3, 3, 3)
    plt.imshow(HM_Summary["NeRF_after_alignment"]["Image"], vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("MAE: " + str(Results[1, 0]))
    plt.title("After Alignment")
    plt.colorbar()
    plt.subplot(3, 3, 4)
    plt.imshow(GT_Clipped, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 5)
    plt.imshow(BA_clipped, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 6)
    plt.imshow(AA_clipped, vmin=min_val, vmax=max_val)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 3, 8)
    E_img = HM_Summary["NeRF_before_alignment"]["Image"] - HM_Summary["Ground_Truth"]
    while np.all(E_img[:, 0] != E_img[:, 0]):
        E_img = E_img[:, 1::]
    while np.all(E_img[0, :] != E_img[0, :]):
        E_img = E_img[1::, :]

    while np.all(E_img[:, -1] != E_img[:, -1]):
        E_img = E_img[:, 0:-1]
    while np.all(E_img[-1, :] != E_img[-1, :]):
        E_img = E_img[0:-1, :]

    max_range = np.nanmax(np.abs(E_img))
    plt.imshow(E_img, cmap="bwr", vmin=-max_range, vmax=max_range)
    plt.ylabel("Alt. Error Img")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(3, 3, 9)
    E_img = HM_Summary["NeRF_after_alignment"]["Image"] - HM_Summary["Ground_Truth"]
    while np.all(E_img[:, 0] != E_img[:, 0]):
        E_img = E_img[:, 1::]
    while np.all(E_img[0, :] != E_img[0, :]):
        E_img = E_img[1::, :]

    while np.all(E_img[:, -1] != E_img[:, -1]):
        E_img = E_img[:, 0:-1]
    while np.all(E_img[-1, :] != E_img[-1, :]):
        E_img = E_img[0:-1, :]

    # max_range = np.nanmax(np.abs(E_img))
    plt.imshow(E_img, cmap="bwr", vmin=-max_range, vmax=max_range)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.savefig(output_loc + "/NeRF_HM.png")
    plt.close("all")
