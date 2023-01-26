# coding=utf-8
import numpy as np
import torch as t
from tqdm import tqdm
from matplotlib import pyplot as plt
from all_NeRF import world_angle_2_local_vec, show_dict_struc
from misc import sample_pt_coarse, zero_invalid_pts
from T_NeRF_Full_2 import get_PV
from .mg_Img_Eval import sig
from tabulate import tabulate

def Sample_Walk_Points_Shadow(P_imgs, testing_idx, points_in_space, points_across_angles, points_in_view, thresh = 5):

    testing_points = []
    training_points = []

    testing_views = []
    training_views = []

    for i in range(len(P_imgs)):
        if i in testing_idx:
            testing_points.append(P_imgs[i].sun_el_and_az)
            testing_views.append([90-P_imgs[i].off_Nadir_from_IMD, P_imgs[i].Azmuth_from_IMD])
        else:
            training_points.append(P_imgs[i].sun_el_and_az)
            training_views.append([90 - P_imgs[i].off_Nadir_from_IMD, P_imgs[i].Azmuth_from_IMD])

    testing_points = np.array(testing_points)
    training_points = np.array(training_points)
    training_views = np.array(training_views)
    testing_views = np.array(testing_views)

    close_walking_points = [[[],[]]]
    n_close_walking_points = 0
    c = 0
    while n_close_walking_points < points_across_angles**2:
        xy = np.stack(np.meshgrid(np.linspace(np.min(training_points[:,0])-thresh,np.max(training_points[:,0])+thresh, points_across_angles+c),
                              np.linspace(np.min(training_points[:,1])-thresh,np.max(training_points[:,1])+thresh, points_across_angles+c), indexing="ij"), -1).reshape([-1,1,2])
        dist = np.sqrt(np.sum(np.abs(xy - np.expand_dims(training_points, 0))**2, 2))
        dist = np.min(dist, 1)
        good = dist < thresh
        n_close_walking_points = np.sum(good)
        close_walking_points = xy[good,0]
        c += 1

    close_walking_views = [[[], []]]
    n_close_walking_views = 0
    c = 0
    while n_close_walking_views < points_in_view ** 2:
        xy = np.stack(np.meshgrid(
            np.linspace(np.min(training_views[:, 0]) - thresh, np.max(training_views[:, 0]) + thresh,
                        points_across_angles + c),
            np.linspace(0, 360, points_across_angles + c, endpoint=False), indexing="ij"), -1).reshape([-1, 1, 2])
        dist = np.sqrt(np.sum(np.abs(xy - np.expand_dims(training_views, 0)) ** 2, 2))
        dist = np.min(dist, 1)
        good = dist < thresh
        n_close_walking_views = np.sum(good)
        close_walking_views = xy[good, 0]
        c += 1


    all_walking_points = np.stack(np.meshgrid(np.linspace(5,90, points_across_angles), np.linspace(0,360,points_across_angles, endpoint=False), indexing="ij"), -1).reshape([-1, 2])
    all_walking_views = np.stack(
        np.meshgrid(np.linspace(np.min(training_views[:,0])-5, 90, points_in_view), np.linspace(0, 360, points_in_view, endpoint=False),
                    indexing="ij"), -1).reshape([-1, 2])

    ground_points = np.stack(np.meshgrid(np.linspace(-1,1, points_in_space), np.linspace(-1,1,points_in_space), indexing="ij"), -1).reshape([-1, 2])


    return training_points, testing_points, close_walking_points, all_walking_points, ground_points, testing_views, training_views, close_walking_views, all_walking_views

def eval_shadow_data(shadow_net, shadow_angles, ground_points, Z_points, world_center_LLA, W2L_H, max_batch_size, device):
    with t.no_grad():
        Zero_Tool = zero_invalid_pts()

        Results_Vis_Exact = np.zeros([shadow_angles.shape[0], ground_points.shape[0], Z_points, 1])
        Results_Vis_Est = np.zeros([shadow_angles.shape[0], ground_points.shape[0], Z_points, 1])
        Results_Sky_Col = np.zeros([shadow_angles.shape[0], 3])

        sun_el_and_az_vec_original = np.array([world_angle_2_local_vec(shadow_angles[i,0], shadow_angles[i,1], world_center_LLA, W2L_H) for i in tqdm(range(shadow_angles.shape[0]), desc="Converting Solar Angles to vecs", leave=False)])
        sun_el_and_az_vec = sun_el_and_az_vec_original / sun_el_and_az_vec_original[:, -1::]
        tops = t.tensor(np.expand_dims(np.concatenate([ground_points, np.zeros([ground_points.shape[0],1])], 1), 0) + np.expand_dims(sun_el_and_az_vec, 1)).float()
        bots = t.tensor(np.expand_dims(np.concatenate([ground_points, np.zeros([ground_points.shape[0],1])], 1), 0) - np.expand_dims(sun_el_and_az_vec, 1)).float()
        max_step = max(max_batch_size // (Z_points), 1)
        for i in tqdm(range(tops.shape[0]), desc="Rendering shadows...", leave=False):
            for j in range(0, tops.shape[1], max_step):
                j_end = min(j + max_step, tops.shape[1])
                Xs, deltas = sample_pt_coarse(tops[i, j:j_end], bots[i, j:j_end], Z_points, eval_mode=True)
                deltas[Zero_Tool(Xs)] = 0.
                solar_vec = t.stack([t.tensor(sun_el_and_az_vec_original[i]).float()] * (Xs.shape[0] * Xs.shape[1]), 0)


                rho, vis, sky_col = shadow_net.forward_Solar(Xs.reshape([-1,3]).to(device), solar_vec.to(device), None)
                rho = rho.cpu().reshape([Xs.shape[0], Z_points, 1])
                vis = vis.cpu().reshape([Xs.shape[0], Z_points, 1])

                PV = get_PV(rho, deltas)

                Results_Vis_Exact[i,j:j_end] = PV.numpy()
                Results_Vis_Est[i, j:j_end] = vis
                if j == 0:
                    sky_col = sky_col.cpu().reshape([Xs.shape[0], Z_points, -1])
                    Results_Sky_Col[i] = sky_col[0,0].numpy()
        return Results_Vis_Exact, Results_Vis_Est, Results_Sky_Col


def Test_Shadow_Points(shadow_net, training_points, testing_points, close_walking_points, all_walking_points, ground_points, world_center_LLA, W2L_H, device, Z_points = 96, max_batch_size = 15000, full_return = True):

    Shadow_Summary = {"Ground_Points":ground_points, "Sun_El_Az":{"Training":training_points, "Testing":testing_points, "Near_Walk":close_walking_points, "Full_Walk":all_walking_points}}

    print("Getting training point data...")
    Results_Vis_Exact, Results_Vis_Est, Results_Sky_Col = eval_shadow_data(shadow_net, training_points, ground_points, Z_points, world_center_LLA, W2L_H, max_batch_size, device)
    ans = {"Exact_Vis":Results_Vis_Exact, "Est_Vis":Results_Vis_Est, "Sky_Col":Results_Sky_Col}
    Shadow_Summary["Training_Results"] = ans
    print("Getting testing point data...")
    Results_Vis_Exact, Results_Vis_Est, Results_Sky_Col = eval_shadow_data(shadow_net, testing_points, ground_points, Z_points, world_center_LLA, W2L_H, max_batch_size, device)
    ans = {"Exact_Vis": Results_Vis_Exact, "Est_Vis": Results_Vis_Est, "Sky_Col": Results_Sky_Col}
    Shadow_Summary["Testing_Results"] = ans
    print("Getting near point data...")
    Results_Vis_Exact, Results_Vis_Est, Results_Sky_Col = eval_shadow_data(shadow_net, close_walking_points, ground_points, Z_points, world_center_LLA, W2L_H, max_batch_size, device)
    ans = {"Exact_Vis": Results_Vis_Exact, "Est_Vis": Results_Vis_Est, "Sky_Col": Results_Sky_Col}
    Shadow_Summary["Near_Results"] = ans
    print("Getting full point data...")
    Results_Vis_Exact, Results_Vis_Est, Results_Sky_Col = eval_shadow_data(shadow_net, all_walking_points, ground_points, Z_points, world_center_LLA, W2L_H, max_batch_size, device)
    ans = {"Exact_Vis": Results_Vis_Exact, "Est_Vis": Results_Vis_Est, "Sky_Col": Results_Sky_Col}
    Shadow_Summary["Full_Results"] = ans

    if full_return == False:
        Shadow_Summary = _Summary_to_short(Shadow_Summary)

    return Shadow_Summary


def shadow_anaylysis(Ground_Points, Solar_el_az, Results_Dict):
    # print(Ground_Points.shape, Solar_el_az.shape)
    # show_dict_struc(Results_Dict)
    #
    # print(Results_Dict["Exact_Vis"].shape, Results_Dict["Est_Vis"].shape)
    direct_loss = np.mean((Results_Dict["Exact_Vis"]- Results_Dict["Est_Vis"])**2)
    avg_error = np.mean(np.abs(Results_Dict["Exact_Vis"] - Results_Dict["Est_Vis"]))

    thresh_GT = Results_Dict["Exact_Vis"] > .5
    thresh_Est = Results_Dict["Est_Vis"] > .5

    TP = np.sum(thresh_GT * thresh_Est)
    TN = np.sum(((~thresh_GT) * (~thresh_Est)))
    FP = np.sum((~thresh_GT) * thresh_Est)
    FN = np.sum(thresh_GT * (~thresh_Est))

    Acc = (TP + TN) / (TP + TN + FP + FN)
    Prec_Sun = (TP) / (TP + FP)
    Recall_Sun = (TP) / (TP + FN)
    Prec_Shadow = (TN) / (TN + FN)
    Recall_Shadow = (TN) / (TN + FP)

    Surf_Dist = np.sum(thresh_GT, 2) - np.sum(thresh_Est, 2)
    avg_offset = np.mean(np.abs(Surf_Dist))

    # print(TP, TN, FP, FN)
    # print(Acc, Prec_Sun, Recall_Sun, Prec_Shadow, Recall_Shadow)
    # print(direct_loss, avg_error, avg_offset)
    ans_dict = {"Acc":Acc, "Prec_Sun":Prec_Sun, "Recall_Sun":Recall_Sun, "Prec_Shadow":Prec_Shadow, "Recall_Shadow":Recall_Shadow, "Loss":direct_loss, "Avg_Error":avg_error, "Avg_Offset":avg_offset}
    return ans_dict

def _Summary_to_short(Shadow_Summary):
    final_ans = {}
    final_ans["Training"] = shadow_anaylysis(Shadow_Summary["Ground_Points"], Shadow_Summary["Sun_El_Az"]["Training"],
                                             Shadow_Summary["Training_Results"])
    final_ans["Testing"] = shadow_anaylysis(Shadow_Summary["Ground_Points"], Shadow_Summary["Sun_El_Az"]["Testing"],
                                            Shadow_Summary["Testing_Results"])
    final_ans["Near"] = shadow_anaylysis(Shadow_Summary["Ground_Points"], Shadow_Summary["Sun_El_Az"]["Near_Walk"],
                                         Shadow_Summary["Near_Results"])
    final_ans["Full"] = shadow_anaylysis(Shadow_Summary["Ground_Points"], Shadow_Summary["Sun_El_Az"]["Full_Walk"],
                                         Shadow_Summary["Full_Results"])

    return final_ans

def Orgainize_Output_Imgs_Shadows(Shadow_Summary, output_loc, already_short = False):
    # show_dict_struc(Shadow_Summary)
    #
    # print(Shadow_Summary["Ground_Points"].shape)
    # print(Shadow_Summary["Ground_Points"].reshape([20,20,2])[:,:,0])
    # shadow_len_exact = np.sum(Shadow_Summary["Full_Results"]["Exact_Vis"] <= .5, 2)
    # shadow_len_est = np.sum(Shadow_Summary["Full_Results"]["Est_Vis"] <= .5, 2)
    #
    # diff = shadow_len_exact - shadow_len_est
    # avg_abs_diff = np.mean((diff), 0).reshape([int(diff.shape[0]**.5), int(diff.shape[0]**.5)])
    # val = np.max(np.abs(avg_abs_diff))
    # plt.imshow(avg_abs_diff, cmap="bwr", vmin=-val, vmax=val)
    # plt.colorbar()
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    if already_short == False:
        final_ans = _Summary_to_short(Shadow_Summary)
    else:
        final_ans = Shadow_Summary
    table_col_labels = (
        "Case", "Acc", "Prec. Sun", "Recall Sun", "Prec. Shadow", "Recall Shadow", "Loss", "Avg. Error", "Avg. Offset")
    Summary_table = []
    for a_key in final_ans.keys():
        Summary_table.append([a_key] + [final_ans[a_key][i] for i in final_ans[a_key].keys()])

    fout = open(output_loc + "/Shadow_Summary.txt", "w")
    fout.write("Overview\n")
    fout.write(tabulate(Summary_table, headers=table_col_labels))
    fout.close()
    # exit()

    return final_ans

