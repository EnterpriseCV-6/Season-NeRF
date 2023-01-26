# coding=utf-8
import torch as t
from tqdm import tqdm
import numpy as np
from T_NeRF_Eval_Utils.mg_Img_Eval import get_PV, component_render_by_dir
from T_NeRF_Full_2.mg_eval import load_from_input_dict
from all_NeRF import show_dict_struc
import pickle
import csv

def eval_solar_data(rendered_dict):
    Is_Solar_GT = rendered_dict["Exact_Solar"] > .5
    Is_Solar_Est = rendered_dict["Est_Solar_Vis"] > .5

    n_TP = np.sum(Is_Solar_GT * Is_Solar_Est)
    n_TN = np.sum(~Is_Solar_GT * ~Is_Solar_Est)
    n_FP = np.sum(~Is_Solar_GT * Is_Solar_Est)
    n_FN = np.sum(Is_Solar_GT * ~Is_Solar_Est)
    # print(n_TP, n_TN, n_FP, n_FN, (n_TP + n_TN) / (n_TP + n_TN + n_FP + n_FN), n_TP / (n_TP + .5 * n_FP + .5 * n_FN), n_TN / (n_TN + .5 * n_FP + .5 * n_FN))

    all_shadows = {"TP":n_TP, "TN":n_TN, "FP":n_FP, "FN":n_FN}

    PS = get_PV(t.tensor(rendered_dict["Rho"]), t.tensor(rendered_dict["Deltas"])).numpy() * (
                1 - np.exp(-rendered_dict["Rho"] * rendered_dict["Deltas"]))

    Is_Solar_at_surf_GT = np.sum(rendered_dict["Exact_Solar"] * PS, 1) > .3
    Is_Solar_at_surf_Est = np.sum(rendered_dict["Est_Solar_Vis"] * PS, 1) > .3

    n_TP = np.sum(Is_Solar_at_surf_GT * Is_Solar_at_surf_Est)
    n_TN = np.sum(~Is_Solar_at_surf_GT * ~Is_Solar_at_surf_Est)
    n_FP = np.sum(~Is_Solar_at_surf_GT * Is_Solar_at_surf_Est)
    n_FN = np.sum(Is_Solar_at_surf_GT * ~Is_Solar_at_surf_Est)
    # print(n_TP, n_TN, n_FP, n_FN, (n_TP + n_TN) / (n_TP + n_TN + n_FP + n_FN), n_TP / (n_TP + .5 * n_FP + .5 * n_FN), n_TN / (n_TN + .5 * n_FP + .5 * n_FN))
    sub_shadows = {"TP": n_TP, "TN": n_TN, "FP": n_FP, "FN": n_FN}

    ans_Dict = {"All_Solar_Vis":all_shadows, "Surface_Solar_Vis":sub_shadows}
    return ans_Dict


def advanced_solar(input_dict, out_img_size):
    P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device = load_from_input_dict(
        input_dict)

    sat_thetas = np.linspace(0, 360, 4, endpoint=False)
    sat_el_phis = np.linspace(60, 90, 3, endpoint=False)

    solar_thetas = np.linspace(0, 360, 6, endpoint=False)
    solar_el_phis = np.linspace(15, 90, 4, endpoint=False)

    steps = 4*3*6*4
    # out_img_size = (16,16,8)#(128,128,64)


    Full_ans = {"All_Solar_Vis":
                    {"TP": -np.ones([4, 3, 6, 4], dtype=int), "TN":-np.ones([4, 3, 6, 4]),
                     "FP":-np.ones([4, 3, 6, 4]), "FN":-np.ones([4, 3, 6, 4])},
                "Surface_Solar_Vis":
                    {"TP": -np.ones([4, 3, 6, 4], dtype=int), "TN": -np.ones([4, 3, 6, 4]),
                     "FP": -np.ones([4, 3, 6, 4]), "FN": -np.ones([4, 3, 6, 4])},
                "Keys":{"Idx_1_sat_azmuth":sat_thetas,
                        "Idx_2_sat_el":sat_thetas,
                        "Idx_3_solar_azmuth":solar_thetas,
                        "Idx_4_solar_el":solar_el_phis}}


    with tqdm(total=steps, leave=True, desc="Shadow Step") as pbar:
        for i in range(sat_thetas.shape[0]):
            for j in range(sat_el_phis.shape[0]):
                for k in range(solar_thetas.shape[0]):
                    for ell in range(solar_el_phis.shape[0]):
                        test = component_render_by_dir(a_t_nerf, [sat_el_phis[j], sat_thetas[i]], [solar_el_phis[ell], solar_thetas[k]], 0.0, out_img_size, W2C=image_builder.world_center_LLA, W2L_H=image_builder.W2L_H, device=device)
                        ans = eval_solar_data(test)
                        for a_key1 in ans.keys():
                            for a_key2 in ans[a_key1].keys():
                                Full_ans[a_key1][a_key2][i,j,k,ell] = ans[a_key1][a_key2]
                        pbar.update(1)
    return Full_ans

def _get_stats(TP, TN, FP, FN):
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Prec_P = TP / (TP + FP)
    Recall_P = TP / (TP + FN)
    Prec_N = TN / (TN + FN)
    Recall_N = TN / (TN + FP)

    Acc_Balanced = (Prec_P + Prec_N) / 2

    F1_P = 2 * Prec_P * Recall_P / (Prec_P + Recall_P)
    F1_N = 2 * Prec_N * Recall_N / (Prec_N + Recall_N)

    Shadow_Prev = (TN + FP) / (TP + TN + FP + FN)

    return Acc, Acc_Balanced, Shadow_Prev, Prec_P, Recall_P, Prec_N, Recall_N, F1_P, F1_N

def _shadow_analyasis(loaded_data):
    base_keys = ["Accuracy", "Accuracy_Balanced", "Shadow_Prevalence", "Solar_Precision", "Solar_Recall", "Shadow_Precision", "Shadow_Recall", "Solar_F1", "Shadow_F1"]
    all_TP, all_TN, all_FP, all_FN = 0,0,0,0
    for a_region in loaded_data.keys():
        TP = loaded_data[a_region]["All_Solar_Vis"]["TP"]
        TN = loaded_data[a_region]["All_Solar_Vis"]["TN"]
        FP = loaded_data[a_region]["All_Solar_Vis"]["FP"]
        FN = loaded_data[a_region]["All_Solar_Vis"]["FN"]

        all_TP, all_TN, all_FP, all_FN = all_TP + np.sum(TP), all_TN + np.sum(TN), all_FP + np.sum(FP), all_FN + np.sum(FN)

        ans = _get_stats(TP, TN, FP, FN)
        for i in range(9):
            loaded_data[a_region]["All_Solar_Vis"][base_keys[i]] = ans[i]

        ans = _get_stats(np.sum(TP), np.sum(TN), np.sum(FP), np.sum(FN))
        loaded_data[a_region]["All_Solar_Vis"]["Overall"] = {}
        for i in range(9):
            loaded_data[a_region]["All_Solar_Vis"]["Overall"][base_keys[i]] = ans[i]
    # ans = _get_stats(all_TP, all_FN, all_FP, all_FN)
    # All_Solar_Vis_Overall = {}
    # for i in range(9):
    #     All_Solar_Vis_Overall[base_keys[i]] = ans[i]

    all_TP, all_TN, all_FP, all_FN = 0, 0, 0, 0
    for a_region in loaded_data.keys():
        TP = loaded_data[a_region]["Surface_Solar_Vis"]["TP"]
        TN = loaded_data[a_region]["Surface_Solar_Vis"]["TN"]
        FP = loaded_data[a_region]["Surface_Solar_Vis"]["FP"]
        FN = loaded_data[a_region]["Surface_Solar_Vis"]["FN"]

        all_TP, all_TN, all_FP, all_FN = all_TP + np.sum(TP), all_TN + np.sum(TN), all_FP + np.sum(FP), all_FN + np.sum(FN)

        ans = _get_stats(TP, TN, FP, FN)
        for i in range(9):
            loaded_data[a_region]["Surface_Solar_Vis"][base_keys[i]] = ans[i]

        ans = _get_stats(np.sum(TP), np.sum(TN), np.sum(FP), np.sum(FN))
        loaded_data[a_region]["Surface_Solar_Vis"]["Overall"] = {}
        for i in range(9):
            loaded_data[a_region]["Surface_Solar_Vis"]["Overall"][base_keys[i]] = ans[i]

    # ans = _get_stats(all_TP, all_FN, all_FP, all_FN)
    # Surface_Solar_Vis_Overall = {}
    # for i in range(9):
    #     Surface_Solar_Vis_Overall[base_keys[i]] = ans[i]

    # Final_data = {"Original":loaded_data, "All_Overview":All_Solar_Vis_Overall, "Surface_Overview":Surface_Solar_Vis_Overall}

    return loaded_data



def load_and_analyze_shadow_file(file_name):
    fin = open(file_name, "rb")
    loaded_Data = pickle.load(fin)
    fin.close()
    show_dict_struc(loaded_Data)
    loaded_Data = _shadow_analyasis(loaded_Data)

    all_data_table = []
    for a_key in loaded_Data.keys():
        print(a_key)
        S = loaded_Data[a_key]["Surface_Solar_Vis"]["TP"].shape
        for i in range(S[0]):
            for j in range(S[1]):
                for k in range(S[2]):
                    for ell in range(S[3]):
                        a,b,c,d = loaded_Data[a_key]["Keys"]["Idx_1_sat_azmuth"][i],  loaded_Data[a_key]["Keys"]["Idx_2_sat_el"][j],  loaded_Data[a_key]["Keys"]["Idx_3_solar_azmuth"][k],  loaded_Data[a_key]["Keys"]["Idx_4_solar_el"][ell]

                        TP = loaded_Data[a_key]["Surface_Solar_Vis"]["TP"][i,j,k,ell]
                        TN = loaded_Data[a_key]["Surface_Solar_Vis"]["TN"][i,j,k,ell]
                        FP = loaded_Data[a_key]["Surface_Solar_Vis"]["FP"][i,j,k,ell]
                        FN = loaded_Data[a_key]["Surface_Solar_Vis"]["FN"][i,j,k,ell]

                        all_data_table.append([a_key, a,b,c,d, TP, TN, FP, FN])
    print(all_data_table)
    fields = ["Region", "Sat Azmuth", "Sat El", "Solar Az", "Solar El", "TP", "TN", "FP", "FN"]

    with open('./test_csv', "w") as f:
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(all_data_table)
        f.close()

    exit()

    show_dict_struc(loaded_Data)
    for a_key in loaded_Data.keys():
        print(a_key)
        print(loaded_Data[a_key]["Surface_Solar_Vis"]["Overall"])
        print(loaded_Data[a_key]["All_Solar_Vis"]["Overall"])
    exit()
