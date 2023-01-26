# coding=utf-8
from tqdm import tqdm
import numpy as np
from misc import sample_pt_coarse, zero_invalid_pts
import torch as t
import torch.utils.data as tdata
from T_NeRF_Full_2 import encode_time, get_PV
from matplotlib import pyplot as plt
from copy import deepcopy
from all_NeRF import show_dict_struc
import cv2 as cv
from T_NeRF_Eval_Utils import mask_ssim, mask_PSNR
from all_NeRF.mg_EM_Imgs import mg_EM, compare_EM_imgs
from all_NeRF import world_angle_2_local_vec, CV_reshape
from tabulate import tabulate

def _internal_render(the_network, tops, bots, a_sun_el_az_vec, a_year_frac, out_img_size, max_batch_size, include_exact_solar, device):
    Zero_Tool = zero_invalid_pts()
    key_list = ["World_Points", "Deltas", "Rho", "Base_Col", "Est_Solar_Vis", "Sky_Col", "Output_class", "Adjust_col"]
    out_size_last_idx = [3, 1, 1, 3, 1, 3, the_network.n_classes, the_network.n_classes * 3]
    if include_exact_solar:
        key_list += ["Exact_Solar"]
        out_size_last_idx += [1]
    Results_Dict = {}
    idx = 0
    for a_key in key_list:
        if a_key != "Adjust_col":
            Results_Dict[a_key] = np.zeros([tops.shape[0], out_img_size[2], out_size_last_idx[idx]])
        else:
            Results_Dict[a_key] = np.zeros([tops.shape[0], out_img_size[2], the_network.n_classes, 3])
        idx += 1

    if include_exact_solar == False:
        step_size = max_batch_size // out_img_size[2]
    else:
        step_size = max_batch_size // (out_img_size[2] ** 2)

    for i in tqdm(range(0, tops.shape[0], step_size), desc="Rendering image...", leave=False):
        i_end = min(i + step_size, tops.shape[0])
        pts, deltas = sample_pt_coarse(tops[i:i_end], bots[i:i_end], out_img_size[2], eval_mode=True,
                                       include_end_pt=True)
        deltas[Zero_Tool(pts)] = 0.
        solar_angles = t.tensor([a_sun_el_az_vec] * ((pts.shape[0]) * out_img_size[2])).float().reshape(
            [-1, 3]).to(device)
        times = t.tensor([encode_time(a_year_frac)] * ((pts.shape[0]) * out_img_size[2])).float().reshape(
            [-1, 4]).to(device)
        outs = the_network.forward_seperate(pts.reshape([-1, 3]).to(device), solar_angles, times)
        idx = 2
        for j in range(5):
            Results_Dict[key_list[idx]][i:i_end] = outs[j].reshape([pts.shape[0], out_img_size[2], -1]).cpu().numpy()
            idx += 1
        Results_Dict[key_list[idx]][i:i_end] = outs[-1].reshape(
            [pts.shape[0], out_img_size[2], the_network.n_classes, -1]).cpu().numpy()
        Results_Dict[key_list[0]][i:i_end] = pts.numpy()
        Results_Dict[key_list[1]][i:i_end] = deltas.numpy()

        if include_exact_solar:
            new_bots = pts.reshape([-1, 3])
            S = (1. - new_bots[:, 2]) / a_sun_el_az_vec[2]
            new_tops = (new_bots + S.reshape([-1, 1]) * a_sun_el_az_vec.reshape([1, -1])).float()
            new_pts, new_deltas = sample_pt_coarse(new_tops, new_bots, out_img_size[2], eval_mode=True,
                                                   include_end_pt=True)
            bad_points = Zero_Tool(new_pts)
            new_deltas[bad_points] = 0.
            rhos = the_network.forward_Classic_Sigma_Only(new_pts.reshape([-1, 3]).to(device)).reshape(
                [new_tops.shape[0], out_img_size[2], 1]).cpu()

            PV_solar_exact = t.exp(-t.sum((rhos * new_deltas)[:, 0:-1, :], 1)).reshape(
                [pts.shape[0], out_img_size[2], 1])
            Results_Dict["Exact_Solar"][i:i_end] = PV_solar_exact.numpy()

    return Results_Dict

def component_render_by_P(the_network, a_P_img, out_img_size:tuple, device, max_batch_size = 150000, include_exact_solar = True):
    with t.no_grad():
        XY = np.stack(np.meshgrid(np.linspace(0, a_P_img.img.shape[0]-1, out_img_size[0]), np.linspace(0, a_P_img.img.shape[1]-1, out_img_size[1]), indexing="ij"), -1)
        XY = np.round(XY).astype(int).reshape([-1,2])
        x,y,z = a_P_img.invert_P(XY[:,0], XY[:,1], 1.)
        tops = np.stack([x,y,np.ones_like(x)], -1)
        x,y,z = a_P_img.invert_P(XY[:,0], XY[:,1], -1.)
        bots = np.stack([x, y, -np.ones_like(x)], -1)

        good = (tops[:,0] >= -1) * (tops[:,1] <= 1) * (bots[:,0] >= -1) * (bots[:,1] <= 1) * \
               (tops[:, 1] >= -1) * (tops[:, 0] <= 1) * (bots[:, 1] >= -1) * (bots[:, 0] <= 1)
        tops = t.tensor(tops[good]).float()
        bots = t.tensor(bots[good]).float()
        Results_Dict = _internal_render(the_network, tops, bots, a_P_img.sun_el_and_az_vec, a_P_img.get_year_frac(), out_img_size, max_batch_size, include_exact_solar, device)

        Results_Dict["Image_Points_in_GT_Img"] = XY[good]
        XY_img = np.stack(np.meshgrid(np.arange(out_img_size[0]),
                                  np.arange(out_img_size[1]), indexing="ij"), -1).reshape([-1, 2])[good]

        Results_Dict["Image_Points"] = XY_img
    return Results_Dict

def component_render_by_dir(the_network, view_el_az, sun_el_az, time_frac, out_img_size:tuple, W2C, W2L_H, device, max_batch_size = 150000, include_exact_solar = True):
    with t.no_grad():
        XYZ = np.stack(np.meshgrid(np.linspace(1, -1, out_img_size[0]), np.linspace(-1, 1, out_img_size[1]), indexing="ij"), -1).reshape([-1,2])
        XYZ = np.concatenate([XYZ, np.zeros([XYZ.shape[0], 1])], 1)
        view_vec = world_angle_2_local_vec(view_el_az[0], view_el_az[1], W2C, W2L_H)
        sun_el_and_az_vec = world_angle_2_local_vec(sun_el_az[0], sun_el_az[1], W2C, W2L_H)
        tops = XYZ + np.expand_dims(view_vec/view_vec[2], 0)
        bots = XYZ - np.expand_dims(view_vec / view_vec[2], 0)

        tops = t.tensor(tops).float()
        bots = t.tensor(bots).float()
        Results_Dict = _internal_render(the_network, tops, bots, sun_el_and_az_vec, time_frac,
                                        out_img_size, max_batch_size, include_exact_solar, device)

        XY_img = np.stack(np.meshgrid(np.arange(out_img_size[0]),
                                      np.arange(out_img_size[1]), indexing="ij"), -1).reshape([-1,2])


        Results_Dict["Image_Points"] = XY_img
    return Results_Dict

def sig(X):
    return 1/(1+np.exp(-X))

def inv_sig(X):
    return -np.log(1/X - 1)

def get_imgs_from_Img_Dict(Img_Dict, out_img_size:tuple, use_classic_shadows:bool):
    with t.no_grad():
        Sky_Col = Img_Dict["Sky_Col"][0, 0]
        Est_Out_Class = Img_Dict["Output_class"][0, 0]

        PS = get_PV(t.tensor(Img_Dict["Rho"]), t.tensor(Img_Dict["Deltas"])).numpy() * (1-np.exp(-Img_Dict["Rho"] * Img_Dict["Deltas"]))

        Base_Cols = np.sum(PS * sig(Img_Dict["Base_Col"]), 1)
        Base_Col_img = np.zeros([out_img_size[0], out_img_size[1], 3]) * np.nan
        Base_Col_img[Img_Dict["Image_Points"][:,0], Img_Dict["Image_Points"][:,1]] = Base_Cols

        Shadow_Effects = np.sum(PS * Img_Dict["Est_Solar_Vis"], 1)[:,0]
        Shadow_Mask_unscaled = np.zeros([out_img_size[0], out_img_size[1]]) * np.nan
        Shadow_Mask_unscaled[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Shadow_Effects
        Shadow_Mask = sig((Shadow_Mask_unscaled-.2)*30)

        Shadow_Adjust_mask = np.expand_dims(Shadow_Mask, -1) + np.expand_dims((1 - Shadow_Mask), -1) * Sky_Col.reshape([1, 1, 3])



        if "Exact_Solar" in Img_Dict.keys():
            Shadow_Effects = np.sum(PS * Img_Dict["Exact_Solar"], 1)[:, 0]
            Shadow_Mask_unscaled_exact = np.zeros([out_img_size[0], out_img_size[1]]) * np.nan
            Shadow_Mask_unscaled_exact[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Shadow_Effects
            Shadow_Mask_exact = sig((Shadow_Mask_unscaled_exact - .2) * 30)

            Shadow_Adjust_mask_exact = np.expand_dims(Shadow_Mask_exact, -1) + np.expand_dims((1 - Shadow_Mask_exact), -1) * Sky_Col.reshape([1, 1, 3])
            # Shadow_Adjust_mask_exact = np.expand_dims(Shadow_Mask_exact + (1 - Shadow_Mask_exact), 2) * Sky_Col.reshape([1, 1, 3])


        Est_Col_Img = np.zeros([out_img_size[0], out_img_size[1], 3]) * np.nan
        Cols_season = np.sum(PS * sig(Img_Dict["Base_Col"] + (np.expand_dims(Img_Dict["Output_class"], 2) @ Img_Dict["Adjust_col"])[:,:,0,:]), 1)
        Est_Col_Img[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Cols_season


        Extreme_Imgs = []
        for i in range(Img_Dict["Adjust_col"].shape[2]):
            an_img = np.zeros([out_img_size[0], out_img_size[1], 3]) * np.nan
            Cols = np.sum(PS * sig(Img_Dict["Base_Col"] + Img_Dict["Adjust_col"][:,:,i]), 1)
            an_img[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Cols
            Extreme_Imgs.append(an_img)

        if use_classic_shadows:
            shadow_term = Img_Dict["Est_Solar_Vis"] + (1-Img_Dict["Est_Solar_Vis"]) * Img_Dict["Sky_Col"]
            classic_col_adj = sig(Img_Dict["Base_Col"] + (np.expand_dims(Img_Dict["Output_class"], 2) @ Img_Dict["Adjust_col"])[:,:,0,:])
            classic_col_adj *= shadow_term
            classic_shadow_img = np.sum(PS * classic_col_adj, 1)
            quasi_shadow_mask = classic_shadow_img / (Cols_season + 1e-8)
            Shadow_Adjust_mask[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = quasi_shadow_mask
            if "Exact_Solar" in Img_Dict.keys():
                shadow_term = Img_Dict["Exact_Solar"] + (1 - Img_Dict["Exact_Solar"]) * Img_Dict["Sky_Col"]
                classic_col_adj = sig(
                    Img_Dict["Base_Col"] + (np.expand_dims(Img_Dict["Output_class"], 2) @ Img_Dict["Adjust_col"])[:, :,
                                           0, :])
                classic_col_adj *= shadow_term
                classic_shadow_img = np.sum(PS * classic_col_adj, 1)
                quasi_shadow_mask = classic_shadow_img / (Cols_season + 1e-8)
                Shadow_Adjust_mask_exact[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = quasi_shadow_mask


        Results = {"Base_Img":Base_Col_img, "Season_Adj_Img":Est_Col_Img, "Extreme_Imgs":Extreme_Imgs, "Shadow_Adjust":Shadow_Adjust_mask, "Shadow_Mask":Shadow_Mask, "Raw_Shadow_Mask":Shadow_Mask_unscaled, "Sky_Col":Sky_Col, "Time_Class":Est_Out_Class}
        if "Exact_Solar" in Img_Dict.keys():
            #"Shadow_Adjust":Shadow_Adjust_mask, "Shadow_Mask":Shadow_Mask, "Raw_Shadow_Mask":Shadow_Mask_unscaled
            Results["Shadow_Adjust_Exact"] = Shadow_Adjust_mask_exact
            Results["Shadow_Mask_Exact"] = Shadow_Mask_exact
            Results["Raw_Shadow_Mask_Exact"] = Shadow_Mask_unscaled_exact

    return Results

def get_imgs_from_Img_Dict_t_step(Img_Dict, out_img_size:tuple, class_vecs_array):
    with t.no_grad():
        Sky_Col = Img_Dict["Sky_Col"][0, 0]

        PS = get_PV(t.tensor(Img_Dict["Rho"]), t.tensor(Img_Dict["Deltas"])).numpy() * (1-np.exp(-Img_Dict["Rho"] * Img_Dict["Deltas"]))

        Base_Cols = np.sum(PS * sig(Img_Dict["Base_Col"]), 1)
        Base_Col_img = np.zeros([out_img_size[0], out_img_size[1], 3]) * np.nan
        Base_Col_img[Img_Dict["Image_Points"][:,0], Img_Dict["Image_Points"][:,1]] = Base_Cols

        Shadow_Effects = np.sum(PS * Img_Dict["Est_Solar_Vis"], 1)[:,0]
        Shadow_Mask_unscaled = np.zeros([out_img_size[0], out_img_size[1]]) * np.nan
        Shadow_Mask_unscaled[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Shadow_Effects
        Shadow_Mask = sig((Shadow_Mask_unscaled-.2)*30)

        Shadow_Adjust_mask = np.expand_dims(Shadow_Mask, -1) + np.expand_dims((1 - Shadow_Mask), -1) * Sky_Col.reshape([1, 1, 3])



        if "Exact_Solar" in Img_Dict.keys():
            Shadow_Effects = np.sum(PS * Img_Dict["Exact_Solar"], 1)[:, 0]
            Shadow_Mask_unscaled_exact = np.zeros([out_img_size[0], out_img_size[1]]) * np.nan
            Shadow_Mask_unscaled_exact[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Shadow_Effects
            Shadow_Mask_exact = sig((Shadow_Mask_unscaled_exact - .2) * 30)

            Shadow_Adjust_mask = np.expand_dims(Shadow_Mask_exact, -1) + np.expand_dims((1 - Shadow_Mask_exact), -1) * Sky_Col.reshape([1, 1, 3])
            # Shadow_Adjust_mask_exact = np.expand_dims(Shadow_Mask_exact + (1 - Shadow_Mask_exact), 2) * Sky_Col.reshape([1, 1, 3])


        all_Imgs = []
        for i in tqdm(range(class_vecs_array.shape[0]), leave=False, desc="Walking through times"):
            Est_Col_Img = np.zeros([out_img_size[0], out_img_size[1], 3]) * np.nan
            Cols = np.sum(PS * sig(Img_Dict["Base_Col"] + (class_vecs_array[i].reshape([1,1,-1]) @ Img_Dict["Adjust_col"])[:,:,0,:]), 1)
            Est_Col_Img[Img_Dict["Image_Points"][:, 0], Img_Dict["Image_Points"][:, 1]] = Cols
            all_Imgs.append(Est_Col_Img * Shadow_Adjust_mask)
        all_Imgs = np.array(all_Imgs)
    return all_Imgs



class pt_loader(tdata.Dataset):
    def __init__(self, Img_Dict, GT, use_exact_vis = False):
        PS = get_PV(t.tensor(Img_Dict["Rho"]), t.tensor(Img_Dict["Deltas"])).numpy() * (
                    1 - np.exp(-Img_Dict["Rho"] * Img_Dict["Deltas"]))
        Base_Cols = Img_Dict["Base_Col"]
        Adjust_Cols = Img_Dict["Adjust_col"]
        if use_exact_vis:
            Solar_Vis = Img_Dict["Exact_Solar"]
        else:
            Solar_Vis = Img_Dict["Est_Solar_Vis"]

        self.PS = t.tensor(PS).float()
        self.Base_Cols = t.tensor(Base_Cols).float()
        self.Adjust_Cols = t.tensor(Adjust_Cols).float()
        self.Solar_Vis = t.tensor(sig((np.sum(PS * Solar_Vis, 1) - .2) * 30)).float()
        self.GT = t.tensor(GT).float()
        # self._calls = 0


    def __len__(self):
        return self.PS.shape[0]

    def __getitem__(self, item):
        return self.PS[item], self.Base_Cols[item], self.Adjust_Cols[item], self.Solar_Vis[item], self.GT[item]

def Grad_Descent_Seasonal_Align(Results_Dict, target_img, batch_size = 15000, steps = 100):
    # print(Results_Dict.keys())
    # print(Results_Dict["Output_class"].shape)
    # print(Results_Dict["Image_Points_in_GT_Img"].shape)

    Adjust_Time = t.nn.Parameter(t.tensor(np.log(Results_Dict["Output_class"][0,0])).reshape([1,1,-1,1]).float(), requires_grad=True)
    Sky_Col = t.nn.Parameter(t.tensor(inv_sig(Results_Dict["Sky_Col"])[0, 0]).reshape([1,-1]).float(), requires_grad=True)

    GT_Cols = target_img[Results_Dict["Image_Points_in_GT_Img"][:, 0], Results_Dict["Image_Points_in_GT_Img"][:, 1]]

    the_loader = pt_loader(Results_Dict, GT_Cols)
    the_loss_calc = t.nn.MSELoss()

    the_opt = t.optim.Adam(params=[Adjust_Time, Sky_Col], lr=.1)
    data_caller = t.utils.data.DataLoader(the_loader, batch_size = batch_size, shuffle=True)
    # print(len(the_loader))
    step_counter = 0
    sig_t = t.nn.Sigmoid()
    with tqdm(total=steps, leave=False, desc="Align Season") as pbar:
        while True:
            for batch_idx, (PS, Base_Cols, Adjust_Cols, Solar_Vis, GT) in enumerate(data_caller):
                the_opt.zero_grad()
                Rendered_Col = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * t.exp(Adjust_Time)/t.sum(t.exp(Adjust_Time)), 2)), 1) * (Solar_Vis + (1 - Solar_Vis) * sig_t(Sky_Col))
                error = the_loss_calc(Rendered_Col, GT)
                error.backward()
                the_opt.step()
                # print(step_counter, error.item(), (t.exp(Adjust_Time)/t.sum(t.exp(Adjust_Time))).detach()[0,0,:,0], sig_t(Sky_Col).detach()[0])


                step_counter += 1
                if step_counter > steps:
                    break
                pbar.update(1)
            if step_counter > steps:
                break
    return (t.exp(Adjust_Time)/t.sum(t.exp(Adjust_Time))).detach()[:,:,:,0], sig_t(Sky_Col.detach())

def Grad_Descent_Seasonal_Align_v2(Results_Dict, target_img, t0, network, device, batch_size = 15000, steps = 100):
    # print(Results_Dict.keys())
    # print(Results_Dict["Output_class"].shape)
    # print(Results_Dict["Image_Points_in_GT_Img"].shape)

    Adjust_Time = t.nn.Parameter(t.tensor([t0]).float(), requires_grad=True)
    Sky_Col = t.nn.Parameter(t.tensor(inv_sig(Results_Dict["Sky_Col"])[0, 0]).reshape([1,-1]).float(), requires_grad=True)

    GT_Cols = target_img[Results_Dict["Image_Points_in_GT_Img"][:, 0], Results_Dict["Image_Points_in_GT_Img"][:, 1]]

    # print(GT_Cols.shape)
    print(Adjust_Time)

    the_loss_calc = t.nn.MSELoss()

    the_opt = t.optim.Adam(params=[Adjust_Time], lr=.1)
    # print(len(the_loader))
    step_counter = 0
    sig_t = t.nn.Sigmoid()

    PS = get_PV(t.tensor(Results_Dict["Rho"]), t.tensor(Results_Dict["Deltas"])).numpy() * (
            1 - np.exp(-Results_Dict["Rho"] * Results_Dict["Deltas"]))
    Base_Cols = Results_Dict["Base_Col"]
    Adjust_Cols = Results_Dict["Adjust_col"]
    # if use_exact_vis:
    #     Solar_Vis = Results_Dict["Exact_Solar"]
    # else:
    Solar_Vis = t.tensor(Results_Dict["Est_Solar_Vis"]).float().to(device)

    PS = t.tensor(PS).float().to(device)
    Base_Cols = t.tensor(Base_Cols).float().to(device)
    Adjust_Cols = t.tensor(Adjust_Cols).float().to(device)
    Solar_Vis = sig_t((t.sum(PS * Solar_Vis, 1) - .2) * 30)
    GT = t.tensor(GT_Cols).float().to(device)


    for i in tqdm(range(steps), leave=False, desc="Align Season"):
        the_opt.zero_grad()
        ti = t.stack([t.cos(Adjust_Time* 2 * np.pi), t.sin(Adjust_Time* 2 * np.pi), t.cos(Adjust_Time* 2 * np.pi), t.sin(Adjust_Time* 2 * np.pi)], 1).reshape([1,4])
        tv = network.get_class_only(ti.to(device)).reshape([1,1,-1, 1])
        Rendered_Col = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv, 2)), 1) * (Solar_Vis + (1 - Solar_Vis) * sig_t(Sky_Col.to(device)))
        error = the_loss_calc(Rendered_Col, GT)
        error.backward()
        the_opt.step()
        if (i+1) % 10 == 0:
            print(i, Adjust_Time.item(), error.item())

    ti = t.stack([t.cos(Adjust_Time * 2 * np.pi), t.sin(Adjust_Time * 2 * np.pi), t.cos(Adjust_Time * 2 * np.pi),
                  t.sin(Adjust_Time * 2 * np.pi)], 1).reshape([1, 4])
    tv = network.get_class_only(ti.to(device)).reshape([1, 1, -1, 1])


    return tv.detach()[:,:,:,0].cpu(), sig_t(Sky_Col.detach()), Adjust_Time.cpu().detach().item()

def Grad_Descent_Seasonal_Align_v3(Results_Dict, target_img, t0, network, device, batch_size = 15000, steps = 100, use_classic_shadows=False):
    if use_classic_shadows:
        return _grad_descent_v3_classic_shadows(Results_Dict, target_img, t0, network, device)
    else:
        return _grad_descent_v3(Results_Dict, target_img, t0, network, device)

def _grad_descent_v3(Results_Dict, target_img, t0, network, device):
    ts = t.tensor([t0] + list(np.linspace(0, 1, 366))).float()

    ts_scaled = t.stack([t.cos(ts * 2 * np.pi), t.sin(ts * 2 * np.pi), t.cos(ts * 2 * np.pi),
                         t.sin(ts * 2 * np.pi)], 1).reshape([ts.shape[0], -1])
    with t.no_grad():
        tv = network.get_class_only(ts_scaled.to(device))

        GT_Cols = target_img[Results_Dict["Image_Points_in_GT_Img"][:, 0], Results_Dict["Image_Points_in_GT_Img"][:, 1]]
        GT = t.tensor(GT_Cols).float().to(device)
        the_loss_calc = t.nn.MSELoss()
        sig_t = t.nn.Sigmoid()

        PS = get_PV(t.tensor(Results_Dict["Rho"]), t.tensor(Results_Dict["Deltas"])).numpy() * (
                1 - np.exp(-Results_Dict["Rho"] * Results_Dict["Deltas"]))
        Base_Cols = Results_Dict["Base_Col"]
        Adjust_Cols = Results_Dict["Adjust_col"]
        # if use_exact_vis:
        #     Solar_Vis = Results_Dict["Exact_Solar"]
        # else:
        Solar_Vis = t.tensor(Results_Dict["Est_Solar_Vis"]).float().to(device)

        PS = t.tensor(PS).float().to(device)
        Base_Cols = t.tensor(Base_Cols).float().to(device)
        Adjust_Cols = t.tensor(Adjust_Cols).float().to(device)
        Solar_Vis = sig_t((t.sum(PS * Solar_Vis, 1) - .2) * 30)

        good = (Solar_Vis < .99)[:, 0]
        GT_good = GT[good]
        Solar_Vis_good = Solar_Vis[good]

        scores = np.ones(ts.shape[0])
        all_Sky_cols = np.zeros([ts.shape[0], 3])

        for i in tqdm(range(ts.shape[0]), leave=False, desc="Align Season"):
            # Rendered_Col_no_sky = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1,1,-1,1]), 2)), 1) * (Solar_Vis + (1 - Solar_Vis) * Sky_Col)
            A = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1, 1, -1, 1]), 2)), 1)
            Y = GT_good - A[good] * Solar_Vis_good
            X = (1 - Solar_Vis_good) * A[good]
            T1 = 1 / t.sum(X * X, 0)
            T2 = t.sum(X * Y, 0)
            Sky_Col = t.clamp(T1 * T2, 0, 1)
            Rendered_Col = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1, 1, -1, 1]), 2)), 1) * (
                        Solar_Vis + (1 - Solar_Vis) * Sky_Col)
            error = the_loss_calc(Rendered_Col, GT)
            scores[i] = error.cpu().item()
            all_Sky_cols[i] = Sky_Col.cpu().numpy()

        best_idx = np.argmin(scores)
        # print("Base Time:", t0)
        # print("Best Time:", ts[best_idx].item())
        # print("Best Sky Col:", all_Sky_cols[best_idx])
        # plt.scatter(ts.cpu().numpy(), scores)
        # plt.show()

        adj_vec = tv[best_idx]
        # print(adj_vec)

        # exit()

    return adj_vec.cpu(), t.tensor(all_Sky_cols[best_idx]).reshape([1, 1, 3]).float(), ts[best_idx].item()

def _grad_descent_v3_classic_shadows(Results_Dict, target_img, t0, network, device):
    ts = t.tensor([t0] + list(np.linspace(0, 1, 366))).float()

    ts_scaled = t.stack([t.cos(ts * 2 * np.pi), t.sin(ts * 2 * np.pi), t.cos(ts * 2 * np.pi),
                         t.sin(ts * 2 * np.pi)], 1).reshape([ts.shape[0], -1])
    with t.no_grad():
        tv = network.get_class_only(ts_scaled.to(device))

        GT_Cols = target_img[Results_Dict["Image_Points_in_GT_Img"][:, 0], Results_Dict["Image_Points_in_GT_Img"][:, 1]]
        GT = t.tensor(GT_Cols).float().to(device)
        the_loss_calc = t.nn.MSELoss()
        sig_t = t.nn.Sigmoid()

        PS = get_PV(t.tensor(Results_Dict["Rho"]), t.tensor(Results_Dict["Deltas"])).numpy() * (
                1 - np.exp(-Results_Dict["Rho"] * Results_Dict["Deltas"]))
        Base_Cols = Results_Dict["Base_Col"]
        Adjust_Cols = Results_Dict["Adjust_col"]
        # if use_exact_vis:
        #     Solar_Vis = Results_Dict["Exact_Solar"]
        # else:
        Solar_Vis = t.tensor(Results_Dict["Est_Solar_Vis"]).float().to(device)

        PS = t.tensor(PS).float().to(device)
        Base_Cols = t.tensor(Base_Cols).float().to(device)
        Adjust_Cols = t.tensor(Adjust_Cols).float().to(device)

        # good = (Solar_Vis < .99)[:, 0]
        # GT_good = GT[good]
        # Solar_Vis_good = Solar_Vis[good]

        scores = np.ones(ts.shape[0])
        all_Sky_cols = np.zeros([ts.shape[0], 3])

        for i in tqdm(range(ts.shape[0]), leave=False, desc="Align Season"):
            # Rendered_Col_no_sky = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1,1,-1,1]), 2)), 1) * (Solar_Vis + (1 - Solar_Vis) * Sky_Col)
            Y = GT - t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1, 1, -1, 1]), 2)) * (Solar_Vis), 1)
            X = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1, 1, -1, 1]), 2)) * (1-Solar_Vis), 1)
            good = t.sum(X * X, 0) > 0
            T1 = 1 / t.sum(X * X, 0)[good]
            T2 = t.sum(X * Y, 0)[good]
            Sky_Col = t.clamp(T1 * T2, 0, 1)
            Rendered_Col = t.sum(PS * sig_t(Base_Cols + t.sum(Adjust_Cols * tv[i].reshape([1, 1, -1, 1]), 2)) * (
                    Solar_Vis + (1 - Solar_Vis) * Sky_Col), 1)
            error = the_loss_calc(Rendered_Col, GT)
            scores[i] = error.cpu().item()
            all_Sky_cols[i] = Sky_Col.cpu().numpy()

        best_idx = np.argmin(scores)
        # print("Base Time:", t0)
        # print("Best Time:", ts[best_idx].item())
        # print("Best Sky Col:", all_Sky_cols[best_idx])
        # plt.scatter(ts.cpu().numpy(), scores)
        # plt.show()

        adj_vec = tv[best_idx]
        # print(adj_vec)

        # exit()
    # print(t.tensor(all_Sky_cols[best_idx]).reshape([1, 1, 3]).float())
    return adj_vec.cpu(), t.tensor(all_Sky_cols[best_idx]).reshape([1, 1, 3]).float(), ts[best_idx].item()



def eval_Rendering(the_network, a_P_img, device, out_img_size, out_img_solar_size = None, season_steps = 100, use_classic_shadows = False):
    Results_Dict = component_render_by_P(the_network, a_P_img, out_img_size, device=device, include_exact_solar=False)
    imgs_dict = get_imgs_from_Img_Dict(Results_Dict, out_img_size, use_classic_shadows)

    Results_aligned_dict = deepcopy(Results_Dict)
    new_time, new_sky_col, adj_time = Grad_Descent_Seasonal_Align_v3(Results_aligned_dict, a_P_img.img, a_P_img.get_year_frac(), the_network, device, steps=season_steps, use_classic_shadows=use_classic_shadows)
    # new_time, new_sky_col = Grad_Descent_Seasonal_Align(Results_aligned_dict, a_P_img.img, steps=season_steps)
    Results_aligned_dict["Output_class"] = np.zeros_like(Results_aligned_dict["Output_class"]) + new_time.numpy()
    Results_aligned_dict["Sky_Col"] = np.zeros_like(Results_aligned_dict["Sky_Col"]) + new_sky_col.numpy()
    imgs_aligned_dict = get_imgs_from_Img_Dict(Results_aligned_dict, out_img_size, use_classic_shadows)

    GT = CV_reshape(a_P_img.img, out_img_size[0:2])
    # ans = {"Images":imgs_dict, "Data":Results_Dict, "Seasonal_Aligned_Imgs":imgs_aligned_dict, "Aligned_Vals":(new_time, new_sky_col)}
    ans = {"Images": imgs_dict, "Seasonal_Aligned_Imgs": imgs_aligned_dict, "Aligned_Vals": (new_time, new_sky_col), "Ground_Truth":GT}

    if out_img_solar_size is not None:
        Results_Dict_solar = component_render_by_P(the_network, a_P_img, out_img_solar_size, device=device,
                                             include_exact_solar=True)
        imgs_dict_solar = get_imgs_from_Img_Dict(Results_Dict_solar, out_img_solar_size, use_classic_shadows)

        Results_aligned_dict_solar = deepcopy(Results_Dict_solar)
        new_time, new_sky_col, adj_time = Grad_Descent_Seasonal_Align_v3(Results_aligned_dict_solar, a_P_img.img, a_P_img.get_year_frac(), the_network, device, steps=season_steps, use_classic_shadows=use_classic_shadows)
        # new_time, new_sky_col = Grad_Descent_Seasonal_Align(Results_aligned_dict_solar, a_P_img.img, steps=season_steps)
        Results_aligned_dict_solar["Output_class"] = np.zeros_like(Results_aligned_dict_solar["Output_class"]) + new_time.numpy()
        Results_aligned_dict_solar["Sky_Col"] = np.zeros_like(Results_aligned_dict_solar["Sky_Col"]) + new_sky_col.numpy()
        imgs_aligned_dict_solar = get_imgs_from_Img_Dict(Results_aligned_dict_solar, out_img_solar_size, use_classic_shadows)

        # exact_solar_ans = {"Images": imgs_dict_solar, "Data": Results_Dict_solar, "Seasonal_Aligned_Imgs": imgs_aligned_dict_solar,
        #        "Aligned_Vals": (new_time, new_sky_col)}
        GT = CV_reshape(a_P_img.img, out_img_solar_size[0:2])
        exact_solar_ans = {"Images": imgs_dict_solar,
                           "Seasonal_Aligned_Imgs": imgs_aligned_dict_solar,
                           "Aligned_Vals": (new_time, new_sky_col), "Ground_Truth":GT}

    else:
        exact_solar_ans = {}
    return ans, exact_solar_ans


def image_quality_metric_gauntlet(Img_GT, Img_Est, SSIM_Scale, EM_Scale):
    # plt.subplot(1,2,1)
    # plt.imshow(Img_GT)
    # plt.subplot(1,2,2)
    # plt.imshow(Img_Est)
    # plt.show()

    mask = np.all(Img_GT == Img_GT, 2) * np.all(Img_Est == Img_Est, 2)

    PSNR = mask_PSNR(Img_GT, Img_Est, mask)
    SSIM, valid_SSIM_pts = mask_ssim(Img_GT, Img_Est, mask, window_size=SSIM_Scale)
    SSIM = np.mean(SSIM[valid_SSIM_pts])
    mean_L2 = np.nanmean(np.sqrt(np.sum((Img_GT - Img_Est)**2, 2)))

    Img_GT[mask] *= np.NaN

    EM1 = mg_EM(Img_GT)
    EM2 = mg_EM(Img_Est)
    EM, _, _ = compare_EM_imgs(EM1, EM2)
    # print(mean_L2, PSNR, SSIM, EM)
    return mean_L2, PSNR, SSIM, EM * EM_Scale

def eval_img_dict(GT_img, img_dict, use_exact_shadow, SSIM_size, EM_Scale_F = 1.):
    GT_img_resized = cv.resize(GT_img, tuple(img_dict["Images"]["Base_Img"].shape[0:2]))

    if use_exact_shadow:
        est_img = img_dict["Images"]["Shadow_Adjust_Exact"] * img_dict["Images"]["Season_Adj_Img"]
        # show_dict_struc(img_dict)
    else:
        est_img = img_dict["Images"]["Shadow_Adjust"] * img_dict["Images"]["Season_Adj_Img"]

    results_Table = np.zeros([4,4])
    # mean_L2, PSNR, SSIM, EM = image_quality_metric_gauntlet(GT_img_resized, img_dict["Images"]["Base_Img"])
    # mean_L2, PSNR, SSIM, EM = image_quality_metric_gauntlet(GT_img_resized, img_dict["Images"]["Season_Adj_Img"])
    # mean_L2, PSNR, SSIM, EM = image_quality_metric_gauntlet(GT_img_resized, est_img)
    results_Table[0] = image_quality_metric_gauntlet(np.copy(GT_img_resized), img_dict["Images"]["Base_Img"], SSIM_size, EM_Scale_F)
    results_Table[1] = image_quality_metric_gauntlet(np.copy(GT_img_resized), img_dict["Images"]["Season_Adj_Img"], SSIM_size, EM_Scale_F)
    results_Table[2] = image_quality_metric_gauntlet(np.copy(GT_img_resized), est_img, SSIM_size, EM_Scale_F)
    if use_exact_shadow:
        results_Table[3] = image_quality_metric_gauntlet(np.copy(GT_img_resized), img_dict["Seasonal_Aligned_Imgs"]["Season_Adj_Img"] * img_dict["Seasonal_Aligned_Imgs"]["Shadow_Adjust_Exact"], SSIM_size, EM_Scale_F)
    else:
        results_Table[3] = image_quality_metric_gauntlet(np.copy(GT_img_resized),
                                                         img_dict["Seasonal_Aligned_Imgs"]["Season_Adj_Img"] *
                                                         img_dict["Seasonal_Aligned_Imgs"]["Shadow_Adjust"], SSIM_size, EM_Scale_F)

    row_labels = ["Base Image", "Season without shadow Image", "Full Image", "Seasonal Aligned Image"]
    col_labels = ["mean L2", "PSNR", "SSIM", "EM"]

    img_dict["Scores"] = {"Table":results_Table, "Row_Labels":row_labels, "Col_Labels":col_labels}

    return img_dict

def Full_Eval_Imgs(a_t_nerf, P_imgs, training_idx, testing_idx, device, out_img_size, solar_img_size, include_training_imgs, SSIM_size = 13, season_steps = 100, use_classic_shadows = False):
    print("Eval. Testing images...")
    Imgs_results = {"Testing":{}, "Training":{}}

    Scale = np.array(out_img_size) / np.array(solar_img_size)
    SSIM_size_scaled = max(SSIM_size // np.mean(Scale[0:2]), 1)
    scale_f = 1.0

    for i in tqdm(testing_idx):
        img_dict, exact_solar_dict = eval_Rendering(a_t_nerf, P_imgs[i], device, out_img_size=out_img_size, out_img_solar_size=solar_img_size, season_steps = season_steps, use_classic_shadows=use_classic_shadows)
        img_dict = eval_img_dict(P_imgs[i].img, img_dict, use_exact_shadow=False, SSIM_size=SSIM_size)
        exact_solar_dict = eval_img_dict(P_imgs[i].img, exact_solar_dict, use_exact_shadow=True, SSIM_size=SSIM_size_scaled, EM_Scale_F = scale_f)
        Imgs_results["Testing"][P_imgs[i].img_name] = {}
        Imgs_results["Testing"][P_imgs[i].img_name]["Standard"] = img_dict
        Imgs_results["Testing"][P_imgs[i].img_name]["Exact_Solar"] = exact_solar_dict

    if include_training_imgs:
        print("Eval. Training images...")
        for i in tqdm(training_idx):
            img_dict, exact_solar_dict = eval_Rendering(a_t_nerf, P_imgs[i], device, out_img_size=out_img_size,
                                                        out_img_solar_size=solar_img_size, use_classic_shadows=use_classic_shadows)
            img_dict = eval_img_dict(P_imgs[i].img, img_dict, use_exact_shadow=False, SSIM_size=SSIM_size)
            exact_solar_dict = eval_img_dict(P_imgs[i].img, exact_solar_dict, use_exact_shadow=True, SSIM_size=SSIM_size_scaled, EM_Scale_F = scale_f)
            Imgs_results["Training"][P_imgs[i].img_name] = {}
            Imgs_results["Training"][P_imgs[i].img_name]["Standard"] = img_dict
            Imgs_results["Training"][P_imgs[i].img_name]["Exact_Solar"] = exact_solar_dict

    return Imgs_results

def get_Shadow_score(Raw_Mask, Raw_Mask_Exact, Mask, Mask_Exact):
    avg_Error = np.nanmean(np.abs(Raw_Mask - Raw_Mask_Exact))

    good = (Mask_Exact == Mask_Exact) * (Mask == Mask)
    in_Sun_GT = Mask_Exact[good] > .5
    in_Sun_Est = Mask[good] > .5


    TP = np.sum(in_Sun_GT * in_Sun_Est)
    TN = np.sum((~in_Sun_GT) * (~in_Sun_Est))
    FP = np.sum((~in_Sun_GT) * in_Sun_Est)
    FN = np.sum(in_Sun_GT * (~in_Sun_Est))

    acc = (TP + TN) / (TP + TN + FP + FN)
    Prec_Sun = TP / (TP + FP) if TP + FP > 0 else 0
    Recall_Sun = TP / (TP + FN) if TP + FN > 0 else 0
    Prec_Shadow = TN / (TN + FN) if TN + FN > 0 else 0
    Recall_Shadow = TN / (TN + FP) if TN + FP > 0 else 0

    return avg_Error, TP, TN, FP, FN, acc, Prec_Sun, Recall_Sun, Prec_Shadow, Recall_Shadow


def Orgainize_Output_Imgs_Imgs(Img_Summary, P_imgs, output_loc):
    # show_dict_struc(Img_Summary["Testing"])

    mu_Results_est_shadow = np.zeros([4,4])
    mu_Results_exact_shadow = np.zeros([4, 4])
    Shadow_Table = []
    for a_key in Img_Summary["Testing"].keys():
        mu_Results_est_shadow += Img_Summary["Testing"][a_key]["Standard"]["Scores"]["Table"] / len(Img_Summary["Testing"].keys())
        mu_Results_exact_shadow += Img_Summary["Testing"][a_key]["Exact_Solar"]["Scores"]["Table"] / len(Img_Summary["Testing"].keys())

        A = Img_Summary["Testing"][a_key]["Exact_Solar"]["Images"]["Raw_Shadow_Mask"]
        B = Img_Summary["Testing"][a_key]["Exact_Solar"]["Images"]["Raw_Shadow_Mask_Exact"]
        C = Img_Summary["Testing"][a_key]["Exact_Solar"]["Images"]["Shadow_Mask"]
        D = Img_Summary["Testing"][a_key]["Exact_Solar"]["Images"]["Shadow_Mask_Exact"]
        avg_Error, TP, TN, FP, FN, acc, Prec_Sun, Recall_Sun, Prec_Shadow, Recall_Shadow = get_Shadow_score(A,B,C,D)
        Shadow_Table.append([a_key, avg_Error, TP, TN, FP, FN, np.round(acc, 3), np.round(Prec_Sun, 3), np.round(Recall_Sun, 3), np.round(Prec_Shadow, 3), np.round(Recall_Shadow, 3)])
    shadow_Col_Labels = ("Region", "Average Error", "TP", "TN", "FP", "FN", "Acc", "Prec. Sun", "Recall Sun", "Prec. Shadow", "Recall Shadow")

    fout = open(output_loc + "/Testing_Shadow_Scores.txt", "w")
    fout.write(tabulate(Shadow_Table, headers = shadow_Col_Labels))
    fout.close()


    fout = open(output_loc + "/Img_Scores.txt", "w")
    c = 0
    for a_key in Img_Summary["Testing"].keys():
        if c == 0:
            table1 = [[Img_Summary["Testing"][a_key]["Standard"]["Scores"]["Row_Labels"][i]] + list(
                mu_Results_est_shadow[i]) for i in range(4)]
            fout.write("Average Est. Shadows\n")
            fout.write(tabulate(table1, headers = ["Image Type"] + list(
                Img_Summary["Testing"][a_key]["Standard"]["Scores"]["Col_Labels"])))
            fout.write("\n")
            table1 = [[Img_Summary["Testing"][a_key]["Exact_Solar"]["Scores"]["Row_Labels"][i]] + list(mu_Results_exact_shadow[i]) for i in range(4)]
            fout.write("Average Exact. Shadows\n")
            fout.write(tabulate(table1, headers=["Image Type"] + list(
                Img_Summary["Testing"][a_key]["Exact_Solar"]["Scores"]["Col_Labels"])))
            fout.write("\n\n\n")
            c += 1


        table1 = [[Img_Summary["Testing"][a_key]["Standard"]["Scores"]["Row_Labels"][i]] + list(Img_Summary["Testing"][a_key]["Standard"]["Scores"]["Table"][i]) for i in range(4)]
        fout.write(a_key + " Est. Shadows\n")
        fout.write(tabulate(table1, headers=["Image Type"] + list(Img_Summary["Testing"][a_key]["Standard"]["Scores"]["Col_Labels"])))
        fout.write("\n")
        table1 = [[Img_Summary["Testing"][a_key]["Exact_Solar"]["Scores"]["Row_Labels"][i]] + list(
            Img_Summary["Testing"][a_key]["Exact_Solar"]["Scores"]["Table"][i]) for i in range(4)]
        fout.write(a_key + " Exact. Shadows\n")
        fout.write(tabulate(table1, headers=["Image Type"] + list(
            Img_Summary["Testing"][a_key]["Exact_Solar"]["Scores"]["Col_Labels"])))
        fout.write("\n\n\n")
    fout.close()


    for an_img in Img_Summary["Testing"].keys():
        base_img = None
        for a_P_img in P_imgs:
            if a_P_img.img_name == an_img:
                base_img = a_P_img.img
                break

        plt.figure(figsize=(8,6), dpi=100)
        plt.subplot(2,3,1)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Raw_Shadow_Mask"], vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.title("Estimated Shadows")
        plt.ylabel("Analog Shadows")
        plt.subplot(2, 3, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title("Exact Shadows")
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Raw_Shadow_Mask_Exact"], vmin=0, vmax=1)
        plt.subplot(2, 3, 3)
        plt.xticks([])
        plt.yticks([])
        plt.title("Accuracy")
        diff = Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Raw_Shadow_Mask_Exact"] - Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Raw_Shadow_Mask"]
        plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 3, 4)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Shadow_Mask"], vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("Shadow Mask")
        plt.subplot(2, 3, 5)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Shadow_Mask_Exact"], vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 3, 6)
        diff = Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Shadow_Mask_Exact"] - \
               Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Shadow_Mask"]
        acc = np.mean(np.abs(diff[diff == diff]) < .05)
        plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Shadow Acc: " + str(np.round(acc,3)))
        plt.tight_layout()
        plt.savefig(output_loc + "/" + an_img + "_Shadows.png")
        plt.close("all")



        # print(Img_Summary["Testing"][an_img]["Standard"]["Scores"])
        Sum_Table = np.round(Img_Summary["Testing"][an_img]["Standard"]["Scores"]["Table"], 2)

        plt.figure(figsize=(12,6), dpi=200)
        if base_img is not None:
            plt.subplot(2,5,1)
            plt.imshow(base_img)
            plt.xticks([])
            plt.yticks([])
            plt.title("Ground Truth")
            plt.ylabel("Est Shadow")
        plt.subplot(2, 5, 2)
        plt.imshow(Img_Summary["Testing"][an_img]["Standard"]["Images"]["Base_Img"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[0, 1]) + " , SSIM: " + str(Sum_Table[0, 2]))
        plt.title("Base")
        plt.subplot(2, 5, 3)
        plt.imshow(Img_Summary["Testing"][an_img]["Standard"]["Images"]["Season_Adj_Img"])
        plt.xticks([])
        plt.yticks([])
        plt.title("Seasons, no Shadows")
        plt.xlabel("PSNR: " + str(Sum_Table[1, 1]) + " , SSIM: " + str(Sum_Table[1, 2]))
        plt.subplot(2, 5, 4)
        plt.imshow(Img_Summary["Testing"][an_img]["Standard"]["Images"]["Shadow_Adjust"] * Img_Summary["Testing"][an_img]["Standard"]["Images"]["Season_Adj_Img"])
        plt.title("Seasons and Shadows")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[2, 1]) + " , SSIM: " + str(Sum_Table[2, 2]))
        plt.subplot(2, 5, 5)
        plt.imshow(Img_Summary["Testing"][an_img]["Standard"]["Seasonal_Aligned_Imgs"]["Shadow_Adjust"] *
                   Img_Summary["Testing"][an_img]["Standard"]["Seasonal_Aligned_Imgs"]["Season_Adj_Img"])
        plt.title("Seasonal Alignment")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[3, 1]) + " , SSIM: " + str(Sum_Table[3, 2]))

        Sum_Table = np.round(Img_Summary["Testing"][an_img]["Exact_Solar"]["Scores"]["Table"], 2)
        if base_img is not None:
            plt.subplot(2,5,6)
            plt.imshow(base_img)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("Exact Shadows")
        plt.subplot(2, 5, 7)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Base_Img"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[0, 1]) + " , SSIM: " + str(Sum_Table[0, 2]))
        plt.subplot(2, 5, 8)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Season_Adj_Img"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[1, 1]) + " , SSIM: " + str(Sum_Table[1, 2]))
        plt.subplot(2, 5, 9)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Shadow_Adjust_Exact"] *
                   Img_Summary["Testing"][an_img]["Exact_Solar"]["Images"]["Season_Adj_Img"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[2, 1]) + " , SSIM: " + str(Sum_Table[2, 2]))
        plt.subplot(2, 5, 10)
        plt.imshow(Img_Summary["Testing"][an_img]["Exact_Solar"]["Seasonal_Aligned_Imgs"]["Shadow_Adjust_Exact"] *
                   Img_Summary["Testing"][an_img]["Exact_Solar"]["Seasonal_Aligned_Imgs"]["Season_Adj_Img"])
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("PSNR: " + str(Sum_Table[2, 1]) + " , SSIM: " + str(Sum_Table[2, 2]))
        plt.tight_layout()

        plt.savefig(output_loc + "/" + an_img + "_Colors.png")
        plt.close("all")