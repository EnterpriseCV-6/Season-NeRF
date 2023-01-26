from pre_NeRF import P_img
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch as t

F_save = None

def get_img_from_P_img(network, a_P_img:P_img, out_img_size = None, time = None, sun_vec = None, device="cpu", use_TQDM = True, use_classic_solar = False):

    max_batch_size = 50000

    if out_img_size is None:
        out_img_size = np.array([a_P_img.img.shape[0], a_P_img.img.shape[1], 96])
    if time is None:
        time = a_P_img.time_obj.get_time_encode()[1:3]
    if sun_vec is None:
        sun_vec = a_P_img.sun_el_and_az_vec

    Xs = np.linspace(0, a_P_img.img.shape[0], out_img_size[0], endpoint=False)
    Ys = np.linspace(0, a_P_img.img.shape[1], out_img_size[1], endpoint=False)

    Xs, Ys = np.meshgrid(Xs, Ys, indexing="ij")
    Xs = np.ravel(Xs)
    Ys = np.ravel(Ys)

    X0, Y0, Z0 = a_P_img.invert_P(Xs, Ys, np.ones_like(Xs))
    X1, Y1, Z1 = a_P_img.invert_P(Xs, Ys, -np.ones_like(Xs))

    Ts = np.linspace(0, 1, out_img_size[2]).reshape([1,-1,1])
    XYZ = np.expand_dims(np.stack([X0, Y0, Z0], 1), 1) * (1-Ts) + Ts * np.expand_dims(np.stack([X1, Y1, Z1], 1), 1)

    XYZ = np.reshape(XYZ, [-1,3])
    Sun_Angle_input = t.tensor(sun_vec.reshape([1, 3])).float().to(device)
    Time_input = t.tensor(time).float().reshape([1,2]).to(device)

    # all_Rho, all_Col, all_Solar_Vis, all_Sky_Col = None, None, None, None
    all_Rho, all_Col, all_Solar_Vis, all_Sky_Col = np.zeros([np.prod(out_img_size), 1]), np.zeros([np.prod(out_img_size), 3]), np.zeros([np.prod(out_img_size), 1]), np.zeros([np.prod(out_img_size), 3])

    with t.no_grad():
        network.eval()
        if use_TQDM:
            for i in tqdm(range(0, XYZ.shape[0], max_batch_size)):
                i_end = min(i+max_batch_size, XYZ.shape[0])
                XYZ_input = t.tensor(XYZ[i:i_end]).float().to(device)

                Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col = network(XYZ_input, Sun_Angle_input * t.ones_like(XYZ_input), (t.ones([XYZ_input.shape[0], 2], device=device) * Time_input).float())
                all_Rho[i:i_end], all_Col[i:i_end], all_Solar_Vis[i:i_end], all_Sky_Col[i:i_end] = Rho.cpu().numpy(), Col.cpu().numpy(), Solar_Vis.cpu().numpy(), Sky_Col.cpu().numpy()
        else:
            for i in range(0, XYZ.shape[0], max_batch_size):
                i_end = min(i + max_batch_size, XYZ.shape[0])
                XYZ_input = t.tensor(XYZ[i:i_end]).float().to(device)

                Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col = network(XYZ_input,
                                                                                 Sun_Angle_input * t.ones_like(
                                                                                     XYZ_input), (t.ones(
                        [XYZ_input.shape[0], 2], device=device) * Time_input).float())
                all_Rho[i:i_end], all_Col[i:i_end], all_Solar_Vis[i:i_end], all_Sky_Col[
                                                                            i:i_end] = Rho.cpu().numpy(), Col.cpu().numpy(), Solar_Vis.cpu().numpy(), Sky_Col.cpu().numpy()

    XYZ = XYZ.reshape([out_img_size[0], out_img_size[1], out_img_size[2], 3])

    deltas = np.sqrt(np.sum((XYZ[:,:,0:-1] - XYZ[:,:,1::])**2, 3))
    deltas = np.expand_dims(np.concatenate([deltas, deltas[:,:,-1::]], 2), -1)

    XYZ_mask = np.all(np.abs(XYZ) < 1, 3, keepdims=True)

    all_Rho = all_Rho.reshape([out_img_size[0], out_img_size[1], out_img_size[2], 1]) * XYZ_mask
    all_Col = all_Col.reshape([out_img_size[0], out_img_size[1], out_img_size[2], 3])
    all_Solar_Vis = all_Solar_Vis.reshape([out_img_size[0], out_img_size[1], out_img_size[2], 1])
    all_Sky_Col = all_Sky_Col.reshape([out_img_size[0], out_img_size[1], out_img_size[2], 3])

    P_E = 1-np.exp(-all_Rho*deltas)
    P_vis = np.exp(-np.cumsum(np.concatenate([np.zeros([out_img_size[0], out_img_size[1], 1, 1]), all_Rho*deltas], 2), 2)[:,:,0:-1])
    # global F_save
    # if F_save is None:
    #     F_save = all_Sky_Col
    # print(F_save[0,0,0])

    F = np.array([.25, .25, .25]).reshape([1,1,1,3])
    Solar_Vis3 = t.sigmoid(t.tensor((np.sum(all_Solar_Vis * P_E * P_vis, 2)-.2)*30)).detach().numpy()
    if use_classic_solar:
        Col_Img = np.sum(P_E*P_vis*((all_Solar_Vis + (1-all_Solar_Vis) * all_Sky_Col) * all_Col), 2)
    else:
        Col_Img = np.sum(P_E * P_vis * all_Col, 2) * (Solar_Vis3 + (1 - Solar_Vis3) * np.mean(all_Sky_Col, 2))
    return Col_Img, np.all(np.any(np.abs(XYZ <= 1), 3), 2)




