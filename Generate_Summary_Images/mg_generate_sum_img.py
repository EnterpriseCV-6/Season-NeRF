# coding=utf-8
import datetime
from matplotlib import pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
import numpy as np

#IMPORTANT number of walking shadows must equal number of walking times!
def gen_sum(walking_views, walking_shadows, walking_times, image_builder, out_img_size = 128, output_path = None):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ncol = walking_times.shape[0] + 1
    nrow = walking_shadows.shape[0]
    # fig, ax = plt.subplots(ncols=walking_times.shape[0], nrows=nrow, figsize=(ncol+1,nrow+1))
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)))
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1.-0.5/(nrow+1),
                           bottom=0.5/(nrow+1),
                           left=0.5/(ncol+1),
                           right=1-0.5/(ncol+1))

    date_1 = datetime.datetime.strptime("01/01/21", "%m/%d/%y")
    all_res = []
    r = 0
    for a_walking_shadow in tqdm(walking_shadows, leave=True):
        c = 0
        for a_waking_time in tqdm(walking_times, leave=False):
            # print(walking_views[r], a_walking_shadow, a_waking_time)
            res, mask = image_builder.render_img(walking_views[r], a_walking_shadow, a_waking_time, out_img_size=out_img_size)
            all_res.append([res, mask])
            ax = plt.subplot(gs[r,c])
            ax.imshow(res["Col_Img"])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                end_date = date_1 + datetime.timedelta(days=365.24 * a_waking_time)
                ax.set_title(months[end_date.month - 1] + ". " + str(end_date.day))
            if c == 0:
                # the_lab = "(" + str(np.round(90-walking_views[r][0])) + ", " + str(np.round(walking_views[r][1])) + ")\n(" + str(np.round(a_walking_shadow[0])) + ", " + str(np.round(a_walking_shadow[1])) + ")"
                the_lab = str(r+1)
                ax.set_ylabel(the_lab, rotation="horizontal", labelpad=5)#, rotation=0, labelpad=23)
            c += 1
        ax = plt.subplot(gs[r,c])
        out_img = (1 / (1 + np.exp(-30 * (res["Shadow_Mask"] - .2))))
        out_img[~mask] = np.NaN
        ax.imshow(out_img, vmin = 0, vmax = 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title("Shadow Mask")
        r += 1
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close("all")

def gen_sum_compare(P_imgs, seasonal_proto_idx, image_builder, out_img_size=128, output_path = None, time_splits = 12, time_recurse=1, time_subsets = 1):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ncol =  len(seasonal_proto_idx)
    nrow = len(seasonal_proto_idx) + 1
    # fig, ax = plt.subplots(ncols=walking_times.shape[0], nrows=nrow, figsize=(ncol+1,nrow+1))
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)))
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1))

    for i in range(len(seasonal_proto_idx)):
        ax = plt.subplot(gs[0, i])
        ax.imshow(P_imgs[seasonal_proto_idx[i]].img * np.expand_dims(P_imgs[seasonal_proto_idx[i]].get_mask(), -1))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Ground Truth")
        ax.set_title(months[P_imgs[seasonal_proto_idx[i]].time_obj.month - 1] + ". " + str(
            P_imgs[seasonal_proto_idx[i]].time_obj.day))

    for i in tqdm(range(len(seasonal_proto_idx)), leave=True):
        name, sat_el_and_az, sun_el_and_az, year_frac = P_imgs[seasonal_proto_idx[i]].get_meta_data()
        for j in tqdm(range(len(seasonal_proto_idx)), leave=False):
            ax = plt.subplot(gs[j+1, i])
            best_t, all_data = get_opt_time_img(P_imgs[seasonal_proto_idx[i]].img[0::32, 0::32], P_imgs[seasonal_proto_idx[i]].get_mask()[0::32, 0::32], image_builder, sat_el_and_az, sun_el_and_az, 64, time_splits, time_recurse, time_subsets)
            res, mask = image_builder.render_img(sat_el_and_az, sun_el_and_az, best_t, out_img_size=out_img_size)
            # print(all_data)
            print(best_t)
            # plt.figure()
            # plt.subplot(1,2,1)
            ax.imshow(res["Col_Img"]*np.expand_dims(mask,-1))
            # plt.subplot(1,2,2)
            # plt.imshow(P_imgs[seasonal_proto_idx[i]].img[0::16, 0::16] * np.expand_dims(P_imgs[seasonal_proto_idx[i]].get_mask()[0::16, 0::16], -1))
            # plt.show()
            ax.set_xticks([])
            ax.set_yticks([])




    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close("all")

# from T_NeRF_Eval_Utils import get_img_from_P_img
def  get_opt_time_img(base_img, base_img_mask, image_builder, sat_el_and_az, sun_el_and_az, out_img_size, time_splits = 12, time_recurse=2, time_subsets = 3):
    ts = np.linspace(0,1, time_splits, endpoint=False)

    bin_edges = [np.linspace(0, 360, 19), np.linspace(0, 100, 6), np.linspace(0, 100, 6)]
    img_HSL_base = np.array([hsluv.rgb_to_hsluv(a) for a in base_img[base_img_mask].reshape([-1, 3])])
    base_img_sig = mg_EM.get_Sig_advanced(img_HSL_base, bin_edges, dist_thresh=10, show_process=False, thresh=0.001)

    best_EM_dist = -1.
    best_t = 0.
    all_data = []
    some_Ts = []
    some_EM = []
    for a_t in ts:#tqdm(ts, leave=False,desc="Time Walk"):
        res, mask = image_builder.render_img(sat_el_and_az, sun_el_and_az, a_t, out_img_size=out_img_size)
        EM_dist, psnr, ssim = img_sim_scores(base_img, res["Col_Img"], Img1_sig=base_img_sig, Img1_mask=base_img_mask, Img2_mask=mask)
        if best_EM_dist < 0 or EM_dist < best_EM_dist:
            best_EM_dist = EM_dist
            best_t = a_t
        all_data.append([a_t, EM_dist, psnr, ssim])
        some_EM.append(EM_dist)
        some_Ts.append(a_t)

    # plt.scatter(some_Ts, some_EM)

    for i in range(time_recurse-1):
        best = np.argsort(some_EM)
        some_EM = []
        some_Ts2 = []
        diff = (some_Ts[1] - some_Ts[0])/2
        ts_adjust = np.linspace(-diff, diff, time_splits // time_subsets)
        for j in range(time_subsets):
            new_ts = some_Ts[best[j]] + ts_adjust
            for a_t in new_ts:#tqdm(new_ts, leave=False, desc="Time Walk"):
                res, mask = image_builder.render_img(sat_el_and_az, sun_el_and_az, a_t, out_img_size=out_img_size)
                EM_dist, psnr, ssim = img_sim_scores(base_img, res["Col_Img"], Img1_sig=base_img_sig, Img2_mask=mask)
                if best_EM_dist < 0 or EM_dist < best_EM_dist:
                    best_EM_dist = EM_dist
                    best_t = a_t
                all_data.append([a_t, EM_dist, psnr, ssim])
                some_EM.append(EM_dist)
                some_Ts2.append(a_t)
        some_Ts = np.copy(some_Ts2)
        # plt.scatter(some_Ts, some_EM)

    all_data = np.array(all_data)

    # plt.show()



    return best_t, all_data


import all_NeRF.mg_EM_Imgs as mg_EM
from T_NeRF_Eval_Utils import mask_ssim, mask_PSNR
import hsluv
def img_sim_scores(Img1, Img2, Img1_sig = None, Img2_sig = None, Img1_mask = None, Img2_mask = None):
    Img1_sig = None
    Img2_sig = None
    if Img1_mask is None:
        Img1_mask = np.ones([Img1.shape[0], Img1.shape[1]], dtype=bool)
    if Img2_mask is None:
        Img2_mask = np.ones([Img2.shape[0], Img2.shape[1]], dtype=bool)
    if Img1_sig is None or Img2_sig is None:
        # bin_edges = [np.linspace(0, 360, 19), np.linspace(0, 100, 6), np.linspace(0, 100, 6)]
        bin_edges = [np.linspace(0, 1, 9), np.linspace(0, 1, 9), np.linspace(0, 1, 9)]
        if Img1_sig is None:
            # img_HSL_scaled_1 = np.array([hsluv.rgb_to_hsluv(a) for a in Img1[Img1_mask].reshape([-1,3])])
            # Img1_sig = mg_EM.get_Sig_advanced(img_HSL_scaled_1, bin_edges, dist_thresh=10, show_process=False, thresh=0.001)
            img_HSL_scaled_1 = Img1[Img1_mask].reshape([-1, 3])
            Img1_sig = mg_EM.get_Sig_advanced(img_HSL_scaled_1, bin_edges, dist_thresh=1./16., show_process=False,
                                              thresh=0.001)
        if Img2_sig is None:
            # img_HSL_scaled_2 = np.array([hsluv.rgb_to_hsluv(a) for a in Img2[Img2_mask].reshape([-1, 3])])
            # Img2_sig = mg_EM.get_Sig_advanced(img_HSL_scaled_2, bin_edges, dist_thresh=10, show_process=False, thresh=0.001)
            img_HSL_scaled_2 = Img2[Img2_mask].reshape([-1, 3])
            Img2_sig = mg_EM.get_Sig_advanced(img_HSL_scaled_2, bin_edges, dist_thresh=1./16., show_process=False,
                                              thresh=0.001)
    EM_dist, _, _ = mg_EM.EM_sig_Compare(Img1_sig, Img2_sig)

    if np.all(Img1.shape == Img2.shape):
        ssim, ssim_mask = mask_ssim(Img1, Img2, Img1_mask * Img2_mask)
        ssim = np.mean(ssim[ssim_mask])
        psnr = mask_PSNR(Img1, Img2, Img1_mask * Img2_mask)
    else:
        ssim = -5.
        psnr = -1.
    return EM_dist, psnr, ssim
