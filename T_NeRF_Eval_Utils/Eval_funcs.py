import torch as t
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage.interpolation import rotate
from scipy.signal import convolve2d
from tqdm import tqdm
import hsluv
from T_NeRF_Eval_Utils.mg_image_from_P_img import get_img_from_P_img

def softplus(X):
    Y = np.copy(X)
    Y[Y < 20] = np.log(1+np.exp(Y[Y<20]))
    return Y

def sigmoid(X):
    return 1/(1+np.exp(-X))

def get_gaussian_kernel2d(window_size: (int, int), sigma: (float, float)):
    X = np.arange(window_size[0]) - window_size[0]//2
    X = np.exp(-X**2 /  (2 * sigma[0]**2))
    X = X / np.sum(X)

    Y = np.arange(window_size[1]) - window_size[1] // 2
    Y = np.exp(-Y ** 2 / (2 * sigma[1] ** 2))
    Y = Y / np.sum(Y)
    K = X.reshape([-1,1]) @ Y.reshape([1,-1])
    # print(K)

    return K

def mask_PSNR(img1, img2, mask):
    PSNR = -10 * np.log10(np.mean((img1 - img2)[mask] ** 2))
    return PSNR

def mask_filter(img, kernel):
    # print(kernel.shape)
    # print(img.shape)
    ans = convolve(img, kernel, mode="nearest")
    # if len(img.shape) == 2:
    #     ans = convolve2d(img, kernel, mode = "same")
    # else:
    #     ans = np.stack([convolve2d(img[:,:,i], kernel[:,:,0], mode = "same") for i in range(3)], 2)
    # print(ans.shape)
    # # exit()
    return ans

def mask_ssim(img1, img2, mask, window_size: int = 13, max_val: float = 1.0, eps: float = 1e-12):
    kernel = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))

    valid_pts = mask_filter(1.-mask*1., kernel) == 0
    kernel = np.expand_dims(kernel, -1)


    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1, mu2 = mask_filter(img1, kernel), mask_filter(img2, kernel)
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    mu1_sq_f, mu2_sq_f = mask_filter(img1 ** 2, kernel), mask_filter(img2 ** 2, kernel)
    mu1_mu2_f = mask_filter(img1 * img2, kernel)

    sigma1_sq = mu1_sq_f - mu1_sq
    sigma2_sq = mu2_sq_f - mu2_sq
    sigma12 = mu1_mu2_f - mu1_mu2

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    # plt.subplot(1,3,1)
    # plt.imshow(img1)
    # plt.subplot(1,3,2)
    # plt.imshow(img2)
    # print(np.mean(num/den))
    # plt.subplot(1,3,3)
    # plt.imshow(num/den)
    # plt.colorbar()
    # plt.show()

    return num / (den + eps*0), valid_pts


# def ssim(image_pred, image_gt):
#     """
#     image_pred and image_gt: (1, 3, H, W)
#     important: kornia==0.5.3
#     """
#     from kornia.losses import ssim as ssim_
#     # from kornia.filters import get_gaussian_kernel2d
#
#     # K = get_gaussian_kernel2d((3,3), (1.5,1.5))
#     # img = t.tensor(image_pred).reshape([1, 3, image_pred.shape[0], image_pred.shape[1]])
#     # print(filter2d(img, K))
#
#
#     image_pred_t = t.tensor(np.moveaxis(np.expand_dims(image_pred, 0), -1, 1))
#     image_gt_t = t.tensor(np.moveaxis(np.expand_dims(image_gt, 0), -1, 1))
#     # print(ssim_(image_pred_t, image_gt_t, 3).shape)
#     return t.mean(ssim_(image_pred_t, image_gt_t, 3)), ssim_(image_pred_t, image_gt_t, 3)

def full_eval_P_img(network, a_P_img, n_sample_pts, device, step_size = 1, max_batch_size = 5000, bounds = None, use_classic_solar = False):
    # print(a_P_img.img.shape)

    img_loc = []
    GT_Col = []
    Ray_pts = []
    H = np.linspace(1, -1, n_sample_pts)
    # print("Setting up data")
    for i in range(0, a_P_img.img.shape[0], step_size):
        for j in range(0, a_P_img.img.shape[1], step_size):
            X, Y, Z = a_P_img.invert_P(i, j, H)
            good = (X >= -1) * (X <= 1) * (Y <= 1) * (Y >= -1)
            if np.all(good):
                XYZ = np.stack([X[good], Y[good], Z[good]], 1)
                Ray_pts.append(XYZ)
                img_loc.append([i//step_size,j//step_size])
                GT_Col.append(a_P_img.img[i,j])
    Ray_pts = np.array(Ray_pts)
    GT_Col = np.array(GT_Col)
    img_loc = np.array(img_loc)


    GT_img = np.zeros([a_P_img.img.shape[0]//step_size+1, a_P_img.img.shape[1]//step_size+1, a_P_img.img.shape[2]])
    GT_img[img_loc[:,0], img_loc[:,1]] = GT_Col

    img_dict = {"GT_Img": GT_img}

    mask = np.zeros([a_P_img.img.shape[0]//step_size+1, a_P_img.img.shape[1]//step_size+1], dtype=bool)
    mask[img_loc[:, 0], img_loc[:, 1]] = True
    img_dict["Valid_Pt_Mask"] = mask


    SA = t.tensor(np.ones([max_batch_size, 3]) * a_P_img.sun_el_and_az_vec.reshape([1,3])).float().to(device)
    T = t.tensor(np.ones([max_batch_size, 4]) * np.array(a_P_img.time_obj.get_time_encode())[1::].reshape([1,4])).float().to(device)


    deltas = np.sqrt(np.sum((Ray_pts[:,1::] - Ray_pts[:,0:-1])**2, 2, keepdims=True))
    deltas = np.concatenate([deltas, np.mean(deltas, 1, keepdims=True)], 1)


    full_input = t.tensor(Ray_pts).reshape([-1,3]).float()
    network.eval()

    all_Rhos = t.zeros([full_input.shape[0], 1])
    all_Cols = t.zeros([full_input.shape[0], a_P_img.img.shape[2]])
    all_Solar_Vis = t.zeros([full_input.shape[0], 1])
    all_Sky_Cols = t.zeros([full_input.shape[0], a_P_img.img.shape[2]])
    all_output_class = t.zeros([full_input.shape[0], network.n_classes])
    all_Adjust_cols = t.zeros([full_input.shape[0], network.n_classes, a_P_img.img.shape[2]])


    # print("Applying NeRF")
    with t.no_grad():
        for i in range(0, full_input.shape[0], max_batch_size):
            batch_end = min(i+max_batch_size, full_input.shape[0])
            if batch_end == full_input.shape[0]:
                SA = SA[0:batch_end-i]
                T = T[0:batch_end - i]
            Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col = network.forward_full_eval(full_input[i:batch_end].to(device), SA, T)
            all_Rhos[i:batch_end] = Rho.cpu()
            all_Cols[i:batch_end] = Col.cpu()
            all_Solar_Vis[i:batch_end] = Solar_Vis.cpu()
            all_Sky_Cols[i:batch_end] = Sky_Col.cpu()
            all_output_class[i:batch_end] = output_class.cpu()
            all_Adjust_cols[i:batch_end] = Adjust_col.cpu()

    all_Rhos = all_Rhos.reshape([Ray_pts.shape[0], Ray_pts.shape[1], 1]).numpy()
    all_Cols = all_Cols.reshape([Ray_pts.shape[0], Ray_pts.shape[1], a_P_img.img.shape[2]]).numpy()
    all_Solar_Vis = t.sigmoid(all_Solar_Vis.reshape(all_Rhos.shape)).numpy()
    all_Sky_Cols = t.sigmoid(all_Sky_Cols.reshape(all_Cols.shape)).numpy()
    all_Adjust_cols = all_Adjust_cols.reshape([Ray_pts.shape[0], Ray_pts.shape[1], network.n_classes, a_P_img.img.shape[2]]).numpy()
    all_output_class = all_output_class[0].numpy()

    PE = 1-np.exp(-all_Rhos * deltas)
    P_Vis = np.exp(-np.cumsum(np.concatenate([np.zeros([all_Rhos.shape[0], 1, 1]), all_Rhos * deltas], 1), 1)[:, 0:-1])
    P_Surf = PE * P_Vis

    H_map = np.sum(P_Surf * Ray_pts[:,:,2:3], 1)/np.sum(P_Surf, 1)

    HM = np.zeros([a_P_img.img.shape[0] // step_size + 1, a_P_img.img.shape[1] // step_size + 1]) + np.NaN
    HM[img_loc[:, 0], img_loc[:, 1]] = H_map[:,0]
    img_dict["HM"] = HM

    # print("Base Sky Color:", all_Sky_Cols[0,0])
    base_Sky_Color = all_Sky_Cols * (1-all_Solar_Vis) + all_Solar_Vis

    Solar_Vis3 = sigmoid((np.sum(all_Solar_Vis * P_Surf, 1) - .2) * 30)
    Sky_Col_Adj = (Solar_Vis3 + (1 - Solar_Vis3) * np.mean(all_Sky_Cols, 1))
    Class_terms = [np.zeros(network.n_classes), all_output_class]
    scores = {}
    results_sat_form = {}
    sample_sat_form = {}
    for i in range(network.n_classes):
        new_C = np.zeros(network.n_classes)
        new_C[i] = 1
        Class_terms.append(new_C)
    for i in range(len(Class_terms)):
        C = Class_terms[i].reshape([1,1,-1,1])
        adjusted_Color = sigmoid(all_Cols + np.sum(C * all_Adjust_cols, 2))
        if use_classic_solar:
            output_Cols = np.sum(adjusted_Color * base_Sky_Color * P_Surf, 1)
        else:
            output_Cols = np.sum(adjusted_Color * P_Surf, 1) * Sky_Col_Adj

        a_img = np.zeros([a_P_img.img.shape[0] // step_size + 1, a_P_img.img.shape[1] // step_size + 1, a_P_img.img.shape[2]])
        a_img[img_loc[:, 0], img_loc[:, 1]] = output_Cols

        PSNR = -10*np.log10(np.sum((a_img - GT_img)**2) / (a_P_img.img.shape[2] * np.sum(mask)))

        # mask[mask == False] = True
        # their_ssim, their_img = ssim(a_img, GT_img)
        # print(np.any(SSIM != 1, 2))
        # SSIM = np.mean(SSIM[np.any(SSIM < 0.999, 2)])
        # plt.subplot(1,2,2)
        # plt.imshow(np.moveaxis(their_img[0].numpy(), 0, -1))
        # plt.colorbar()
        # plt.show()


        SSIM, vp = mask_ssim(a_img, GT_img, mask, 3)
        SSIM = np.sum(np.mean(SSIM, 2) * vp) / np.sum(vp)

        if i == 0:
            img_dict["Base_Img"] = a_img
            scores["Base_Img_PSNR"] = PSNR
            scores["Base_Img_SSIM"] = SSIM
        elif i == 1:
            img_dict["Ideal_Time_Img"] = a_img
            scores["Ideal_Time_Img_PSNR"] = PSNR
            scores["Ideal_Time_Img_SSIM"] = SSIM

            H_map_full_loc = np.sum(P_Surf * Ray_pts, 1) / np.sum(P_Surf, 1)
            if bounds is not None:
                LLA = (H_map_full_loc + 1) / 2 * np.expand_dims((bounds[:,1] - bounds[:,0]), 0) + np.expand_dims(bounds[:,0], 0)
            else:
                LLA = bounds

            # print(LLA)
            # print(LLA.shape)
            # print(HM.shape)
            # print(Ray_pts.shape)
            # plt.imshow(HM)
            # plt.show()
            # exit()


            results_sat_form["rgbs"] = t.tensor(GT_img).view([-1,3])
            results_sat_form["rgb_coarse"] = t.tensor(a_img).view([-1,3])
            results_sat_form["LLA"] = LLA
            sample_sat_form["h"] = a_img.shape[0]
            sample_sat_form["w"] = a_img.shape[1]

            # plt.imshow(a_img)
            # plt.show()
        else:
            img_dict["Class_" + str(i-1)] = a_img
            scores["Class_" + str(i-1) + "_PSNR"] = PSNR
            scores["Class_" + str(i-1) + "_SSIM"] = SSIM
        # print(PSNR, SSIM)

    scores["Sky_Col"], scores["Ideal_Class_Output"] = all_Sky_Cols[0, 0], all_output_class

    return img_dict, scores, [sample_sat_form, results_sat_form]


def gen_results(network, img_shape, n_samples, device, max_batch_size):
    XYZ = np.stack(
        np.meshgrid(np.arange(0, img_shape[0]), np.arange(0, img_shape[1]), np.arange(n_samples), indexing="ij"), -1)
    XYZ = XYZ.reshape([-1, 3])

    s = np.array([1 / img_shape[0], 1 / img_shape[1], 1 / n_samples])

    all_Rhos = np.zeros([img_shape[0], img_shape[1], n_samples, 1])
    all_Cols = np.zeros([img_shape[0], img_shape[1], n_samples, 3])
    delta = 2 / n_samples
    network.eval()
    with t.no_grad():
        for i in range(0, XYZ.shape[0], max_batch_size):
            batch_end = min(i + max_batch_size, XYZ.shape[0])

            xyz = XYZ[i:batch_end] * s * 2 - 1
            xyz[:, 2] *= -1

            Rho = network.forward_Classic_Sigma_Only(t.tensor(xyz).float().to(device))
            Col = network.G_NeRF_net.forward_color_only(t.tensor(xyz).float().to(device))
            all_Rhos[XYZ[i:batch_end, 0], XYZ[i:batch_end, 1], XYZ[i:batch_end, 2]] = Rho.cpu()
            all_Cols[XYZ[i:batch_end, 0], XYZ[i:batch_end, 1], XYZ[i:batch_end, 2]] = Col.cpu()
    P_E = 1 - np.exp(-all_Rhos * delta)[:, :, :, 0]
    P_Vis = np.exp(
        -np.cumsum(np.concatenate([np.zeros([all_Rhos.shape[0], all_Rhos.shape[1], 1, 1]), all_Rhos * delta], 2), 2)[:,
         :, 0:-1])[:, :, :, 0]
    P_Surf = P_E * P_Vis

    return all_Rhos, P_E, P_Vis, P_Surf, all_Cols

def eval_HM(network, GT, h_range, n_samples, device, max_batch_size):
    XYZ = np.stack(np.meshgrid(np.arange(0, GT.shape[0]), np.arange(0, GT.shape[1]), np.arange(n_samples), indexing="ij"), -1)
    XYZ = XYZ.reshape([-1,3])

    s = np.array([1/GT.shape[0], 1/GT.shape[1], 1/n_samples])

    all_Rhos = np.zeros([GT.shape[0], GT.shape[1], n_samples, 1])
    delta = 2 / n_samples
    network.eval()
    with t.no_grad():
        for i in range(0, XYZ.shape[0], max_batch_size):
            batch_end = min(i + max_batch_size, XYZ.shape[0])

            xyz = XYZ[i:batch_end] * s * 2 - 1
            xyz[:,2] *= -1

            Rho = network.forward_Classic_Sigma_Only(t.tensor(xyz).float().to(device))
            all_Rhos[XYZ[i:batch_end, 0], XYZ[i:batch_end, 1], XYZ[i:batch_end, 2]] = Rho.cpu()
    P_E = 1-np.exp(-all_Rhos * delta)[:,:,:,0]
    P_Vis = np.exp(-np.cumsum(np.concatenate([np.zeros([all_Rhos.shape[0], all_Rhos.shape[1], 1, 1]), all_Rhos * delta], 2), 2)[:, :, 0:-1])[:,:,:,0]
    P_Surf = P_E * P_Vis
    Est_HM = np.sum(P_Surf * np.linspace(1,-1, n_samples).reshape([1,1,-1]), 2) / np.sum(P_Surf, 2)

    Surf_PDF = P_Surf / np.sum(P_Surf, 2, keepdims=True)
    conf_range = np.zeros([Surf_PDF.shape[0], Surf_PDF.shape[1],3])
    for i in range(conf_range.shape[0]):
        for j in range(conf_range.shape[1]):
            start_x = np.argmax(Surf_PDF[i,j])
            value = Surf_PDF[i,j,start_x]
            z0 = start_x
            z1 = start_x+1
            while value < .67 and (z0 != 0 or z1 != Surf_PDF.shape[2]) :
                z0 = max(0, z0-1)
                z1 = min(z1+1, Surf_PDF.shape[2])
                value = np.sum(Surf_PDF[i,j, z0:z1])
            conf_range[i,j,0] = z0
            conf_range[i,j,1] = z1
    conf_range[:,:,2] = (conf_range[:,:,1] - conf_range[:,:,0]) / Surf_PDF.shape[2] * (h_range[1] - h_range[0])
    print(np.nanmean(conf_range[:,:,2]), np.nanmedian(conf_range[:,:,2]))
    # Est_HM = Surf_PDF.shape[2] - np.argmax(Surf_PDF, 2) - 1
    # Est_HM = (Est_HM / Surf_PDF.shape[2])*2-1


    h0 = h_range[0]
    h1 = h_range[1]

    Est_HM = (Est_HM + 1) / 2 * (h1 - h0) + h0
    GT = (GT + 1) / 2 * (h1 - h0) + h0
    Est_HM = Est_HM + np.nanmean((GT - Est_HM).ravel())
    diff = Est_HM - GT

    # K = get_gaussian_kernel2d((3,3), (1.5,1.5))
    # GT = mask_filter(GT, K)

    # valid = (Est_HM - GT) == (Est_HM - GT)
    # n_valid = np.sum(valid)
    # X = np.ones([n_valid, 2])
    # X[:,1] = Est_HM[valid]
    # Y = GT[valid].reshape([-1,1])
    # A = np.linalg.inv(X.T @ X) @ X.T @ Y

    # diff = (Est_HM * A[1,0] + A[0,0] - GT) * h_range
    # bias = np.mean(diff[diff == diff])
    # std_dev = np.std(diff[diff == diff])
    # print(std_dev)
    # exit()

    # plt.subplot(2,3,1)
    # plt.imshow(GT)
    # plt.subplot(2,3,2)
    # plt.imshow(Est_HM)
    # plt.subplot(2,3,3)
    # plt.imshow(diff)
    # plt.colorbar()
    #
    # plt.subplot(2,3,4)
    # plt.imshow(Surf_PDF.shape[2] - np.argmax(Surf_PDF, 2) - 1)
    # plt.subplot(2,3,5)
    # plt.imshow(conf_range[:,:,2])
    # plt.colorbar()
    #
    #
    # plt.show()

    diff = np.ravel(diff[diff == diff])
    # diff = diff - bias

    MAE = np.mean(np.abs(diff))
    RMSE = np.sqrt(np.mean(diff**2))
    Acc = np.sum(np.abs(diff) <= 1) / diff.shape[0]
    median_error = np.median(np.abs(diff))
    # print(bias)
    print("Before Alignment")
    print(MAE, RMSE, Acc, median_error)
    #exit()
    scores_before = {"MAE":MAE, "RMSE":RMSE, "Acc_1_m":Acc, "Median":median_error}
    Imgs = {"GT":GT, "Est_HM_no_Shift":Est_HM}

    # Est_HM = (Est_HM + 1) / 2 * (h1 - h0) + h0
    # GT = (GT + 1) / 2 * (h1 - h0) + h0
    # Est_HM = Est_HM + np.nanmean((GT - Est_HM).ravel())

    best_RMSE = RMSE
    shifts = np.array([[-1,-1],
                       [-1, 0],
                       [-1, 1],
                       [0, -1],
                       [0, 0],
                       [0, 1],
                       [1, -1],
                       [1, 0],
                       [1,1]])
    rotations = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])

    best_case = (0,-1)
    overall_change = np.array([0,0,0])
    steps = 0
    while best_case[0] != -1 and steps < 100:
        steps += 1
        best_case = (-1, -1)
        for i in range(shifts.shape[0]):
            for j in range(rotations.shape[0]):
                # GT_star = apply_shift_and_R(GT, shifts[i], rotations[j])
                # Est_HM = Est_HM + np.nanmean((GT_star - Est_HM).ravel())
                # RSME2 = np.sqrt(np.nanmean((Est_HM - GT_star)**2))
                HM_star = apply_shift_and_R(Est_HM, shifts[i], rotations[j])
                HM_star = HM_star + np.nanmean((GT - HM_star).ravel())
                RSME2 = np.sqrt(np.nanmean((HM_star - GT) ** 2))
                if RSME2 < best_RMSE:
                    best_RMSE = RSME2
                    best_case = (i,j)
        if best_case[0] != -1:
            Est_HM = apply_shift_and_R(Est_HM, shifts[best_case[0]], rotations[best_case[1]])
            overall_change[0] += shifts[best_case[0], 0]
            overall_change[1] += shifts[best_case[0], 1]
            overall_change[2] += rotations[best_case[1]]
        print(steps, overall_change, best_RMSE)
    Est_HM = Est_HM + np.nanmean((GT - Est_HM).ravel())
    diff = Est_HM - GT
    diff_no_ravel = Est_HM - GT

    diff = np.ravel(diff[diff == diff])
    # diff = diff - bias

    MAE = np.mean(np.abs(diff))
    RMSE = np.sqrt(np.mean(diff ** 2))
    Acc = np.sum(np.abs(diff) <= 1) / diff.shape[0]
    median_error = np.median(np.abs(diff))
    # print(bias)
    print("After Alignment")
    print(MAE, RMSE, Acc, median_error)
    scores_after = {"MAE": MAE, "RMSE": RMSE, "Acc_1_m": Acc, "Median": median_error, "Shift_x_y_deg":overall_change}
    print("Shift")
    print(overall_change)

    # plt.subplot(2,3,4)
    # plt.imshow(GT)
    # plt.subplot(2,3,5)
    # plt.imshow(Est_HM)
    # plt.subplot(2,3,6)
    # plt.imshow(diff_no_ravel)
    # plt.colorbar()
    # plt.show()
    Imgs["Est_HM_after_Shift"] = Est_HM

    # exit()
    return Imgs, scores_before, scores_after

    # exit()

def apply_shift_and_R(Img, shift, R):
    new_Img = np.copy(Img)
    if shift[0] == -1:
        new_Img = np.concatenate([new_Img, np.ones([1, new_Img.shape[1]]) * np.NaN], 0)[0:-1]
    elif shift[0] == 1:
        new_Img = np.concatenate([np.ones([1, new_Img.shape[1]]) * np.NaN, new_Img], 0)[1::]

    if shift[1] == -1:
        new_Img = np.concatenate([new_Img, np.ones([new_Img.shape[0], 1]) * np.NaN], 1)[:, 0:-1]
    elif shift[1] == 1:
        new_Img = np.concatenate([np.ones([new_Img.shape[0], 1]) * np.NaN, new_Img], 1)[:, 1::]

    new_Img = rotate(new_Img, angle=R, reshape=False, cval=np.mean(new_Img), prefilter=True, order=5)


    # plt.imshow(new_Img)
    # plt.show()
    # exit()

    return new_Img

def apply_shift_and_R_steps(Img, steps):
    new_Img = np.copy(Img)
    for a_step in steps:
        new_Img = apply_shift_and_R(new_Img, a_step[0], a_step[1])

    return new_Img



def find_align(GT, Est_HM, max_iters = 50, improve_thresh = 1e-8):
    diff = Est_HM - GT
    diff = np.ravel(diff[diff == diff])
    # diff = diff - bias

    MAE = np.mean(np.abs(diff))
    RMSE = np.sqrt(np.mean(diff ** 2))
    Acc = np.sum(np.abs(diff) <= 1) / diff.shape[0]
    median_error = np.median(np.abs(diff))
    # print(bias)
    print("Before Alignment")
    print(MAE, RMSE, Acc, median_error)
    best_RMSE = RMSE
    shifts = np.array([[-1, -1],
                       [-1, 0],
                       [-1, 1],
                       [0, -1],
                       [0, 0],
                       [0, 1],
                       [1, -1],
                       [1, 0],
                       [1, 1]])
    rotations = np.array([-5,-4,-3,-2,-1, 0, 1,2,3,4,5])

    best_case = (0, -1)
    overall_change = np.array([0, 0, 0])
    steps = 0
    align_steps = []
    while best_case[0] != -1 and steps < max_iters:
        steps += 1
        best_case = (-1, -1)
        for i in range(shifts.shape[0]):
            for j in range(rotations.shape[0]):
                # GT_star = apply_shift_and_R(GT, shifts[i], rotations[j])
                # Est_HM = Est_HM + np.nanmean((GT_star - Est_HM).ravel())
                # RSME2 = np.sqrt(np.nanmean((Est_HM - GT_star)**2))
                HM_star = apply_shift_and_R(Est_HM, shifts[i], rotations[j])
                HM_star = HM_star + np.nanmean((GT - HM_star).ravel())
                RSME2 = np.sqrt(np.nanmean((HM_star - GT) ** 2))
                if RSME2 < best_RMSE - improve_thresh:
                    best_RMSE = RSME2
                    best_case = (i, j)
        if best_case[0] != -1:
            Est_HM = apply_shift_and_R(Est_HM, shifts[best_case[0]], rotations[best_case[1]])
            overall_change[0] += shifts[best_case[0], 0]
            overall_change[1] += shifts[best_case[0], 1]
            overall_change[2] += rotations[best_case[1]]
            align_steps.append((shifts[best_case[0]], rotations[best_case[1]]))
        print(steps, overall_change, best_RMSE)
    Est_HM = Est_HM + np.nanmean((GT - Est_HM).ravel())
    diff = Est_HM - GT

    diff = np.ravel(diff[diff == diff])
    # diff = diff - bias

    MAE = np.mean(np.abs(diff))
    RMSE = np.sqrt(np.mean(diff ** 2))
    Acc = np.sum(np.abs(diff) <= 1) / diff.shape[0]
    median_error = np.median(np.abs(diff))
    print("After Alignment")
    print(MAE, RMSE, Acc, median_error)
    scores_after = {"MAE": MAE, "RMSE": RMSE, "Acc_1_m": Acc, "Median": median_error, "Shift_x_y_deg": overall_change}
    print("Shift")
    print(overall_change)

    return overall_change, align_steps


def eval_solar_walk(network, a_P_img, sun_el_and_az_test_vecs, device, out_img_size, use_HSLuv = False, use_classic_solar = False, thresh = .75):
    Solar_Imgs = {"Base_Solar_Vec":a_P_img.sun_el_and_az_vec}
    compare_vec = []
    c = 1
    print("Solar Walk")
    for a_P_img_sun_angle_only in tqdm(sun_el_and_az_test_vecs):
        out_img, valid_mask = get_img_from_P_img(network, a_P_img, time=a_P_img.time_obj.get_time_encode()[1:3],
                                                 out_img_size=out_img_size,
                                                 sun_vec=a_P_img_sun_angle_only, device=device,
                                                 use_TQDM=False, use_classic_solar = use_classic_solar)

        if use_HSLuv:
            for x_idx in range(out_img.shape[0]):
                for y_idx in range(out_img.shape[1]):
                    out_img[x_idx, y_idx] = hsluv.hsluv_to_rgb(
                        out_img[x_idx, y_idx] * np.array([360., 100., 100.]))

        Solar_Imgs[str(c)] = {"Solar_Vec":a_P_img_sun_angle_only, "Img":out_img, "Mask":valid_mask}
        compare_vec.append(out_img[valid_mask])
        c += 1

    compare_vec = np.array(compare_vec)
    end_pt = int(thresh*compare_vec.shape[1])
    score_matrix = np.zeros([compare_vec.shape[0], compare_vec.shape[0], 2])-1
    for i in range(compare_vec.shape[0]):
        for j in range(i+1, compare_vec.shape[0]):
            diff = np.sort(np.sqrt(np.sum((compare_vec[i] - compare_vec[j])**2, 1)))
            score = np.mean(diff[0:end_pt])
            score_full = np.mean(diff)
            score_matrix[i,j] = score, score_full
    Solar_Imgs["Score_Full_Score_Matrix"] = score_matrix
    return Solar_Imgs

def eval_season_walk(network, a_P_img, encoded_times, device, out_img_size, use_HSLuv = False, use_classic_solar = False, thresh = .75):
    Time_Imgs = {"Base_Time":a_P_img.time_obj.get_time_frac()}
    compare_vec = []
    c = 1
    print("Time Walk")
    for a_time in tqdm(encoded_times):
        out_img, valid_mask = get_img_from_P_img(network, a_P_img, time=a_time,
                                                 out_img_size=out_img_size,
                                                 sun_vec=a_P_img.sun_el_and_az_vec, device=device,
                                                 use_TQDM=False, use_classic_solar = use_classic_solar)

        if use_HSLuv:
            for x_idx in range(out_img.shape[0]):
                for y_idx in range(out_img.shape[1]):
                    out_img[x_idx, y_idx] = hsluv.hsluv_to_rgb(
                        out_img[x_idx, y_idx] * np.array([360., 100., 100.]))

        Time_Imgs[str(c)] = {"Time_Frac":np.arccos(a_time[0]), "Img":out_img, "Mask":valid_mask}
        compare_vec.append(out_img[valid_mask])
        c += 1

    compare_vec = np.array(compare_vec)
    end_pt = int(thresh*compare_vec.shape[1])
    score_matrix = np.zeros([compare_vec.shape[0], compare_vec.shape[0], 2])-1
    for i in range(compare_vec.shape[0]):
        for j in range(i+1, compare_vec.shape[0]):
            diff = np.sort(np.sqrt(np.sum((compare_vec[i] - compare_vec[j])**2, 1)))
            score = np.mean(diff[0:end_pt])
            score_full = np.mean(diff)
            score_matrix[i,j] = score, score_full
    Time_Imgs["Score_Full_Score_Matrix"] = score_matrix

    return Time_Imgs


def summarize_P_imgs(P_imgs, num_training = 3, extra_extreme_steps = 7):
    Is_Testing = np.zeros(len(P_imgs), dtype=bool)
    Is_Testing[np.linspace(0, len(P_imgs) - 1, num_training, dtype=int)] = True
    P_img_summary = {}
    P_img_summary["Basic_Info"] = {"Is_Testing": Is_Testing}

    el_az_angle, el_az_vec = [], []
    for a_P_img in P_imgs:
        el_az_angle.append(a_P_img.sun_el_and_az)
        el_az_vec.append(a_P_img.sun_el_and_az_vec)

    el_az_angle = np.array(el_az_angle)
    el_az_vec = np.array(el_az_vec)

    idx = np.argsort(el_az_angle[:,0])
    new_solar_el_az_angle = np.zeros([el_az_angle.shape[0]-1, el_az_angle.shape[1]])
    new_solar_el_az_vec = np.zeros([el_az_angle.shape[0]-1, 3])
    for i in range(idx.shape[0]-1):
        new_solar_el_az_angle[i] = (el_az_angle[idx[i]] + el_az_angle[idx[i+1]])/2
        new_solar_el_az_vec[i] = P_imgs[0].world_angle_2_local_vec(new_solar_el_az_angle[i,0], new_solar_el_az_angle[i,1])

    extra_solar_el = np.repeat(np.linspace(max(np.min(el_az_angle[:,0]) - 10, 0), min(np.max(el_az_angle[:,0])+10, 90), extra_extreme_steps), extra_extreme_steps)
    extra_solar_az = np.tile(np.linspace(max(np.min(el_az_angle[:,1]) - 10, 0),
                                         min(np.max(el_az_angle[:, 1]) + 10, 360), extra_extreme_steps, endpoint=True), extra_extreme_steps)
    extra_solar_el_az = np.stack([extra_solar_el, extra_solar_az],1)
    extra_solar_el_az_vec = np.zeros([extra_solar_el_az.shape[0], 3])
    for i in range(extra_solar_el_az.shape[0]):
        extra_solar_el_az_vec[i] = P_imgs[0].world_angle_2_local_vec(extra_solar_el_az[i,0], extra_solar_el_az[i,1])

    P_img_summary["Basic_Info"]["Solar_Angle"] = {"El_Az_Angle":el_az_angle, "El_Az_Vec":el_az_vec}
    P_img_summary["Solar_Near_info"] = {"El_Az_Angle":new_solar_el_az_angle, "El_Az_Vec":new_solar_el_az_vec}
    P_img_summary["Solar_Far_info"] = {"El_Az_Angle": extra_solar_el_az, "El_Az_Vec": extra_solar_el_az_vec}



    year_frac = []
    for a_P_img in P_imgs:
        year_frac.append(a_P_img.get_year_frac())
    year_frac = np.array(year_frac)
    idx = np.argsort(year_frac)
    new_year_frac = np.zeros(year_frac.shape[0])
    for i in range(year_frac.shape[0]-1):
        new_year_frac[i] = (year_frac[idx[i]] + year_frac[idx[i+1]])/2
    extra_year_frac = np.linspace(0,1, 24)
    P_img_summary["Basic_Info"]["Time_Info"] = year_frac
    P_img_summary["Time_Near_Info"] = new_year_frac
    P_img_summary["Time_Far_Info"] = extra_year_frac



    el_az_camera_angle, el_az_camera_vec = [], []
    for a_P_img in P_imgs:
        el_az_camera_angle.append([90-a_P_img.off_Nadir_from_IMD, a_P_img.Azmuth_from_IMD])
        el_az_camera_vec.append(a_P_img.world_angle_2_local_vec(el_az_camera_angle[-1][0], el_az_camera_angle[-1][1]))
    el_az_camera_angle = np.array(el_az_camera_angle)
    el_az_camera_vec = np.array(el_az_camera_vec)

    idx = np.argsort(el_az_camera_angle[:, 0])
    new_camera_el_az_angle = np.zeros([el_az_camera_angle.shape[0] - 1, el_az_camera_angle.shape[1]])
    new_camera_el_az_vec = np.zeros([el_az_camera_angle.shape[0] - 1, 3])
    for i in range(idx.shape[0] - 1):
        new_camera_el_az_angle[i] = (el_az_camera_angle[idx[i]] + el_az_camera_angle[idx[i + 1]]) / 2
        new_camera_el_az_vec[i] = P_imgs[0].world_angle_2_local_vec(new_camera_el_az_angle[i, 0],
                                                                   new_camera_el_az_angle[i, 1])

    extra_camera_el = np.repeat(np.linspace(max(np.min(el_az_camera_angle[:, 0]) - 10, 0),
                                            min(np.max(el_az_camera_angle[:, 0]) + 10, 90), extra_extreme_steps), extra_extreme_steps)
    extra_camera_az = np.tile(np.linspace(max(np.min(el_az_camera_angle[:, 1]) - 10, 0),
                                         min(np.max(el_az_camera_angle[:, 1]) + 10, 360), extra_extreme_steps, endpoint=True),
                             extra_extreme_steps)
    extra_camera_el_az = np.stack([extra_camera_el, extra_camera_az], 1)
    extra_camera_el_az_vec = np.zeros([extra_camera_el_az.shape[0], 3])
    for i in range(extra_camera_el_az.shape[0]):
        extra_camera_el_az_vec[i] = P_imgs[0].world_angle_2_local_vec(extra_camera_el_az[i, 0], extra_camera_el_az[i, 1])

    P_img_summary["Basic_Info"]["Camera_Angle"] = {"El_Az_Angle": el_az_camera_angle, "El_Az_Vec": el_az_camera_vec}
    P_img_summary["Camera_Near_info"] = {"El_Az_Angle": new_camera_el_az_angle, "El_Az_Vec": new_camera_el_az_vec}
    P_img_summary["Camera_Far_info"] = {"El_Az_Angle": extra_camera_el_az, "El_Az_Vec": extra_camera_el_az_vec}

    P_img_summary["Basic_Info"]["Overall"] = {"Time_Info":np.mean(year_frac), "Solar_Angle":np.mean(el_az_angle, 0), "Camera_Angle":np.mean(el_az_camera_angle, 0)}

    el_and_az_test = P_img_summary["Basic_Info"]["Solar_Angle"]["El_Az_Angle"]
    # print(el_and_az_test)
    a_poly = np.flip(np.polyfit(el_and_az_test[:,0], el_and_az_test[:,1], deg=1))
    el0, el1 = np.min(el_and_az_test[:, 0]), np.max(el_and_az_test[:, 0])
    # el0, el1 = np.mean(el_and_az_test[:,0]) - np.std(el_and_az_test[:,0]), np.mean(el_and_az_test[:,0]) + np.std(el_and_az_test[:,0])
    els = np.linspace(el0, el1, 12+2)[1:-1]
    azs = np.sum(np.array([a_poly[i] * els ** i for i in range(a_poly.shape[0])]), 0)
    times = np.linspace(0,1,12, endpoint=False)



    # plt.subplot(1,3,1)
    # plt.title("View Angle")
    # plt.scatter(extra_camera_el_az[:, 0], extra_camera_el_az[:, 1], c="red")
    # plt.scatter(new_camera_el_az_angle[:, 0], new_camera_el_az_angle[:, 1], c="orange")
    # plt.scatter(el_az_camera_angle[~Is_Testing, 0], el_az_camera_angle[~Is_Testing, 1], c="green")
    # plt.scatter(el_az_camera_angle[Is_Testing, 0], el_az_camera_angle[Is_Testing, 1], c="blue")
    # plt.xlabel("Elevation Angle, degrees")
    # plt.ylabel("Azmuth Angle, degrees")
    #
    # plt.subplot(1,3,2)
    # plt.title("Solar Angle")
    # plt.scatter(extra_solar_el_az[:, 0], extra_solar_el_az[:, 1], c="red")
    # plt.scatter(new_solar_el_az_angle[:, 0], new_solar_el_az_angle[:, 1], c="orange")
    # plt.scatter(el_az_angle[~Is_Testing, 0], el_az_angle[~Is_Testing, 1], c="green")
    # plt.scatter(el_az_angle[Is_Testing, 0], el_az_angle[Is_Testing, 1], c="blue")
    #
    # plt.plot(els, azs, "-o")
    #
    # plt.xlabel("Elevation Angle, degrees")
    # plt.ylabel("Azmuth Angle, degrees")
    #
    #
    # plt.subplot(1,3,3)
    # plt.title("Image Time")
    # plt.scatter(extra_year_frac, [3] * extra_year_frac.shape[0], c="red")
    # plt.scatter(new_year_frac, [2] * new_year_frac.shape[0], c="orange")
    # plt.scatter(year_frac[~Is_Testing], [1]*year_frac[~Is_Testing].shape[0], c="green")
    # plt.scatter(year_frac[Is_Testing], [1]*year_frac[Is_Testing].shape[0], c="blue")
    # plt.legend(["Extreme Inputs", "Near Inputs", "Training Inputs", "Testing Inputs"])
    # plt.ylim(0,4)
    # plt.yticks([])
    # plt.xticks(np.linspace(0, 1, 12),
    #            ["Jan.", "Feb.", "Mar.", "Apr.", "May", "Jun.", "Jul.", "Aug", "Sep.", "Oct.", "Nov.", "Dec."])
    #
    # # plt.show()
    # plt.close()

    return P_img_summary, [els, azs, times]



def eval_ortho_img(network, P_imgs, device, out_img_size, use_HSLuv = False):
    print("NOT DONE")
    max_batch_size = 50000
    XYZ = np.stack(
        np.meshgrid(np.arange(0, out_img_size[0]), np.arange(0, out_img_size[1]), np.arange(out_img_size[2]), indexing="ij"), -1)
    XYZ = XYZ.reshape([-1, 3])

    s = np.array([1 / out_img_size[0], 1 / out_img_size[1], 1 / out_img_size[2]])

    all_Rhos = np.zeros([out_img_size[0], out_img_size[1], out_img_size[2], 1])
    delta = 2 / out_img_size[2]
    network.eval()

    all_Solar_Angles = np.array([a_P_img.sun_el_and_az for a_P_img in P_imgs])
    print(all_Solar_Angles)
    plt.scatter(all_Solar_Angles[:,0], all_Solar_Angles[:,1])
    plt.show()
    # exit()

    with t.no_grad():
        for i in tqdm(range(0, XYZ.shape[0], max_batch_size)):
            batch_end = min(i + max_batch_size, XYZ.shape[0])

            xyz = XYZ[i:batch_end] * s * 2 - 1
            xyz[:, 2] *= -1

            Rho = network.forward_Classic_Sigma_Only(t.tensor(xyz).float().to(device))
            all_Rhos[XYZ[i:batch_end, 0], XYZ[i:batch_end, 1], XYZ[i:batch_end, 2]] = Rho.cpu()
    P_E = 1 - np.exp(-all_Rhos * delta)[:, :, :, 0]
    P_Vis = np.exp(
        -np.cumsum(np.concatenate([np.zeros([all_Rhos.shape[0], all_Rhos.shape[1], 1, 1]), all_Rhos * delta], 2), 2)[:,
         :, 0:-1])[:, :, :, 0]
    P_Surf = P_E * P_Vis
    Est_HM = np.sum(P_Surf * np.linspace(1, -1, P_Surf.shape[2]).reshape([1, 1, -1]), 2) / np.sum(P_Surf, 2)
    print(Est_HM.shape)
    plt.imshow(Est_HM)
    plt.show()
    exit()

