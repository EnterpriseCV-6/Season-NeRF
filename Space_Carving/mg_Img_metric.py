# from sewar.full_ref import mse, rmse, psnr, uqi, ssim, msssim, ergas, scc, rase, sam
# from image_similarity_measures.quality_metrics import fsim, issm, sre

import sewar.full_ref
# import image_similarity_measures.quality_metrics
import numpy as np
import torch as t
from scipy.signal import convolve
# from matplotlib import pyplot as plt
from Space_Carving.mg_phase_cong import very_fast_phasecong as pc

# from skimage.metrics import structural_similarity
import math
import cv2

def _ehs(x, y):
    """
    Entropy-Histogram Similarity measure
    """
    H = (np.histogram2d(x.flatten(), y.flatten()))[0]

    return -np.sum(np.nan_to_num(H * np.log2(H)))


def _edge_c(x, y):
    """
    Edge correlation coefficient based on Canny detector
    """
    # Use 100 and 200 as thresholds, no indication in the paper what was used
    g = cv2.Canny((x * 0.0625).astype(np.uint8), 100, 200)
    h = cv2.Canny((y * 0.0625).astype(np.uint8), 100, 200)

    g0 = np.mean(g)
    h0 = np.mean(h)

    numerator = np.sum((g - g0) * (h - h0))
    denominator = np.sqrt(np.sum(np.square(g-g0)) * np.sum(np.square(h-h0)))

    return numerator / denominator


def issm(org_img: np.ndarray, pred_img: np.ndarray) -> float:
    """
    Information theoretic-based Statistic Similarity Measure

    Note that the term e which is added to both the numerator as well as the denominator is not properly
    introduced in the paper. We assume the authers refer to the Euler number.
    """

    # Variable names closely follow original paper for better readability
    x = org_img * 4096
    y = pred_img * 4096
    A = 0.3
    B = 0.5
    C = 0.7

    ehs_val = _ehs(x, y)
    canny_val = _edge_c(x, y)

    numerator = canny_val * ehs_val * (A + B) + math.e
    denominator = A * canny_val * ehs_val + B * ehs_val + C * ssim(x, y) + math.e

    return np.nan_to_num(numerator / denominator)


def ssim(org_img: np.ndarray, pred_img: np.ndarray, max_p=4095) -> float:
    """
    Structural SIMularity index
    """

    return structural_similarity(org_img, pred_img, data_range=max_p, multichannel=True)



def _dumb_metric(img1, img2):
   return 1.

class _dumb_psnr():
    def __init__(self, max_val):
        self.max_val = max_val

    def __call__(self, img1, img2):
        return sewar.full_ref.psnr(img1, img2, self.max_val)

class _dumb_ssim():
    def __init__(self, max_val):
        self.max_val = max_val

    def __call__(self, img1, img2):
        return sewar.ssim(img1, img2, MAX=self.max_val, ws=img1.shape[0])[0]

class _dumb_uqi():
    def __init__(self, max_val):
        self.max_val = max_val

    def __call__(self, img1, img2):
        return sewar.uqi(img1, img2, ws=img1.shape[0]//2*2-1)

class _dumb_rase():
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        return sewar.rase(img1, img2, ws=3)

class _dumb_ergas():
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        return sewar.ergas(img1, img2, ws=3, r=1)

class _dumb_ms_ssim():
    def __init__(self, max_val, weights, K1, K2):
        self.max_val = max_val
        self.weights = weights
        self.k1 = K1
        self.k2 = K2

    def __call__(self, img1, img2):
        return sewar.msssim(img1, img2, weights=self.weights, ws=11, K1=self.k1, K2 = self.k2, MAX=self.max_val)

    # def __call__(self, img1, img2):
    #     img1D, img2D = np.copy(img1), np.copy(img2)
    #     # print(img1D.shape)
    #     mask = np.ones([2,2,1])/4.
    #     ms_ssim_score = 1.
    #     LCS = 1
    #     for i in range(len(self.weights)):
    #         LCS, CS = sewar.ssim(img1D, img2D, ws = img1D.shape[0], K1 = self.k1, K2 = self.k2, MAX = self.max_val)
    #         if i != len(self.weights)-1:
    #             img1D = convolve(img1D, mask, mode="same")
    #             img2D = convolve(img2D, mask, mode="same")
    #             img1D = img1D[::2, ::2]
    #             img2D = img2D[::2, ::2]
    #             # print(img1D.shape)
    #             ms_ssim_score *= CS ** self.weights[i]
    #
    #     return ms_ssim_score * LCS ** self.weights[-1]




class Img_Met():
    def __init__(self):
        self._cheat = _dumb_metric

    def apply_cheat(self, Imgs):
        ans = []
        for z in range(Imgs.shape[0]):
            ans.append(np.zeros([Imgs.shape[1], Imgs.shape[1]]))
            for n1 in range(Imgs.shape[1]):
                for n2 in range(Imgs.shape[1]):
                    ans[-1][n1, n2] = self._cheat(Imgs[z,n1], Imgs[z, n2])
        ans = np.stack(ans)
        return ans


    def apply_numpy(self, Imgs):
        pass

    def apply_tensor(self, Imgs):
        pass

class mg_mse(Img_Met):
    def __init__(self):
        super(mg_mse, self).__init__()
        self._cheat = sewar.full_ref.mse

    def apply_numpy(self, Imgs):
        diff = np.expand_dims(Imgs, 1) - np.expand_dims(Imgs, 2)
        ans = np.mean(diff**2, (3,4,5))
        return ans

    def apply_tensor(self, Imgs):
        diff = t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)
        ans = t.mean(diff ** 2, (3, 4, 5))
        return ans

class mg_rmse(Img_Met):
    def __init__(self):
        super(mg_rmse, self).__init__()
        self._cheat = sewar.full_ref.rmse

    def apply_numpy(self, Imgs):
        diff = np.expand_dims(Imgs, 1) - np.expand_dims(Imgs, 2)
        ans = np.sqrt(np.mean(diff**2, (3,4,5)))
        return ans

    def apply_tensor(self, Imgs):
        diff = t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)
        ans = t.sqrt(t.mean(diff ** 2, (3, 4, 5)))
        return ans

class mg_psnr(Img_Met):
    def __init__(self, max_val = 1, eps = 1e-10):
        super(mg_psnr, self).__init__()
        self._cheat = _dumb_psnr(max_val)
        self.k = 20*np.log10(max_val)
        self.eps = eps

    def apply_numpy(self, Imgs):
        diff = np.expand_dims(Imgs, 1) - np.expand_dims(Imgs, 2)
        ans =  self.k - 10*np.log10(np.mean(diff**2, (3,4,5)) + self.eps)
        return ans

    def apply_tensor(self, Imgs):
        diff = t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)
        ans = self.k - 10*t.log10(t.mean(diff**2, (3,4,5)) + self.eps)
        return ans

class mg_ssim_v0(Img_Met):
    def __init__(self, max_val=1, eps=1e-10):
        super(mg_ssim_v0, self).__init__()
        self._cheat = _dumb_ssim(max_val)
        self.k = 20 * np.log10(max_val)
        self.eps = eps

    def apply_numpy(self, Imgs):
        M = 1.
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * M) ** 2
        C2 = (K2 * M) ** 2
        img_channels = Imgs.shape[-1]
        out_img_size = Imgs.shape[-2] * Imgs.shape[-3]

        all_imgs_mus = np.mean(Imgs, (2, 3, 4))
        all_imgs_vars = np.var(Imgs, (2, 3, 4), ddof=1)
        all_sub_imgs = Imgs - np.expand_dims(np.expand_dims(np.expand_dims(all_imgs_mus, 2), 2), 2)

        mu_x_mu_y_part = 2 * np.expand_dims(all_imgs_mus, 1) * np.expand_dims(all_imgs_mus, 2) + C1
        mu_x_plus_mu_y_part = np.expand_dims(all_imgs_mus, 1) ** 2 + np.expand_dims(all_imgs_mus, 2) ** 2 + C1
        var_x_plus_var_y_part = np.expand_dims(all_imgs_vars, 1) + np.expand_dims(all_imgs_vars, 2) + C2
        sigma_xy_part = 2 * np.sum(
            np.expand_dims(all_sub_imgs, 1) * np.expand_dims(all_sub_imgs, 2),
            (3, 4, 5)) / (img_channels * (out_img_size) - 1) + C2
        ans = mu_x_mu_y_part * sigma_xy_part / (mu_x_plus_mu_y_part * var_x_plus_var_y_part)
        return ans

    def apply_tensor(self, Imgs):
        M = 1.
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * M) ** 2
        C2 = (K2 * M) ** 2
        img_channels = Imgs.shape[-1]
        out_img_size = Imgs.shape[-2] * Imgs.shape[-3]

        all_imgs_mus = t.mean(Imgs, (2, 3, 4))
        all_imgs_vars = t.var(Imgs, (2, 3, 4), unbiased=True)
        all_sub_imgs = Imgs - t.unsqueeze(t.unsqueeze(t.unsqueeze(all_imgs_mus, 2), 2), 2)

        mu_x_mu_y_part = 2 * t.unsqueeze(all_imgs_mus, 1) * t.unsqueeze(all_imgs_mus, 2) + C1
        mu_x_plus_mu_y_part = t.unsqueeze(all_imgs_mus, 1) ** 2 + t.unsqueeze(all_imgs_mus, 2) ** 2 + C1
        var_x_plus_var_y_part = t.unsqueeze(all_imgs_vars, 1) + t.unsqueeze(all_imgs_vars, 2) + C2
        sigma_xy_part = 2 * t.sum(
            t.unsqueeze(all_sub_imgs, 1) * t.unsqueeze(all_sub_imgs, 2),
            (3, 4, 5)) / (img_channels * (out_img_size) - 1) + C2
        ans = mu_x_mu_y_part * sigma_xy_part / (mu_x_plus_mu_y_part * var_x_plus_var_y_part)
        return ans

class mg_ssim(Img_Met):
    def __init__(self, max_val=1., K1 = 0.01, K2 = 0.03):
        super(mg_ssim, self).__init__()
        self._cheat = _dumb_ssim(max_val)

        M = max_val
        self.C1 = (K1 * M) ** 2
        self.C2 = (K2 * M) ** 2

    def apply_numpy(self, Imgs):
        out_img_size = Imgs.shape[-2] * Imgs.shape[-3]

        all_imgs_mus = np.mean(Imgs, (2, 3))
        all_imgs_vars = np.var(Imgs, (2, 3), ddof=1)
        all_sub_imgs = Imgs - np.expand_dims(np.expand_dims(all_imgs_mus, 2), 2)

        mu_x_mu_y_part = 2 * np.expand_dims(all_imgs_mus, 1) * np.expand_dims(all_imgs_mus, 2) + self.C1
        mu_x_plus_mu_y_part = np.expand_dims(all_imgs_mus, 1) ** 2 + np.expand_dims(all_imgs_mus, 2) ** 2 + self.C1
        var_x_plus_var_y_part = np.expand_dims(all_imgs_vars, 1) + np.expand_dims(all_imgs_vars, 2) + self.C2
        sigma_xy_part = 2 * np.sum(
            np.expand_dims(all_sub_imgs, 1) * np.expand_dims(all_sub_imgs, 2),
            (3, 4)) / ((out_img_size) - 1) + self.C2
        ans = np.mean(mu_x_mu_y_part * sigma_xy_part / (mu_x_plus_mu_y_part * var_x_plus_var_y_part), 3)

        return ans

    def apply_tensor(self, Imgs):
        out_img_size = Imgs.shape[-2] * Imgs.shape[-3]

        all_imgs_mus = t.mean(Imgs, (2, 3))
        all_imgs_vars = t.var(Imgs, (2, 3), unbiased=True)
        all_sub_imgs = Imgs - t.unsqueeze(t.unsqueeze(all_imgs_mus, 2), 2)

        mu_x_mu_y_part = 2 * t.unsqueeze(all_imgs_mus, 1) * t.unsqueeze(all_imgs_mus, 2) + self.C1
        mu_x_plus_mu_y_part = t.unsqueeze(all_imgs_mus, 1) ** 2 + t.unsqueeze(all_imgs_mus, 2) ** 2 + self.C1
        var_x_plus_var_y_part = t.unsqueeze(all_imgs_vars, 1) + t.unsqueeze(all_imgs_vars, 2) + self.C2
        sigma_xy_part = 2 * t.sum(
            t.unsqueeze(all_sub_imgs, 1) * t.unsqueeze(all_sub_imgs, 2),
            (3, 4)) / ((out_img_size) - 1) + self.C2
        ans = t.mean(mu_x_mu_y_part * sigma_xy_part / (mu_x_plus_mu_y_part * var_x_plus_var_y_part), 3)
        return ans

class mg_sam(Img_Met):
    def __init__(self):
        super(mg_sam, self).__init__()
        self._cheat = sewar.full_ref.sam

    def apply_numpy(self, Imgs):
        ts = np.reshape(Imgs, (Imgs.shape[0], Imgs.shape[1], Imgs.shape[2] * Imgs.shape[3], Imgs.shape[4]))
        rs = np.copy(ts)
        ts = np.expand_dims(ts, 1)
        rs = np.expand_dims(rs, 2)

        num = np.sum(ts * rs, 3)
        den_A = np.sqrt(np.sum(ts**2, 3))
        den_B = np.sqrt(np.sum(rs**2, 3))

        ans = np.mean(np.arccos(np.clip(num/ (den_A * den_B), 0, 1)), 3)

        return ans

    def apply_tensor(self, Imgs):
        ts = t.reshape(Imgs, (Imgs.shape[0], Imgs.shape[1], Imgs.shape[2] * Imgs.shape[3], Imgs.shape[4]))
        rs = ts.clone()
        ts = t.unsqueeze(ts, 1)
        rs = t.unsqueeze(rs, 2)

        num = t.sum(ts * rs, 3)
        den_A = t.sqrt(t.sum(ts ** 2, 3))
        den_B = t.sqrt(t.sum(rs ** 2, 3))

        ans = t.mean(t.acos(t.clamp(num / (den_A * den_B), 0, 1)), 3)
        return ans

class mg_uqi(mg_ssim):
    def __init__(self, max_val=1.):
        super(mg_uqi, self).__init__(max_val=max_val, K1=0.01, K2=0.03)
        self._cheat = _dumb_uqi(max_val)

class mg_ms_ssim(Img_Met):
    def __init__(self, max_val=1., K1=0.01, K2=0.03, weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]):
        super(mg_ms_ssim, self).__init__()
        # weights = [1]
        self._cheat = _dumb_ms_ssim(max_val, weights, K1, K2)
        self.weights = weights

        M = max_val
        self.C1 = (K1 * M) ** 2
        self.C2 = (K2 * M) ** 2
        self.C3 = self.C2 / 2
        self.n_weights = len(weights)

    def _get_compoents_np(self, Imgs):
        all_imgs_mus = np.mean(Imgs, (2, 3))
        all_imgs_vars = np.var(Imgs, (2, 3), ddof=1)
        all_sub_imgs = Imgs - np.expand_dims(np.expand_dims(all_imgs_mus, 2), 2)
        sigmaxy = np.sum(
            np.expand_dims(all_sub_imgs, 1) * np.expand_dims(all_sub_imgs, 2),
            (3, 4)) / ((Imgs.shape[-2] * Imgs.shape[-3]) - 1)

        lum = (2 * np.expand_dims(all_imgs_mus, 1) * np.expand_dims(all_imgs_mus, 2) + self.C1) / (np.expand_dims(all_imgs_mus**2, 1) + np.expand_dims(all_imgs_mus**2, 2) + self.C1)
        con = (2 * np.sqrt(np.expand_dims(all_imgs_vars, 1) * np.expand_dims(all_imgs_vars, 2)) + self.C2) / (np.expand_dims(all_imgs_vars, 1) + np.expand_dims(all_imgs_vars, 2) + self.C2)
        struc = (sigmaxy + self.C3) / (np.sqrt(np.expand_dims(all_imgs_vars, 1) * np.expand_dims(all_imgs_vars, 2)) + self.C3)

        return lum, con, struc

    def apply_numpy(self, Imgs):
        Imgs_DS = np.copy(Imgs)
        sim_val = 1.
        L = 1
        mask = np.ones([1,1,2,2,1])
        # x = np.linspace(-3,3, 11).reshape([-1,1])
        # mask = np.exp(-(x**2 + x.T**2)/1.5)
        # mask.resize([1,1,11,11,1])

        # print(np.round(mask, 3))
        # exit()

        mask = mask / np.sum(mask)
        for i in range(self.n_weights):
            L, C, S = self._get_compoents_np(Imgs_DS)
            # plt.figure()
            # plt.imshow(Imgs_DS[0, 0])
            sim_val *= pow((C * S).astype(np.complex), self.weights[i])
            if i != self.n_weights - 1:
                # print(Imgs_DS.shape)
                Imgs_DS = convolve(Imgs_DS, mask, mode="same")
                # print(Imgs_DS.shape)
                Imgs_DS = Imgs_DS[:,:,::2, ::2, :]
                # print(Imgs_DS.shape)

        # plt.show()
        # exit()



        sim_val *= pow(L.astype(complex), self.weights[-1])
        ans = np.mean(np.real(sim_val), -1)

        return ans

    def apply_tensor(self, Imgs):
        out_img_size = Imgs.shape[-2] * Imgs.shape[-3]

        all_imgs_mus = t.mean(Imgs, (2, 3))
        all_imgs_vars = t.var(Imgs, (2, 3), unbiased=True)
        all_sub_imgs = Imgs - t.unsqueeze(t.unsqueeze(all_imgs_mus, 2), 2)

        mu_x_mu_y_part = 2 * t.unsqueeze(all_imgs_mus, 1) * t.unsqueeze(all_imgs_mus, 2) + self.C1
        mu_x_plus_mu_y_part = t.unsqueeze(all_imgs_mus, 1) ** 2 + t.unsqueeze(all_imgs_mus, 2) ** 2 + self.C1
        var_x_plus_var_y_part = t.unsqueeze(all_imgs_vars, 1) + t.unsqueeze(all_imgs_vars, 2) + self.C2
        sigma_xy_part = 2 * t.sum(
            t.unsqueeze(all_sub_imgs, 1) * t.unsqueeze(all_sub_imgs, 2),
            (3, 4)) / ((out_img_size) - 1) + self.C2
        ans = t.mean(mu_x_mu_y_part * sigma_xy_part / (mu_x_plus_mu_y_part * var_x_plus_var_y_part), 3)
        return ans

class mg_sre(Img_Met):
    def __init__(self, eps = 1e-10):
        super(mg_sre, self).__init__()
        self._cheat = image_similarity_measures.quality_metrics.sre
        self.eps = eps

    def apply_numpy(self, Imgs):
        A = np.expand_dims(np.mean(Imgs, (2,3))**2, 2)
        B = np.linalg.norm((np.expand_dims(Imgs, 1) - np.expand_dims(Imgs, 2)), axis=(3,4))
        n = Imgs.shape[2] * Imgs.shape[3]
        B = B/n
        B[B < self.eps] = self.eps
        ans = np.log10(A/B)

        return 10*np.mean(ans, 3)

    def apply_tensor(self, Imgs):
        A = t.unsqueeze(t.mean(Imgs, (2, 3)) ** 2, 2)
        B = t.norm((t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)), p="fro", dim=(3, 4)) + self.eps
        n = Imgs.shape[2] * Imgs.shape[3]
        B = B / n
        B[B < self.eps] = self.eps
        ans = t.log10(A / B)

        return 10 * t.mean(ans, 3)

class mg_rase(Img_Met):
    def __init__(self, eps = 1e-10):
        super(mg_rase, self).__init__()
        self.eps = eps
        self._cheat = _dumb_rase()

    def apply_numpy(self, Imgs):
        RMSE = np.sqrt(np.mean((np.expand_dims(Imgs, 1) - np.expand_dims(Imgs, 2))**2, (3,4,5)))
        M = np.expand_dims(np.mean(Imgs, (2,3,4)), 2)
        ans = RMSE/(M + self.eps)
        return ans

    def apply_tensor(self, Imgs):
        RMSE = t.sqrt(t.mean((t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)) ** 2, (3, 4, 5)))
        M = t.unsqueeze(t.mean(Imgs, (2, 3, 4)), 2)
        ans = RMSE / (M + self.eps)
        return ans

class mg_ERGAS(Img_Met):
    def __init__(self, eps = 1e-10):
        super(mg_ERGAS, self).__init__()
        self.eps = eps
        self._cheat = _dumb_ergas()

    def apply_numpy(self, Imgs, r = 1.):
        RMSE = np.mean((np.expand_dims(Imgs, 1) - np.expand_dims(Imgs, 2))**2, (3,4))
        M = np.expand_dims(np.mean(Imgs, (2,3)), 2)
        ans = np.sqrt(np.mean(RMSE/(M**2 + self.eps), 3)) * r
        return ans

    def apply_tensor(self, Imgs, r = 1.):
        RMSE = t.mean((t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)) ** 2, (3, 4))
        M = t.unsqueeze(t.mean(Imgs, (2, 3)), 2)
        ans = t.sqrt(t.mean(RMSE / (M ** 2 + self.eps), 3)) * r
        return ans

class mg_fsim(Img_Met):
    def __init__(self, eps = 1e-10):
        super(mg_fsim, self).__init__()
        self.eps = eps
        self._cheat = image_similarity_measures.quality_metrics.fsim

    def _sim(self, X, constant):
        numerator = 2 * np.expand_dims(X, 1) * np.expand_dims(X, 2) + constant
        denominator = np.expand_dims(X, 1) ** 2 + np.expand_dims(X, 2) ** 2 + constant

        return numerator / denominator


    def apply_numpy(self, Imgs, r = 1.):
        alpha, beta = 1., 1.
        grad_Filter1 = np.array([[-3,0,3],
                                [-10,0,10],
                                [-3,0,3]])
        grad_Filter2 = grad_Filter1.T
        grad_Filter1.resize([1, 1, 3, 3, 1])
        grad_Filter2.resize([1, 1, 3, 3, 1])
        img_grads = np.sqrt(convolve(Imgs, grad_Filter1, mode="same")**2 + convolve(Imgs, grad_Filter2, mode="same")**2)
        pcs_2dim = pc(Imgs, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pcs_2dim_sum = np.sum(pcs_2dim[4], 2)
        S_g = self._sim(img_grads, constant=160)
        S_pc = self._sim(pcs_2dim_sum, constant=0.85)
        S_l = (S_pc ** alpha) * (S_g ** beta)


        norm = np.zeros(S_l.shape)
        for i in range(S_l.shape[1]):
            for j in range(S_l.shape[2]):
                norm[:, i, j] = np.maximum(pcs_2dim_sum[:, i], pcs_2dim_sum[:, j])

        numerator = np.sum(S_l * norm, (3,4))
        denominator = np.sum(norm, (3,4))
        ans = np.mean(numerator/denominator, -1)
        return ans

    def apply_tensor(self, Imgs, r = 1.):
        RMSE = t.mean((t.unsqueeze(Imgs, 1) - t.unsqueeze(Imgs, 2)) ** 2, (3, 4))
        M = t.unsqueeze(t.mean(Imgs, (2, 3)), 2)
        ans = t.sqrt(t.mean(RMSE / (M ** 2 + self.eps), 3)) * r
        return ans


class mg_issm(Img_Met):
    def __init__(self, max_val=1., K1 = 0.01, K2 = 0.03):
        super(mg_issm, self).__init__()
        self._cheat = issm
        self.C1 = 1
        self.C2 = 1

        self.simm_op = mg_ssim(max_val, K1, K2)

    def apply_numpy(self, Imgs):
        # out_img_size = Imgs.shape[-2] * Imgs.shape[-3]
        #
        # all_imgs_mus = np.mean(Imgs, (2, 3))
        # all_imgs_vars = np.var(Imgs, (2, 3), ddof=1)
        # all_sub_imgs = Imgs - np.expand_dims(np.expand_dims(all_imgs_mus, 2), 2)
        #
        # mu_x_mu_y_part = 2 * np.expand_dims(all_imgs_mus, 1) * np.expand_dims(all_imgs_mus, 2) + self.C1
        # mu_x_plus_mu_y_part = np.expand_dims(all_imgs_mus, 1) ** 2 + np.expand_dims(all_imgs_mus, 2) ** 2 + self.C1
        # var_x_plus_var_y_part = np.expand_dims(all_imgs_vars, 1) + np.expand_dims(all_imgs_vars, 2) + self.C2
        # sigma_xy_part = 2 * np.sum(
        #     np.expand_dims(all_sub_imgs, 1) * np.expand_dims(all_sub_imgs, 2),
        #     (3, 4)) / ((out_img_size) - 1) + self.C2
        # ans = np.mean(mu_x_mu_y_part * sigma_xy_part / (mu_x_plus_mu_y_part * var_x_plus_var_y_part), 3)
        ans = self.apply_cheat(Imgs)

        return ans

    def _ehs(self, Imgs):
        """
        Entropy-Histogram Similarity measure
        """

        H = (np.histogram2d(Imgs[0,0,:,:,0].cpu().numpy().flatten(), Imgs[0,1,:,:,0].cpu().numpy().flatten(), bins=3))[0]
        print(H)
        print(np.nan_to_num(H * np.log2(H)))
        exit()

        return -np.sum(np.nan_to_num(H * np.log2(H)))

    def apply_tensor(self, Imgs):
        ssim_scores = self.simm_op.apply_tensor(Imgs)
        print(ssim_scores.shape)

        print(Imgs.shape)
        A = 0.3
        B = 0.5
        C = 0.7

        ehs_val = self._ehs(Imgs)
        canny_val = _edge_c(x, y)

        numerator = canny_val * ehs_val * (A + B) + math.e
        denominator = A * canny_val * ehs_val + B * ehs_val + C * ssim_scores + math.e

        exit()
        return numerator / denominator