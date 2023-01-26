import numpy as np
import hsluv
import cv2 as cv

from scipy.stats import binned_statistic_dd
from tqdm import tqdm
from scipy.spatial.kdtree import KDTree
import scipy.sparse as sp_graph_sparse
from matplotlib import pyplot as plt

def get_Sig(pt_list, bin_edges):
    n_bins_per_edge = bin_edges.shape[1]
    H, edges = np.histogramdd(pt_list, bins=bin_edges, normed=False)
    # H = H / np.sum(H)
    # H = H / np.max(H)*100

    Sig = []
    for i in range(n_bins_per_edge-1):
        for j in range(n_bins_per_edge-1):
            for k in range(n_bins_per_edge-1):
                Sig.append([edges[0][i], edges[1][j], edges[2][k], H[i, j, k]])
    Sig = np.array(Sig)
    Sig[:,3] = Sig[:,3] / np.sum(Sig[:,3])
    return Sig


def prune_sig(Sig, thresh = 0.001):
    relevant_Sig = Sig[:,3] >= thresh
    Pruned_Sig = Sig[relevant_Sig]
    if np.sum(Pruned_Sig[:,3]) < .95:
        print("Warning: Pruning removing more than 5% of the data.")
    return Pruned_Sig, relevant_Sig

def get_Sig_advanced(img, bin_edges, dist_thresh, thresh = 0.001, show_process = False):
    if show_process:
        print("Building Centroids (Phase 1)...")
    counts = binned_statistic_dd(img, values=[1] * img.shape[0], bins=bin_edges, statistic="count")[0]
    ret1 = binned_statistic_dd(img, values=img[:, 0], bins=bin_edges, statistic="mean")[0]
    ret2 = binned_statistic_dd(img, values=img[:, 1], bins=bin_edges, statistic="mean")[0]
    ret3 = binned_statistic_dd(img, values=img[:, 2], bins=bin_edges, statistic="mean")[0]

    centroids = np.stack([ret1, ret2, ret3], 0)
    counts_R = counts.reshape([-1])
    good = counts_R > 0
    if show_process:
        print("Done")
        print("Initial Bin Count:", counts_R.shape[0])
        print("Non-Zero Bin Count:", np.sum(good))
        print("Phase 1 done.")
    counts_R = counts_R[good]
    centroids_R = centroids.reshape([3,-1])[:,good].T

    if show_process:
        print("Building initial Sigs for Phase 2")
    the_KD_Tree = KDTree(centroids_R)
    ans = the_KD_Tree.query_ball_point(centroids_R, r=dist_thresh)
    merges = []
    Sig = []
    for i in range(centroids_R.shape[0]):
        if len(ans[i]) == 1:
            Sig.append([centroids_R[i,0], centroids_R[i,1], centroids_R[i,2], counts_R[i]])
        else:
            merges.append(ans[i])

    if show_process:
        print("Adding merged Sigs for Phase 2")
    n_merges = len(merges)
    empty_set = set([])
    M = []
    for i in range(n_merges):
        has_intersect = False
        merge_set = set(merges[i])
        for j in range(len(M)):
            if (M[j] & merge_set) != empty_set:
                M[j] = M[j] | merge_set
                has_intersect = True
                break
        if has_intersect == False:
            M.append(merge_set)
    for i in M:
        a_sig = [0.,0.,0.,0.]
        for idx in list(i):
            n0 = a_sig[3]
            n1 = counts_R[idx]
            for j in range(3):
                a_sig[j] = (n0 * a_sig[j] + centroids_R[idx,j] * n1) / (n0 + n1)
            a_sig[3] = n0 + n1
        Sig.append(a_sig)
    Sig = np.array(Sig)
    Sig[:,3] = Sig[:,3] / np.sum(Sig[:,3])
    if show_process:
        print("Number of Sigs after Merging:", Sig.shape[0])
        print("Phase 2 done.")
        print("Pruning Sigs.")

    Sig, _ = prune_sig(Sig, thresh)
    if show_process:
        print("Sigs after Pruning:", Sig.shape[0])
        print("Sig Recall:", np.sum(Sig[:,3]))

    return Sig



def EM_img_Compare(img1, img2, bins_per_edge = 8, use_hsluv = False):
    img1r = img1.reshape([-1, 3])
    img2r = img2.reshape([-1, 3])


    if use_hsluv:
        bin_edges = np.array([np.linspace(0, 360, bins_per_edge + 1), np.linspace(0, 100, bins_per_edge + 1),
                              np.linspace(0, 100, bins_per_edge + 1)])
        img1r = np.array(
            [hsluv.rgb_to_hsluv(img1r[j]) for j in range(0, img1r.shape[0], 1)])
        img2r = np.array(
            [hsluv.rgb_to_hsluv(img2r[j]) for j in range(0, img2r.shape[0], 1)])
    else:
        bin_edges = np.array([np.linspace(0, 1, bins_per_edge + 1), np.linspace(0, 1, bins_per_edge + 1),
                              np.linspace(0, 1, bins_per_edge + 1)])


    img1_sig = get_Sig(img1r, bin_edges)
    img1_sig, _ = prune_sig(img1_sig)
    img1_sig[:,3] = img1_sig[:,3] / np.sum(img1_sig[:,3])

    img2_sig = get_Sig(img2r, bin_edges)
    img2_sig, _ = prune_sig(img2_sig)
    img2_sig[:, 3] = img2_sig[:, 3] / np.sum(img2_sig[:, 3])

    EM, lowerbound, flow_matrix = EM_sig_Compare(img1_sig, img2_sig)

    return EM

def EM_img_Compare_v2(img1, img2, bins_per_edge = 8, use_hsluv = False, prune_thresh = .001):
    img1r = img1.reshape([-1, 3])
    img2r = img2.reshape([-1, 3])


    # if use_hsluv:
    #     bin_edges = np.array([np.linspace(0, 360, bins_per_edge + 1), np.linspace(0, 100, bins_per_edge + 1),
    #                           np.linspace(0, 100, bins_per_edge + 1)])
    #     img1r = np.array(
    #         [hsluv.rgb_to_hsluv(img1r[j]) for j in range(0, img1r.shape[0], 1)])
    #     img2r = np.array(
    #         [hsluv.rgb_to_hsluv(img2r[j]) for j in range(0, img2r.shape[0], 1)])
    # else:
    bin_edges = np.array([np.linspace(0, 1, bins_per_edge + 1), np.linspace(0, 1, bins_per_edge + 1),
                          np.linspace(0, 1, bins_per_edge + 1)])
    dist_thresh = (bin_edges[0,1]-bin_edges[0,0])/2


    img1_sig = get_Sig_advanced(img1r, bin_edges, dist_thresh, prune_thresh)
    img2_sig = get_Sig_advanced(img2r, bin_edges, dist_thresh, prune_thresh)
    EM, lowerbound, flow_matrix = EM_sig_Compare(img1_sig, img2_sig)

    return EM

def EM_sig_Compare(Sig1, Sig2):
    #Reorder sigs so that it is Weight by XYZ rather than XYZ by weight
    EM_sig_order1 = np.concatenate([Sig1[:, 3::], Sig1[:, 0:3]], 1)
    EM_sig_order2 = np.concatenate([Sig2[:, 3::], Sig2[:, 0:3]], 1)
    EM, lowerbound, flow_matrix = cv.EMD(EM_sig_order1.astype(np.float32), EM_sig_order2.astype(np.float32), cv.DIST_L1)
    return EM, lowerbound, flow_matrix

def EM_naive_Sig(img, bin_edges):
    counts = binned_statistic_dd(img, values=[1] * img.shape[0], bins=bin_edges, statistic="count")[0]
    ret1, _, bin_ids = binned_statistic_dd(img, values=img[:, 0], bins=bin_edges, statistic="mean", expand_binnumbers=False)
    ret2 = binned_statistic_dd(img, values=img[:, 1], bins=bin_edges, statistic="mean", )[0]
    ret3 = binned_statistic_dd(img, values=img[:, 2], bins=bin_edges, statistic="mean")[0]

    centroids = np.stack([ret1, ret2, ret3], -1)
    return centroids, counts, bin_ids.T

class mg_EM():
    def __init__(self, RGB_img:np.ndarray):
        self._img = RGB_img
        self._valid_color_spaces = ["RGB", "HSLUV"]
        self._sig = None

    def _merge_sigs(self, Sigs, dist_thresh, bin_labels, use_bin_labels, show_process):
        if show_process:
            print("Merging Sigs...")


        centroids_R = Sigs[:,0:3]
        the_KD_Tree = KDTree(centroids_R)
        ans = the_KD_Tree.query_ball_point(centroids_R, r=dist_thresh)

        c = 0
        edge_array = np.zeros([Sigs.shape[0], Sigs.shape[0]], dtype=bool)
        for a_sim_set in ans:
            for sim_case in a_sim_set:
                edge_array[c, sim_case] = True
            c += 1


        graph = sp_graph_sparse.csr_matrix(edge_array)
        n_compoents, bin_ids = sp_graph_sparse.csgraph.connected_components(graph, directed=False)

        if use_bin_labels:
            new_bin_labels = np.copy(bin_labels)
            for i in np.arange(Sigs.shape[0]):
                new_bin_labels[bin_labels == i] = bin_ids[i]
        else:
            new_bin_labels = bin_labels

        new_Sigs = np.zeros([n_compoents, 4])
        for i in range(Sigs.shape[0]):
            n0 = new_Sigs[bin_ids[i], 3]
            n1 = Sigs[i,3]
            new_Sigs[bin_ids[i], 0:3] = (new_Sigs[bin_ids[i], 0:3] * n0 + Sigs[i, 0:3] * n1) / (n0 + n1)
            new_Sigs[bin_ids[i], 3] = n0 + n1


        if show_process:
            print("Number of bins after Merge:", n_compoents)

        return new_Sigs, new_bin_labels

    def _RGB_to_LAB(self):
        _img = np.copy(self._img)
        bad_loc = _img != _img
        _img[bad_loc] = 0.
        _img = _img.astype(np.float32)
        img2 = cv.cvtColor(_img, cv.COLOR_RGB2LAB)
        max_bin = (100., 127., 127.)
        min_bin = (0., -127., -127.)
        img2[bad_loc] = np.NaN

        return img2, min_bin, max_bin

    def _LAB_to_RGB(self, img):
        img2 = cv.cvtColor(img, cv.COLOR_LAB2RGB)
        return img2

    def get_Sig(self, bin_size = 12.5, prune_thresh = .001, use_LAB = True, naive=False, show_details=False, get_bin_ids = False):
        if use_LAB:
            Col_Img, min_bin, max_bin = self._RGB_to_LAB()
            Col_Img = Col_Img.reshape([-1,3])
        else:
            Col_Img = self._img.reshape([-1,3])
            min_bin = (0., 0., 0.)
            max_bin = (1., 1., 1.)

        bins = []
        for i in range(3):
            bins.append(np.linspace(min_bin[i], max_bin[i], int((max_bin[i]-min_bin[i])/bin_size)+1))

        if show_details:
            print("Bin Edges:")
            [print(i) for i in bins]

        centroids, counts, bin_ids = EM_naive_Sig(Col_Img, bins)
        if show_details:
            print("Number Sigs:", np.prod(counts.shape))
        Sigs = np.concatenate([centroids.reshape([-1, 3]), counts.reshape([-1, 1])], 1)


        not_nan_Sig = Sigs[:, 3] > 0
        non_zero_sig_count = np.sum(not_nan_Sig)
        if show_details:
            print("Number Sigs (non-zero):", non_zero_sig_count)
        Sigs = Sigs[not_nan_Sig]
        Sigs[:, 3] = Sigs[:, 3] / np.sum(Sigs[:, 3])

        if get_bin_ids:
            U_ids = np.unique(bin_ids)
            new_bin_ids = np.copy(bin_ids)
            for i in range(U_ids.shape[0]):
                new_bin_ids[bin_ids == U_ids[i]] = i
            bin_ids = new_bin_ids


        if naive == False:
            dist_thresh = np.array([(bins[i][1]-bins[i][0])/2 for i in range(3)])
            if len(np.unique(dist_thresh)) > 1 and show_details:
                print("Waning: Unequal distance threshold for channels!")
                print("Distance Threshold for Merging:", dist_thresh)
            dist_thresh = np.mean(dist_thresh)
            if show_details:
                print("Distance Threshold for Merging:", dist_thresh)
            Sigs, bin_ids = self._merge_sigs(Sigs, dist_thresh, bin_ids, get_bin_ids, show_details)
            non_zero_sig_count = Sigs.shape[0]


        if prune_thresh > 0:
            good = Sigs[:,3] > prune_thresh
            prune_recall = np.sum(Sigs[good, -1])
            Sigs = Sigs[good]
            Sigs[:,3] = Sigs[:,3] / np.sum(Sigs[:,3])
            if get_bin_ids:
                removed_idx = np.any(bin_ids.reshape([-1,1]) == np.arange(non_zero_sig_count)[~good].reshape([1,-1]), 1)
                if np.any(removed_idx):
                    bin_ids[removed_idx] = -1
                    U_ids = np.unique(bin_ids)
                    new_bin_ids = np.copy(bin_ids)
                    for i in range(1, U_ids.shape[0]):
                        new_bin_ids[bin_ids == U_ids[i]] = i-1
                    bin_ids = new_bin_ids


            if show_details:
                print("Number Sigs (after prune):", np.sum(good))
                print("Recall (after prune):", prune_recall)
        else:
            prune_recall = 1.

        if not get_bin_ids:
            bin_ids = None

        EM_sig_order = np.concatenate([Sigs[:,3::], Sigs[:,0:3]], 1)
        self._sig = EM_sig_order


        return Sigs, prune_recall, bin_ids

    def get_Sig_adj_img(self, Sigs, bin_ids, use_LAB=True):
        if use_LAB:
            Col_Img, _, _ = self._RGB_to_LAB()
        else:
            Col_Img = self._img.copy()

        valid_mask = np.zeros([self._img.shape[0], self._img.shape[1]], dtype=bool)

        idxs = np.arange(bin_ids.shape[0])
        for i in range(Sigs.shape[0]):
            locs = idxs[bin_ids == i]
            X,Y = np.unravel_index(locs, (self._img.shape[0], self._img.shape[1]))

            valid_mask[X,Y] = True
            Col_Img[X,Y] = Sigs[i,0:3]

        if use_LAB:
            Col_Img = self._LAB_to_RGB(Col_Img)

        return Col_Img, valid_mask

def compare_EM_imgs(Img1:mg_EM, Img2:mg_EM):
    if Img1._sig is None:
        Sigs, prune_recall, bin_ids = Img1.get_Sig(get_bin_ids=True)
        # Col_img, valid_mask = Img1.get_Sig_adj_img(Sigs, bin_ids)
        # plt.subplot(1, 2, 1)
        # plt.imshow(Img1._img)
        # plt.subplot(1,2,2)
        # plt.imshow(Col_img)
        # plt.show()
    if Img2._sig is None:
        Img2.get_Sig()

    # print(Img1._sig)
    # exit()
    EM, lowerbound, flow_matrix = cv.EMD(Img1._sig.astype(np.float32), Img2._sig.astype(np.float32), cv.DIST_L1)
    return EM, lowerbound, flow_matrix