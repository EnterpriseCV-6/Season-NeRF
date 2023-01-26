# coding=utf-8


from matplotlib import pyplot as plt
from matplotlib import gridspec
import datetime
import numpy as np


def _show_all_images(P_imgs, region, idx, skip_show = False):
    nrow = int(np.sqrt(len(P_imgs)))
    ncol = nrow
    while nrow * ncol < len(P_imgs):
        if nrow < ncol:
            nrow += 1
        else:
            ncol += 1

    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)))
    for i in range(len(P_imgs)):
        ax = plt.subplot(nrow, ncol, i + 1)
        if region is None:
            ax.imshow(P_imgs[i].img[0::8, 0::8])
        else:
            ax.imshow(P_imgs[i].img[region[0]:region[1], region[2]:region[3]])
        ax.set_xticks([])
        ax.set_yticks([])
        name, sat_el_and_az, sun_el_and_az, year_frac = P_imgs[i].get_meta_data()
        # title = str(idx[i]) + ", " + P_imgs[i].time_obj.raw_str[5:10]
        # title = str(idx[i]) + ", " + str(sat_el_and_az)
        # title = str(idx[i]) + ", " + str(sun_el_and_az)
        # title = str(idx[i]) + ", " + str(sat_el_and_az[1]) + ", " + str(sun_el_and_az[0])
        title = str(idx[i]) + ", " + P_imgs[i].time_obj.raw_str[5:10] + ", " + str(sat_el_and_az[1]) + ", " + str(sun_el_and_az[0])
        ax.set_title(title)
    if skip_show == False:
        plt.show()

def show_all_images(P_imgs, region = None, autosort = False):
    if autosort == False:
        _show_all_images(P_imgs, region, np.arange(len(P_imgs)))
    else:
        from all_NeRF import mg_EM_Imgs
        import hsluv
        from tqdm import tqdm
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        bin_edges = [np.linspace(0, 360, 19), np.linspace(0, 100, 6), np.linspace(0, 100, 6)]
        dist_matrix = np.zeros([len(P_imgs), len(P_imgs)])
        Sigs = []
        for a_P_img in tqdm(P_imgs):
            a_img = a_P_img.img[0::16, 0::16].reshape([-1,3])
            img_HSL_scaled = np.array([hsluv.rgb_to_hsluv(a) for a in a_img])
            a_Sig = mg_EM_Imgs.get_Sig_advanced(img_HSL_scaled, bin_edges, dist_thresh=10, show_process=False, thresh=0.001)
            Sigs.append(a_Sig)
        for i in tqdm(range(len(P_imgs))):
            for j in range(i+1, len(P_imgs)):
                dist, _, _ = mg_EM_Imgs.EM_sig_Compare(Sigs[i], Sigs[j])
                dist_matrix[i,j] = dist
                # dist_matrix[j,i] = dist

        min_points_per_component = 4
        graph = np.ones([len(P_imgs), len(P_imgs)], dtype=int)
        n_commponets = 1
        labels = [0] * len(P_imgs)
        while n_commponets < 4:

            x,y = np.unravel_index(np.argmax(dist_matrix), [len(P_imgs), len(P_imgs)])
            if dist_matrix[x,y] <= 0:
                break
            dist_matrix[x,y] = -1
            graph[x,y] = 0
            graph[y,x] = 0
            csr_graph = csr_matrix(graph)
            n_commponets, labels = connected_components(csr_graph, directed=False)
            good = True
            for i in range(n_commponets):
                if np.sum(labels == i) < min_points_per_component:
                    good = False
                    break
            if good == False:
                graph[x, y] = 1
                graph[y, x] = 1

        for i in range(n_commponets):
            R_P_imgs = []
            idx = []
            for j in range(len(P_imgs)):
                if labels[j] == i:
                    R_P_imgs.append(P_imgs[j])
                    idx.append(j)
            _show_all_images(R_P_imgs, region, idx, skip_show=True)

        plt.show()


def gen_angle_images(P_imgs, testing_idx, walking_sat_angle, walking_sun_angle, annotate_pts = False, output_path = None):
    # print(walking_sat_angle)
    # print(walking_sun_angle)
    training_sat_angle = []
    training_sun_angle = []
    testing_sat_angle = []
    testing_sun_angle = []
    for i in range(len(P_imgs)):
        name, sat_el_and_az, sun_el_and_az, year_frac = P_imgs[i].get_meta_data()
        if i not in testing_idx:
            training_sat_angle.append(sat_el_and_az)
            training_sun_angle.append(sun_el_and_az)
        else:
            testing_sat_angle.append(sat_el_and_az)
            testing_sun_angle.append(sun_el_and_az)
    training_sat_angle = np.array(training_sat_angle)
    training_sun_angle = np.array(training_sun_angle)
    testing_sat_angle = np.array(testing_sat_angle)
    testing_sun_angle = np.array(testing_sun_angle)

    training_sat_angle[:, 0] = 90 - training_sat_angle[:, 0]
    x, y = np.cos(np.deg2rad(training_sat_angle[:, 1])) * training_sat_angle[:, 0], np.sin(
        np.deg2rad(training_sat_angle[:, 1])) * training_sat_angle[:, 0]
    testing_sat_angle[:, 0] = 90 - testing_sat_angle[:, 0]
    x2, y2 = np.cos(np.deg2rad(testing_sat_angle[:, 1])) * testing_sat_angle[:, 0], np.sin(np.deg2rad(
        testing_sat_angle[:, 1])) * testing_sat_angle[:, 0]

    if walking_sat_angle.shape[0] > 0:
        walking_sat_angle[:, 0] = 90 - walking_sat_angle[:, 0]
        x3, y3 = np.cos(np.deg2rad(walking_sat_angle[:, 1])) * walking_sat_angle[:, 0], np.sin(np.deg2rad(
            walking_sat_angle[:, 1])) * walking_sat_angle[:, 0]

    plt.figure(figsize=(12, 6), dpi=80)
    plt.subplot(1, 2, 1)
    plt.axhline(c="black")
    plt.axvline(c="black")
    X = np.linspace(-32, 32, 50)
    Y = np.linspace(-32, 32, 50)
    Z = np.sqrt(X.reshape([-1, 1]) ** 2 + Y.reshape([1, -1]) ** 2)
    ax = plt.contour(X, Y, Z)
    plt.clabel(ax, inline=True, fontsize=10)

    a = plt.scatter(x, y)
    b = plt.scatter(x2, y2)
    if walking_sat_angle.shape[0] > 0:
        c = plt.scatter(x3, y3, c="red")
    if annotate_pts:
        for i in range(x3.shape[0]):
            plt.annotate(str(i+1), (x3[i], y3[i]))
    plt.scatter(0, 0, c="purple")
    if walking_sat_angle.shape[0] > 0:
        plt.legend([a, b, c], ["Training Point", "Testing Point", "Walking Point"])
    else:
        plt.legend([a, b], ["Training Point", "Testing Point"])
    plt.title("Overview of Satellite Off-Nadir and Azmuth Angles")
    plt.xlim(-32, 32)
    plt.ylim(-32, 32)

    plt.subplot(1, 2, 2)

    x, y = np.cos(np.deg2rad(training_sun_angle[:, 1])) * training_sun_angle[:, 0], np.sin(
        np.deg2rad(training_sun_angle[:, 1])) * training_sun_angle[:, 0]
    x2, y2 = np.cos(np.deg2rad(testing_sun_angle[:, 1])) * testing_sun_angle[:, 0], np.sin(np.deg2rad(
        testing_sun_angle[:, 1])) * testing_sun_angle[:, 0]

    if walking_sun_angle.shape[0] > 0:
        x3, y3 = np.cos(np.deg2rad(walking_sun_angle[:, 1])) * walking_sun_angle[:, 0], np.sin(np.deg2rad(
            walking_sun_angle[:, 1])) * walking_sun_angle[:, 0]

    plt.axhline(c="black")
    plt.axvline(c="black")
    X = np.linspace(5, -60, 50)
    Y = np.linspace(-5, 60, 50)
    Z = np.sqrt(X.reshape([-1, 1]) ** 2 + Y.reshape([1, -1]) ** 2)
    ax = plt.contour(X, Y, Z)
    plt.clabel(ax, inline=True, fontsize=10)

    a = plt.scatter(x, y)
    b = plt.scatter(x2, y2)
    if walking_sun_angle.shape[0] > 0:
        c = plt.scatter(x3, y3, c="red")
    if annotate_pts:
        for i in range(x3.shape[0]):
            plt.annotate(str(i + 1), (x3[i], y3[i]))
    plt.scatter(0, 0, c="purple")
    if walking_sun_angle.shape[0] > 0:
        plt.legend([a, b, c], ["Training Point", "Testing Point", "Walking Point"])
    else:
        plt.legend([a, b], ["Training Point", "Testing Point"])
    plt.title("Overview of Solar Elevation and Azmuth Angles")
    plt.xlim(-60, 5)
    plt.ylim(-5, 60)

    plt.tight_layout()
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close("all")

def get_walking_Points(P_imgs, n_walking_view, n_walking_sun, n_walking_points_time, min_day_sep, unused_idx = ()):
    walking_times = np.linspace(0, 1, n_walking_points_time, endpoint=False)

    training_time = []
    training_sun_el_az = []
    training_view_el_az = []
    for i in range(len(P_imgs)):
        if i not in list(unused_idx):
            name, sat_el_and_az, sun_el_and_az, year_frac = P_imgs[i].get_meta_data()
            training_time.append(year_frac)
            training_sun_el_az.append(sun_el_and_az)
            training_view_el_az.append(sat_el_and_az)
    training_time = np.array(training_time)
    training_sun_el_az = np.array(training_sun_el_az)
    training_view_el_az = np.array(training_view_el_az)

    min_el = max(np.min(training_sun_el_az[:,0]) - 5., 0.)
    max_el = min(np.max(training_sun_el_az[:,0]) + 5., 90.)

    the_gen = np.poly1d(np.polyfit(training_sun_el_az[:,0], training_sun_el_az[:,1], deg=3))
    walk_sun = np.linspace(min_el, max_el, n_walking_sun)
    walk_sun = np.stack([walk_sun, the_gen(walk_sun)], 1)

    dist_thresh = min_day_sep / 365.24
    if min_day_sep > 0:
        n = 1
        dist = np.abs(walking_times.reshape([-1, 1]) - training_time.reshape([1, -1]))
        dist[dist > .5] = 1. - dist[dist > .5]
        dist = np.min(dist, 1)
        good = dist <= dist_thresh
        while np.sum(good) < n_walking_points_time:
            walking_times = np.linspace(0, 1, n_walking_points_time + n, endpoint=False)
            dist = np.abs(walking_times.reshape([-1, 1]) - training_time.reshape([1, -1]))
            dist[dist > .5] = 1. - dist[dist > .5]
            dist = np.min(dist, 1)
            good = dist <= dist_thresh
            n += 1
        walking_times = walking_times[good]

    min_view = max(np.min(training_view_el_az[:,0])-5, 0.)
    # k = max(int(np.round((90-min_view) / 5.)), 1)
    k = 1
    view_el = np.linspace(90, min_view, n_walking_view+1)[1::]
    view_az = np.linspace(0, 360*k, n_walking_view)
    walk_view = np.stack([view_el, view_az], 1)
    return walk_view, walk_sun, walking_times







def show_proto_images(P_imgs, training_idx, testing_idx, seasonal_proto_idx, walking_times, output_path = None):
    fig = plt.figure(figsize=((len(seasonal_proto_idx)+1)*5, 6))
    training_time = np.array([P_imgs[i].get_year_frac() for i in training_idx])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i in range(len(seasonal_proto_idx)):
        a_ax = fig.add_subplot(1, len(seasonal_proto_idx) + 1, i + 2)
        a_ax.imshow(P_imgs[seasonal_proto_idx[i]].img)
        a_ax.set_xticks([])
        a_ax.set_yticks([])
        a_ax.set_title(months[P_imgs[seasonal_proto_idx[i]].time_obj.month - 1] + ". " + str(
            P_imgs[seasonal_proto_idx[i]].time_obj.day))

    testing_idx_small = []
    for i in testing_idx:
        if i not in seasonal_proto_idx:
            testing_idx_small.append(i)

    ax = fig.add_subplot(1, len(seasonal_proto_idx) + 1, 1, polar=True)
    testing_time = np.array([P_imgs[i].get_year_frac() for i in testing_idx_small])
    proto_time = np.array([P_imgs[i].get_year_frac() for i in seasonal_proto_idx])
    ax.scatter(training_time * 2 * np.pi, np.ones_like(training_time) + np.random.rand(training_time.shape[0]) * .4)

    # ax.scatter(proto_time * 2 * np.pi, np.ones_like(proto_time) + np.random.rand(proto_time.shape[0]) * .4)

    ax.scatter(testing_time * 2 * np.pi, np.ones_like(testing_time) + np.random.rand(testing_time.shape[0]) * .4)
    ax.scatter(proto_time * 2 * np.pi, np.ones_like(proto_time) + np.random.rand(proto_time.shape[0]) * .4, c="green")

    if walking_times.shape[0] > 0:
        ax.scatter(walking_times * 2 * np.pi, np.ones_like(walking_times) + .2,
               c="red")
    ax.set_rmax(1.5)
    ax.set_rticks([])
    angles = np.linspace(0, 360, 12, endpoint=False)
    ax.set_thetagrids(angles, months)

    # ax.legend(["Training Point", "non-Prototypical Testing Point", "Prototypical Testing Point", "Walking Point"], loc=10)
    # ax.legend(["Training Point", "Testing Point", "Walking Point"], loc=10)
    ax.set_title("Overview of Data Times")
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
        plt.close("all")
