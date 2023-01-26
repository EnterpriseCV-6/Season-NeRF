# coding=utf-8
# from T_NeRF_Full_2.Quick_Run import Quick_Run_Net
# from pre_NeRF.P_Img import P_img
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

def walk(walking_points_dictionary, image_builder, out_img_size:int):
    sat_el_and_az = walking_points_dictionary["Sat_el_and_az"]
    sun_el_and_az = walking_points_dictionary["Solar_el_and_az"]
    time_fracs = walking_points_dictionary["Time_Frac"]
    Out_image_store = np.empty([sat_el_and_az.shape[0], sun_el_and_az.shape[0], time_fracs.shape[0]], dtype=dict)
    for i in tqdm(range(Out_image_store.shape[0]), position=0, desc="Camera Angle Loop"):
        camera_el_and_az = sat_el_and_az[i]
        for j in tqdm(range(Out_image_store.shape[1]), position=1, leave=False, desc="Solar Loop"):
            solar_el_and_az = sun_el_and_az[j]
            for k in tqdm(range(Out_image_store.shape[2]), position=2, leave=False, desc="Time Loop"):
                time_frac = time_fracs[k]
                out_imgs_dict, mask = image_builder.render_img(camera_el_and_az, solar_el_and_az, time_frac, out_img_size)
                new_entry = {"Col_Img":out_imgs_dict["Col_Img"], "Shadow_Img":out_imgs_dict["Shadow_Mask"], "Mask":mask, "Sat_el_az":camera_el_and_az, "Solar_el_az":solar_el_and_az, "Time_Frac":time_frac}
                Out_image_store[i,j,k] = new_entry
    return Out_image_store


def get_P_img_info(P_imgs, testing_img_names, n_walking_pts = 12):
    ans_dict = {"Training_Image_Meta_Data":{"Names":[], "Sat_el_and_az":[], "Solar_el_and_az":[], "Time_Frac":[]}, "Testing_Image_Meta_Data":{"Names":[], "Sat_el_and_az":[], "Solar_el_and_az":[], "Time_Frac":[]}}

    for a_P_img in P_imgs:
        name, sat_angle, sun_angle, time = a_P_img.get_meta_data()
        if name in testing_img_names:
            key = "Testing_Image_Meta_Data"
        else:
            key = "Training_Image_Meta_Data"
        ans_dict[key]["Names"].append(name)
        ans_dict[key]["Sat_el_and_az"].append(sat_angle)
        ans_dict[key]["Solar_el_and_az"].append(sun_angle)
        ans_dict[key]["Time_Frac"].append(time)

    walking_sat_angles = get_walking_points_sat(ans_dict, n_walking_pts,2)
    walking_sun_angles, gen_poly = get_walking_points_sun(ans_dict, n_walking_pts)
    walking_times = get_walking_points_time(ans_dict, n_walking_pts)

    ans_dict["Walking_Points"] = {"Sat_el_and_az":walking_sat_angles, "Solar_el_and_az":walking_sun_angles, "Time_Frac":walking_times}

    return ans_dict

def get_walking_points_time(summary_dict, n_points):
    training_time = np.array(summary_dict["Training_Image_Meta_Data"]["Time_Frac"])
    testing_time = np.array(summary_dict["Testing_Image_Meta_Data"]["Time_Frac"])
    walking_times = np.linspace(0,1, n_points, endpoint=False)

    # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # ax.scatter(training_time*2*np.pi, np.ones_like(training_time) + np.random.rand(training_time.shape[0])*.4)
    # ax.scatter(testing_time * 2 * np.pi, np.ones_like(testing_time) + np.random.rand(testing_time.shape[0])*.4)
    # ax.scatter(walking_times*2*np.pi,  np.ones_like(walking_times) + np.random.rand(walking_times.shape[0])*.4, c="red")
    # ax.set_rmax(1.5)
    # ax.set_rticks([])
    # angles = np.linspace(0,360, 12, endpoint=False)
    # months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # ax.set_thetagrids(angles, months)
    #
    # plt.legend(["Training Point", "Testing Point", "Walking Point"])
    # plt.title("Overview of Times")
    # plt.show()

    return walking_times






def get_walking_points_sun(summary_dict, n_points):
    training_sat_angle = np.array(summary_dict["Training_Image_Meta_Data"]["Solar_el_and_az"])
    x, y = np.cos(np.deg2rad(training_sat_angle[:,1]))*training_sat_angle[:,0], np.sin(np.deg2rad(training_sat_angle[:,1])) * training_sat_angle[:,0]
    testing_sat_angle = np.array(summary_dict["Testing_Image_Meta_Data"]["Solar_el_and_az"])
    x2, y2 = np.cos(np.deg2rad(testing_sat_angle[:, 1])) * testing_sat_angle[:, 0], np.sin(np.deg2rad(
        testing_sat_angle[:, 1])) * testing_sat_angle[:, 0]

    a_poly = np.polyfit(training_sat_angle[:,0], training_sat_angle[:,1], deg=3)

    min_el, max_el = 20, 70

    els = np.linspace(min_el, max_el, n_points)
    thetas = np.poly1d(a_poly)(els)
    walking_sun_els_and_azs = np.stack([els, thetas], 1)
    x3, y3 = np.cos(np.deg2rad(thetas)) * (els), np.sin(np.deg2rad(thetas)) * (els)

    # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # ax.scatter(training_sat_angle[:, 1]/180*np.pi, training_sat_angle[:, 0])
    # ax.scatter(testing_sat_angle[:, 1]/180*np.pi, testing_sat_angle[:, 0])
    # ax.scatter(thetas/180*np.pi, els, c="red")
    # ax.legend(["Training Point", "Testing Point", "Walking Point"])
    # plt.title("Overview of Satellite Off-Nadir and Azmuth Angles")
    # plt.show()

    # plt.axhline(c="black")
    # plt.axvline(c="black")
    # X = np.linspace(5, -60 ,50)
    # Y = np.linspace(-5, 60,50)
    # Z = np.sqrt(X.reshape([-1,1])**2 + Y.reshape([1,-1])**2)
    # ax = plt.contour(X, Y, Z)
    # plt.clabel(ax, inline=True, fontsize=10)
    #
    # a = plt.scatter(x, y)
    # b = plt.scatter(x2, y2)
    # c = plt.scatter(x3, y3, c = "red")
    # plt.scatter(0,0, c="purple")
    # plt.legend([a,b,c], ["Training Point", "Testing Point", "Walking Point"])
    # plt.title("Overview of Solar Elevation and Azmuth Angles")
    # plt.xlim(-60,5)
    # plt.ylim(-5,60)
    # plt.show()
    return walking_sun_els_and_azs, a_poly

def get_walking_points_sat(summary_dict, n_points, n_loops):
    training_sat_angle = np.array(summary_dict["Training_Image_Meta_Data"]["Sat_el_and_az"])
    training_sat_angle[:,0] = 90-training_sat_angle[:,0]
    x, y = np.cos(np.deg2rad(training_sat_angle[:,1]))*training_sat_angle[:,0], np.sin(np.deg2rad(training_sat_angle[:,1])) * training_sat_angle[:,0]
    testing_sat_angle = np.array(summary_dict["Testing_Image_Meta_Data"]["Sat_el_and_az"])
    testing_sat_angle[:, 0] = 90 - testing_sat_angle[:, 0]
    x2, y2 = np.cos(np.deg2rad(testing_sat_angle[:, 1])) * testing_sat_angle[:, 0], np.sin(np.deg2rad(
        testing_sat_angle[:, 1])) * testing_sat_angle[:, 0]

    min_el, max_el = 60, 89
    min_theta, max_theta = 0, 360

    els = np.linspace(min_el, max_el, n_points)
    thetas = np.linspace(min_theta, max_theta*n_loops, n_points)
    walking_sat_els_and_azs = np.stack([els, thetas], 1)
    x3, y3 = np.cos(np.deg2rad(thetas)) * (90-els), np.sin(np.deg2rad(thetas)) * (90-els)


    # fig, ax = plt.subplots(subplot_kw={"projection":"polar"})
    # ax.scatter(training_sat_angle[:, 1]/180*np.pi, training_sat_angle[:,0])
    # ax.scatter(testing_sat_angle[:, 1]/180*np.pi, testing_sat_angle[:, 0])
    # ax.scatter(thetas, (90-els)/180*np.pi, c="red")
    # ax.legend(["Training Point", "Testing Point", "Walking Point"])
    # plt.title("Overview of Satellite Off-Nadir and Azmuth Angles")
    # plt.show()


    # plt.axhline(c="black")
    # plt.axvline(c="black")
    # X = np.linspace(-32,32,50)
    # Y = np.linspace(-32,32,50)
    # Z = np.sqrt(X.reshape([-1,1])**2 + Y.reshape([1,-1])**2)
    # ax = plt.contour(X, Y, Z)
    # plt.clabel(ax, inline=True, fontsize=10)
    #
    # a = plt.scatter(x, y)
    # b = plt.scatter(x2, y2)
    # c = plt.scatter(x3, y3, c = "red")
    # plt.scatter(0,0, c="purple")
    # plt.legend([a,b,c], ["Training Point", "Testing Point", "Walking Point"])
    # plt.title("Overview of Satellite Off-Nadir and Azmuth Angles")
    # plt.xlim(-32,32)
    # plt.ylim(-32,32)
    #
    # plt.show()
    return walking_sat_els_and_azs


def compare_P_imgs(P_imgs, eval_tool, out_img_size):
    n = len(P_imgs)
    sat_angle = []
    sun_angle = []
    time = []
    names = []
    for a_P_img in P_imgs:
        a_name, a_sat_angle, a_sun_angle, a_time = a_P_img.get_meta_data()
        sat_angle.append(a_sat_angle)
        sun_angle.append(a_sun_angle)
        time.append(a_time)
        names.append(a_name)

    ans_summary = {"Meta_Data":{"Names":names, "Sat_el_and_az":sat_angle, "Solar_el_and_az":sun_angle, "Time_Frac":time}}
    ans_summary["Images"] = []
    ans_summary["Shadow_Images"] = []
    ans_summary["Masks"] = []
    ans_summary["Key"] = []

    idx = np.arange(0,n)
    combinations = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1).reshape([-1,3])

    for i in tqdm(range(combinations.shape[0])):
        a,b,c = combinations[i]
        img_dict, mask = eval_tool.render_img(sat_angle[a], sun_angle[b], time[c], out_img_size=out_img_size)
        ans_summary["Images"].append(img_dict["Col_Img"])
        ans_summary["Shadow_Images"] = [img_dict["Shadow_Mask"]]
        ans_summary["Masks"] = [mask]
        ans_summary["Key"] = [combinations[i]]
    return ans_summary
