# coding=utf-8

import torch as t
from opt import get_opts
import pickle
from mg_get_DSM import get_DSM
from tqdm import tqdm
import numpy as np
from T_NeRF_Eval_Utils import load_t_nerf, eval_HM, full_eval_P_img, eval_solar_walk, eval_season_walk
from T_NeRF_Full_2 import Quick_Run
from all_NeRF import mg_detailed_eval as eval_lib
from all_NeRF.basic_functions import show_dict_struc
import Generate_Summary_Images
import os

def eval_T_NeRF(args):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    # args = get_opts(exp_Name=input_dict["Exp_Name"],
    #                 region=input_dict["Region"],
    #                 num_time_class=input_dict["Time_Classes"],
    #                 use_type_2_solar=input_dict["Solar_Loss_Type_2"],
    #                 use_reg=input_dict["Reg_Terms"],
    #                 use_prior=not input_dict["Skip_Prior_Start"],
    #                 use_solar=not input_dict["Ignore_Solar"],
    #                 use_time=not input_dict["Ignore_Time"],
    #                 log_loc=input_dict["Log_File"],
    #                 use_MSE_loss=input_dict["MSE_Loss"],
    #                 force_write_json=False)


    full_ans = {}
    a_t_nerf, testing_imgs = load_t_nerf(args)
    a_t_nerf.eval()
    a_t_nerf = a_t_nerf.to(device)

    if args.skip_Bundle_Adjust == False:
        refined = "_Refined"
    else:
        refined = ""

    # fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    # P_imgs = pickle.load(fin)
    # fin.close()
    # world_center = P_imgs[0].get_world_center()
    # H = P_imgs[0].S
    # img_render_tool = Quick_Run.Quick_Run_Net(a_t_nerf, args, world_center, H, device)
    # from matplotlib import pyplot as plt
    # for i in np.linspace(0, 360, 3, endpoint=False):
    #     plt.figure()
    #     a_out_img = img_render_tool.render_img([60, i], [80, 47], 0.25, 256)
    #     plt.imshow(a_out_img)
    # plt.show()
    # exit()

    # if input_dict["Ignore_Solar"]:
    #     a_t_nerf.ignore_solar()

    training_DSM, GT_DSM = get_DSM(args, device=device)

    if args.skip_Bundle_Adjust == False:
        refined = "_Refined"
    else:
        refined = ""
    bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + refined + ".npy")
    print("Eval HM.")
    Imgs, scores_before, scores_after = eval_HM(a_t_nerf, GT_DSM, (bounds_LLA[2,0], bounds_LLA[2,1]), (args.n_samples + args.n_importance), device, max_batch_size=5000)

    full_ans = {"Height_Info":{"Height_Map_Imgs":Imgs, "Height_Map_Scores_Before_Alignment":scores_before, "Height_Map_Scores_After_Alignment":scores_after}, "Image_Info":{}}

    fout = open(args.logs_dir + "/Analysis.pickle", "wb")
    pickle.dump(full_ans, fout)
    fout.close()

    if args.skip_Bundle_Adjust == False:
        refined = "_Refined"
    else:
        refined = ""

    fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()

    # summarize_P_imgs(P_imgs)
    # exit()

    # eval_ortho_img(a_t_nerf, P_imgs, device, [128, 128, args.n_samples], args.use_HSLuv)
    # exit()

    sun_vec_list = np.array([a_P_img.sun_el_and_az_vec for a_P_img in P_imgs])
    angle_order = np.argsort(np.arctan2(sun_vec_list[:,2], np.sqrt(sun_vec_list[:,0]**2+sun_vec_list[:,1]**2)))
    sun_vec_list = sun_vec_list[angle_order]

    full_sun_list = np.zeros([sun_vec_list.shape[0]*2-1, 3])
    for i in range(len(sun_vec_list)-1):
        full_sun_list[2*i] = sun_vec_list[i]
        full_sun_list[2*i+1] = (sun_vec_list[i] + sun_vec_list[i+1])/2
        full_sun_list[2*i+1] = full_sun_list[2*i+1] / np.sqrt(np.sum(full_sun_list[2*i+1]**2))
    full_sun_list[-1] = sun_vec_list[-1]

    encoded_times = list(np.sort([a_P_img.time_obj.get_time_frac()[1] for a_P_img in P_imgs]))
    for i in np.linspace(0,1,52):
        encoded_times.append(i)
    encoded_times = np.array(encoded_times)
    encoded_times = np.stack([np.cos(2*np.pi*encoded_times), np.sin(2*np.pi*encoded_times)], 1)


    # indx = np.linspace(0,len(P_imgs)-1, 3, endpoint=True, dtype=int)
    indx = []
    valid_names = np.array([a_P_img.img_name for a_P_img in P_imgs])
    all_good = True
    for a_testing_name in testing_imgs:
        matches = np.where(a_testing_name == valid_names)[0]
        if matches.shape[0] == 0:
            all_good = False
            print("Unable to find match for testing name", a_testing_name)
        elif matches.shape[0] > 1:
            all_good = False
            print("Multiple matches for testing name", a_testing_name)
        else:
            indx.append(matches[0])
    indx = np.array(indx)
    if all_good == False:
        exit()

    c = 1
    for i in tqdm(indx):
        a_P_img = P_imgs[i]

        img_dict, scores, sat_form_data = full_eval_P_img(a_t_nerf, a_P_img, args.n_samples + args.n_importance, device, step_size = 4, max_batch_size = 100000, bounds=bounds_LLA, use_classic_solar=args.Solar_Type_2)
        if c == 1:
            Time_Results = eval_season_walk(a_t_nerf, a_P_img, encoded_times, device, [128, 128, args.n_samples], args.use_HSLuv, args.Solar_Type_2)
            Solar_Results = eval_solar_walk(a_t_nerf, a_P_img, full_sun_list, device, [128, 128, args.n_samples], args.use_HSLuv, args.Solar_Type_2)
            a_ans = {"Is_Testing_Img": (a_P_img.img_name in testing_imgs), "Imgs": img_dict, "Scores": scores, "Solar_Results":Solar_Results, "Time_Results":Time_Results}
        else:
            a_ans = {"Is_Testing_Img":(a_P_img.img_name in testing_imgs), "Imgs":img_dict, "Scores":scores}
        full_ans["Image_Info"][a_P_img.img_name] = a_ans

        fout = open(args.logs_dir + "/Analysis.pickle", "wb")
        pickle.dump(full_ans, fout)
        fout.close()

        fout = open(args.logs_dir + "/G_NeRF_Sat_Form_" + str(c) + ".pickle", "wb")
        pickle.dump(sat_form_data, fout)
        fout.close()
        c += 1


def load_from_input_args(args):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    a_t_nerf, testing_imgs = load_t_nerf(args)
    a_t_nerf.eval()
    a_t_nerf = a_t_nerf.to(device)

    if args.skip_Bundle_Adjust == False:
        refined = "_Refined"
    else:
        refined = ""

    fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()

    training_DSM, GT_DSM = get_DSM(args, device=device)

    if args.skip_Bundle_Adjust == False:
        refined = "_Refined"
    else:
        refined = ""
    bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + refined + ".npy")

    image_builder = Quick_Run.Quick_Run_Net(a_t_nerf, args, P_imgs[0].get_world_center(), P_imgs[0].S, device,
                                            use_tqdm=False, use_full_solar=False)
    return P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device


def load_from_input_dict(input_dict):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    args = get_opts(exp_Name=input_dict["Exp_Name"],
                    region=input_dict["Region"],
                    num_time_class=input_dict["Time_Classes"],
                    use_type_2_solar=input_dict["Solar_Loss_Type_2"],
                    use_reg=input_dict["Reg_Terms"],
                    use_prior=not input_dict["Skip_Prior_Start"],
                    use_solar=not input_dict["Ignore_Solar"],
                    use_time=not input_dict["Ignore_Time"],
                    log_loc=input_dict["Log_File"],
                    use_MSE_loss=input_dict["MSE_Loss"],
                    force_write_json=False)

    # full_ans = {"Height_Info": {"Height_Map_Imgs": Imgs, "Height_Map_Scores_Before_Alignment": scores_before,
    #                             "Height_Map_Scores_After_Alignment": scores_after}, "Image_Info": {}}
    a_t_nerf, testing_imgs = load_t_nerf(args)
    a_t_nerf.eval()
    a_t_nerf = a_t_nerf.to(device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""

    fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()

    training_DSM, GT_DSM = get_DSM(args, device=device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""
    bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + refined + ".npy")

    image_builder = Quick_Run.Quick_Run_Net(a_t_nerf, args, P_imgs[0].get_world_center(), P_imgs[0].S, device,
                                            use_tqdm=False, use_full_solar=False)
    return P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device

def eval_T_NeRF_2(input_dict):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    args = get_opts(exp_Name=input_dict["Exp_Name"],
                    region=input_dict["Region"],
                    num_time_class=input_dict["Time_Classes"],
                    use_type_2_solar=input_dict["Solar_Loss_Type_2"],
                    use_reg=input_dict["Reg_Terms"],
                    use_prior=not input_dict["Skip_Prior_Start"],
                    use_solar=not input_dict["Ignore_Solar"],
                    use_time=not input_dict["Ignore_Time"],
                    log_loc=input_dict["Log_File"],
                    use_MSE_loss=input_dict["MSE_Loss"],
                    force_write_json=False)


    full_ans = {"Image_Info": {}}
    # full_ans = {"Height_Info": {"Height_Map_Imgs": Imgs, "Height_Map_Scores_Before_Alignment": scores_before,
    #                             "Height_Map_Scores_After_Alignment": scores_after}, "Image_Info": {}}
    a_t_nerf, testing_imgs = load_t_nerf(args)
    a_t_nerf.eval()
    a_t_nerf = a_t_nerf.to(device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""

    fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()

    training_DSM, GT_DSM = get_DSM(args, device=device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""
    bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + refined + ".npy")

    image_builder = Quick_Run.Quick_Run_Net(a_t_nerf, args, P_imgs[0].get_world_center(), P_imgs[0].S, device, use_tqdm=False, use_full_solar=False)

    # print("Eval HM.")
    # Imgs, scores_before, scores_after = eval_HM(a_t_nerf, GT_DSM, (bounds_LLA[2, 0], bounds_LLA[2, 1]),
    #                                             (args.n_samples + args.n_importance), device, max_batch_size=5000)
    #
    # full_ans = {"Height_Info": {"Height_Map_Imgs": Imgs, "Height_Map_Scores_Before_Alignment": scores_before,
    #                             "Height_Map_Scores_After_Alignment": scores_after}, "Image_Info": {}}

    print("Eval Imgs")
    for a_idx in tqdm(range(len(P_imgs))):
        a_P_img = P_imgs[a_idx]
        img_dict, scores, sat_form_data = full_eval_P_img(a_t_nerf, a_P_img, args.n_samples + args.n_importance, device,
                                                          step_size=4, max_batch_size=100000, bounds=bounds_LLA)
        meta_info = {"Is_Testing_Img": (a_P_img.img_name in testing_imgs), "Name":a_P_img.img_name, "Cam_el_and_az":(90-a_P_img.off_Nadir_from_IMD, a_P_img.Azmuth_from_IMD), "Solar_el_and_az":a_P_img.sun_el_and_az, "Time_Frac":a_P_img.get_year_frac()}
        a_ans = {"Img_Meta_Data":meta_info, "Imgs": img_dict, "Scores": scores}
        full_ans["Image_Info"][a_P_img.img_name] = a_ans
    print("P Img Summary")
    sum = eval_lib.get_P_img_info(P_imgs, testing_imgs, n_walking_pts=12)
    full_ans["Meta_Data_Summary"] = sum
    print("P Img Walk")
    Walk_Results = eval_lib.walk(sum["Walking_Points"], image_builder, 128)
    full_ans["Walk_Results"] = Walk_Results

    # from matplotlib import pyplot as plt
    # for k in range(2):
    #     c = 1
    #     for i in range(2):
    #         for j in range(2):
    #             plt.subplot(2,2, c)
    #             plt.imshow(Walk_Results[k,i,j]["Col_Img"])
    #             c += 1
    #     plt.show()
    show_dict_struc(full_ans, max_r_level=2)
    fout = open(args.logs_dir + "/Full_Analysis.pickle", "wb")
    pickle.dump(full_ans, fout)
    fout.close()

def eval_T_NeRF_3(input_dict, save_imgs):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    args = get_opts(exp_Name=input_dict["Exp_Name"],
                    region=input_dict["Region"],
                    num_time_class=input_dict["Time_Classes"],
                    use_type_2_solar=input_dict["Solar_Loss_Type_2"],
                    use_reg=input_dict["Reg_Terms"],
                    use_prior=not input_dict["Skip_Prior_Start"],
                    use_solar=not input_dict["Ignore_Solar"],
                    use_time=not input_dict["Ignore_Time"],
                    log_loc=input_dict["Log_File"],
                    use_MSE_loss=input_dict["MSE_Loss"],
                    force_write_json=False)

    if save_imgs:
        path = input_dict["Log_File"] + "/" + input_dict["Exp_Name"] + "/Summary_Imgs"
        try:
            os.mkdir(path)
        except:
            print("Output folder already exists")
    else:
        path = None

    full_ans = {"Image_Info": {}}
    # full_ans = {"Height_Info": {"Height_Map_Imgs": Imgs, "Height_Map_Scores_Before_Alignment": scores_before,
    #                             "Height_Map_Scores_After_Alignment": scores_after}, "Image_Info": {}}
    a_t_nerf, testing_imgs = load_t_nerf(args)
    a_t_nerf.eval()
    a_t_nerf = a_t_nerf.to(device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""

    fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()

    training_DSM, GT_DSM = get_DSM(args, device=device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""
    bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + refined + ".npy")

    image_builder = Quick_Run.Quick_Run_Net(a_t_nerf, args, P_imgs[0].get_world_center(), P_imgs[0].S, device,
                                            use_tqdm=False, use_full_solar=False)

    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    import datetime


    # Generate_Summary_Images.show_all_images(P_imgs)#, [500, 1000, 1000, 1500])
    n_walking_points = 8
    n_walking_view = 4
    n_walking_sun = 4
    min_day_sep = 20
    view_proto_idx = []
    solar_proto_idx = []
    # #OMA_248
    #
    # #OMA_281
    # seasonal_proto_idx = [0, 11, 18, 37]
    # testing_idx = seasonal_proto_idx + [29, 39]
    print(len(P_imgs))
    # exit()
    training_idx = []
    testing_idx = []
    for i in range(len(P_imgs)):
        if P_imgs[i].img_name not in testing_imgs:
            training_idx.append(i)
        else:
            print(P_imgs[i].img_name)
            testing_idx.append(i)
    seasonal_proto_idx = testing_idx

    walk_view, walk_sun, walking_times = Generate_Summary_Images.get_walking_Points(P_imgs, n_walking_view, n_walking_sun, n_walking_points, min_day_sep)
    walk_view = np.array(walk_view)
    walk_sun = np.array(walk_sun)
    walking_times = np.array(walking_times)

    sub_walking_times = walking_times[np.linspace(0, walking_times.shape[0], num=8, endpoint=False, dtype=int)]
    sub_walk_view = walk_view[np.linspace(0, walk_view.shape[0], num=4, endpoint=False, dtype=int)]
    sub_walk_sun = walk_sun[np.linspace(0, walk_sun.shape[0], num=4, endpoint=False, dtype=int)]

    Generate_Summary_Images.gen_angle_images(P_imgs, testing_idx, walk_view, walk_sun, annotate_pts=True, output_path = path + "/Angle_data.png")
    Generate_Summary_Images.show_proto_images(P_imgs, training_idx, testing_idx, seasonal_proto_idx, walking_times, output_path = path + "/proto_data.png")


    Generate_Summary_Images.gen_sum(sub_walk_view, sub_walk_sun, sub_walking_times, image_builder, out_img_size=128, output_path = path+ "/sum_data.png")

    # exit()
    # dist_thresh = min_day_sep / 365.24
    #
    # training_idx = []
    # testing_idx = []
    # for i in range(len(P_imgs)):
    #     if (i in view_proto_idx) or (i in solar_proto_idx) or (i in seasonal_proto_idx):
    #         testing_idx.append(i)
    #     else:
    #         training_idx.append(i)
    #
    # training_time = np.array([P_imgs[i].get_year_frac() for i in training_idx])
    # walking_times = np.linspace(0, 1, n_walking_points, endpoint=False)
    # if min_day_sep > 0:
    #     n = 1
    #     dist = np.abs(walking_times.reshape([-1, 1]) - training_time.reshape([1, -1]))
    #     dist[dist > .5] = 1. - dist[dist > .5]
    #     dist = np.min(dist, 1)
    #     good = dist <= dist_thresh
    #     while np.sum(good) < n_walking_points:
    #         walking_times = np.linspace(0, 1, n_walking_points + n, endpoint=False)
    #         dist = np.abs(walking_times.reshape([-1, 1]) - training_time.reshape([1, -1]))
    #         dist[dist > .5] = 1. - dist[dist > .5]
    #         dist = np.min(dist, 1)
    #         good = dist <= dist_thresh
    #         n += 1
    #     walking_times = walking_times[good]
    #
    # Generate_Summary_Images.show_proto_images(P_imgs, training_idx, testing_idx, seasonal_proto_idx, walking_times)
    #
    # exit()
    #
    # months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # walking_shadows = np.array([[30, 0], [50, 10], [70, 20], [90, 30]])
    # walking_views = np.array([[90, 0], [80, 90], [70, 180], [60, 270]])
    #
    # # fig = plt.figure()
    # # months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # # for i in range(len(seasonal_proto_idx)):
    # #     a_ax = fig.add_subplot(1,len(seasonal_proto_idx)+1,i+2)
    # #     a_ax.imshow(P_imgs[seasonal_proto_idx[i]].img)
    # #     a_ax.set_xticks([])
    # #     a_ax.set_yticks([])
    # #     a_ax.set_title(months[P_imgs[seasonal_proto_idx[i]].time_obj.month-1] + ". " + str(P_imgs[seasonal_proto_idx[i]].time_obj.day))
    # #
    # #
    # # testing_idx_small = []
    # # for i in testing_idx:
    # #     if i not in seasonal_proto_idx:
    # #         testing_idx_small.append(i)
    # #
    # # ax = fig.add_subplot(1, len(seasonal_proto_idx)+1, 1, polar=True)
    # # testing_time = np.array([P_imgs[i].get_year_frac() for i in testing_idx_small])
    # # proto_time = np.array([P_imgs[i].get_year_frac() for i in seasonal_proto_idx])
    # # ax.scatter(training_time * 2 * np.pi, np.ones_like(training_time) + np.random.rand(training_time.shape[0]) * .4)
    # # ax.scatter(testing_time * 2 * np.pi, np.ones_like(testing_time) + np.random.rand(testing_time.shape[0]) * .4)
    # # ax.scatter(proto_time * 2 * np.pi, np.ones_like(proto_time) + np.random.rand(proto_time.shape[0]) * .4, c="green")
    # # ax.scatter(walking_times * 2 * np.pi, np.ones_like(walking_times) + .2,
    # #            c="red")
    # # ax.set_rmax(1.5)
    # # ax.set_rticks([])
    # # angles = np.linspace(0, 360, 12, endpoint=False)
    # # ax.set_thetagrids(angles, months)
    # #
    # # ax.legend(["Training Point", "Testing Point", "Prototypical Tesing Point", "Walking Point"], loc=10)
    # # ax.set_title("Overview of Data Times")
    # #
    # # plt.show()
    #
    # ncol = walking_times.shape[0] + 1
    # nrow = walking_shadows.shape[0]
    # # fig, ax = plt.subplots(ncols=walking_times.shape[0], nrows=nrow, figsize=(ncol+1,nrow+1))
    # fig = plt.figure(figsize=((ncol + 1), (nrow + 1)))
    # gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
    #                        top=1.-0.5/(nrow+1),
    #                        bottom=0.5/(nrow+1),
    #                        left=0.5/(ncol+1),
    #                        right=1-0.5/(ncol+1))
    #
    # date_1 = datetime.datetime.strptime("01/01/21", "%m/%d/%y")
    # all_res = []
    # r = 0
    # for a_walking_shadow in tqdm(walking_shadows, leave=True):
    #     c = 0
    #     for a_waking_time in tqdm(walking_times, leave=False):
    #         res, mask = image_builder.render_img(walking_views[r], a_walking_shadow, a_waking_time, out_img_size=128)
    #         all_res.append([res, mask])
    #         ax = plt.subplot(gs[r,c])
    #         ax.imshow(res["Col_Img"])
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         if r == 0:
    #             end_date = date_1 + datetime.timedelta(days=365.24 * a_waking_time)
    #             ax.set_title(months[end_date.month - 1] + ". " + str(end_date.day))
    #         if c == 0:
    #             the_lab = "(" + str(np.round(90-walking_views[r][0])) + ", " + str(np.round(walking_views[r][1])) + ")\n(" + str(np.round(a_walking_shadow[0])) + ", " + str(np.round(a_walking_shadow[1])) + ")"
    #
    #             ax.set_ylabel(the_lab, rotation="horizontal", labelpad=23)#, rotation=0, labelpad=23)
    #         c += 1
    #     ax = plt.subplot(gs[r,c])
    #     out_img = (1 / (1 + np.exp(-30 * (res["Shadow_Mask"] - .2))))
    #     out_img[~mask] = np.NaN
    #     ax.imshow(out_img, vmin = 0, vmax = 1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     if r == 0:
    #         ax.set_title("Shadow Mask")
    #     r += 1
    # plt.show()
    # print()

def eval_T_NeRF_4(input_dict, show_img = False):
    import all_NeRF.mg_EM_Imgs as mg_EM
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    args = get_opts(exp_Name=input_dict["Exp_Name"],
                    region=input_dict["Region"],
                    num_time_class=input_dict["Time_Classes"],
                    use_type_2_solar=input_dict["Solar_Loss_Type_2"],
                    use_reg=input_dict["Reg_Terms"],
                    use_prior=not input_dict["Skip_Prior_Start"],
                    use_solar=not input_dict["Ignore_Solar"],
                    use_time=not input_dict["Ignore_Time"],
                    log_loc=input_dict["Log_File"],
                    use_MSE_loss=input_dict["MSE_Loss"],
                    force_write_json=False)

    full_ans = {"Image_Info": {}}
    # full_ans = {"Height_Info": {"Height_Map_Imgs": Imgs, "Height_Map_Scores_Before_Alignment": scores_before,
    #                             "Height_Map_Scores_After_Alignment": scores_after}, "Image_Info": {}}
    a_t_nerf, testing_imgs = load_t_nerf(args)
    a_t_nerf.eval()
    a_t_nerf = a_t_nerf.to(device)

    if show_img == False:
        path = input_dict["Log_File"] + "/" + input_dict["Exp_Name"] + "/Summary_Imgs"
        try:
            os.mkdir(path)
        except:
            print("Output folder already exists")
    else:
        path = None

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""

    fin = open(args.cache_dir + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
    P_imgs = pickle.load(fin)
    fin.close()

    training_DSM, GT_DSM = get_DSM(args, device=device)

    if args.use_Bundle_Adjust:
        refined = "_Refined"
    else:
        refined = ""
    bounds_LLA = np.load(args.cache_dir + "/bounds_LLA" + refined + ".npy")

    from matplotlib import pyplot as plt
    from matplotlib import gridspec

    training_idx = []
    testing_idx = []
    for i in range(len(P_imgs)):
        if P_imgs[i].img_name not in testing_imgs:
            training_idx.append(i)
        else:
            print(P_imgs[i].img_name)
            testing_idx.append(i)
    seasonal_proto_idx = testing_idx

    from T_NeRF_Eval_Utils import get_img_from_P_img
    from T_NeRF_Eval_Utils import mask_ssim, mask_PSNR

    ncol = len(seasonal_proto_idx)
    nrow = 2
    fig = plt.figure(figsize=((ncol + 1), (nrow + 1)))
    gs = gridspec.GridSpec(nrows=nrow, ncols=ncol, wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1),
                           bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1),
                           right=1 - 0.5 / (ncol + 1))
    times = np.linspace(0, 1, 24)
    for i in range(len(seasonal_proto_idx)):
        best_time = -1
        best_score = -1
        best_img = []
        for a_time in times:
            e_time = [np.cos(2*np.pi*a_time), np.sin(2*np.pi*a_time)]
            a_img, mask = get_img_from_P_img(a_t_nerf, P_imgs[seasonal_proto_idx[i]], (128,128,96), device=device, time=e_time)
            score, valid_score = mask_ssim(a_img, P_imgs[seasonal_proto_idx[i]].img[0::16, 0::16], mask)
            score = np.mean(score[valid_score])
            if score > best_score:
                best_score = score
                best_time = a_time
                best_img = a_img.copy()
            print(a_time, score)
        ax = plt.subplot(gs[0, i])
        ax.imshow(P_imgs[seasonal_proto_idx[i]].img[0::16, 0::16] * np.expand_dims(P_imgs[seasonal_proto_idx[i]].get_mask()[0::16, 0::16], -1))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Ground Truth")
        ax = plt.subplot(gs[1, i])
        ax.imshow(best_img * np.expand_dims(mask, -1))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Rendered Image")
    if show_img:
        plt.show()
    else:
        plt.savefig(path + "/Season_Fit.png")
        plt.close("all")