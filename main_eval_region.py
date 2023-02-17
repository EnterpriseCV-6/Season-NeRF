# coding=utf-8

from Generate_Summary_Images import gen_angle_images, show_proto_images
import T_NeRF_Eval_Utils as evals
import numpy as np
import pickle
from T_NeRF_Full_2.mg_eval import load_from_input_args
import os

def regional_eval(args, output_loc, quick_mode = True):
    try:
        os.mkdir(output_loc)
    except:
        print("Warning: output directory", output_loc, "already exists!")

    use_classic_solar = args.Solar_Type_2#input_dict["Solar_Loss_Type_2"]


    # fout = open(output_loc + "/Input_Dict.pickle", "wb")
    # pickle.dump(input_dict, fout)
    # fout.close()

    P_imgs, a_t_nerf, image_builder, bounds_LLA, GT_DSM, training_DSM, testing_imgs, device = load_from_input_args(args)
    testing_idx = []
    training_idx = []
    for i in range(len(P_imgs)):
        if P_imgs[i].img_name in testing_imgs:
            testing_idx.append(i)
        else:
            training_idx.append(i)
    seasonal_proto_idx = np.array([])
    walking_times = np.array([])

    print("Direct Evaluation of Testing Images...")
    print("Get Data Overview Images")
    gen_angle_images(P_imgs, testing_idx, np.array([]), np.array([]), output_path=output_loc + "/Data_Sat_and_Sun_pose.png")
    show_proto_images(P_imgs, training_idx, testing_idx, testing_idx, walking_times, output_path=output_loc + "/Prototypical_Imgs.png")
    print("Eval of Height Maps")
    HM_Summary = evals.Full_Eval_HM(image_builder, GT_DSM, training_DSM, bounds_LLA[2])
    fout = open(output_loc + "/HM_Summary.pickle", "wb")
    pickle.dump(HM_Summary, fout)
    fout.close()
    # fin = open(output_loc + "/HM_Summary.pickle", "rb")
    # HM_Summary = pickle.load(fin)
    # fin.close()
    evals.Orgainize_Output_Imgs_HM(HM_Summary, output_loc)


    print("Eval of Image Quality")
    if quick_mode:
        Img_Summary = evals.Full_Eval_Imgs(a_t_nerf, P_imgs, training_idx, testing_idx, device, season_steps=25,
                                           out_img_size=(256, 256, 96), solar_img_size=(64, 64, 96),
                                           include_training_imgs=False, use_classic_shadows=use_classic_solar)
    else:
        Img_Summary = evals.Full_Eval_Imgs(a_t_nerf, P_imgs, training_idx, testing_idx, device, season_steps= 100, out_img_size=(512,512,96), solar_img_size=(256,256,96), include_training_imgs=False, use_classic_shadows=use_classic_solar)

    fout = open(output_loc + "/Img_Summary.pickle", "wb")
    try:
        pickle.dump(Img_Summary, fout)
        fout.close()
    except:
        if not fout.closed:
            fout.close()
        print("Unable to pickle Img Summary in a single file, trying seperate files.")
        for a_key in Img_Summary["Testing"].keys():
            fout = open(output_loc + "/Img_Summary_" + a_key +".pickle", "wb")
            try:
                pickle.dump(Img_Summary["Testing"][a_key], fout)
            except:
                print("Unable to save file Img_Summary[Testing]["+a_key+"]")
            fout.close()
    # fin = open(output_loc + "/Img_Summary.pickle", "rb")
    # Img_Summary = pickle.load(fin)
    # fin.close()
    evals.Orgainize_Output_Imgs_Imgs(Img_Summary, P_imgs, output_loc)

    print("Proof of Solar Claims")
    if quick_mode:
        training_points, testing_points, close_walking_points, all_walking_points, ground_points, testing_views, training_views, close_walking_views, all_walking_views  = evals.Sample_Walk_Points_Shadow(
        P_imgs, testing_idx, points_in_space=64, points_across_angles=20, points_in_view=1)
    else:
        training_points, testing_points, close_walking_points, all_walking_points, ground_points, testing_views, training_views, close_walking_views, all_walking_views = evals.Sample_Walk_Points_Shadow(
            P_imgs, testing_idx, points_in_space=64, points_across_angles=42, points_in_view=1)
    Shadow_Summary_Short = evals.Test_Shadow_Points(a_t_nerf, training_points, testing_points, close_walking_points, all_walking_points, ground_points, P_imgs[0].get_world_center(), P_imgs[0].S, device=device, full_return=False)

    # Shadow_Summary = evals.Test_Shadow_Points(a_t_nerf, training_points, testing_points, close_walking_points,
    #                                           all_walking_points, ground_points, P_imgs[0].get_world_center(),
    #                                           P_imgs[0].S, device=device, full_return=True)
    # try:
    #     fout = open(output_loc + "/Shadow_Summary.pickle", "wb")
    #     pickle.dump(Shadow_Summary, fout)
    #     fout.close()
    # except:
    #     print("Warning: Shadow Summary too big to save!")
    #     fout = open(output_loc + "/Shadow_Summary_Warning.txt", "w")
    #     fout.write("Unable to save shadow summary, too much data (more than 4 GB).\n")
    #     fout.close()


    # fin = open(output_loc + "/Shadow_Scores_Summary.pickle", "rb")
    # Shadow_Summary = pickle.load(fin)
    # fin.close()
    Shadow_Scores = evals.Orgainize_Output_Imgs_Shadows(Shadow_Summary_Short, output_loc, already_short=True)
    fout = open(output_loc + "/Shadow_Scores_Summary.pickle", "wb")
    pickle.dump(Shadow_Scores, fout)
    fout.close()

    print("Proof of Seasonal Claims")
    if quick_mode:
        Season_Summary = evals.Full_Eval_Seasons(a_t_nerf, P_imgs, device=device, out_img_size=(128, 128, 96),
                                                 n_walking_sun_angles=3, n_walking_view_angles=4, n_walking_times=5,
                                                 use_exact_solar=False, use_classic_shadows=use_classic_solar)
    else:
        Season_Summary = evals.Full_Eval_Seasons(a_t_nerf, P_imgs, device=device, out_img_size=(128, 128, 96),
                                              n_walking_sun_angles = 5, n_walking_view_angles = 11, n_walking_times = 12, use_exact_solar=False, use_classic_shadows=use_classic_solar)
    fout = open(output_loc + "/Season_Summary.pickle", "wb")
    pickle.dump(Season_Summary, fout)
    fout.close()
    # fin = open(output_loc +  "/Season_Summary.pickle", "rb")
    # Season_Summary = pickle.load(fin)
    # fin.close()
    evals.Orgainize_Output_Seasons(Season_Summary, P_imgs, training_idx, np.array(testing_idx[0:3]), output_loc)


def multi_region_merge(path_to_data, indiv_data_loc, output_dir_name):
    temp = path_to_data.split("/")
    path_to_data = ""
    for i in range(len(temp)-1):
        path_to_data = path_to_data + temp[i] + "/"
    path_to_data = path_to_data[0:-1]
    print(path_to_data)
    print(indiv_data_loc)
    print(os.listdir(path_to_data))

    try:
        os.mkdir(path_to_data + "/" + output_dir_name)
    except:
        print("Directory " + path_to_data + "/" + output_dir_name + " already exists!")
    output_path = path_to_data + "/" + output_dir_name

    valid_areas = []
    for a_file in os.listdir(path_to_data):
        if os.path.exists(path_to_data + "/" + a_file + "/" + indiv_data_loc):
            valid_areas.append(path_to_data + "/" + a_file + "/" + indiv_data_loc)
    valid_areas = np.sort(valid_areas)
    print("Valid Region Labels:")
    [print(valid_areas[i].split("/")[-2]) for i in range(len(valid_areas))]

    # evals.merge_season_walk(valid_areas, output_path)

    evals.merge_area_overviews(valid_areas, output_path)

    evals.merge_HMs(valid_areas, output_path)
    evals.merge_imgs(valid_areas, output_path)
    evals.merge_imgs_shadows(valid_areas, output_path)
    evals.merge_seasons(valid_areas, output_path)
    print("DONE")

def multi_season_merge(path_to_data, indiv_data_loc, output_dir_name):
    print(path_to_data)
    print(indiv_data_loc)
    print(os.listdir(path_to_data))

    try:
        os.mkdir(path_to_data + "/" + output_dir_name)
    except:
        print("Directory " + path_to_data + "/" + output_dir_name + " already exists!")
    output_path = path_to_data + "/" + output_dir_name

    valid_areas = []
    for a_file in os.listdir(path_to_data):
        if os.path.exists(path_to_data + "/" + a_file + "/" + indiv_data_loc):
            valid_areas.append(path_to_data + "/" + a_file + "/" + indiv_data_loc)
    valid_areas = np.sort(valid_areas)
    print("Valid Region Labels:")
    [print(valid_areas[i].split("/")[-2]) for i in range(len(valid_areas))]

    evals.merge_season_walk_split(valid_areas, output_path)
    exit()
    evals.merge_season_walk(valid_areas, output_path)