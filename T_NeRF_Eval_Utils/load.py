import torch as t
# from T_NeRF_Full.T_NeRF_net_v2 import T_NeRF
from T_NeRF_Full_2.T_NeRF_net_v2 import T_NeRF as T_NeRF2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from all_NeRF.basic_functions import show_dict_struc

def giffify(set_of_imgs, output_path):
    frames = []
    for a_image in set_of_imgs:
        if str(a_image.dtype) != "uint8":
            a_image = np.array(255 * a_image, dtype=np.uint8)
        Pil_img = Image.fromarray(a_image).convert("RGB")
        frames.append(Pil_img)
    frame_one = frames[0]
    frame_one.save(output_path, format="GIF", append_images = frames, save_all=True, duration = 100, loop = 0)

def load_t_nerf(args, model_name = "Final_Model.nn", get_testing_imgs = True):
    thenet = T_NeRF2(args.fc_units, args.number_low_frequency_cases)
    thenet.load_state_dict(t.load(args.logs_dir + "/" + model_name))
    testing_imgs = []
    if get_testing_imgs:
        fin = open(args.logs_dir + "/Testing_Imgs.txt", "r")
        testing_imgs = fin.readlines()
        fin.close()
        testing_imgs = [testing_imgs[i][0:-1] for i in range(len(testing_imgs))]

    return thenet, testing_imgs

def load_results(args):
    fin = open(args.logs_dir + "/Analysis.pickle", "rb")
    # fin = open(args.logs_dir + "/Analysis.pickle", "rb")
    Results:dict = pickle.load(fin)
    fin.close()

    Results_imgs = Results["Height_Info"]
    vmin, vmax = np.min(Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"]), np.max(
        Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"])
    vmin = min(vmin, np.nanmin(Results_imgs["Height_Map_Imgs"]["GT"]))
    vmax = min(vmax, np.nanmax(Results_imgs["Height_Map_Imgs"]["GT"]))

    plt.subplot(2,3,1)
    plt.title("Ground Truth Lidar")
    plt.imshow(Results_imgs["Height_Map_Imgs"]["GT"], vmin=vmin, vmax=vmax)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.subplot(2,3,2)
    plt.title("Est HM, before Alignment")
    plt.xticks([],[])
    plt.yticks([],[])
    plt.imshow(Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.title("Error")
    plt.imshow(Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"] - Results_imgs["Height_Map_Imgs"]["GT"])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    R1 = Results_imgs["Height_Map_Scores_Before_Alignment"]
    ans = "MAE: " + str(np.round(R1["MAE"], 3)) + "\nRMSE: " + str(np.round(R1["RMSE"], 3)) + "\n% in 1 m: " + str(
        np.round(R1["Acc_1_m"], 3)) \
          + "\nMedian: " + str(np.round(R1["Median"], 3))
    # plt.text(5,5, ans, bbox={'facecolor':'white', 'pad': 10})

    plt.xlabel(ans)
    plt.subplot(2,3,4)
    plt.imshow(Results_imgs["Height_Map_Imgs"]["GT"], vmin=vmin, vmax=vmax)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.subplot(2,3,5)
    plt.title("Est HM, after Alignment")
    plt.colorbar()
    plt.imshow(Results_imgs["Height_Map_Imgs"]["Est_HM_after_Shift"], vmin=vmin, vmax=vmax)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.subplot(2,3,6)
    plt.imshow(Results_imgs["Height_Map_Imgs"]["GT"] - Results_imgs["Height_Map_Imgs"]["Est_HM_after_Shift"])
    plt.xticks([], [])
    plt.yticks([], [])
    R1 = Results_imgs["Height_Map_Scores_After_Alignment"]
    ans = "MAE: " + str(np.round(R1["MAE"], 3)) +  "\nRMSE: " + str(np.round(R1["RMSE"], 3)) + "\n% in 1 m: " +  str(np.round(R1["Acc_1_m"], 3))\
          + "\nMedian: " + str(np.round(R1["Median"], 3))
    # plt.text(5,5, ans, bbox={'facecolor':'white', 'pad': 10})

    plt.xlabel(ans)
    plt.colorbar()

    R1 = Results_imgs["Height_Map_Scores_Before_Alignment"]
    print("Before Alignment")
    print(R1["MAE"], R1["RMSE"], R1["Acc_1_m"], R1["Median"])
    R1 = Results_imgs["Height_Map_Scores_After_Alignment"]
    print("After Alignment")
    print(R1["MAE"], R1["RMSE"], R1["Acc_1_m"], R1["Median"])
    print("Shift")
    print(R1["Shift_x_y_deg"])

    plt.show()

    show_dict_struc(Results)
    running_PSNR = 0
    running_SSIM = 0

    running_PSNR_train = 0
    running_SSIM_train = 0
    n = 0
    n_train = 0
    for i in Results["Image_Info"].keys():
        if Results["Image_Info"][i]["Is_Testing_Img"]:
            n += 1
            running_PSNR += Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_PSNR"]
            running_SSIM += Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_SSIM"]
        else:
            n_train += 1
            running_PSNR_train += Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_PSNR"]
            running_SSIM_train += Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_SSIM"]
    if n != 0:
        print("Avg Testing PSNR:", running_PSNR / n)
        print("Avg Testing SSIM:", running_SSIM / n)
    if n_train != 0:
        print("Avg Training PSNR:", running_PSNR_train / n_train)
        print("Avg Training SSIM:", running_SSIM_train / n_train)

    # names = list(Results["Image_Info"].keys())
    # # print(k)
    # ell = [0,7,12,30]
    # # ell = np.linspace(0, len(names), num=4, endpoint=False, dtype=int)
    # c = 1
    #
    # for j in range(4):
    #     i = names[ell[j]]
    #     print("Img Name:", i)
    #     plt.subplot(3, 4, c)
    #     plt.imshow(Results["Image_Info"][i]["Imgs"]["GT_Img"])
    #     if j != 0:
    #         plt.axis("off")
    #     else:
    #         plt.ylabel("Ground Truth Image")
    #         plt.yticks([],[])
    #         plt.xticks([],[])
    #     c += 1
    # for j in range(4):
    #     i = names[ell[j]]
    #     print("Img Name:", i)
    #     plt.subplot(3, 4, c)
    #     plt.imshow(Results["Image_Info"][i]["Imgs"]["Base_Img"])
    #     if j != 0:
    #         plt.axis("off")
    #     else:
    #         plt.ylabel("Image Before Temporal Adjustment")
    #         plt.yticks([], [])
    #         plt.xticks([], [])
    #     c += 1
    # for j in range(4):
    #     i = names[ell[j]]
    #     print("Img Name:", i)
    #     plt.subplot(3, 4, c)
    #     plt.imshow(Results["Image_Info"][i]["Imgs"]["Ideal_Time_Img"])
    #     if j != 0:
    #         plt.axis("off")
    #     else:
    #         plt.ylabel("Image After Temporal Adjustment")
    #         plt.yticks([], [])
    #         plt.xticks([], [])
    #     c += 1
    #
    # plt.show()

    c2 = 0
    for i in Results["Image_Info"].keys():
        print("Img Name:", i, c2)
        print("Is Testing:", Results["Image_Info"][i]["Is_Testing_Img"])
        print("Sky Color:", Results["Image_Info"][i]["Scores"]["Sky_Col"])
        print("Ideal Time PSNR, SSIM:", Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_PSNR"], Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_SSIM"])
        print("Class Output:", Results["Image_Info"][i]["Scores"]["Ideal_Class_Output"])
        c = 1
        for j in Results["Image_Info"][i]["Imgs"].keys():
            if j != "Valid_Pt_Mask":
                plt.subplot(2,4,c)
                plt.title(j)
                plt.imshow(Results["Image_Info"][i]["Imgs"][j])
                plt.yticks([],[])
                plt.xticks([],[])
                if j != "GT_Img" and j != "HM":
                    PSNR = Results["Image_Info"][i]["Scores"][j + "_PSNR"]
                    SSIM = Results["Image_Info"][i]["Scores"][j + "_SSIM"]
                    plt.xlabel("PSNR: " + str(np.round(PSNR, 3)) + " --- SSIM: " + str(np.round(SSIM, 3)))
                c += 1

        plt.show()
        c2 += 1





    # for a_key in Results.keys():
    #     print(a_key)
    #     for a_key_2 in Results[a_key].keys():
    #         print("   " + a_key_2)
    #         for a_key_3 in Results[a_key][a_key_2].keys():
    #             print("      " + a_key_3)


def get_HM_img(Results, path, show_instead_of_save = False):
    Results_imgs = Results["Height_Info"]
    vmin, vmax = np.min(Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"]), np.max(
        Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"])
    vmin = min(vmin, np.nanmin(Results_imgs["Height_Map_Imgs"]["GT"]))
    vmax = min(vmax, np.nanmax(Results_imgs["Height_Map_Imgs"]["GT"]))

    plt.figure(figsize=(16, 12), dpi=80)

    plt.subplot(2, 3, 1)
    plt.title("Ground Truth Lidar")
    plt.imshow(Results_imgs["Height_Map_Imgs"]["GT"], vmin=vmin, vmax=vmax)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.subplot(2, 3, 2)
    plt.title("Est HM, before Alignment")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title("Error")
    plt.imshow(Results_imgs["Height_Map_Imgs"]["Est_HM_no_Shift"] - Results_imgs["Height_Map_Imgs"]["GT"])
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    R1 = Results_imgs["Height_Map_Scores_Before_Alignment"]
    ans = "MAE: " + str(np.round(R1["MAE"], 3)) + "\nRMSE: " + str(np.round(R1["RMSE"], 3)) + "\n% in 1 m: " + str(
        np.round(R1["Acc_1_m"], 3)) \
          + "\nMedian: " + str(np.round(R1["Median"], 3))
    # plt.text(5,5, ans, bbox={'facecolor':'white', 'pad': 10})

    plt.xlabel(ans)
    plt.subplot(2, 3, 4)
    plt.imshow(Results_imgs["Height_Map_Imgs"]["GT"], vmin=vmin, vmax=vmax)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.subplot(2, 3, 5)
    plt.title("Est HM, after Alignment")
    plt.colorbar()
    plt.imshow(Results_imgs["Height_Map_Imgs"]["Est_HM_after_Shift"], vmin=vmin, vmax=vmax)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.subplot(2, 3, 6)
    plt.imshow(Results_imgs["Height_Map_Imgs"]["GT"] - Results_imgs["Height_Map_Imgs"]["Est_HM_after_Shift"])
    plt.xticks([], [])
    plt.yticks([], [])
    R1 = Results_imgs["Height_Map_Scores_After_Alignment"]
    ans = "MAE: " + str(np.round(R1["MAE"], 3)) + "\nRMSE: " + str(np.round(R1["RMSE"], 3)) + "\n% in 1 m: " + str(
        np.round(R1["Acc_1_m"], 3)) \
          + "\nMedian: " + str(np.round(R1["Median"], 3))
    # plt.text(5,5, ans, bbox={'facecolor':'white', 'pad': 10})

    plt.xlabel(ans)
    plt.colorbar()

    R1 = Results_imgs["Height_Map_Scores_Before_Alignment"]
    print("Before Alignment")
    print(R1["MAE"], R1["RMSE"], R1["Acc_1_m"], R1["Median"])
    R1 = Results_imgs["Height_Map_Scores_After_Alignment"]
    print("After Alignment")
    print(R1["MAE"], R1["RMSE"], R1["Acc_1_m"], R1["Median"])
    print("Shift")
    print(R1["Shift_x_y_deg"])

    plt.tight_layout()
    if show_instead_of_save:
        plt.show()
    else:
        plt.savefig(path + "/HM.png")
        plt.close("all")

def get_Solar_Walk_Img(Results, path, show_instead_of_save = False):
    for a_key in Results["Image_Info"].keys():
        if "Solar_Results" in Results["Image_Info"][a_key].keys():
            temp_store = []
            temp_store_idx = []
            temp_store_vec = []
            Base_Vec = -1
            Base_Matrix = np.zeros([1,1,2])
            for another_key in Results["Image_Info"][a_key]["Solar_Results"].keys():
                if another_key == "Base_Solar_Vec":
                    Base_Vec = Results["Image_Info"][a_key]["Solar_Results"][another_key]
                elif another_key == "Score_Full_Score_Matrix":
                    Base_Matrix = Results["Image_Info"][a_key]["Solar_Results"][another_key]
                    Base_Matrix[Base_Matrix < 0] = 0
                    for temp in range(2):
                        Base_Matrix[:,:,temp] += Base_Matrix[:,:,temp].T
                else:
                    temp_store.append(Results["Image_Info"][a_key]["Solar_Results"][another_key]["Img"])
                    temp_store_idx.append(int(another_key))
                    temp_store_vec.append(Results["Image_Info"][a_key]["Solar_Results"][another_key]["Solar_Vec"])

            if show_instead_of_save == False:
                giffify(temp_store, path + "/Shadow_Walk.gif")

            n = len(temp_store)
            c = np.sqrt(n / (16.*9))
            C = int(np.round(c*16))
            R = int(np.round(c*9))
            while R*C < n:
                C += 1
            plt.figure(figsize=(32, 18), dpi=80)
            for i in range(n):
                plt.subplot(R,C,i+1)
                plt.imshow(temp_store[i])
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Shadow_Walk.png")
                plt.close("all")

            plt.figure(figsize=(8, 8), dpi=80)
            a,b = np.unravel_index(np.argmax(Base_Matrix[:,:,0]), [Base_Matrix.shape[0], Base_Matrix.shape[1]])
            plt.subplot(2,2,1)
            plt.imshow(temp_store[0])
            plt.title("Max. Shadow")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2,2,2)
            plt.imshow(temp_store[-1])
            plt.title("Min. Shadow")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2,2,3)
            plt.imshow(temp_store[a])
            plt.title("Worst Pair Img 1")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2,2,4)
            plt.imshow(temp_store[b])
            plt.title("Worst Pair Img 2")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Shadow_Compare.png")
                plt.close("all")

            plt.figure(figsize=(16, 9), dpi=80)
            plt.subplot(1,2,1)
            plt.title("Shadow Error, Shadow Rejection")
            plt.matshow(Base_Matrix[:,:,0], fignum=0)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.title("Shadow Error, No Shadow Rejection")
            plt.matshow(Base_Matrix[:, :, 1], fignum=0)
            plt.colorbar()
            plt.tight_layout()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Shadow_Scores.png")
                plt.close("all")

def get_Time_Walk_Img(Results, path, show_instead_of_save = False):
    counts = 0
    for a_key in Results["Image_Info"].keys():
        if "Solar_Results" in Results["Image_Info"][a_key].keys():
            for another_key in Results["Image_Info"][a_key]["Solar_Results"].keys():
                if another_key != "Base_Solar_Vec" and another_key != "Score_Full_Score_Matrix":
                    counts += 1
            break
    counts = (counts+1)//2

    for a_key in Results["Image_Info"].keys():
        if "Time_Results" in Results["Image_Info"][a_key].keys():
            temp_store = []
            temp_store_idx = []
            temp_store_vec = []
            Base_Vec = -1
            Base_Matrix = np.zeros([1, 1, 2])
            for another_key in Results["Image_Info"][a_key]["Time_Results"].keys():
                if another_key == "Base_Time":
                    Base_Vec = Results["Image_Info"][a_key]["Time_Results"][another_key]
                elif another_key == "Score_Full_Score_Matrix":
                    Base_Matrix = Results["Image_Info"][a_key]["Time_Results"][another_key]
                    Base_Matrix[Base_Matrix < 0] = 0
                    for temp in range(2):
                        Base_Matrix[:, :, temp] += Base_Matrix[:, :, temp].T
                else:
                    temp_store.append(Results["Image_Info"][a_key]["Time_Results"][another_key]["Img"])
                    temp_store_idx.append(int(another_key))
                    temp_store_vec.append(Results["Image_Info"][a_key]["Time_Results"][another_key]["Time_Frac"])
            # print(np.array(temp_store_vec)/(2*np.pi))
            # exit()

            n = counts
            c = np.sqrt(n / (16. * 9))
            C = int(np.round(c * 16))
            R = int(np.round(c * 9))
            while R * C < n:
                C += 1
            plt.figure(figsize=(32, 18), dpi=80)
            for i in range(n):
                plt.subplot(R, C, i + 1)
                plt.imshow(temp_store[i])
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Data_Set_Time_Walk.png")
                plt.close("all")
                giffify(temp_store[0:counts], path + "/Data_Set_Time_Walk.gif")

            n = len(temp_store_vec) - counts
            c = np.sqrt(n / (16. * 9))
            C = int(np.round(c * 16))
            R = int(np.round(c * 9))
            while R * C < n:
                C += 1
            plt.figure(figsize=(32, 18), dpi=80)
            for i in range(n):
                plt.subplot(R, C, i + 1)
                plt.imshow(temp_store[i + counts])
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Time_Walk.png")
                plt.close("all")
                giffify(temp_store[counts::], path + "/Time_Walk.gif")

            plt.figure(figsize=(8, 8), dpi=80)
            plt.title("Season Change")
            plt.matshow(Base_Matrix[:, :, 1], fignum=0)
            plt.colorbar()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Time_Scores.png")
                plt.close("all")

def show_indiv_img_Results(Results, path, show_instead_of_save = False):
    c2 = 0
    for i in Results["Image_Info"].keys():
        if i != "Solar_Results" and i != "Time_Results":
            # print("Img Name:", i, c2)
            # print("Is Testing:", Results["Image_Info"][i]["Is_Testing_Img"])
            # print("Sky Color:", Results["Image_Info"][i]["Scores"]["Sky_Col"])
            # print("Ideal Time PSNR, SSIM:", Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_PSNR"],
            #       Results["Image_Info"][i]["Scores"]["Ideal_Time_Img_SSIM"])
            # print("Class Output:", Results["Image_Info"][i]["Scores"]["Ideal_Class_Output"])
            c = 1
            n = len(Results["Image_Info"][i]["Imgs"].keys())-1
            R = n // 4
            while R*4 < n:
                R += 1
            plt.figure(figsize=(24, 6*R), dpi=80)
            for j in Results["Image_Info"][i]["Imgs"].keys():
                if j != "Valid_Pt_Mask":
                    plt.subplot(R, 4, c)
                    plt.title(j)
                    plt.imshow(Results["Image_Info"][i]["Imgs"][j])
                    plt.yticks([], [])
                    plt.xticks([], [])
                    if j != "GT_Img" and j != "HM":
                        PSNR = Results["Image_Info"][i]["Scores"][j + "_PSNR"]
                        SSIM = Results["Image_Info"][i]["Scores"][j + "_SSIM"]
                        plt.xlabel("PSNR: " + str(np.round(PSNR, 3)) + " --- SSIM: " + str(np.round(SSIM, 3)))
                    c += 1
            plt.tight_layout()
            if show_instead_of_save:
                plt.show()
            else:
                plt.savefig(path + "/Img_" + str(i) + ".png")
                plt.close("all")
            c2 += 1

def load_results_2(arg_dict):
    fin = open(arg_dict.logs_dir + "/Analysis.pickle", "rb")#open(arg_dict["Log_File"] + "/" + arg_dict["Exp_Name"] + "/Analysis.pickle", "rb")
    Results:dict = pickle.load(fin)
    fin.close()
    path = arg_dict.logs_dir + "/Output" #arg_dict["Log_File"] + "/" + arg_dict["Exp_Name"] + "/Output"
    try:
        os.mkdir(path)
    except:
        print("Output folder already exists")

    # show_dict_struc(Results)

    get_HM_img(Results, path)
    get_Solar_Walk_Img(Results, path)
    get_Time_Walk_Img(Results, path)
    show_indiv_img_Results(Results, path)

def build_walk_summary(summary_dict, path):
    training_sat_angle = np.array(summary_dict["Training_Image_Meta_Data"]["Sat_el_and_az"])
    training_sat_angle[:, 0] = 90 - training_sat_angle[:, 0]
    x, y = np.cos(np.deg2rad(training_sat_angle[:, 1])) * training_sat_angle[:, 0], np.sin(
        np.deg2rad(training_sat_angle[:, 1])) * training_sat_angle[:, 0]
    testing_sat_angle = np.array(summary_dict["Testing_Image_Meta_Data"]["Sat_el_and_az"])
    testing_sat_angle[:, 0] = 90 - testing_sat_angle[:, 0]
    x2, y2 = np.cos(np.deg2rad(testing_sat_angle[:, 1])) * testing_sat_angle[:, 0], np.sin(np.deg2rad(
        testing_sat_angle[:, 1])) * testing_sat_angle[:, 0]

    walking_sat_angle = np.array(summary_dict["Walking_Points"]["Sat_el_and_az"])
    walking_sat_angle[:, 0] = 90 - walking_sat_angle[:, 0]
    x3, y3 = np.cos(np.deg2rad(walking_sat_angle[:, 1])) * walking_sat_angle[:, 0], np.sin(np.deg2rad(
        walking_sat_angle[:, 1])) * walking_sat_angle[:, 0]

    plt.figure(figsize=(12,6), dpi=80)
    plt.subplot(1,2,1)
    plt.axhline(c="black")
    plt.axvline(c="black")
    X = np.linspace(-32,32,50)
    Y = np.linspace(-32,32,50)
    Z = np.sqrt(X.reshape([-1,1])**2 + Y.reshape([1,-1])**2)
    ax = plt.contour(X, Y, Z)
    plt.clabel(ax, inline=True, fontsize=10)

    a = plt.scatter(x, y)
    b = plt.scatter(x2, y2)
    c = plt.scatter(x3, y3, c = "red")
    plt.scatter(0,0, c="purple")
    plt.legend([a,b,c], ["Training Point", "Testing Point", "Walking Point"])
    plt.title("Overview of Satellite Off-Nadir and Azmuth Angles")
    plt.xlim(-32,32)
    plt.ylim(-32,32)

    plt.subplot(1,2,2)

    training_sat_angle = np.array(summary_dict["Training_Image_Meta_Data"]["Solar_el_and_az"])
    x, y = np.cos(np.deg2rad(training_sat_angle[:, 1])) * training_sat_angle[:, 0], np.sin(
        np.deg2rad(training_sat_angle[:, 1])) * training_sat_angle[:, 0]
    testing_sat_angle = np.array(summary_dict["Testing_Image_Meta_Data"]["Solar_el_and_az"])
    x2, y2 = np.cos(np.deg2rad(testing_sat_angle[:, 1])) * testing_sat_angle[:, 0], np.sin(np.deg2rad(
        testing_sat_angle[:, 1])) * testing_sat_angle[:, 0]

    walking_sat_angle = np.array(summary_dict["Walking_Points"]["Solar_el_and_az"])
    x3, y3 = np.cos(np.deg2rad(walking_sat_angle[:, 1])) * walking_sat_angle[:, 0], np.sin(np.deg2rad(
        walking_sat_angle[:, 1])) * walking_sat_angle[:, 0]

    plt.axhline(c="black")
    plt.axvline(c="black")
    X = np.linspace(5, -60 ,50)
    Y = np.linspace(-5, 60,50)
    Z = np.sqrt(X.reshape([-1,1])**2 + Y.reshape([1,-1])**2)
    ax = plt.contour(X, Y, Z)
    plt.clabel(ax, inline=True, fontsize=10)

    a = plt.scatter(x, y)
    b = plt.scatter(x2, y2)
    c = plt.scatter(x3, y3, c = "red")
    plt.scatter(0,0, c="purple")
    plt.legend([a,b,c], ["Training Point", "Testing Point", "Walking Point"])
    plt.title("Overview of Solar Elevation and Azmuth Angles")
    plt.xlim(-60,5)
    plt.ylim(-5,60)

    plt.tight_layout()
    plt.savefig(path + "/Angle_Data.png")

    training_time = np.array(summary_dict["Training_Image_Meta_Data"]["Time_Frac"])
    testing_time = np.array(summary_dict["Testing_Image_Meta_Data"]["Time_Frac"])
    walking_times = summary_dict["Walking_Points"]["Time_Frac"]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.scatter(training_time*2*np.pi, np.ones_like(training_time) + np.random.rand(training_time.shape[0])*.4)
    ax.scatter(testing_time * 2 * np.pi, np.ones_like(testing_time) + np.random.rand(testing_time.shape[0])*.4)
    ax.scatter(walking_times*2*np.pi,  np.ones_like(walking_times) + np.random.rand(walking_times.shape[0])*.4, c="red")
    ax.set_rmax(1.5)
    ax.set_rticks([])
    angles = np.linspace(0,360, 12, endpoint=False)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_thetagrids(angles, months)

    plt.legend(["Training Point", "Testing Point", "Walking Point"], loc=10)
    plt.title("Overview of Times")
    plt.savefig(path + "/Time_Data.png")
    plt.close("all")

def summarize_walk_imgs(Walks, Metadata_dict, path):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i in range(Walks.shape[0]):
        c = 1
        plt.figure(figsize=(12,12), dpi=80)
        for j in range(Walks.shape[1]):
            for k in range(Walks.shape[2]):
                plt.subplot(Walks.shape[1], Walks.shape[2], c)
                if j == 0:
                    plt.title(months[(int(np.round((Walks[i,j,k]["Time_Frac"])*12)))])
                if k == 0:
                    plt.ylabel(str(np.round(Walks[i,j,k]["Solar_el_az"][0], 1)))
                plt.imshow(Walks[i,j,k]["Col_Img"])
                plt.xticks([])
                plt.yticks([])
                c += 1
            # exit()

        plt.subplots_adjust(left=0.05,
                            bottom=0.05,
                            right=0.95,
                            top=0.95,
                            wspace=0.0,
                            hspace=0.0)
        el, az = Walks[i, 0, 0]["Sat_el_az"]
        plt.savefig(path + "/Output_" + str(int(el)) + "_" + str(int(az)) + ".png")
        plt.close("all")

    for i in range(Walks.shape[0]):
        c = 1
        plt.figure(figsize=(12, 12), dpi=80)
        for j in range(Walks.shape[1]):
            for k in range(Walks.shape[2]):
                plt.subplot(Walks.shape[1], Walks.shape[2], c)
                if j == 0:
                    plt.title(months[(int(np.round((Walks[i, j, k]["Time_Frac"]) * 12)))])
                if k == 0:
                    plt.ylabel(str(np.round(Walks[i, j, k]["Solar_el_az"][0], 1)))
                plt.imshow(Walks[i, j, k]["Shadow_Img"])
                plt.xticks([])
                plt.yticks([])
                c += 1
            # exit()

        plt.subplots_adjust(left=0.05,
                            bottom=0.05,
                            right=0.95,
                            top=0.95,
                            wspace=0.0,
                            hspace=0.0)
        el, az = Walks[i, 0, 0]["Sat_el_az"]
        plt.savefig(path + "/Output_Shadow_" + str(int(el)) + "_" + str(int(az)) + ".png")
        plt.close("all")

    # exit()

def load_results_2_adv(arg_dict):
    fin = open(arg_dict["Log_File"] + "/" + arg_dict["Exp_Name"] + "/Full_Analysis.pickle", "rb")
    Results: dict = pickle.load(fin)
    fin.close()
    path = arg_dict["Log_File"] + "/" + arg_dict["Exp_Name"] + "/Output_Adv"
    try:
        os.mkdir(path)
    except:
        print("Output folder already exists")

    path_2 = arg_dict["Log_File"] + "/" + arg_dict["Exp_Name"] + "/Output_Adv/Indiv"
    try:
        os.mkdir(path_2)
    except:
        print("Output Indiv folder already exists")

    # show_dict_struc(Results, 2)
    summarize_walk_imgs(Results["Walk_Results"], Results["Meta_Data_Summary"], path)
    build_walk_summary(Results["Meta_Data_Summary"], path)

    # get_HM_img(Results, path)
    show_indiv_img_Results(Results, path_2)


    # exit()