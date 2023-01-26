# coding=utf-8

from matplotlib import pyplot as plt
from opt import get_opts
from mg_pre_NeRF import run_pre_NeRF
from mg_get_DSM import get_DSM
from mg_Pt_holder import setup_quick_loader
import torch as t
import json
import os
from T_NeRF_Full_2.mg_eval import eval_T_NeRF
from T_NeRF_Full_2.Net_Tool_2 import T_NeRF_Net_Tool
from T_NeRF_Eval_Utils import load_results_2, load_results_2_adv
import main_eval_region
import numpy as np
from tqdm import tqdm

def get_exp_name(input_dict, expanded_exp_name = ""):
    exp_name = input_dict["Region"] + "_T" + str(input_dict["Time_Classes"])
    for a_key in input_dict.keys():
        if isinstance(input_dict[a_key], bool) and input_dict[a_key]:
            exp_name += "_" + a_key
    exp_name += expanded_exp_name
    return exp_name

def run_test(input_dict, testing_region_name_list_loc = None, eval_only = False):
    print("Running test...")
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    args = get_opts(exp_Name=get_exp_name(input_dict),
                    region=input_dict["Region"],
                    num_time_class=input_dict["Time_Classes"],
                    use_type_2_solar=input_dict["Solar_Loss_Type_2"],
                    use_reg=input_dict["Reg_Terms"],
                    use_prior= not input_dict["Skip_Prior_Start"],
                    use_solar= not input_dict["Ignore_Solar"],
                    use_time= not input_dict["Ignore_Time"],
                    log_loc=input_dict["Log_File"],
                    use_MSE_loss=input_dict["MSE_Loss"],
                    force_write_json=True)

    if testing_region_name_list_loc is not None:
        args.testing_image_names = testing_region_name_list_loc
    fout = open(input_dict["Log_File"] + "/" + get_exp_name(input_dict) + "/Args.json", "w")
    json.dump(input_dict, fout)
    fout.close()

    P_imgs, _ = run_pre_NeRF(args)


    if eval_only == False:
        H = P_imgs[0].S
        WC = P_imgs[0].get_world_center()
        setup_quick_loader(args)
        training_DSM, GT_DSM = get_DSM(args, device=device)
        setup_T_NeRF(args, training_DSM, GT_DSM, device, H, WC)

        print("Evaluating Results...")
        eval_T_NeRF(input_dict)
        print("Generating Results...")
        load_results_2(input_dict)

    use_quick_mode = (args.max_train_steps < 40000)
    main_eval_region.regional_eval(input_dict, input_dict["Log_File"] + "/" + input_dict["Exp_Name"] + "/Detailed_Output", quick_mode=(use_quick_mode))

    print("Updating Output Table...")
    if eval_only == False:
        main_eval_region.multi_region_merge(input_dict["Log_File"], "Detailed_Output", output_dir_name="Full_Summary")


def setup_T_NeRF(args, training_DSM, GT_DSM, device, H, WC):
    n_steps = args.max_train_steps
    net_tool = T_NeRF_Net_Tool(args, training_DSM, GT_DSM, device, H, WC)
    # net_tool.check_data_dict(False)
    # exit()

    # net_tool.find_lr(target_steps=100, init_steps=100, est_lr=.03, max_restarts=5)
    # exit()

    print("Number Training Steps:", n_steps)
    print("Number Epochs:", np.round(net_tool.get_num_epochs(), 3))
    print("Save Points:")
    for i in  net_tool.sub_section_outputs:
        print(i)
    # exit()
    for i in tqdm(range(n_steps)):
        net_tool.step()
        if i == 0:
            net_tool.eval_img(0)
    t.save(net_tool.network.state_dict(), args.logs_dir + "/Final_Model.nn")



def _main():
    loc_of_testing_region_files = "/mnt/cloudNAS2/Michael/Home/Pycharm_Projects/NeRF_SC_v35/Testing_Regions"
    log_file_loc = "/mnt/cloudNAS2/Michael/Home/Public_Pycharm_Output/NeRF_SC_v3"
    # log_file_name = "Logs_for_IEEE"
    log_file_name = "Logs_SIRENv2"#"Logs_Prototypical_Imgs_T8"#"Logs_Prototypical_Imgs_Bigger_Solar_BCE"#

    regions = ["OMA_124", "OMA_132", "OMA_248", "OMA_281", "OMA_374"]
    eval_only = False

    region = regions[3]#"OMA_132"
    log_loc = log_file_loc + "/" + log_file_name

    try:
        os.mkdir(log_loc)
    except:
        print("Directory " + log_loc + " exists.")

    # times = [4,6,8,2,12,16,1]
    base_time = 4

    input_dict = {"Time_Classes": base_time,
                  "MSE_Loss": False,
                  "Skip_Prior_Start": False,
                  "Solar_Loss_Type_2": False,
                  "Reg_Terms": False,
                  "Ignore_Solar": False,
                  "Ignore_Time": False,
                  "Log_File": log_loc, "Region": region}
    input_dict["Exp_Name"] = get_exp_name(input_dict)
    run_test(input_dict, loc_of_testing_region_files + "/" + region + ".txt", eval_only)


if __name__ == '__main__':
    _main()
