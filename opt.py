"""
This script defines the input parameters that can be customized from the command line
"""
import sys
import argparse
import datetime
import json
import os

#["JAX_004", "JAX_068", "JAX_105", "JAX_260", "OMA_248", "OMA_281", "OMA_374"]
def get_opts(loaded_args = None, exp_Name = None, region = None, num_time_class = None, use_type_2_solar = None,
             use_reg = None, use_prior = None, log_loc = None, force_write_json = False, use_solar = None, use_time = None,
             use_MSE_loss = None):
    parser = argparse.ArgumentParser()

    # input paths
    #default="OMA_248"
    if region is None:
        parser.add_argument('--site_name', type=str, required=True,
        help="Name of area to perform tests on")
    else:
        parser.add_argument('--site_name', type=str, required=False, default=region,
                        help="Name of area to perform tests on")
    #default="/mnt/cloudNAS3/Michael_CN3/Home/Documents/Sat_Imgs/IEEE_Data/Images",
    parser.add_argument('--root_dir', type=str, required=True, help='root directory of the input dataset')
    #default="/mnt/cloudNAS3/Michael_CN3/Home/Documents/Sat_Imgs/IEEE_Data/Track3-Metadata",
    parser.add_argument('--rpc_dir', type=str, required=False, help='Directory where the Corrected RPCs are located')
    parser.add_argument("--ckpts_dir", type=str, default="ckpts",
                        help="output directory to save trained models")
    #default = "/mnt/cloudNAS2/Michael/Home/Public_Pycharm_Output/NeRF_SC_v3/logs"
    if log_loc is None:
        parser.add_argument("--logs_dir", type=str, required=True, help="output directory to save experiment logs")
    else:
        parser.add_argument("--logs_dir", type=str,
                            default=log_loc,
                            help="output directory to save experiment logs")
    #default="/mnt/cloudNAS3/Michael_CN3/Home/Documents/Sat_Imgs/IEEE_Data/Images"
    parser.add_argument('--gt_dir', type=str, required=True, help='directory where the ground truth DSM is located (if available)')
    #default="/mnt/cloudNAS3/Michael_CN3/Home/Pycharm_Projects/NeRF_SC_v3/Cache",
    parser.add_argument('--cache_dir', type=str, required=True, help='directory where cache for the current dataset is found, should already exist and contain the correct RPCs.')
    #default="TEST"
    if exp_Name is None:
        # other basic stuff and dataset options
        parser.add_argument("--exp_name", type=str, required=True, help="experiment name")
    else:
        # other basic stuff and dataset options
        parser.add_argument("--exp_name", type=str, default=exp_Name,
                            help="experiment name")


    parser.add_argument('--min_height', type=float, default=None,
                        help="Minimum height of region (only needed if GT not provided)")
    parser.add_argument('--max_height', type=float, default=None,
                        help="Maximum height of region (only needed if GT not provided)")
    # parser.add_argument('--cache_dir', type=str, default="./Cache",
    #                     help='directory where cache for the current dataset is found')
    parser.add_argument('--use_Bundle_Adjust', action="store_true", default=True,
                        help="Apply Bundle Adjustment to Satellite Images")
    parser.add_argument('--DSM_Mode', type=str, default="Space_Carve", choices=["Space_Carve", "Stereo", "LiDAR", "None"],
                        help="Method to aquire DSM")
    parser.add_argument('--testing_size', type=int, default=3,
                        help="Number of Images to reserve for testing.")
    parser.add_argument("--testing_image_names", type=str, required=False,
                        help="Optional Argument to give the location of file containing the names of the images reserved for testing, will override testing_size argument!")
    # parser.add_argument("--ckpt_path", type=str, default=None,
    #                     help="pretrained checkpoint path to load")
    # parser.add_argument('--data', type=str, default='sat', choices=['sat', 'blender'],
    #                     help='type of dataset')
    parser.add_argument("--model", type=str, default="t-nerf", choices=["g-nerf", "t-nerf"],
                        help="which NeRF to use")
    parser.add_argument("--camera_model", type=str, default="Pinhole", choices=["Pinhole", "RPC"],
                        help="which camera model to use")
    parser.add_argument("--gpu_id", type=int, default=0, required=False,
                        help="GPU that will be used")

    # training and network configuration
    parser.add_argument('--lr', type=float, default=(10**((-5.86-1))),
                        help='max learning rate')
    parser.add_argument('--lr_alpha_scale', type=float, default=100, #1000
                        help='Scale lr for alpha parameters.')
    parser.add_argument('--batch_size', type=int, default=512,#1280,
                        help='batch size (number of input rays per iteration)')
    parser.add_argument('--img_training_downscale', type=int, default=4,
                        help='downscale factor for the training input images')
    parser.add_argument('--img_validation_downscale', type=int, default=8,
                        help='downscale factor for the testing images')
    parser.add_argument('--max_train_steps', type=int, default=50000,
                        help='number of training iterations')
    parser.add_argument('--n_saves', type=int, default=24*4*0+25,#*0+3,
                        help="Number of saves during training")
    parser.add_argument('--use_advanced_solar', action='store_true', default=True,
                        help='Use my solar mode rather than S-NeRFs')
    parser.add_argument('--fc_units', type=int, default=512,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers', type=int, default=8,
                        help='number of fully connected layers in the main block of layers')
    parser.add_argument('--n_samples', type=int, default=96,
                        help='number of coarse scale discrete points per input ray')
    parser.add_argument('--n_importance', type=int, default=0,
                        help='number of fine scale discrete points per input ray')
    parser.add_argument('--pose_encode_size', type=int, default=10,
                        help="Size of encoding for position")
    parser.add_argument('--view_angle_size', type=int, default=4,
                        help="Size of encoding for view angle, set to zero to not use view angle")
    parser.add_argument('--sun_angle_size', type=int, default=4,
                        help="Size of encoding for sun angle")
    parser.add_argument('--temporal_size', type=int, default=3,
                        help="Size of encoding for time information")
    if num_time_class is None:
        parser.add_argument('--number_low_frequency_cases', type=int, default=4,
                            help="Number of cases low frequency events will be divided into, only relevent for T-NeRF")
    else:
        parser.add_argument('--number_low_frequency_cases', type=int, default=num_time_class,
                            help="Number of cases low frequency events will be divided into, only relevent for T-NeRF")
    # parser.add_argument('--seperate_fine', action='store_true',
    #                     help='creates a second network for fine output')
    # parser.add_argument('--noise_std', type=float, default=0.0,
    #                     help='standard deviation of noise added to sigma to regularize')
    parser.add_argument('--chunk', type=int, default=1024*10,
                        help='maximum number of rays that can be processed at once without memory issues')

    # other sat-nerf specific stuff
    parser.add_argument('--sc_lambda', type=float, default=0.03*0+1*0+0.03,
                        help='float that multiplies the solar correction auxiliary loss')
    parser.add_argument('--ds_lambda', type=float, default=0.03,
                        help='float that multiplies the depth supervision auxiliary loss')
    parser.add_argument('--p_lambda', type=float, default=0.03,
                        help='float that multiplies the prior supervision auxiliary loss when using alpha sigma')
    if use_prior is None:
        parser.add_argument('--jump_start', action='store_true', default=True,
                            help='Use DSM to guide early training')
    else:
        parser.add_argument('--jump_start', action='store_true', default=use_prior,
                            help='Use DSM to guide early training')
    parser.add_argument('--weight_training_samples', action='store_true', default=False,
                        help="Weight training regions based on distance between regions")
    parser.add_argument('--use_auto_balance', action='store_true', default=False,
                        help="Use loss balance compuation from Multi-loss weighting with coefficent of variations")
    parser.add_argument("--consistency_loss_weight", type=float, default=.003*0,
                        help="Weight term for consistency loss") #currently compares Expected vs maximum surface NOT different rays
    # parser.add_argument("--expected_max_loss_weight", type=float, default=0.03,
    #                     help="Weight term for expected_max loss")
    # parser.add_argument('--ds_drop', type=float, default=0.25,
    #                     help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--ds_drop', type=float, default=0.2,
                        help='portion of training where DSM will be used, effects both depth supervision loss and jump start')
    # parser.add_argument('--ds_noweights', action='store_true',
    #                     help='do not use reprojection errors to weight depth supervision loss')
    parser.add_argument('--first_beta_portion', type=float, default=.3,
                        help='portion of training steps at which the beta will NOT be used')
    # parser.add_argument('--t_embbeding_tau', type=int, default=4,
    #                     help='portion of training steps at which the depth supervision loss will be dropped')
    # parser.add_argument('--t_embbeding_vocab', type=int, default=30,
    #                     help='portion of training steps at which the depth supervision loss will be dropped')
    parser.add_argument('--use_HSLuv', action='store_true', default=False, help="Use HSLuv instead of RGB training.")

    if use_type_2_solar is None:
        parser.add_argument('--Solar_Type_2', action='store_true', default=False, help="Use Solar Error according to S-NeRF and Sat-NeRF.")
    else:
        parser.add_argument('--Solar_Type_2', action='store_true', default=use_type_2_solar, help="Use Solar Error according to S-NeRF and Sat-NeRF.")

    if use_reg is None:
        parser.add_argument('--Use_Reg', action='store_true', default=True, help="Use reg. terms")
    else:
        parser.add_argument('--Use_Reg', action='store_true', default=use_reg,
                            help="Use reg. terms")

    if use_solar is None:
        parser.add_argument('--Use_Solar', action='store_true', default=True, help="Consider solar rays, will override Use_Solar_Type_2")
    else:
        parser.add_argument('--Use_Solar', action='store_true', default=use_solar,
                            help="Consider solar rays, will override Use_Solar_Type_2")

    if use_time is None:
        parser.add_argument('--Use_Time', action='store_true', default=True,
                            help="Consider solar rays, will override Use_Solar_Type_2")
    else:
        parser.add_argument('--Use_Time', action='store_true', default=use_time,
                            help="Use Time input")

    if use_MSE_loss is None:
        parser.add_argument('--Use_MSE_loss', action='store_true', default=False,
                            help="Use MSE loss instead of adapative loss")
    else:
        parser.add_argument('--Use_MSE_loss', action='store_true', default=use_MSE_loss,
                            help="Use MSE loss instead of adapative loss")




    print(sys.argv)
    args = parser.parse_args()

    exp_id = args.config_name if args.exp_name is None else args.exp_name
    # args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), exp_id)
    # args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m_%d"), exp_id)
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}/{}".format(args.logs_dir, args.exp_name), exist_ok=True)
    os.makedirs("{}/{}".format(args.cache_dir, args.site_name), exist_ok=True)
    args.cache_dir = "{}/{}".format(args.cache_dir, args.site_name)
    args.logs_dir = "{}/{}".format(args.logs_dir, args.exp_name)
    if (exp_Name is None and region is None) or force_write_json:
        with open("{}/opts.json".format(args.logs_dir), "w") as f:
            json.dump(vars(args), f, indent=2)

    return args
