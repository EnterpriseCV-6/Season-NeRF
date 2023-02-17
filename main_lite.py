import argparse
import os
import json
import sys

from main import run_test

def get_opts(loaded_args = None, use_MSE_loss = None, num_time_class = None, use_prior = None, use_type_2_solar = None, use_solar = None):
    parser = argparse.ArgumentParser()


    parser.add_argument('--IO_Location', type=str, required=True,
                        help="Location of files for input and output.")


    #Args auto-filled in because of lite mode
    parser.add_argument('--site_name', type=str, required=False, default="OMA_281",
                        help="Name of site to run Season-NeRF on, ex JAX_068, OMA_084.")
    parser.add_argument('--exp_name', type=str, required=False, default="OMA_281_Lite",
                        help="Name of Experiment.")


    #Overrides assumptions made by IO_Location argument.
    #All are required args for file locs IF IO_Location IS NOT USED!!!.
    parser.add_argument('--cache_dir', type=str, required=False,
                        help='directory where cache for the current dataset is found')
    parser.add_argument('--root_dir', type=str, required=False,
                        help="Location of Images for site.")
    parser.add_argument('--rpc_dir', type=str, required=False,
                        help="Location of RPC files.")
    parser.add_argument('--logs_dir', type=str, required=False,
                        help="Location to store Log outputs.")
    parser.add_argument("--testing_image_names", type=str, required=False,
                        help="Location of txt file containing name of testing images.")

    #Optional Training Modes:
    if use_MSE_loss is None:
        parser.add_argument('--Use_MSE_loss', action='store_true', default=False,
                            help="Use MSE loss instead of adapative loss")
    else:
        parser.add_argument('--Use_MSE_loss', action='store_true', default=use_MSE_loss,
                            help="Use MSE loss instead of adapative loss")
    if use_prior is None:
        parser.add_argument('--jump_start', action='store_true', default=True,
                            help='Use DSM to guide early training')
    else:
        parser.add_argument('--jump_start', action='store_true', default=use_prior,
                            help='Use DSM to guide early training')
    if use_type_2_solar is None:
        parser.add_argument('--Solar_Type_2', action='store_true', default=False,
                            help="Use Solar Error according to S-NeRF and Sat-NeRF.")
    else:
        parser.add_argument('--Solar_Type_2', action='store_true', default=use_type_2_solar,
                            help="Use Solar Error according to S-NeRF and Sat-NeRF.")

    #Optional Args for pre-processing
    parser.add_argument('--skip_Bundle_Adjust', action='store_true', required=False, default=False,
                        help="Use Bundle Adjusted RPCs for process.")
    parser.add_argument('--gt_dir', type=str, required=False,
                        help="Location of Ground Truth Lidar, if different than location of images")

    #Optional Args for training Network
    parser.add_argument('--img_training_downscale', type=int, default=4,
                        help='downscale factor for the training input images')
    parser.add_argument('--img_validation_downscale', type=int, default=8,
                        help='downscale factor for the testing images')
    parser.add_argument('--max_train_steps', type=int, default=5000,
                        help='number of training iterations')
    parser.add_argument('--n_samples', type=int, default=96,
                        help='number of coarse scale discrete points per input ray')
    parser.add_argument('--n_saves', type=int, default=10,
                        help="Number of saves during training")
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size (number of image rays per iteration, number of solar rays is always 2* number of image rays)')
    parser.add_argument('--lr', type=float, default=(10 ** (-4.86) * 3),
                        help='max learning rate')
    parser.add_argument('--lr_alpha_scale', type=float, default=1000,
                        help='Scale lr for alpha parameters.')

    parser.add_argument('--fc_units', type=int, default=512,
                        help='number of fully connected units in the main block of layers')
    parser.add_argument('--fc_layers', type=int, default=8,
                        help='number of fully connected layers in the main block of layers')

    parser.add_argument('--sc_lambda', type=float, default=0.03,
                        help='float that multiplies the solar correction auxiliary loss')
    parser.add_argument('--ds_lambda', type=float, default=0.03,
                        help='float that multiplies the depth supervision auxiliary loss')
    parser.add_argument('--p_lambda', type=float, default=0.03,
                        help='float that multiplies the prior supervision auxiliary loss when using alpha sigma')

    if num_time_class is None:
        parser.add_argument('--number_low_frequency_cases', type=int, default=4,
                            help="Number of cases low frequency events will be divided into, only relevent for T-NeRF")
    else:
        parser.add_argument('--number_low_frequency_cases', type=int, default=num_time_class,
                            help="Number of cases low frequency events will be divided into, only relevent for T-NeRF")



    #Args that should NEVER change and NEVER be used but exist for legacy reasons
    parser.add_argument("--camera_model", type=str, default="Pinhole", choices=["Pinhole", "RPC"],
                        help="which camera model to use")
    parser.add_argument("--gpu_id", type=int, default=0, required=False,
                        help="GPU that will be used")
    parser.add_argument('--weight_training_samples', action='store_true', default=False,
                        help="Weight training regions based on distance between regions")
    parser.add_argument('--DSM_Mode', type=str, default="Space_Carve",
                        choices=["Space_Carve", "Stereo", "LiDAR", "None"],
                        help="Method to aquire DSM")
    parser.add_argument('--chunk', type=int, default=1024 * 10,
                        help='maximum number of rays that can be processed at once without memory issues')
    parser.add_argument('--n_importance', type=int, default=0,
                        help='number of fine scale discrete points per input ray')
    parser.add_argument('--use_HSLuv', action='store_true', default=False, help="Use HSLuv instead of RGB training.")
    parser.add_argument('--Use_Reg', action='store_true', default=True, help="Use reg. terms")
    parser.add_argument('--use_auto_balance', action='store_true', default=False,
                        help="Use loss balance compuation from Multi-loss weighting with coefficent of variations")

    if use_solar is None:
        parser.add_argument('--Use_Solar', action='store_true', default=True,
                            help="Consider solar rays, will override Use_Solar_Type_2")
    else:
        parser.add_argument('--Use_Solar', action='store_true', default=use_solar,
                            help="Consider solar rays, will override Use_Solar_Type_2")

    print(sys.argv)
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = args.IO_Location + "/Cache"
    if args.root_dir is None:
        args.root_dir = args.IO_Location + "/IEEE_Data/Images"
    if args.rpc_dir is None:
        args.rpc_dir = args.IO_Location + "/IEEE_Data/Track3-Metadata"
    if args.logs_dir is None:
        args.logs_dir = args.IO_Location + "/Logs"
    if args.testing_image_names is None:
        args.testing_image_names = args.IO_Location + "/Testing_Imgs"

    if args.gt_dir is None:
        args.gt_dir = args.root_dir

    exp_id = args.config_name if args.exp_name is None else args.exp_name
    # args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), exp_id)
    # args.exp_name = "{}_{}".format(datetime.datetime.now().strftime("%Y-%m_%d"), exp_id)
    print("\nRunning {} - Using gpu {}\n".format(args.exp_name, args.gpu_id))

    os.makedirs("{}/{}".format(args.logs_dir, args.exp_name), exist_ok=True)
    os.makedirs("{}/{}".format(args.cache_dir, args.site_name), exist_ok=True)
    args.cache_dir = "{}/{}".format(args.cache_dir, args.site_name)
    args.logs_dir = "{}/{}".format(args.logs_dir, args.exp_name)
    args.testing_image_names = "{}/{}.txt".format(args.testing_image_names, args.site_name)

    with open("{}/opts.json".format(args.logs_dir), "w") as f:
        json.dump(vars(args), f, indent=2)

    return args





def _main():
    args = get_opts()
    run_test(args, eval_only=False)



if __name__ == '__main__':
    _main()