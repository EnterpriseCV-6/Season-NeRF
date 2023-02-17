import numpy as np
import argparse
from misc import load_args_from_json
import torch as t
from T_NeRF_Full_2.T_NeRF_net_v2 import T_NeRF
from T_NeRF_Eval_Utils.mg_Img_Eval import component_render_by_dir, get_imgs_from_Img_Dict
from matplotlib import pyplot as plt

def get_opts():
    parser = argparse.ArgumentParser()


    parser.add_argument('--Model_Location', type=str, required=True,
                        help="Location of files for input and output.")

    parser.add_argument('--VA', type=float, nargs=2, required=True,
                        help="View elevation and azimuth angles in degrees.")
    parser.add_argument('--SA', type=float, nargs=2, required=True,
                        help="Solar elevation and azimuth angles in degrees")
    parser.add_argument('--tf', type=float, required=True,
                        help="Time Fraction")


    #Other Control Args
    parser.add_argument('--Output_Size', type=int, nargs=3, required=False, default=(256,256,96),
                        help="Size of output image (n_rows, n_cols, sample per ray).")
    parser.add_argument('--Save_Name', type=str, required=False,
                        help="Save image as Save_Name INSTEAD OF displaying image.")

    #Other Useful args
    parser.add_argument('--ignore_progess', action='store_true', required=False, default=False,
                        help="Do not display rendering progress bars.")
    parser.add_argument('--full_outputs', action='store_true', required=False, default=False,
                        help="Display shadow mask and additional seasonal images.")
    parser.add_argument('--exact_shadow', action='store_true', required=False, default=False,
                        help="Use exact shadow mask instead of estimated shadow mask.")
    parser.add_argument('--Seasonal_Alignment', type=str, required=False,
                        help="Image to perform seasonal alignment against, if not past Seasonal alignment is NOT used.")
    parser.add_argument('--Force_CPU', action='store_true', required=False, default=False,
                        help="Use CPU for rendering, even if GPU is available.")

    args = parser.parse_args()
    return args

def load_t_nerf(args, model_name = "Final_Model.nn"):
    thenet = T_NeRF(args.fc_units, args.number_low_frequency_cases)
    thenet.load_state_dict(t.load(args.logs_dir + "/" + model_name, map_location=t.device("cpu")))

    return thenet



def load_model(file_loc):
    network_args = load_args_from_json(file_loc + "/opts.json")
    a_t_nerf = load_t_nerf(network_args)
    return a_t_nerf, network_args



def _main():
    args = get_opts()
    device = t.device("cuda:0" if t.cuda.is_available() and args.Force_CPU == False else "cpu")

    the_model, network_args = load_model(args.Model_Location)

    W2C_W2L_H = np.load(args.Model_Location + "/W2C_W2L_H.npy", allow_pickle=True).item()


    the_model.eval()
    the_model = the_model.to(device)

    view = args.VA
    sun = args.SA
    time = args.tf
    out_img_size = args.Output_Size
    use_exact_solar = False
    use_classic_shadows = False

    raw_data = component_render_by_dir(the_model, view, sun, time, out_img_size,
                                       W2C=W2C_W2L_H.get("W2C"), W2L_H=W2C_W2L_H.get("W2L_H"),
                                       include_exact_solar=use_exact_solar, device=device)
    imgs = get_imgs_from_Img_Dict(raw_data, out_img_size, use_classic_shadows)
    output = {"Season_Adj_Img": imgs["Season_Adj_Img"], "Shadow_Adjust": imgs["Shadow_Adjust"],
                       "Shadow_Mask": imgs["Shadow_Mask"], "Time_Class": imgs["Time_Class"]}
    plt.imshow(imgs["Season_Adj_Img"] * imgs["Shadow_Adjust"])
    plt.show()
    plt.imshow(imgs["Shadow_Mask"])
    plt.show()




if __name__ == '__main__':
    _main()