import pre_NeRF
import gdal
import numpy as np
from lego_rvl.rpc.rpc import RPC_py as RPC
import pickle
from os.path import exists

def check_cache(args):
    print("Checking cache for P_imgs and bounds...")

    path_head = args.cache_dir
    if args.skip_Bundle_Adjust == False:
        refined = "_Refined"
    else:
        refined = ""

    if exists(path_head + "/bounds_LLA" + refined + ".npy") and exists(path_head + "/P_imgs_" + args.camera_model + refined + ".pickle"):
        cache_hit = True
        fin = open(path_head + "/P_imgs_" + args.camera_model + refined + ".pickle", "rb")
        P_imgs = pickle.load(fin)
        fin.close()
        bounds_LLA = np.load(path_head + "/bounds_LLA" + refined + ".npy")
    else:
        cache_hit = False
        P_imgs = None
        bounds_LLA = None
    print("Done")

    return cache_hit, P_imgs, bounds_LLA

def run_pre_NeRF(args):
    cache_hit, P_imgs, bounds_LLA = check_cache(args)
    if cache_hit == False:
        print("Unable to find Requested files in cache, creating and caching for future use.")
        sat_img_list = pre_NeRF.load_sat_imgs(args.site_name, args.root_dir, args.rpc_dir)
        #Use DSM to get height range of area, if DSM is not available use supplied min and max height
        if args.gt_dir is not None:
            geotiff = gdal.Open(args.gt_dir + "/" + args.site_name + "_DSM.tif")
            print(args.gt_dir + "/" + args.site_name + "_DSM.tif")
            img = geotiff.ReadAsArray()
            img[img == -9999.] = np.NaN
            #Increase Area by five meters to ensure fit
            min_h = np.min(img[img==img]) - 5
            max_h = np.max(img[img==img]) + 5
        else:
            min_h = args.min_height
            max_h = args.max_height

        name = args.site_name[0:3]
        if args.skip_Bundle_Adjust == False:
            in_cache = True
            for a_sat_img in sat_img_list:
                if exists(args.cache_dir + "/" + "rpc_" + a_sat_img.img_name + "_corrected.ikono") == False and \
                    exists(args.cache_dir + "/" + "rpc_" + a_sat_img.img_name + "_original.ikono") == False:
                    print("Missing refined RPC, running Bundle Adjust.")
                    in_cache = False
                    break
            if in_cache == False:
                print("Error: Unable to run Bundle Adjustment!")
                print("Please make sure adjusted RPCs are already in cache OR")
                print("Please use arg --skip_Bundle_Adjust.")
                exit()
                pre_NeRF.run_wrapper(sat_img_list, args.cache_dir, tiePtsSet_file=None, flagUseSavedTiePts=False,
                                     the_imdDir=None, config_file="./BA_config.json")
            else:
                print("Found cached RPCs, bundle adjustment not needed.")
            for a_sat_img in sat_img_list:
                id = a_sat_img.img_name[9:11]
                if exists(args.cache_dir + "/" + "rpc_" + a_sat_img.img_name + "_corrected.ikono"):
                    corrected_RPC = RPC().from_file(args.cache_dir + "/" + "rpc_" + a_sat_img.img_name + "_corrected.ikono", args.rpc_dir + "/" + name + "/" + id + ".IMD")
                else:
                    corrected_RPC = RPC().from_file(
                        args.cache_dir + "/" + "rpc_" + a_sat_img.img_name + "_original.ikono",
                        args.rpc_dir + "/" + name + "/" + id + ".IMD")
                a_sat_img.rpc = corrected_RPC

        bounds_LLA = pre_NeRF.find_bounds_sat_img(sat_img_list, (min_h, max_h)).T
        P_imgs = []
        mu_r, sigma_r, min_e_r, max_e_r = 0, 0, 100, 0
        if args.camera_model == "Parallel":  #Model does not exist ... yet.
            print("Using Parallel Lines camera model.")
            print("Using Pinhole camera model.")
            for a_sat_img in sat_img_list:
                a_P_img = pre_NeRF.P_img_Parallel(a_sat_img, num_train_points=10, min_H=min_h, max_H=max_h)
                mu, sigma, min_e, max_e = pre_NeRF.test_accuracy(a_sat_img, a_P_img, 10, min_h, max_h)
                a_P_img.scale_P(bounds_LLA, np.array([[-1., 1.], [-1, 1], [-1, 1]]))
                P_imgs.append(a_P_img)
                mu_r += mu / len(sat_img_list)
                sigma_r += sigma / len(sat_img_list)
                min_e_r = min(min_e, min_e_r)
                max_e_r = max(max_e_r, max_e)
            c_type = "Parallel"
        elif args.camera_model == "Pinhole":
            print("Using Pinhole camera model.")
            for a_sat_img in sat_img_list:
                a_P_img = pre_NeRF.P_img_Pinhole(a_sat_img, num_train_points=10, min_H=min_h, max_H=max_h)
                mu, sigma, min_e, max_e = pre_NeRF.test_accuracy(a_sat_img, a_P_img, 10, min_h, max_h)
                a_P_img.scale_P(bounds_LLA, np.array([[-1., 1.], [-1, 1], [-1, 1]]))
                P_imgs.append(a_P_img)
                mu_r += mu / len(sat_img_list)
                sigma_r += sigma / len(sat_img_list)
                min_e_r = min(min_e, min_e_r)
                max_e_r = max(max_e_r, max_e)
            c_type = "Pinhole"

        else:
            print("Using RPC camera model.")
            for a_sat_img in sat_img_list:
                a_P_img = pre_NeRF.P_img(a_sat_img)
                mu, sigma, min_e, max_e = pre_NeRF.test_accuracy(a_sat_img, a_P_img, 10, min_h, max_h)
                a_P_img.scale_P(bounds_LLA, np.array([[-1.,1.], [-1,1], [-1,1]]))
                P_imgs.append(a_P_img)
                mu_r += mu / len(sat_img_list)
                sigma_r += sigma / len(sat_img_list)
                min_e_r = min(min_e, min_e_r)
                max_e_r = max(max_e_r, max_e)
            c_type = "RPC"
        if args.skip_Bundle_Adjust == False:
            refined =  "_Refined"
        else:
            refined = ""
        print("Model Error Summary: (mean std min error max error)")
        print(mu_r, sigma_r, min_e_r, max_e_r)
        print("Image Bounds (Lat, Lon H)")
        print(bounds_LLA)

        np.save(args.cache_dir + "/bounds_LLA" + refined + ".npy", bounds_LLA)

        fout = open(args.cache_dir + "/P_imgs_" + c_type + refined + ".pickle", "wb")
        pickle.dump(P_imgs, fout)
        fout.close()

    return P_imgs, bounds_LLA


