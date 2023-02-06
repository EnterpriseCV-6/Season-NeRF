from os import listdir
from os.path import isfile, join
from .mg_Sat_Img import sat_img

def load_sat_imgs(site_name, img_location, rpc_location, use_rpcm = True):
    # print(site_name, img_location, rpc_location)
    print("Loading Images")
    return load_imgs_from_location_type_3(img_location, rpc_location, site_name, use_rpcm)




def load_imgs_from_location_type_3(img_location, IMD_location, region_id, use_rpcm):
    img_list = []
    onlyfiles = sorted([f for f in listdir(img_location) if isfile(join(img_location, f))])
    for i in range(len(onlyfiles)):
        file_name = onlyfiles[i].split("_")
        if len(file_name) == 4 and (file_name[0] + "_" + file_name[1]) == region_id:
            print("{}_{}_{}_{}".format(file_name[0], file_name[1], file_name[2], file_name[3]))
            # print(IMD_location)
            IMD_location_full = IMD_location + "/" + file_name[0] + "/" + file_name[2][1::] + ".IMD"

            test_sat_img = sat_img(img_location, None, onlyfiles[i].split(".")[0], has_rpc=False, load_IMD=True, IMD_loc=IMD_location_full, IMD_loc_full=IMD_location_full)
            test_sat_img.load_rpc_from_tif(img_location + "/" +  onlyfiles[i], use_rpcm)
            img_list.append(test_sat_img)
            # print(test_sat_img)
            # exit()

    return img_list

def download_img_data(img_location, rpc_location):
    print(img_location)
    print(rpc_location)
