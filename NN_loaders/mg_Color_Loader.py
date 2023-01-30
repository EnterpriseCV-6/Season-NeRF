import pickle
import numpy as np
from tqdm import tqdm
import torch as t
import torch.utils.data as tdata
from mg_Pt_holder import basic_Ortho_info, basic_NeRF_info
import hsluv

def setup_col_loader(img_list, args, include_ortho, train_mode):
    camera_name = args.camera_model
    cache_path = args.cache_dir
    if args.skip_Bundle_Adjust == False:
        camera_name += "_Refined"
    img_shape = (1,1,1)

    DS = args.img_training_downscale if train_mode else args.img_validation_downscale

    col_objs = []
    all_solar_vecs = []
    for a_name in img_list:
        cache_file_name = a_name + "_" + camera_name + "_Basic_Info_DS_" + str(DS) + ".pickle"
        fout = open(cache_path + "/" + cache_file_name, "rb")
        basic_col_obj = pickle.load(fout)
        fout.close()
        # basic_col_obj:basic_NeRF_info
        all_solar_vecs.append(basic_col_obj.solar_vec)
        if args.use_HSLuv:
            basic_col_obj.valid_img_pts_col = np.array([hsluv.rgb_to_hsluv(an_entry) for an_entry in basic_col_obj.valid_img_pts_col])
            basic_col_obj.valid_img_pts_col = basic_col_obj.valid_img_pts_col / np.array([[360, 100, 100.]])
        col_objs.append(basic_col_obj)
        img_shape = basic_col_obj.img_size

    if include_ortho:
        col_objs.append(basic_Ortho_info(img_shape))

    all_solar_vecs = np.stack(all_solar_vecs, 0)

    return pt_loader(col_objs, all_solar_vecs)

class pt_loader(tdata.Dataset):
    def __init__(self, basic_col_objs, solar_vecs):
        self.img_names = []
        self.img_sizes = []
        self.full_img_size = []
        self.all_data = None
        self.img_ids = None
        self.solar_vecs = solar_vecs
        current_id = 0
        for a_obj in basic_col_objs:
            self.img_sizes.append(a_obj.valid_img_pts.shape[0])
            self.img_names.append(a_obj.img_name)
            self.full_img_size.append((a_obj.img_size[0], a_obj.img_size[1], a_obj.img_size[2]))

            time_info = a_obj.time_info
            # print(time_info.get_time())
            # print(time_info.get_time_frac())
            # print(time_info.get_time_encode())
            # time_data = np.repeat(np.array(time_info.get_time_encode()[1:3]), 2).reshape([1,4])
            time_data = np.array(time_info.get_time_encode()[1:5]).reshape([1, 4])
            # print(time_data.shape)
            # exit()
            # print("NN_loaders/mg_Color_Loader.py")
            # exit()
            img_weight = a_obj.img_weight



            # print(a_obj.valid_img_pts.shape)
            # print(a_obj.valid_world_pts_top.shape)
            # print(a_obj.valid_world_pts_bot.shape)
            # print(a_obj.view_angles_vec.shape)
            # print(a_obj.solar_vec.shape)
            # print(a_obj.valid_img_pts_col.shape)
            some_data = np.concatenate(
                [a_obj.valid_img_pts, a_obj.valid_world_pts_top, a_obj.valid_world_pts_bot,
                 a_obj.view_angles_vec, a_obj.solar_vec.reshape([1,3])*np.ones_like(a_obj.view_angles_vec),
                 time_data * np.ones([a_obj.view_angles_vec.shape[0], 4]), img_weight * np.ones([a_obj.view_angles_vec.shape[0], 1]),
                 a_obj.valid_img_pts_col], 1)
            # print(a_obj.valid_img_pts_col.shape)
            if self.all_data is None:
                self.all_data = some_data
                self.img_ids = [current_id] * some_data.shape[0]
            else:
                self.all_data = np.concatenate([self.all_data, some_data], 0)
                self.img_ids = self.img_ids + ([current_id] * some_data.shape[0])
            current_id += 1

        self.all_data = t.tensor(self.all_data).float()


    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, item):
        # img_pt = self.all_data[item, 0:2]
        # vec_top = self.all_data[item, 2:5]
        # vec_bot = self.all_data[item, 5:8]
        # vec_angle = self.all_data[item, 8:11]
        # sun_angle = self.all_data[item, 11:14]
        # learing_weight = self.all_data[item, 14:15]
        # img_col = self.all_data[item, 15:-1]
        return self.all_data[item]#img_pt, vec_top, vec_bot, vec_angle, sun_angle, img_col

    def get_id(self, item):
        return self.img_ids[item]