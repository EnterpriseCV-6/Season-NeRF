# coding=utf-8
from T_NeRF_Full_2.Eval_Tools_2 import All_in_One_Eval, create_solor_rays_uniform
import numpy as np
import torch as t
from all_NeRF.mg_unit_converter import az_el_2_vec, vec_2_az_el, world_angle_2_local_vec
from tqdm import tqdm


def encode_time(time_frac_year, time_frac_day = 0):
    ans = np.array([np.cos(time_frac_year * 2 * np.pi), np.sin(time_frac_year * 2 * np.pi),
                    np.cos(time_frac_day * 2 * np.pi), np.sin(time_frac_day * 2 * np.pi)])
    return ans

def _transform_output_dict(output_dict, out_img_size):
    out_img = np.zeros([out_img_size, out_img_size, 3])
    out_img_mask = np.zeros([out_img_size, out_img_size], dtype=bool)

    out_img[output_dict["XY"][:,0], output_dict["XY"][:,1]] = output_dict["Rendered_Col"].numpy()
    out_img_mask[output_dict["XY"][:, 0], output_dict["XY"][:, 1]] = True

    all_out_imgs = {"Col_Img":out_img}

    if "Solar_Vis" in output_dict.keys():
        out_img_shadow = np.zeros([out_img_size, out_img_size])
        prob_shadow = t.sum(output_dict["PS"] * output_dict["Solar_Vis"], 1)
        out_img_shadow[output_dict["XY"][:, 0], output_dict["XY"][:, 1]] = prob_shadow[:, 0]
        all_out_imgs["Shadow_Mask"] = out_img_shadow

    if "Est_Solar_Vis" in output_dict.keys():
        out_img_shadow = np.zeros([out_img_size, out_img_size])
        prob_shadow = t.sum(output_dict["PS"] * output_dict["Est_Solar_Vis"], 1)
        out_img_shadow[output_dict["XY"][:, 0], output_dict["XY"][:, 1]] = prob_shadow[:, 0]
        all_out_imgs["Estimated_Shadow_Mask"] = out_img_shadow

    return all_out_imgs, out_img_mask

def _transform_output_dict_for_dsm(output_dict, out_img_size):
    out_img = np.zeros([out_img_size[0], out_img_size[1], 1]) * np.NaN
    out_img[output_dict["XY"][:,0], output_dict["XY"][:,1]] = np.sum(output_dict["PS"].numpy() * np.linspace(1,-1, 96).reshape([1,-1,1]), 1)
    return out_img[:,:,0]

def eval_solar_output(solar_ray_dict):
    from matplotlib import pyplot as plt
    # for a_key in solar_ray_dict.keys():
    #     print(a_key)
    #     print(solar_ray_dict[a_key].shape)
    for i in range(solar_ray_dict["PE"].shape[0]):
        print(t.sigmoid(solar_ray_dict["Sky_Col"][i][0]))
        plt.subplot(1,2,1)
        plt.plot(solar_ray_dict["PV_Exact"][i].cpu().numpy())
        plt.plot(solar_ray_dict["Solar_Vis"][i].cpu().numpy())
        plt.legend(["Exact", "Approximate"])
        plt.subplot(1,2,2)
        plt.plot((solar_ray_dict["PV_Exact"][i].cpu() * solar_ray_dict["PE"][i].cpu()).numpy())
        plt.plot((solar_ray_dict["Solar_Vis"][i].cpu() * solar_ray_dict["PE"][i].cpu()).numpy())
        plt.legend(["Exact", "Approximate"])
        plt.show()



class Quick_Run_Net():
    def __init__(self, network, args, world_center_LLA, World_2_Local_H, device, max_input_size = 50000, use_tqdm = False, use_full_solar = True):
        self.eval_tool = All_in_One_Eval(args, device, 5, False, False, World_2_Local_H, world_center_LLA)
        self.network = network
        self.world_center_LLA = world_center_LLA
        self.W2L_H = World_2_Local_H
        self.n_samples = args.n_samples
        self.n_classes = args.number_low_frequency_cases
        self.use_tqdm = use_tqdm
        self.use_full_solar = use_full_solar

        if self.use_full_solar:
            self.max_input_size = int(np.sqrt(max_input_size // args.n_samples))
        else:
            self.max_input_size = max_input_size // args.n_samples

    def _get_input_dict(self, camera_el_az, solar_el_az, time_frac, out_img_size, region):
        if isinstance(out_img_size, tuple):
            X, Y = np.meshgrid(np.arange(0, out_img_size[0]), np.arange(0, out_img_size[1]), indexing="ij")
        else:
            X, Y = np.meshgrid(np.arange(0, out_img_size), np.arange(0, out_img_size), indexing="ij")
        XY = np.stack([X,Y],2).reshape([-1,2])

        if not isinstance(out_img_size, tuple):
            mids = np.concatenate([XY*2./(out_img_size-1)-1, np.zeros([XY.shape[0], 1])], 1)
        else:
            mids = np.concatenate([XY * 2. / (np.array([[out_img_size[0], out_img_size[1]]]) - 1) - 1, np.zeros([XY.shape[0], 1])], 1)
        if region is not None:
            mids[:, 0] = (mids[:, 0] + 1) / 2 * (region[1] - region[0]) + region[0]
            mids[:, 1] = (mids[:, 1] + 1) / 2 * (region[3] - region[2]) + region[2]
        cam_vec = world_angle_2_local_vec(camera_el_az[0], camera_el_az[1], self.world_center_LLA, self.W2L_H)
        tops = mids + cam_vec / cam_vec[2]
        bots = mids - cam_vec / cam_vec[2]

        good = np.all((bots <= 1) * (bots >= -1) * (tops <= 1) * (tops >= -1), 1)
        tops = t.tensor(tops[good]).float()
        bots = t.tensor(bots[good]).float()
        XY = XY[good]
        if not isinstance(out_img_size, tuple):
            XY[:,0] = out_img_size - XY[:,0] - 1

        solar_vec = t.tensor(np.stack([world_angle_2_local_vec(solar_el_az[0], solar_el_az[1], self.world_center_LLA, self.W2L_H)] * XY.shape[0], 1).T).float()
        # print(solar_vec)
        # exit()
        time_vecs = encode_time(time_frac)
        time_vecs = t.tensor(np.stack([time_vecs] * XY.shape[0], 1).T).float()

        ans_dict = {"Top":tops, "Bot":bots, "XY":XY, "Sun_Angle":solar_vec, "Time_Encoded":time_vecs}
        return ans_dict

    def _build_output_dict(self, n, rendered_col_only = False):
        n_samples = self.n_samples
        if rendered_col_only:
            output_dict = {"Rendered_Col":t.zeros([n, 3])}
        elif self.use_full_solar:
            output_dict = {"Rendered_Col": t.zeros([n, 3]),
                           "PE": t.zeros([n, n_samples, 1]),
                           "PV": t.zeros([n, n_samples, 1]),
                           "PS": t.zeros([n, n_samples, 1]),
                           "Solar_Vis": t.zeros([n, n_samples, 1]),
                           "Sky_Col": t.zeros([n, n_samples, 3]),
                           "Classes": t.zeros([n, n_samples, self.n_classes]),
                           "Adjust": t.zeros([n, n_samples, 3]),
                           "Est_Solar_Vis": t.zeros([n, n_samples, 1]),
                           "Col": t.zeros([n, n_samples, 3])}
        else:
            output_dict = {"Rendered_Col":t.zeros([n, 3]),
                           "PE":t.zeros([n, n_samples, 1]),
                           "PV":t.zeros([n, n_samples, 1]),
                           "PS": t.zeros([n, n_samples, 1]),
                           "Solar_Vis": t.zeros([n, n_samples, 1]),
                           "Sky_Col": t.zeros([n, n_samples, 3]),
                           "Classes": t.zeros([n, n_samples, self.n_classes]),
                           "Adjust": t.zeros([n, n_samples, 3]),
                           "Col": t.zeros([n, n_samples, 3])}
        return output_dict

    #can give a list of solar els and azs to check or give a number of random solar rays to generate (default)
    def solar_ray_acc_check(self, n_rays = 500, solar_el_and_az = None, H = None, cent = None, solar_vec = None):
        # from matplotlib import pyplot as plt
        # from all_NeRF.mg_unit_converter import LLA_get_vec
        test = create_solor_rays_uniform(H, cent)
        if solar_el_and_az is None and solar_vec is None:
            # from all_NeRF.mg_unit_converter import LLAs_get_angle_vec
            starts, ends, solar_angle_vec, Solar_Time, Solar_az_el = test(n_rays, include_times=True)

            # all_rho = []
            # for i in range(solar_angle_vec.shape[0]):
            #     A = np.linalg.inv(H) @ np.array([[solar_angle_vec[i,0], solar_angle_vec[i,1], solar_angle_vec[i,2], 1.]]).T
            #     A = ((A[0:3]/A[3::]).T)[0]
            #     theta, rho, error = LLAs_get_angle_vec(cent, A)
            #     all_rho.append(rho)
            #     # print(theta, rho)
            # plt.hist(all_rho)
            # plt.show()
            #
            # exit()
            # print()
            # exit()

            Solar_dict = {"Top": starts, "Bot": ends, "Sun_Angle": solar_angle_vec, "Time_Encoded": Solar_Time, "Sun_Angle_Az_El":Solar_az_el}
        elif solar_el_and_az is None:
            starts, ends, solar_angle_vec, Solar_Time = test.create_given_vec(n_rays, solar_vec, True)
            Solar_dict = {"Top": starts, "Bot": ends, "Sun_Angle": solar_angle_vec, "Time_Encoded": Solar_Time}
        else:
            print("Manual entry of solar rays as az el not yet implemented!")
            exit()
        with t.no_grad():
            Network_Output_Solar = self.eval_tool.eval_Rho_Only(Solar_dict, self.network, False)
        return Network_Output_Solar


    def render_img(self, camera_el_and_az, solar_el_and_az, time_frac, out_img_size, region=None):#, H, cent):
        # output = self.solar_ray_acc_check(5, H=H, cent=cent)
        # eval_solar_output(output)
        with t.no_grad():
            input_dict = self._get_input_dict(camera_el_and_az, solar_el_and_az, time_frac, out_img_size, region)
            output_dict = self._build_output_dict(input_dict["Top"].shape[0])
            for i in tqdm(range(0, input_dict["Top"].shape[0], self.max_input_size), leave=self.use_tqdm, desc="Rendering Image"):
                i_end = min(i + self.max_input_size, input_dict["Top"].shape[0])
                sub_dict =  {"Top":input_dict["Top"][i:i_end], "Bot":input_dict["Bot"][i:i_end],
                             "Sun_Angle":input_dict["Sun_Angle"][i:i_end], "Time_Encoded":input_dict["Time_Encoded"][i:i_end]}
                if self.use_full_solar == False:
                    sub_output_dict = self.eval_tool.eval(sub_dict, self.network, -1, False)
                else:
                    sub_output_dict = self.eval_tool.eval_exact_solar(sub_dict, self.network, -1, False)
                for a_key in output_dict.keys():
                    output_dict[a_key][i:i_end] = sub_output_dict[a_key]
            # else:
            #     for i in range(0, input_dict["Top"].shape[0], self.max_input_size):
            #         i_end = min(i + self.max_input_size, input_dict["Top"].shape[0])
            #         sub_dict = {"Top": input_dict["Top"][i:i_end], "Bot": input_dict["Bot"][i:i_end],
            #                     "Sun_Angle": input_dict["Sun_Angle"][i:i_end],
            #                     "Time_Encoded": input_dict["Time_Encoded"][i:i_end]}
            #         if self.use_full_solar == False:
            #             sub_output_dict = self.eval_tool.eval(sub_dict, self.network, -1, False)
            #         else:
            #             sub_output_dict = self.eval_tool.eval_exact_solar(sub_dict, self.network, -1, False)
            #         for a_key in output_dict.keys():
            #             output_dict[a_key][i:i_end] = sub_output_dict[a_key]
            output_dict["XY"] = input_dict["XY"]
            out_img, mask = _transform_output_dict(output_dict, out_img_size)

        #out_img Keys: {"Col_Img": out_img, "Shadow_Mask", "Estimated_Shadow_Mask"}, Estimated shadow mask only availble if exact mask used
        return out_img, mask

    def get_DSM(self, out_img_size, region=None):
        camera_el_and_az = [90, 0]
        solar_el_and_az = [90,0]
        time_frac = 0.0
        with t.no_grad():
            input_dict = self._get_input_dict(camera_el_and_az, solar_el_and_az, time_frac, out_img_size, region)
            output_dict = self._build_output_dict(input_dict["Top"].shape[0])
            for i in tqdm(range(0, input_dict["Top"].shape[0], self.max_input_size), leave=self.use_tqdm,
                          desc="Rendering Image"):
                i_end = min(i + self.max_input_size, input_dict["Top"].shape[0])
                sub_dict = {"Top": input_dict["Top"][i:i_end], "Bot": input_dict["Bot"][i:i_end],
                            "Sun_Angle": input_dict["Sun_Angle"][i:i_end],
                            "Time_Encoded": input_dict["Time_Encoded"][i:i_end]}
                sub_output_dict = self.eval_tool.eval(sub_dict, self.network, -1, False)
                for a_key in output_dict.keys():
                    output_dict[a_key][i:i_end] = sub_output_dict[a_key]

            output_dict["XY"] = input_dict["XY"]
            out_img = _transform_output_dict_for_dsm(output_dict, out_img_size)

        return out_img