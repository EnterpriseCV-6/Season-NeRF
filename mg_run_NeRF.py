import hsluv
from tqdm import tqdm
import numpy as np
import torch as t
from torch.utils.tensorboard import SummaryWriter
from misc import sample_pt_coarse
import cv2 as cv
from matplotlib import pyplot as plt
from copy import deepcopy
import NN_loaders


def build_loader(args, img_list, include_ortho, train_mode):
    loader = {"Color_Loader":NN_loaders.setup_col_loader(img_list, args, include_ortho, train_mode)}
    if args.sc_lambda != 0:
        loader["SC_Loader"] = NN_loaders.setup_SC_loader(len(loader["Color_Loader"]))

    return loader

def build_data_loaders(args):
    print("Building Data Loaders...")
    fin = open(args.logs_dir + "/Training_Imgs.txt", "r")
    training_imgs = fin.readlines()
    training_imgs = [lin.rstrip() for lin in training_imgs]
    fin.close()
    fin = open(args.logs_dir + "/Testing_Imgs.txt", "r")
    val_imgs = fin.readlines()
    val_imgs = [lin.rstrip() for lin in val_imgs]
    fin.close()

    # print("WARNING: Using testing sets twice!")
    train_loader, val_loader = build_loader(args, training_imgs, include_ortho=False, train_mode=True), build_loader(args, val_imgs, include_ortho=True, train_mode=False)
    print("Done.")
    return train_loader, val_loader

class Net_tool():
    def __init__(self, args, device, training_DSM, GT_DSM, init_network = True, has_weight_term = False):
        self.args = args
        self.num_rays_to_eval = args.chunk
        self.num_course_samples = args.n_samples
        self.num_fine_samples = args.n_importance
        self.n_saves = args.n_saves
        self.n_steps = args.max_train_steps
        self.batch_size = args.batch_size
        self.n_DSM_samples = args.n_samples + args.n_importance
        self.has_weight_term = has_weight_term

        self.use_solar = args.sc_lambda > 0

        self._step_count = 0
        self.save_points = self.get_output_loc_lin_first(args.max_train_steps, args.n_saves, 500)

        self.training_DSM, self.GT_DSM = training_DSM, GT_DSM

        self.training_DSM_Dense = np.zeros([self.training_DSM.shape[0], self.training_DSM.shape[1], self.n_DSM_samples])
        self.GT_DSM_Dense = np.zeros([self.GT_DSM.shape[0], self.GT_DSM.shape[1], self.n_DSM_samples])
        i = 0
        for h in np.linspace(-1,1, self.n_DSM_samples):
            self.training_DSM_Dense[:,:,i] = (self.training_DSM >= h) + self.training_DSM*0 #Adding DSM * 0 ensures NaN correctly accounted for
            self.GT_DSM_Dense[:,:,i] = (self.GT_DSM >= h) + self.GT_DSM*0 #Adding DSM * 0 ensures NaN correctly accounted for
            i += 1


        # plt.imshow(np.sum(self.GT_DSM_Dense, 2))
        # plt.show()

        self.training_DSM_Dense = t.tensor(self.training_DSM_Dense)
        self.GT_DSM_Dense = t.tensor(self.GT_DSM_Dense)

        # self.training_DS = args.img_training_downscale
        # self.GT_DS = args.img_validation_downscale


        self.train_data, self.val_data = build_data_loaders(args)
        self.train_loader, self.val_loader = {}, {}
        self.iter_train_loader, self.iter_val_loader = {}, {}

        for a_key in self.train_data.keys():
            self.train_loader[a_key] = t.utils.data.DataLoader(self.train_data[a_key], batch_size = args.batch_size, shuffle=True, num_workers=4)
            self.val_loader[a_key] = t.utils.data.DataLoader(self.val_data[a_key], batch_size=args.batch_size, shuffle=True, num_workers=4)
            self.iter_train_loader[a_key] = iter(self.train_loader[a_key])
            self.iter_val_loader[a_key] = iter(self.val_loader[a_key])


        self.device = device
        if init_network:
            pass
            # self.network, self.eval_tool = build_network(args, device, self.training_DSM_Dense)
            # self.optim = t.optim.Adam(self.network.parameters(), lr=1.0)
            # self.sched = t.optim.lr_scheduler.OneCycleLR(self.optim, max_lr = args.lr, total_steps=self.n_steps,
            #                                              base_momentum = 0.05, max_momentum=0.15, cycle_momentum = False)

        self.writer = SummaryWriter(args.logs_dir, comment=args.exp_name)
        self.GT_Cache = None
        self.optim2 = None
        self.sched2 = None

    def _scale_to_DSM(self, pts, use_GT):
        if use_GT:
            K = t.tensor([self.GT_DSM.shape[0]-1, self.GT_DSM.shape[1]-1, self.n_DSM_samples-1], device=pts.device).reshape([1,1,3])
        else:
            K = t.tensor([self.training_DSM.shape[0]-1, self.training_DSM.shape[1]-1, self.n_DSM_samples-1], device=pts.device).reshape(
                [1, 1, 3])
        return ((pts+1)/2 * K).type(t.long)

    def get_Dist(self, top, bot):
        pts, delta = sample_pt_coarse(top, bot, self.n_DSM_samples, eval_mode=True)
        pts_GT_DSM = self._scale_to_DSM(pts, True).cpu().reshape([-1, 3])
        PE_GT = self.GT_DSM_Dense[pts_GT_DSM[:,0], pts_GT_DSM[:,1], pts_GT_DSM[:,2]].reshape([-1, self.n_DSM_samples, 1])
        Prob_Surf = PE_GT * t.cumprod(t.cat([t.ones([PE_GT.shape[0],1,1]), 1-PE_GT], 1), 1)[:,0:-1]
        Surf_Loc_GT = t.sum(Prob_Surf * t.cumsum(delta, 1), 1)/t.sum(Prob_Surf, 1)

        pts_Prior_DSM = self._scale_to_DSM(pts, False).cpu().reshape([-1, 3])
        PE_GT = self.training_DSM_Dense[pts_Prior_DSM[:, 0], pts_Prior_DSM[:, 1], pts_Prior_DSM[:, 2]].reshape(
            [-1, self.n_DSM_samples, 1])
        Prob_Surf = PE_GT * t.cumprod(t.cat([t.ones([PE_GT.shape[0], 1, 1]), 1 - PE_GT], 1), 1)[:, 0:-1]
        # Surf_Loc_Prior = t.sum(Prob_Surf * pts, 1)
        Surf_Loc_Prior = t.sum(Prob_Surf * t.cumsum(delta, 1), 1)/t.sum(Prob_Surf, 1)

        return Surf_Loc_GT, Surf_Loc_Prior

    def data_to_dict(self, data):
        data_dict = {}
        data_dict["Img_Pt"] = data[:, 0:2]
        data_dict["Top"] = data[:, 2:5]
        data_dict["Bot"] = data[:, 5:8]
        data_dict["View_Angle"] = data[:, 8:11]
        data_dict["Sun_Angle"] = data[:, 11:14]
        data_dict["Time_Encoded"] = data[:, 14:18]
        data_dict["Sample_Weight"] = data[:, 18:19]
        data_dict["GT_Color"] = data[:, 19::]

        return data_dict


    def step(self):
        data_dict = self.get_data(eval_mode=False)
        self.train_step(data_dict, self._step_count)

        self._step_count += 1
        if self._step_count in self.save_points or self._step_count == 1:
            print("Evaluating step", self._step_count)
            data_dict = self.get_data(eval_mode=True)
            self.eval_step(data_dict, self._step_count-1)
            self.eval_img(self._step_count-1)
            # exit()

    def eval_img(self, step_count):
        self.network.eval()
        with t.no_grad():
            update_GT_cache = False
            n_pts = len(self.val_data["Color_Loader"])
            BS = self.args.chunk // (self.args.n_samples + self.args.n_importance)
            out_val_images = np.zeros([len(self.val_data["Color_Loader"].full_img_size)]+ list(self.val_data["Color_Loader"].full_img_size[0]))
            out_val_hm = np.zeros([len(self.val_data["Color_Loader"].full_img_size)] + list(self.val_data["Color_Loader"].full_img_size[0][0:2]))
            out_val_MAE = np.zeros([len(self.val_data["Color_Loader"].full_img_size)] + list(
                self.val_data["Color_Loader"].full_img_size[0][0:2]))

            if self.GT_Cache is None:
                self.GT_Cache = np.zeros_like(out_val_images)
                self.GT_Cache_n = np.zeros(out_val_images.shape[0])
                update_GT_cache = True

            for i in tqdm(range(0,n_pts, BS)):
                all_ids = []
                all_data = []
                for j in range(i, min(n_pts, i + BS)):
                    all_ids.append(self.val_data["Color_Loader"].get_id(j))
                    if j == i:
                        all_data = self.val_data["Color_Loader"][j].unsqueeze(0)
                    else:
                        all_data = t.cat([all_data,  self.val_data["Color_Loader"][j].unsqueeze(0)], 0)
                data_dict = self.data_to_dict(all_data)
                data_dict["Dist_to_Surf_GT"], data_dict["Dist_to_Surf_Prior"] = self.get_Dist(data_dict["Top"],
                                                                                              data_dict["Bot"])
                # Sigma_merged, Col, Solar_Vis, Sky_Col, P_E, P_Vis, P_Surf, sample_pts, deltas, Sigma, Sigma_alpha = self.eval_tool.eval(data_dict, self.network, current_step=self.args.max_train_steps, train_mode=False)
                # if self.args.sc_lambda > 0:
                #     rendered_col = t.sum(P_Surf * Col * (Solar_Vis + Sky_Col * (1-Solar_Vis)), 1)
                # else:
                #     rendered_col = t.sum(P_Surf * Col, 1)
                Results_Network = self.eval_tool.eval(
                    data_dict, self.network, current_step=self.args.max_train_steps, train_mode=False)
                P_Surf = Results_Network["PS"]
                deltas = Results_Network["deltas"]
                sample_pts = Results_Network["sample_pts"].to(self.device)
                rendered_col = Results_Network["Rendered_Col"]

                Expected_Surface_Location = t.sum(P_Surf * sample_pts, 1)/(t.sum(P_Surf, 1) + 1e-8)
                Expected_Surface_Dist = t.sum(t.cumsum(deltas.to(self.device), 1) * P_Surf, 1) / t.sum(P_Surf, 1)
                a_MAE = t.abs(data_dict["Dist_to_Surf_GT"].to(self.device) - Expected_Surface_Dist)

                img_pt = data_dict["Img_Pt"].int().cpu().numpy()
                out_val_images[all_ids, img_pt[:,0], img_pt[:,1]] = rendered_col.cpu().numpy()
                out_val_hm[all_ids, img_pt[:,0], img_pt[:,1]] = Expected_Surface_Location[:,2].cpu().numpy()
                out_val_MAE[all_ids, img_pt[:, 0], img_pt[:, 1]] = a_MAE.cpu().numpy()[:,0]
                if update_GT_cache:
                    self.GT_Cache[all_ids, img_pt[:, 0], img_pt[:, 1]] = data_dict["GT_Color"].numpy()
            out_val_hm = (out_val_hm + 1)/2
            img_error = 0
            for i in range(out_val_hm.shape[0]):
                # out_img = cv.applyColorMap((out_val_hm[i]*255).astype(np.uint8), cv.COLORMAP_JET).astype(float)/255
                out_img = out_val_hm[i]
                self.writer.add_image("HM/Img_" + self.val_data["Color_Loader"].img_names[i], np.expand_dims(out_img,0), step_count)
                if i != out_val_hm.shape[0]-1:
                    out_img = np.moveaxis(np.concatenate([self.GT_Cache[i], out_val_images[i]], 1), -1, 0)
                    if update_GT_cache:
                        self.GT_Cache_n[i] = np.sum(np.any(self.GT_Cache[i] != 0, 2))*3
                    img_error += np.sum(np.log(1/2*(self.GT_Cache[i] - out_val_images[i])**2+1))/self.GT_Cache_n[i]

                else:
                    out_img = np.moveaxis(out_val_images[i], -1, 0)
                    a_MAE = out_val_MAE[i]
                    a_MAE = np.mean(a_MAE[a_MAE == a_MAE])
                    self.writer.add_scalar("Testing/Mean_Height_Error", a_MAE, step_count)
                if self.args.use_HSLuv:
                    out_img = out_img * np.array([360., 100, 100]).reshape([3, 1, 1])
                    for x_idx in range(out_img.shape[1]):
                        for y_idx in range(out_img.shape[2]):
                            out_img[:, x_idx, y_idx] = hsluv.hsluv_to_rgb(out_img[:, x_idx, y_idx])

                self.writer.add_image(("Col/Img_" + self.val_data["Color_Loader"].img_names[i]), out_img, step_count)

        img_error = img_error / (out_val_hm.shape[0]-1)
        self.writer.add_scalar("Testing/Overall_Cauchy_Color_Error", img_error, step_count)
        t.save(self.network.state_dict(), self.args.logs_dir + "/Model_" + str(step_count) + ".nn")
        self.network.train()
        # print(self.network.G_NeRF_net.fc1.norm.weight)

    def get_data(self, eval_mode = False):
        is_eof = False
        if eval_mode == False:
            try:
                data = next(self.iter_train_loader["Color_Loader"])
            except StopIteration:
                self.iter_train_loader["Color_Loader"] = iter(self.train_loader["Color_Loader"])
                data = next(self.iter_train_loader["Color_Loader"])
                is_eof = True
        else:
            try:
                data = next(self.iter_val_loader["Color_Loader"])
            except StopIteration:
                self.iter_val_loader["Color_Loader"] = iter(self.val_loader["Color_Loader"])
                data = next(self.iter_val_loader["Color_Loader"])
                is_eof = True
        data_dict = self.data_to_dict(data)

        if self.use_solar:
            if eval_mode == False:
                try:
                    data_dict["SC_Top"], data_dict["SC_Bot"] = next(self.iter_train_loader["SC_Loader"])
                except StopIteration:
                    self.iter_train_loader["SC_Loader"] = iter(self.train_loader["SC_Loader"])
                    data_dict["SC_Top"], data_dict["SC_Bot"] = next(self.iter_train_loader["SC_Loader"])
                    is_eof = True
            else:
                try:
                    data_dict["SC_Top"], data_dict["SC_Bot"] = next(self.iter_val_loader["SC_Loader"])
                except StopIteration:
                    self.iter_val_loader["SC_Loader"] = iter(self.val_loader["Color_Loader"])
                    data_dict["SC_Top"], data_dict["SC_Bot"] = next(self.iter_val_loader["SC_Loader"])
                    is_eof = True

        data_dict["Dist_to_Surf_GT"], data_dict["Dist_to_Surf_Prior"] = self.get_Dist(data_dict["Top"], data_dict["Bot"])
        return data_dict

    def check_data_dict(self, eval_mode = False):
        # print(self.train_data["Color_Loader"].keys())
        if eval_mode:
            all_data = self.data_to_dict(self.val_data["Color_Loader"].all_data)
            img_ids = self.val_data["Color_Loader"].img_ids
        else:
            all_data = self.data_to_dict(self.train_data["Color_Loader"].all_data)
            img_ids = self.train_data["Color_Loader"].img_ids
        print(all_data.keys())

        img_size = int(t.max(all_data["Img_Pt"])+1)
        training_imgs = t.zeros([max(img_ids)+1, img_size, img_size, 3])
        for i in tqdm(range(all_data["Img_Pt"].shape[0])):
            training_imgs[img_ids[i], int(all_data["Img_Pt"][i][0]), int(all_data["Img_Pt"][i][1])] = all_data["GT_Color"][i]

        for i in range(training_imgs.shape[0]):
            plt.imshow(training_imgs[i].numpy())
            plt.show()

        exit()


    def train_step(self, data_dict, current_step):
        self.optim.zero_grad()
        if self.optim2 is not None:
            self.optim2.zero_grad()
        loss = self.eval_tool.get_loss(data_dict, self.network, current_step, train_mode = True)
        total_loss = 0
        # quit_break_bad = False
        for a_key in loss.keys():
            # print(a_key, loss[a_key])
            # if loss[a_key] != loss[a_key]:
            #     quit_break_bad = True
            if self.has_weight_term == False:
                total_loss += loss[a_key]
                self.writer.add_scalar("Training/" + a_key, loss[a_key].item(), current_step)
            else:
                # print(a_key)
                # print(loss[a_key])
                total_loss += loss[a_key][0] * loss[a_key][1]
                self.writer.add_scalar("Training/" + a_key, loss[a_key][0].item(), current_step)
                if self.args.use_auto_balance:
                    self.writer.add_scalar("Training/" + a_key + "_weight", loss[a_key][1], current_step)
        # if quit_break_bad:
        #     exit()

        total_loss.backward()
        # print(self._step_count)
        # print(self.network.get_class_layer.weight)
        # print(self.network.get_class_layer.weight.grad)

        # print(self.eval_tool.ada_loss[0].latent_alpha)
        # print(self.eval_tool.ada_loss[0].latent_alpha.grad)

        self.optim.step()
        if self.optim2 is not None:
            self.optim2.step()
        self.sched.step()
        if self.sched2 is not None:
            self.sched2.step()
        self.writer.add_scalar("LR/Learning_Rate", self.sched.get_last_lr()[0], current_step)

    def eval_step(self, data_dict, current_step):
        with t.no_grad():
            self.network.eval()
            loss = self.eval_tool.get_loss(data_dict, self.network, current_step, train_mode=False)
            self.network.train()
            if self.has_weight_term == False:
                for a_key in loss.keys():
                    self.writer.add_scalar("Testing/" + a_key, loss[a_key].item(), current_step)
            else:
                for a_key in loss.keys():
                    self.writer.add_scalar("Testing/" + a_key, loss[a_key][0].item(), current_step)

    def get_num_epochs(self):
        return  (self.n_steps * self.batch_size) / len(self.train_data["Color_Loader"])

    def get_output_loc(self, n_steps, n_outputs):
        if n_outputs > 0:
            alpha = np.log(n_steps) / np.log(n_outputs)
            ans = (np.arange(1, n_outputs + 1) ** alpha).astype(int)
            ans[-1] = n_steps
        else:
            ans = np.array([n_steps])
        return ans

    def get_output_loc_lin_first(self, n_steps, n_outputs, min_gap):
        if n_outputs * min_gap >= n_steps:
            ans = np.linspace(1, n_steps, n_outputs, dtype=int)
        else:
            ans = self.get_output_loc(n_steps, n_outputs)
            lin = np.arange(n_outputs) * min_gap + 1
            ans = np.maximum(ans, lin)

        return ans

def run_NeRF(args, device, training_DSM, GT_DSM):
    n_steps = args.max_train_steps
    net_tool = Net_tool(args, device, training_DSM, GT_DSM)
    print("Number Training Steps:", n_steps)
    print("Number Epochs:", np.round(net_tool.get_num_epochs(), 3))
    print("Save Points:", net_tool.save_points)
    for i in tqdm(range(n_steps)):
        net_tool.step()
    t.save(net_tool.network.state_dict(), args.logs_dir + "/Final_Model.nn")