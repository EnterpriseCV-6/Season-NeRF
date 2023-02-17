import numpy as np
from misc import get_output_loc_lin_first
from mg_run_NeRF import Net_tool
from .T_NeRF_net_v2 import T_NeRF
import torch as t
# from .Eval_Tools_3_approx_solar import *
from .Eval_Tools_2 import *
from robust_loss_pytorch import AdaptiveLossFunction
from itertools import chain

class T_NeRF_Net_Tool(Net_tool):
    def __init__(self, args, training_DSM, GT_DSM, device, H, WC):
        super(T_NeRF_Net_Tool, self).__init__(args, device, training_DSM, GT_DSM, init_network=False, has_weight_term=True)
        n_steps = args.max_train_steps
        # section_starts = np.linspace(0, n_steps, 4, endpoint=False, dtype=int)
        # output_points = get_output_loc_lin_first(section_starts[1], args.n_saves // 4, min_gap=1000)

        # if args.jump_start:
        #     ps = [0.2, 0.0, 0.0]
        # else:
        #     ps = [0.0, 0.0, 0.0]

        ps = [0.2, 0.0, 0.0]
        ps.append(1-np.sum(ps))

        p1 = int(ps[0] * n_steps)
        p2 = int(ps[1] * n_steps)
        p3 = int(ps[2] * n_steps)
        p4 = n_steps - p3 - p2 - p1#int(.4 * n_steps)
        pi = [p1, p2, p3, p4]

        self.section_starts = np.array([0, p1, p1+p2, p1+p2+p3])
        self.section_Ends = np.array([p1, p1+p2, p1+p2+p3, n_steps])

        # print(section_starts)
        # exit()

        # self.n_steps = n_steps
        # self.section_starts = section_starts
        self.Section_Steps = []
        # self.Section_Ends = []
        for i in range(self.section_starts.shape[0]-1):
            self.Section_Steps.append(int(self.section_starts[i+1] - self.section_starts[i]))
        #     self.Section_Ends.append(section_starts[i+1])
        self.Section_Steps.append(int(n_steps - self.section_starts[-1]))
        # self.Section_Ends.append(int(n_steps))
        self.sub_section_outputs = []
        for i in range(self.section_starts.shape[0]):
            output_points = get_output_loc_lin_first(pi[i], int(args.n_saves*ps[i]), min_gap=1000)
            self.sub_section_outputs.append(self.section_starts[i] + output_points)
        self.sub_section_outputs[-1][-1] = n_steps

        self.learning_mode = -1
        self.network = T_NeRF(args.fc_units, n_classes=args.number_low_frequency_cases, HM=training_DSM).to(self.device)
        self.args = args
        self.lr = args.lr
        self.H = H
        self.WC = WC

        # print(self.train_data["Color_Loader"].solar_vecs)
        # exit()

    def reset_eval(self):
        alphi_hi = 2.99# 2.001
        scale_init = .03 #1.0
        if self.args.Use_MSE_loss:
            ada_loss = None
        elif self.learning_mode == 1:
            ada_loss = AdaptiveLossFunction(3, t.float32, self.device, alpha_hi=alphi_hi, alpha_init=2.0, scale_init=scale_init, scale_lo=0.01)
        else:
            try:
                alpha_start = t.mean(self.eval_tool.ada_loss[0].alpha()).item()
                scale_start = t.mean(self.eval_tool.ada_loss[0].scale()).item()
            except:
                print("WARNING: Unable to load alpha and scale start, using default")
                alpha_start = 2.0
                scale_start = scale_init
            ada_loss = AdaptiveLossFunction(3, t.float32, self.device, alpha_hi=alphi_hi, alpha_init=alpha_start, scale_init=scale_start, scale_lo=0.01)

        if self.learning_mode == 1:
            print("Guided Classic Learning")
            more_ada_loss = AdaptiveLossFunction(1, t.float32, self.device, alpha_hi=alphi_hi, alpha_init=2.0, scale_init=0.5, scale_lo=0.05)
            if self.args.jump_start:
                self.eval_tool = All_in_One_Eval(self.args, self.device, self.section_Ends[self.learning_mode - 1],
                                             use_prior=True, ada_loss = [ada_loss, more_ada_loss], H=self.H, WC=self.WC,
                                                 base_solar_vecs=self.train_data["Color_Loader"].solar_vecs)
            else:
                self.eval_tool = All_in_One_Eval(self.args, self.device, self.section_Ends[self.learning_mode - 1],
                                                 use_prior=False, ada_loss=ada_loss, H=self.H, WC=self.WC,
                                                 base_solar_vecs=self.train_data["Color_Loader"].solar_vecs)
        elif self.learning_mode == 2:
            print("Classic Learning")
            self.eval_tool = All_in_One_Eval(self.args, self.device, self.section_Ends[self.learning_mode - 1],
                                             use_prior=False, ada_loss = ada_loss)
        elif self.learning_mode == 3:
            print("Classic and Seasonal Learning")
            self.eval_tool = All_in_One_Eval(self.args, self.device, self.section_Ends[self.learning_mode - 1],
                                             use_prior=False, ada_loss = ada_loss)
        elif self.learning_mode == 4:
            print("Classic and Seasonal Learning with Outliers")
            self.eval_tool = All_in_One_Eval(self.args, self.device, self.section_Ends[self.learning_mode - 1],
                                             use_prior=False, ada_loss = ada_loss, H=self.H, WC=self.WC, base_solar_vecs=self.train_data["Color_Loader"].solar_vecs)
        elif self.learning_mode == 5:
            print("Seasonal Learning with Outliers")
            print("Not Yet implemented")
            exit()
        else:
            print("Error: Invalid learning mode")
            exit()

        if self.args.Use_MSE_loss:
            self.optim = t.optim.Adam(self.network.parameters(), lr=self.args.lr)
        elif self.learning_mode == 1 and self.args.jump_start:
            self.optim = t.optim.Adam(self.network.parameters(), lr=self.args.lr)
            self.optim2 = t.optim.Adam(chain(self.eval_tool.ada_loss[0].parameters(), self.eval_tool.ada_loss[1].parameters()), lr=self.args.lr*self.args.lr_alpha_scale)
            # self.optim = t.optim.Adam(chain(self.network.parameters(), self.eval_tool.ada_loss[0].parameters(), self.eval_tool.ada_loss[1].parameters()),
            #                           lr=self.args.lr)
        else:
            # self.optim = t.optim.Adam(chain(self.network.parameters(), self.eval_tool.ada_loss.parameters()), lr=self.args.lr)
            self.optim = t.optim.Adam(self.network.parameters(),lr=self.args.lr)
            self.optim2 = t.optim.Adam(self.eval_tool.ada_loss.parameters(), lr=self.args.lr*self.args.lr_alpha_scale)

        self.sched = t.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.lr,
                                                     total_steps=self.Section_Steps[self.learning_mode-1],
                                                     base_momentum=0.85, max_momentum=0.95, cycle_momentum=False)
        if self.args.Use_MSE_loss == False:
            self.sched2 = t.optim.lr_scheduler.OneCycleLR(self.optim2, max_lr=self.lr * self.args.lr_alpha_scale,
                                                     total_steps=self.Section_Steps[self.learning_mode-1],
                                                     base_momentum=0.85, max_momentum=0.95, cycle_momentum=False)



    def step(self):
        mode = np.sum(self._step_count >= self.section_starts)
        if mode != self.learning_mode:
            self.learning_mode = mode
            self.reset_eval()
        data_dict = self.get_data(eval_mode=False)
        self.train_step(data_dict, self._step_count)
        self._step_count += 1
        if self._step_count in self.sub_section_outputs[mode - 1]:
            print("Evaluating step", self._step_count)
            data_dict = self.get_data(eval_mode=True)
            self.eval_step(data_dict, self._step_count - 1)
            self.eval_img(self._step_count - 1)