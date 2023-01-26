# coding=utf-8
# coding=utf-8

import torch as t
from misc import sample_pt_coarse
import numpy as np
from all_NeRF.basic_functions import get_Surface_Strength_Reg_loss
from all_NeRF.basic_functions import general_loss
from all_NeRF.mg_unit_converter import world_angle_2_local_vec
from .Eval_Tools_2 import get_PV


def eval_network(Network, data_dict, train_mode, n_pts, device):
    with t.no_grad():
        Xs, deltas = sample_pt_coarse(data_dict["Top"], data_dict["Bot"], n_pts, not train_mode)
        deltas = deltas.to(device)
        Sun_Vec = t.ones_like(Xs) * data_dict["Sun_Angle"].unsqueeze(1)
        Times_Vec = t.ones(Xs.shape[0], Xs.shape[1], 4, dtype=Xs.dtype, device=Xs.device) * data_dict["Time_Encoded"].unsqueeze(1)

        Rho = Network.G_NeRF_net.forward_Sigma_Only(Xs.reshape([-1, 3]).to(device)).reshape([Xs.shape[0], Xs.shape[1], 1])
        PV = get_PV(Rho, deltas)
        PE = 1 - t.exp(-Rho * deltas)
        PS = PV * PE
        # vals, args = t.topk(PS, n_solar_rays, 1)
        # dir = data_dict["Bot"] - data_dict["Top"]
        # dir = 2*dir/(dir[:,-1::])
        # Solar_bot = t.unsqueeze(data_dict["Top"], 1) + t.unsqueeze(dir, 1) * args.cpu() / n_pts

        most_likely_surface = t.argmax(PS, 1).cpu()
        most_likely_surface = t.cat([t.arange(0, Xs.shape[0]).reshape([-1,1]), most_likely_surface], 1)

        Solar_Bot = Xs[most_likely_surface[:,0], most_likely_surface[:,1]]
        ts = (1 - Solar_Bot[:,2]) / (data_dict["Sun_Angle"][:,2])
        Solar_Top = Solar_Bot + data_dict["Sun_Angle"] * t.unsqueeze(ts, 1)

        Xs_solar, deltas_solar = sample_pt_coarse(Solar_Top, Solar_Bot, n_pts, not train_mode, include_end_pt=False)


    Rho, Rho_Solar, Base_Col, Classes, Adjust = Network.approx_Solar(Xs.reshape([-1, 3]).to(device), Xs_solar.reshape([-1,3]).to(device), Times_Vec.reshape([-1, 4]).to(device))
    Base_Col = Base_Col.reshape([Xs.shape[0], Xs.shape[1], -1])
    Rho = Rho.reshape([Xs.shape[0], Xs.shape[1], 1])
    Classes = Classes.reshape([Xs.shape[0], Xs.shape[1], -1])
    Adjust = Adjust.reshape([Xs.shape[0], Xs.shape[1], -1])
    Rho_Solar = Rho_Solar.reshape([Xs_solar.shape[0], Xs_solar.shape[1], -1])

    Solar_Vis = t.exp(-t.sum(Rho_Solar * deltas_solar.to(device), 1))
    Sky_Col = .2


    return Rho, Base_Col, Solar_Vis, Sky_Col, Classes, Adjust, Xs, deltas

class All_in_One_Eval():
    def __init__(self, args, device, n_steps, use_prior, ada_loss, H, WC, base_solar_vecs = None):
        self.device = device
        self.args = args
        self.n_steps = n_steps
        self.MSE_loss = t.nn.MSELoss()
        self.smooth_L1 = t.nn.SmoothL1Loss()
        self.use_prior = use_prior
        self.use_reg = args.Use_Reg
        self.use_MSE_loss = args.Use_MSE_loss
        self.ada_loss = ada_loss
        self.Sigmoid = t.nn.Sigmoid()
        self.BCE_loss = t.nn.BCELoss()

    def full_eval(self, data_dict, Network, current_step):
        n_pts = self.args.n_samples
        device = self.device

        Xs, deltas = sample_pt_coarse(data_dict["Top"], data_dict["Bot"], n_pts, True)
        deltas = deltas.to(device)
        Sun_Vec = t.ones_like(Xs) * data_dict["Sun_Angle"].unsqueeze(1)
        Times_Vec = t.ones(Xs.shape[0], Xs.shape[1], 4, dtype=Xs.dtype, device=Xs.device) * data_dict[
            "Time_Encoded"].unsqueeze(1)
        Rho, Base_Col, Solar_Vis, Sky_Col, Classes, Adjust = Network(Xs.reshape([-1, 3]).to(device),
                                                                     Sun_Vec.reshape([-1, 3]).to(device),
                                                                     Times_Vec.reshape([-1, 4]).to(device))

        Base_Col = Base_Col.reshape([Xs.shape[0], Xs.shape[1], -1])
        Rho = Rho.reshape([Xs.shape[0], Xs.shape[1], 1])
        Solar_Vis = Solar_Vis.reshape([Xs.shape[0], Xs.shape[1], 1])
        Sky_Col = Sky_Col.reshape([Xs.shape[0], Xs.shape[1], -1])
        Classes = Classes.reshape([Xs.shape[0], Xs.shape[1], -1])
        Adjust = Adjust.reshape([Xs.shape[0], Xs.shape[1], -1])

        # Solar_Vis2 = self.Sigmoid((Solar_Vis-.2)*12.)

        PV = get_PV(Rho, deltas)
        PE = 1 - t.exp(-Rho * deltas)
        PS = PV * PE

        Solar_Vis3 = self.Sigmoid((t.sum(Solar_Vis.detach() * PS, 1) - .2) * 30)
        Rendered_Col = t.sum(PS * self.Sigmoid(Base_Col), 1) * (Solar_Vis3 + (1 - Solar_Vis3) * t.mean(Sky_Col, 1))

        Results = {"Rendered_Col": Rendered_Col, "PE": PE, "PV": PV, "PS": PS, "Solar_Vis": Solar_Vis,
                   "Sky_Col": Sky_Col,
                   "Classes": Classes, "Adjust": Adjust, "Rho": Rho, "Col": Base_Col,
                   "deltas": deltas, "sample_pts": Xs}
        return Results

    def eval(self, data_dict, Network, current_step, train_mode):
        n_pts = self.args.n_samples
        device = self.device

        Rho, Base_Col, Solar_Vis, Sky_Col, Classes, Adjust, Xs, deltas = eval_network(Network, data_dict, train_mode, n_pts, device)

        # Solar_Vis2 = self.Sigmoid((Solar_Vis-.2)*12.)

        PV = get_PV(Rho, deltas)
        PE = 1-t.exp(-Rho * deltas)
        PS = PV * PE

        Col_Adj = -1

        Solar_Vis3 = self.Sigmoid((Solar_Vis-.2)*30)
        Albedo_Color = t.sum(PS * Base_Col, 1)
        Rendered_Col = Albedo_Color * (Solar_Vis3 + (1-Solar_Vis3) * Sky_Col)


        # if self.use_prior:
        #     Model_Trust = current_step / self.n_steps
        #     Rho_Supervised = Network.Supervised_Sample(Xs.reshape([-1, 3]).cpu(), deltas.reshape(-1, 1).cpu())
        #     Rho_Supervised = Rho_Supervised.reshape([Xs.shape[0], Xs.shape[1], 1]).to(device)
        #
        #     PV_Supervised = get_PV(Rho_Supervised, deltas)
        #     PE_Supervised = 1-t.exp(-Rho_Supervised * deltas)
        #     PS_Supervised = PV_Supervised * PE_Supervised
        #     # Rendered_Col_Supervised = t.sum(PS_Supervised * Col_Adj, 1)
        #     Rendered_Col_Supervised = t.sum(PS_Supervised * Base_Col, 1) * (Solar_Vis3 + (1 - Solar_Vis3) * t.mean(Sky_Col, 1))
        #
        #     Rho_Merged = Rho * Model_Trust + Rho_Supervised * (1 - Model_Trust)
        #     PV_Merged = get_PV(Rho_Merged, deltas)
        #     PE_Merged = 1 - t.exp(-Rho_Merged * deltas)
        #     PS_Merged = PV_Merged * PE_Merged
        #     # Rendered_Col_Merged = t.sum(PS_Merged * Col_Adj, 1)
        #     Albedo_Color = t.sum(PS_Merged * Base_Col, 1)
        #     Rendered_Col_Merged = Albedo_Color * (Solar_Vis3 + (1 - Solar_Vis3) * t.mean(Sky_Col, 1))
        #
        #     Results = {"Rendered_Col": Rendered_Col, "PE": PE, "PV": PV, "PS": PS,
        #                "Solar_Vis": Solar_Vis, "Sky_Col": Sky_Col, "Classes": Classes, "Adjust": Adjust,
        #                "Rho": Rho, "Col": Base_Col, "Col_Adj": Col_Adj,
        #                "PV_Supervised":PV_Supervised, "PE_Supervised":PE_Supervised, "PS_Supervised":PS_Supervised, "Rendered_Col_Supervised":Rendered_Col_Supervised,
        #                "PV_Merged": PV_Merged, "PE_Merged": PE_Merged, "PS_Merged": PS_Merged,
        #                "Rendered_Col_Merged": Rendered_Col_Merged, "Rho_Merged":Rho_Merged, "deltas":deltas, "sample_pts":Xs, "Albedo_Color":Albedo_Color}
        # else:
        Results = {"Rendered_Col":Rendered_Col, "PE":PE, "PV":PV, "PS":PS, "Solar_Vis":Solar_Vis, "Sky_Col":Sky_Col,
                   "Classes":Classes, "Adjust":Adjust, "Rho":Rho, "Col":Base_Col, "Col_Adj":Col_Adj, "deltas":deltas, "sample_pts":Xs, "Albedo_Color":Albedo_Color}
        return Results



    def get_loss(self, data_dict, Network, current_step, train_mode):
        n_rays = data_dict["Top"].shape[0]
        n_pts = self.args.n_samples
        device = self.device
        Loss = {}
        weight = {"Color": 1.0, "Solar_Correction": self.args.sc_lambda, "Alpha_Adjust": 1.}
        # entropy_adj = np.log(2) / np.log(self.args.number_low_frequency_cases)
        Network_Output = self.eval(data_dict, Network, current_step, train_mode)


        if self.use_MSE_loss == True:
            if self.use_prior and train_mode:
                Loss_Color = self.MSE_loss(Network_Output["Rendered_Col_Merged"], data_dict["GT_Color"].to(device))
            else:
                Loss_Color = self.MSE_loss(Network_Output["Rendered_Col"], data_dict["GT_Color"].to(device))
            Loss["Color"] = [Loss_Color, weight["Color"]]
            if self.use_prior:
                Loss["Alpha_Adjust"] = [self.MSE_loss(Network_Output["PE"], Network_Output["PE_Supervised"].detach()), weight["Alpha_Adjust"]]
        else:
            Col_diff = Network_Output["Rendered_Col"] - data_dict["GT_Color"].to(device)
            if self.use_prior:
                alpha_diff = (Network_Output["PE"] - Network_Output["PE_Supervised"].detach()).reshape([-1,1])
                Loss["Alpha_Adjust_ada"] = [t.mean(self.ada_loss[1].lossfun(alpha_diff)), weight["Alpha_Adjust"]]
                Loss["Color_ada"] = [t.mean(self.ada_loss[0].lossfun(Col_diff)), weight["Color"]]
                Loss["Color_alpha"] = [t.mean(self.ada_loss[0].alpha().detach()), 1.]
                Loss["Color_width"] = [t.mean(self.ada_loss[0].scale().detach()), 1.]
                Loss["Alpha_Adjust"] = [self.MSE_loss(Network_Output["PE"], Network_Output["PE_Supervised"].detach()), weight["Alpha_Adjust"]]
                Loss["Alpha_alpha"] = [t.mean(self.ada_loss[1].alpha().detach()), 1.]
                Loss["Alpha_width"] = [t.mean(self.ada_loss[1].scale().detach()), 1.]
            else:
                Loss["Color_ada"] = [t.mean(self.ada_loss.lossfun(Col_diff)), weight["Color"]]
                Loss["Color_alpha"] = [t.mean(self.ada_loss.alpha().detach()), 1.]
                Loss["Color_width"] = [t.mean(self.ada_loss.scale().detach()), 1.]

            with t.no_grad():
                if self.use_prior and train_mode:
                    Loss_Color = self.MSE_loss(Network_Output["Rendered_Col_Merged"], data_dict["GT_Color"].to(device))
                else:
                    Loss_Color = self.MSE_loss(Network_Output["Rendered_Col"], data_dict["GT_Color"].to(device))
                Loss["Color"] = [Loss_Color.detach(), weight["Color"]]

        return Loss