# coding=utf-8

import torch as t
from misc import sample_pt_coarse
import numpy as np
from all_NeRF.basic_functions import get_Surface_Strength_Reg_loss
from all_NeRF.basic_functions import general_loss
from all_NeRF.mg_unit_converter import world_angle_2_local_vec




def get_PV(Rhos, Deltas):
    Y = t.cat([t.zeros([Rhos.shape[0], 1, 1], dtype=Rhos.dtype, device=Rhos.device), Rhos * Deltas], 1)
    PV = (t.exp(-t.cumsum(Y, 1))[:,0:-1])
    return PV

def create_solor_rays(n, include_times = False):

    bounds = np.array([[-1.,1], [-1,1], [-1,1]])
    starts = t.ones([n, 3])
    ends = t.ones([n, 3])

    starts[:, 2] = bounds[2, 1]
    ends[:, 2] = bounds[2, 0]

    starts[:, 1] = (bounds[1, 1] - bounds[1, 0]) * t.rand(n) + bounds[1, 0]
    ends[:, 1] = (bounds[1, 1] - bounds[1, 0]) * t.rand(n) + bounds[1, 0]

    starts[:, 0] = (bounds[0, 1] - bounds[0, 0]) * t.rand(n) + bounds[0, 0]
    ends[:, 0] = (bounds[0, 1] - bounds[0, 0]) * t.rand(n) + bounds[0, 0]

    solar_angle_vec = (starts - ends) / t.sqrt(t.sum((starts - ends) ** 2, 1, keepdim=True))
    if include_times == False:
        return starts, ends, solar_angle_vec
    else:
        random_year_day_frac = t.rand([n,2]) * 2 * np.pi
        random_times = t.stack([t.cos(random_year_day_frac[:,0]), t.sin(random_year_day_frac[:,0]), t.cos(random_year_day_frac[:,1]), t.sin(random_year_day_frac[:,1])], 1)

        return starts, ends, solar_angle_vec, random_times

class create_solor_rays_uniform():
    def __init__(self, W2L_H, WCW, base_vecs = None):
        self.W2L = W2L_H
        self.WC = WCW
        self._use_base_vecs = base_vecs is not None
        self._base_vecs = base_vecs
        self._use_base_vecs = False


    def create_given_vec(self, n, solar_angle_vec, include_times = False):
        bounds = np.array([[-1.,1], [-1,1], [-1,1]])
        starts = t.ones([n, 3])

        delta = 2 * (solar_angle_vec / solar_angle_vec[2::])

        starts[:, 0] = t.tensor(bounds[0, 1] - bounds[0, 0]) * t.rand(n) + t.tensor(bounds[0, 0])
        starts[:, 1] = t.tensor(bounds[1, 1] - bounds[1, 0]) * t.rand(n) + t.tensor(bounds[1, 0])

        ends = (starts - np.expand_dims(delta, 0)).float()

        solar_angle_vec = t.stack([t.tensor(solar_angle_vec).float()] * n, 0)
        if include_times == False:
            return starts, ends, solar_angle_vec
        else:
            random_year_day_frac = t.rand([solar_angle_vec.shape[0], 2]) * 2 * np.pi
            random_times = t.stack([t.cos(random_year_day_frac[:, 0]), t.sin(random_year_day_frac[:, 0]),
                                    t.cos(random_year_day_frac[:, 1]), t.sin(random_year_day_frac[:, 1])], 1)

            return starts, ends, solar_angle_vec, random_times

    def __call__(self, n, include_times = False):

        bounds = np.array([[-1.,1], [-1,1], [-1,1]])
        starts = t.ones([n, 3])


        if self._use_base_vecs == False:
            az_el = np.random.random(n * 2).reshape([n, 2]) * np.array([[360, 89]]) + np.array([[-180, 1]])
            solar_angle_vec = np.array([world_angle_2_local_vec(az_el[i][1], az_el[i][0], self.WC, self.W2L) for i in range(n)])
        else:
            az_el = [-1] * n
            samples = np.random.randint(0, self._base_vecs.shape[0], n)
            solar_angle_vec = self._base_vecs[samples]
        # print(self._base_vecs)
        #
        # print(az_el)
        # from matplotlib import pyplot as plt
        # plt.hist(solar_angle_vec[:,2])
        # plt.show()
        # exit()

        delta = 2*(solar_angle_vec / solar_angle_vec[:,2::])

        starts[:, 0] = t.tensor(bounds[0, 1] - bounds[0, 0]) * t.rand(n) + t.tensor(bounds[0, 0])
        starts[:, 1] = t.tensor(bounds[1, 1] - bounds[1, 0]) * t.rand(n) + t.tensor(bounds[1, 0])

        ends = (starts - delta).float()

        solar_angle_vec = t.tensor(solar_angle_vec).float()

        if include_times == False:
            return starts, ends, solar_angle_vec
        else:
            random_year_day_frac = t.rand([n,2]) * 2 * np.pi
            random_times = t.stack([t.cos(random_year_day_frac[:,0]), t.sin(random_year_day_frac[:,0]), t.cos(random_year_day_frac[:,1]), t.sin(random_year_day_frac[:,1])], 1)

            return starts, ends, solar_angle_vec, random_times, az_el


class All_in_One_Eval():
    def __init__(self, args, device, n_steps, use_prior, ada_loss, H, WC, base_solar_vecs = None):
        self.device = device
        self.args = args
        self.n_steps = n_steps
        self.MSE_loss = t.nn.MSELoss()
        self.smooth_L1 = t.nn.SmoothL1Loss()
        self.use_prior = use_prior
        self.use_reg = args.Use_Reg
        self.use_classic_solar = args.Solar_Type_2
        self.use_MSE_loss = args.Use_MSE_loss
        self.ada_loss = ada_loss
        self.solar_creation_tool = create_solor_rays_uniform(H, WC, base_solar_vecs)
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

        if self.use_classic_solar:
            Rendered_Col = t.sum(PS * self.Sigmoid(Base_Col) * (Solar_Vis + (1 - Solar_Vis) * Sky_Col), 1)
        else:
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

        Xs, deltas = sample_pt_coarse(data_dict["Top"], data_dict["Bot"], n_pts, not train_mode)
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
        PE = 1-t.exp(-Rho * deltas)
        PS = PV * PE

        # from matplotlib import pyplot as plt
        # for i in range(Solar_Vis2.shape[0]):
        #     for j in range(Solar_Vis2.shape[1]):
        #         print(Solar_Vis[i,j].item(), Solar_Vis2[i,j].item(), PS[i,j].item())
        #     print(t.sum(PS[i], 0).item())
        #     plt.plot(PS[i].cpu().detach())
        #     plt.plot(Solar_Vis2[i].detach().cpu()*np.max(PS[i].cpu().detach().numpy()))
        #     plt.show()
        #     print()
        # print(Solar_Vis2.shape)
        # exit()

        # if self.args.Use_Solar:
        #     Col_Adj = (Solar_Vis2 + (1-Solar_Vis2) * Sky_Col) * Base_Col
        # else:
        #     Col_Adj = Base_Col
        Col_Adj = -1

        Albedo_Color = t.sum(PS * Base_Col, 1)

        if self.use_classic_solar:
            Rendered_Col = t.sum(PS * Base_Col * (Solar_Vis + (1 - Solar_Vis) * Sky_Col), 1)
        else:
            Solar_Vis3 = self.Sigmoid((t.sum(Solar_Vis.detach() * PS, 1) - .2) * 30)
            Rendered_Col = Albedo_Color * (Solar_Vis3 + (1-Solar_Vis3) * t.mean(Sky_Col, 1))


        if self.use_prior:
            Model_Trust = current_step / self.n_steps
            Rho_Supervised = Network.Supervised_Sample(Xs.reshape([-1, 3]).cpu(), deltas.reshape(-1, 1).cpu())
            Rho_Supervised = Rho_Supervised.reshape([Xs.shape[0], Xs.shape[1], 1]).to(device)

            PV_Supervised = get_PV(Rho_Supervised, deltas)
            PE_Supervised = 1-t.exp(-Rho_Supervised * deltas)
            PS_Supervised = PV_Supervised * PE_Supervised
            # Rendered_Col_Supervised = t.sum(PS_Supervised * Col_Adj, 1)
            if self.use_classic_solar:
                Rendered_Col_Supervised = t.sum(PS_Supervised * Base_Col * (Solar_Vis + (1 - Solar_Vis) * Sky_Col), 1)
            else:
                Rendered_Col_Supervised = t.sum(PS_Supervised * Base_Col, 1) * (Solar_Vis3 + (1 - Solar_Vis3) * t.mean(Sky_Col, 1))

            Rho_Merged = Rho * Model_Trust + Rho_Supervised * (1 - Model_Trust)
            PV_Merged = get_PV(Rho_Merged, deltas)
            PE_Merged = 1 - t.exp(-Rho_Merged * deltas)
            PS_Merged = PV_Merged * PE_Merged
            # Rendered_Col_Merged = t.sum(PS_Merged * Col_Adj, 1)
            Albedo_Color = t.sum(PS_Merged * Base_Col, 1)
            if self.use_classic_solar:
                Rendered_Col_Merged = t.sum(PS_Merged * Base_Col * (Solar_Vis + (1 - Solar_Vis) * Sky_Col), 1)
            else:
                Rendered_Col_Merged = Albedo_Color * (Solar_Vis3 + (1 - Solar_Vis3) * t.mean(Sky_Col, 1))

            Results = {"Rendered_Col": Rendered_Col, "PE": PE, "PV": PV, "PS": PS,
                       "Solar_Vis": Solar_Vis, "Sky_Col": Sky_Col, "Classes": Classes, "Adjust": Adjust,
                       "Rho": Rho, "Col": Base_Col, "Col_Adj": Col_Adj,
                       "PV_Supervised":PV_Supervised, "PE_Supervised":PE_Supervised, "PS_Supervised":PS_Supervised, "Rendered_Col_Supervised":Rendered_Col_Supervised,
                       "PV_Merged": PV_Merged, "PE_Merged": PE_Merged, "PS_Merged": PS_Merged,
                       "Rendered_Col_Merged": Rendered_Col_Merged, "Rho_Merged":Rho_Merged, "deltas":deltas, "sample_pts":Xs, "Albedo_Color":Albedo_Color}
        else:
            Results = {"Rendered_Col":Rendered_Col, "PE":PE, "PV":PV, "PS":PS, "Solar_Vis":Solar_Vis, "Sky_Col":Sky_Col,
                       "Classes":Classes, "Adjust":Adjust, "Rho":Rho, "Col":Base_Col, "Col_Adj":Col_Adj, "deltas":deltas, "sample_pts":Xs, "Albedo_Color":Albedo_Color}
        return Results


    def _get_exact_solar(self, world_pts, sun_angle, Network):
        Sun_Angle_Extend = t.stack([sun_angle]*world_pts.shape[0], 0)
        K = (1 - world_pts[:,2]) / sun_angle[2]
        Tops = world_pts + K.unsqueeze(1) * Sun_Angle_Extend
        sub_data_dict = {"Top":Tops, "Bot":world_pts, "Sun_Angle":Sun_Angle_Extend, "Time_Encoded":t.tensor([[1,1,1,1]])}
        exact_solar_results = self.eval_Rho_Only(sub_data_dict, Network, False)

        exact_vis = exact_solar_results["PV_Exact"][:,-1]
        est_vis = exact_solar_results["Solar_Vis"][:,-1]
        # for i in range(exact_solar_results["PV_Exact"].shape[0]):
        #     print(exact_solar_results["PV_Exact"][i,-5::])
        #     print(exact_solar_results["Solar_Vis"][i, -5::])
        #     print()
        # exit()
        return exact_vis, est_vis



    def eval_exact_solar(self, data_dict, Network, current_step, train_mode):
        Results = self.eval(data_dict, Network, current_step, train_mode)
        # # print(Results["sample_pts"])
        # # print(Results["sample_pts"].shape)
        # print(data_dict["Sun_Angle"])
        # print(data_dict["Sun_Angle"].shape)
        # exit()
        Results["Est_Solar_Vis"] = Results["Solar_Vis"].clone()
        running_error = 0.
        for i in range(data_dict["Sun_Angle"].shape[0]):
            exact_vis, est_vis = self._get_exact_solar(Results["sample_pts"][i], data_dict["Sun_Angle"][i], Network)
            Results["Solar_Vis"][i] = exact_vis
            running_error += t.mean(t.abs(exact_vis - est_vis))

        Results["Col_Adj"] = (Results["Solar_Vis"] + (1-Results["Solar_Vis"]) * Results["Sky_Col"]) * Results["Col"]
        Results["Rendered_Col"] = t.sum(Results["PS"] * Results["Col_Adj"], 1)

        if self.use_classic_solar:
            Results["Rendered_Col"] = t.sum(Results["PS"] * Results["Col"] * (Results["Solar_Vis"]  + (1-Results["Solar_Vis"]) * Results["Sky_Col"]), 1)
        else:
            Solar_Vis_3 = self.Sigmoid((t.sum(Results["Solar_Vis"] * Results["PS"], 1)-.2)*30)
            Results["Rendered_Col"] = t.sum(Results["PS"] * Results["Col"], 1) * (Solar_Vis_3 + (1-Solar_Vis_3) * t.mean(Results["Sky_Col"], 1))
        return Results

    def eval_Rho_Only(self, data_dict, Network, train_mode, current_step = 0):
        device = self.device
        n_pts = self.args.n_samples

        Xs, deltas = sample_pt_coarse(data_dict["Top"], data_dict["Bot"], n_pts, not train_mode, include_end_pt=True)

        deltas = deltas.to(device)
        Sun_Vec = t.ones_like(Xs) * data_dict["Sun_Angle"].unsqueeze(1)
        Times_Vec = t.ones(Xs.shape[0], Xs.shape[1], 4, dtype=Xs.dtype, device=Xs.device) * data_dict[
            "Time_Encoded"].unsqueeze(1)

        Rho, Solar_Vis, Sky_Col = Network.forward_Solar(Xs.reshape([-1, 3]).to(device),
                                                                     Sun_Vec.reshape([-1, 3]).to(device),
                                                                     Times_Vec.reshape([-1, 4]).to(device))

        Rho = Rho.reshape([Xs.shape[0], Xs.shape[1], 1])
        Solar_Vis = Solar_Vis.reshape([Xs.shape[0], Xs.shape[1], 1])
        Sky_Col = Sky_Col.reshape([Xs.shape[0], Xs.shape[1], -1])

        PE_Exact = 1 - t.exp(-Rho * deltas)
        PV_Exact = get_PV(Rho, deltas)

        if self.use_prior:
            Model_Trust = current_step / self.n_steps
            Xs2, deltas2 = Xs.reshape([-1, 3]).cpu(), deltas.reshape(-1, 1).cpu()
            Xs_good = t.all((Xs2 <= 1.) * (Xs2 >= -1.), 1)
            Xs2, deltas2 = Xs2[Xs_good], deltas2[Xs_good]
            Rho_Supervised = Rho.reshape([-1, 1]).detach().clone().cpu()
            Rho_Supervised[Xs_good] = Network.Supervised_Sample(Xs2, deltas2)
            Rho_Supervised = Rho_Supervised.reshape([Xs.shape[0], Xs.shape[1], 1]).to(device)

            # PV_Supervised = get_PV(Rho_Supervised, deltas)
            # PE_Supervised = 1 - t.exp(-Rho_Supervised * deltas)

            Rho_Merged = Rho * Model_Trust + Rho_Supervised * (1 - Model_Trust)
            PV_Merged = get_PV(Rho_Merged, deltas)
            PE_Merged = 1 - t.exp(-Rho_Merged * deltas)
            Results = {"PE": PE_Merged, "PV_Exact": PV_Merged, "Solar_Vis": Solar_Vis, "Sky_Col": Sky_Col}
        else:
            Results = {"PE":PE_Exact, "PV_Exact":PV_Exact, "Solar_Vis":Solar_Vis, "Sky_Col":Sky_Col}
        return Results


    def get_loss(self, data_dict, Network, current_step, train_mode):
        n_rays = data_dict["Top"].shape[0]
        n_pts = self.args.n_samples
        device = self.device
        Loss = {}
        weight = {"Color": 1.0, "Solar_Correction": self.args.sc_lambda, "Alpha_Adjust": 1.}
        # entropy_adj = np.log(2) / np.log(self.args.number_low_frequency_cases)
        Network_Output = self.eval(data_dict, Network, current_step, train_mode)

        if self.args.Use_Solar:
            starts, ends, solar_angle_vec, Solar_Time, Exact_az_el = self.solar_creation_tool(n_rays, include_times=True)
            Solar_dict = {"Top":starts, "Bot":ends, "Sun_Angle":solar_angle_vec, "Time_Encoded":Solar_Time}
            Network_Output_Solar = self.eval_Rho_Only(Solar_dict, Network, train_mode, current_step)

            # print(Network_Output_Solar["Solar_Vis"].shape, Network_Output_Solar["PV_Exact"].shape)

            # print(Network_Output_Solar["PV_Exact"])
            # print(Network_Output_Solar["PV_Exact"].shape)
            # print(self.BCE_loss(Network_Output_Solar["Solar_Vis"], t.round(Network_Output_Solar["PV_Exact"])))
            # Solar_Error = self.BCE_loss(Network_Output_Solar["Solar_Vis"], t.round(Network_Output_Solar["PV_Exact"]))
            # exit()
            Solar_Error = t.mean(t.sum((Network_Output_Solar["Solar_Vis"] - Network_Output_Solar["PV_Exact"].detach())**2, 1))
            # exit()


            Loss["Solar_Correction"] = [Solar_Error, weight["Solar_Correction"]]
            amount_absorb = t.mean(1 - t.sum(Network_Output_Solar["PE"].detach() * Network_Output_Solar["PV_Exact"].detach() * Network_Output_Solar["Solar_Vis"], 1))
            if self.args.Solar_Type_2 == False:
                Loss["Solar_Correction_2"] = [amount_absorb.detach(), weight["Solar_Correction"]]
            else:
                Loss["Solar_Correction_2"] = [amount_absorb, weight["Solar_Correction"]]
            # Loss["Sky_Color_Var"] = [t.mean(t.var(Network_Output_Solar["Sky_Col"], 0)).detach(), weight["Solar_Correction"]]
            # SK_loss = t.mean(t.sum(Network_Output["Sky_Col"]**2, 2))
            if self.args.Solar_Type_2 == False:
                SK_Albedo, _ = t.min(Network_Output["Albedo_Color"], 0)
                SK_Albedo_loss = SK_Albedo[SK_Albedo < .2]
                if SK_Albedo_loss.shape[0] > 0:
                    SK_Albedo_loss = t.sum((1. - SK_Albedo_loss/.2)**2) / Network_Output["Albedo_Color"].shape[0]
                else:
                    SK_Albedo_loss = t.tensor(0.0, device=device)

                SK_loss = ((Network_Output["Sky_Col"] - .5)/.5)
                SK_loss_non_neg = SK_loss[SK_loss > 0]
                if SK_loss_non_neg.shape[0] > 0:
                    SK_loss = t.sum(SK_loss_non_neg**2)/(np.prod(SK_loss.shape))
                    if self.use_prior:
                        SK_loss = SK_loss.detach()
                else:
                    SK_loss = t.tensor(0., device=device)
                Loss["Sky_Color_Var"] = [SK_loss, weight["Solar_Correction"]]
                Loss["Albedo_Color"] = [SK_Albedo_loss, weight["Solar_Correction"]]

        # Classes = Network_Output["Classes"]
        # Adjust = Network_Output["Adjust"]
        #
        # CU_loss = t.max(1 / (Classes.shape[1] * 2) - t.min(t.mean(Classes, 0)), t.tensor([0]).float().to(device)) * \
        #           Classes.shape[1] * 2
        # if self.use_prior and train_mode:
        #     PS = Network_Output["PS_Merged"].detach()
        # else:
        #     PS = Network_Output["PS"].detach()
        # Expected_Value_Loss = self.smooth_L1(Adjust*PS, t.zeros_like(Adjust, requires_grad=False))
        # quasi_entrop_loss = get_quasi_entropy_loss(Classes)
        #
        # if self.use_reg:
        #     Loss["Class_Utilization_Loss"] = [CU_loss[0], 1.]
        #     Loss["Class_Expected_Value"] = [Expected_Value_Loss, 1.]
        #     Loss["Class_quasi_Entropy"] = [quasi_entrop_loss, 1.]
        # else:
        #     Loss["Class_Utilization_Loss"] = [CU_loss[0].detach(), 1.]
        #     Loss["Class_Expected_Value"] = [Expected_Value_Loss.detach(), 1.]
        #     Loss["Class_quasi_Entropy"] = [quasi_entrop_loss.detach(), 1.]

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
                # Loss["Alpha_Adjust_ada"] = [t.mean(self.ada_loss[1].lossfun(alpha_diff)), weight["Alpha_Adjust"]]
                Loss["Alpha_Adjust_ada"] = [t.mean(self.ada_loss[1].lossfun(alpha_diff)), weight["Alpha_Adjust"]]
                Loss["Color_ada"] = [t.mean(self.ada_loss[0].lossfun(Col_diff)), weight["Color"]]
                Loss["Color_alpha"] = [t.mean(self.ada_loss[0].alpha().detach()), 1.]
                Loss["Color_width"] = [t.mean(self.ada_loss[0].scale().detach()), 1.]
                Loss["Alpha_Adjust"] = [self.MSE_loss(Network_Output["PE"], Network_Output["PE_Supervised"].detach()), weight["Alpha_Adjust"]]
                Scale = t.mean(self.ada_loss[0].scale().detach())**2
                Loss["Solar_Correction"][1] = Loss["Solar_Correction"][1] / Scale
                Loss["Solar_Correction_2"][1] = Loss["Solar_Correction_2"][1] / Scale
                Loss["Alpha_alpha"] = [t.mean(self.ada_loss[1].alpha().detach()), 1.]
                Loss["Alpha_width"] = [t.mean(self.ada_loss[1].scale().detach()), 1.]
                # print(Loss["Color_alpha"], self.ada_loss[0].alpha())
            else:
                Loss["Color_ada"] = [t.mean(self.ada_loss.lossfun(Col_diff)), weight["Color"]]
                Loss["Color_alpha"] = [t.mean(self.ada_loss.alpha().detach()), 1.]
                Loss["Color_width"] = [t.mean(self.ada_loss.scale().detach()), 1.]

                Scale = t.mean(self.ada_loss.scale().detach()) ** 2
                Loss["Solar_Correction"][1] = Loss["Solar_Correction"][1] / Scale
                Loss["Solar_Correction_2"][1] = Loss["Solar_Correction_2"][1] / Scale
            with t.no_grad():
                if self.use_prior and train_mode:
                    Loss_Color = self.MSE_loss(Network_Output["Rendered_Col_Merged"], data_dict["GT_Color"].to(device))
                else:
                    Loss_Color = self.MSE_loss(Network_Output["Rendered_Col"], data_dict["GT_Color"].to(device))
                Loss["Color"] = [Loss_Color.detach(), weight["Color"]]

                # if self.use_prior:
                #     Loss["Alpha_Adjust"] = [
                #         self.MSE_loss(Network_Output["PE"].detach(), Network_Output["PE_Supervised"].detach()),
                #         weight["Alpha_Adjust"]]



        return Loss

def get_quasi_entropy_loss(Classes):
    # print(Classes.shape)
    # n_classes = Classes.shape[-1]
    ans = t.zeros_like(Classes, requires_grad=False)
    ans[t.max(Classes.detach(), -1, keepdim=True)[0] == Classes.detach()] = 1.
    error = t.mean(t.mean(ans - Classes, 1)**2)
    return error