import torch as t
from misc import PE_Encode, Sine_Last, dumb_Encode
from misc import SineLayer2 as SineLayer
from numpy import pi as PI
import numpy as np
from T_NeRF_Full_2.G_NeRF import G_NeRF_Net_Classic


USE_ADVANCED_SOLAR_MODE = False

def _r_walk(a_module:t.nn.Module, update_norm, i = ""):
    for a_mod in a_module.children():
        if isinstance(a_mod, t.nn.BatchNorm1d):
            a_mod.track_running_stats = update_norm
            a_mod.training = update_norm

        # print(i, type(a_mod), isinstance(a_mod, t.nn.BatchNorm1d))
        _r_walk(a_mod, update_norm, i + " ")

class T_NeRF(t.nn.Module):
    def __init__(self, layer_width, n_classes = 4, HM = np.array([[0], [0]])):
        super(T_NeRF, self).__init__()
        use_extended_encoding = True

        use_SIREN2 = True

        self.allow_other_temporal_adjust = False
        self.hm = t.tensor(HM, requires_grad=False)
        self._hm_const = t.tensor(self.hm.shape).reshape([1,2])-1
        self.G_NeRF_net = G_NeRF_Net_Classic(layer_width=layer_width, extended_encoding=use_extended_encoding)

        if use_SIREN2:
            self.Time_Enocder = dumb_Encode()
            self.time_layer_1 = SineLayer(2, layer_width, is_first=True)
        else:
            self.Time_Enocder = PE_Encode(2, use_Extend_encoding=use_extended_encoding)
            self.time_layer_1 = SineLayer(4*2 + 2*use_extended_encoding, layer_width, is_first=True)
        self.time_layer_2 = SineLayer(layer_width, layer_width)
        self.get_class_layer = t.nn.Linear(layer_width, n_classes)

        if self.allow_other_temporal_adjust:
            self.adjust_layer_1 = SineLayer(layer_width//2 + layer_width, layer_width)
        else:
            self.adjust_layer_1 = SineLayer(layer_width//2, layer_width)
        self.adjust_layer_2 = SineLayer(layer_width, layer_width)
        self.adjust_layer_3 = SineLayer(layer_width, layer_width)

        self.adjust_col = t.nn.Linear(layer_width, n_classes*3)
        self.adjust_rho = t.nn.Linear(layer_width, n_classes)
        self.adjust_solar_vis = t.nn.Linear(layer_width, n_classes)
        self.adjust_sky_col = t.nn.Linear(layer_width, n_classes*3)
        self.n_classes = n_classes


        self.SoftMax = t.nn.Softmax(1)
        self.Softplus = t.nn.Softplus()
        self.Sigmoid = t.nn.Sigmoid()#Sine_Last()#

        # self.solar_vis_adj1 = SineLayer(1,10, is_first=True)
        # self.solar_vis_adj2 = t.nn.Linear(10,2)



        self.batch_params_freeze = False
        self._ignore_solar = False

    def ignore_solar(self):
        print("WARNING: IGNORINT SOLAR TERM IN NETWORK, THIS CANNOT BE UNDONE!!!")
        print("THIS SHOULD ONLY BE DONE IN ORDER TO USE LEGACY EVALUATIONS!!!")
        self._ignore_solar = True

    def _process_time(self, Time):
        return Time[:,0:2]

    def forward(self, X, Solar_Angle, Time):
        Rho, Col, Solar_Vis, Sky_Col, X_Encode = self.G_NeRF_net.forward_link_mode(X, Solar_Angle)
        Time_Encode = self.time_layer_2(self.time_layer_1(self.Time_Enocder(self._process_time(Time))))
        output_class = self.SoftMax(self.get_class_layer(Time_Encode))

        if self.allow_other_temporal_adjust:
            Y = self.adjust_layer_1(t.cat([X_Encode, Time_Encode], 1))
        else:
            Y = self.adjust_layer_1(X_Encode)
        Y = self.adjust_layer_2(Y)
        Y = self.adjust_layer_3(Y)

        Adj = self.adjust_col(Y).reshape(X.shape[0], self.n_classes, -1)
        Adjust_col = t.sum(Adj * output_class.unsqueeze(2), 1)


        Rho = self.Softplus(Rho)
        Col = self.Sigmoid(Col + Adjust_col)
        # if self._ignore_solar:
        #     Solar_Vis = t.ones_like(Rho, requires_grad=False)
        # else:
        Solar_Vis = self.Sigmoid(Solar_Vis)

        Sky_Col = self.Sigmoid(Sky_Col)

        # Adv_Solar_vis = self.solar_vis_adj2(self.solar_vis_adj1(t.ones([X.shape[0],1]).to(X.device)))
        # a = self.Softplus(Adv_Solar_vis[:,0:1])
        # b = self.Sigmoid(Adv_Solar_vis[:,1:2])*.5-.25
        # Solar_Vis_Adj = 1/(1+t.exp(-a*(Solar_Vis - .5 - b)))

        return Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col

    def approx_Solar(self, X, X_solar, Time):
        X_Encode, Rho, Col = self.G_NeRF_net.forward_Position(t.cat([X, X_solar], 0))
        Col = Col[0:X.shape[0]]
        X_Encode = X_Encode[0:X.shape[0]]
        Time_Encode = self.time_layer_2(self.time_layer_1(self.Time_Enocder(self._process_time(Time))))
        output_class = self.SoftMax(self.get_class_layer(Time_Encode))

        if self.allow_other_temporal_adjust:
            Y = self.adjust_layer_1(t.cat([X_Encode, Time_Encode], 1))
        else:
            Y = self.adjust_layer_1(X_Encode)
        Y = self.adjust_layer_2(Y)
        Y = self.adjust_layer_3(Y)

        Adj = self.adjust_col(Y).reshape(X.shape[0], self.n_classes, -1)
        Adjust_col = t.sum(Adj * output_class.unsqueeze(2), 1)


        Rho = self.Softplus(Rho)
        Col = self.Sigmoid(Col + Adjust_col)

        return Rho[0:X.shape[0]], Rho[X.shape[0]::], Col, output_class, Adjust_col

    #Same as forward except Col is not combined with Adjust_col and Adjust_col is not combined with output_class
    def forward_seperate(self, X, Solar_Angle, Time):
        Rho, Col, Solar_Vis, Sky_Col, X_Encode = self.G_NeRF_net.forward_link_mode(X, Solar_Angle)
        Time_Encode = self.time_layer_2(self.time_layer_1(self.Time_Enocder(self._process_time(Time))))
        output_class = self.SoftMax(self.get_class_layer(Time_Encode))

        if self.allow_other_temporal_adjust:
            Y = self.adjust_layer_1(t.cat([X_Encode, Time_Encode], 1))
        else:
            Y = self.adjust_layer_1(X_Encode)
        Y = self.adjust_layer_2(Y)
        Y = self.adjust_layer_3(Y)

        Adj = self.adjust_col(Y).reshape(X.shape[0], self.n_classes, -1)
        Adjust_col = Adj

        Rho = self.Softplus(Rho)
        Solar_Vis = self.Sigmoid(Solar_Vis)

        Sky_Col = self.Sigmoid(Sky_Col)

        return Rho, Col, Solar_Vis, Sky_Col, output_class, Adjust_col


    def forward_Solar(self, X, Solar_Angle, Time):
        Rho, Solar_Vis, X_Encode, Sky_Col = self.G_NeRF_net.forward_link_mode_Solar_Training_with_sky_color(X, Solar_Angle)

        return self.Softplus(Rho), self.Sigmoid(Solar_Vis), Sky_Col


    def get_class_only(self, Time):
        Time_Encode = self.time_layer_2(self.time_layer_1(self.Time_Enocder(self._process_time(Time))))
        output_class = self.SoftMax(self.get_class_layer(Time_Encode))
        return output_class


    # def forward_Classic(self, X, Solar_Angle):
    #     return self.G_NeRF_net(X, Solar_Angle)
    #
    def forward_Classic_Sigma_Only(self, X):
        return self.G_NeRF_net.forward_Sigma_Only(X)
    #
    # def forward_Classic_Solar(self, X, Solar_Angle):
    #     return self.G_NeRF_net.forward_Solar(X, Solar_Angle)

    def Supervised_Sample(self, world_pts, delta):
        xy = ((world_pts[:, 0:2] + 1) / 2 * self._hm_const).long()
        Prob_exist_hm = (self.hm[xy[:, 0], xy[:, 1]] >= world_pts[:, 2]).float()
        Prob_exist_hm[Prob_exist_hm > .99] = 0.99
        Prob_exist_hm = -t.log(1 - Prob_exist_hm.unsqueeze(1)) / delta

        return Prob_exist_hm

    #Does not add color adjust and does not compute final color adjustment and does not apply last non-lin
    def forward_full_eval(self, X, Solar_Angle, Time):
        Rho, Col, Solar_Vis, Sky_Col, X_Encode = self.G_NeRF_net.forward_link_mode(X, Solar_Angle)
        Time_Encode = self.time_layer_2(self.time_layer_1(self.Time_Enocder(self._process_time(Time))))
        output_class = self.SoftMax(self.get_class_layer(Time_Encode))

        if self.allow_other_temporal_adjust:
            Y = self.adjust_layer_1(t.cat([X_Encode, Time_Encode], 1))
        else:
            Y = self.adjust_layer_1(X_Encode)
        Y = self.adjust_layer_2(Y)
        Y = self.adjust_layer_3(Y)

        Adj = self.adjust_col(Y).reshape(X.shape[0], self.n_classes, -1)

        Rho = self.Softplus(Rho)

        Solar_Vis = self.Sigmoid(Solar_Vis)

        Sky_Col = self.Sigmoid(Sky_Col)

        return Rho, Col, Solar_Vis, Sky_Col, output_class, Adj