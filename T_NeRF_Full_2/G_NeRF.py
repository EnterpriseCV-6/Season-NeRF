from misc import PE_Encode, Sine_Last, dumb_Encode
from misc import SineLayer as SineLayer
import torch as t
import numpy as np

class G_NeRF_Net_Classic(t.nn.Module):
    def __init__(self, layer_width = 512, expand_size_pose = 10, expand_size_solar_angle = 4, num_out_channels = 3, extended_encoding = False):
        super(G_NeRF_Net_Classic, self).__init__()
        # print("WARNING G_NeRF_Net_Classic is forcing non-extended encoding!")
        use_extended_encoding = extended_encoding
        self.ignore_hue = False
        self.use_norm = True

        self.use_SIREN2 = False
        if self.use_SIREN2:
            expand_size_pose = 0
            expand_size_solar_angle = 0

        self._expand_size_pose = expand_size_pose
        self._expand_size_solar_angle = expand_size_solar_angle
        lw = layer_width
        lw2 = max(layer_width//2, 1)
        lw4 = max(layer_width//4, 1)

        if expand_size_pose != 0:
            input_size = 3*(expand_size_pose*2 + use_extended_encoding)
        else:
            input_size = 3

        if expand_size_solar_angle != 0:
            input_size_solar = 3 * (expand_size_solar_angle * 2 + use_extended_encoding)
        else:
            input_size_solar = 3

        if self.use_SIREN2:
            self.PE_encoder = dumb_Encode()
            self.PE_encoder_solar = dumb_Encode()
        else:
            self.PE_encoder = PE_Encode(expand_size_pose, use_extended_encoding)
            self.PE_encoder_solar = PE_Encode(expand_size_solar_angle, use_extended_encoding)

        self.fc1 = SineLayer(input_size, lw, is_first=True)
        self.fc2 = SineLayer(lw, lw, use_norm=self.use_norm)
        self.fc3 = SineLayer(lw, lw, use_norm=self.use_norm)
        self.fc4 = SineLayer(lw, lw, use_norm=self.use_norm)
        self.fc5 = SineLayer(lw + input_size, lw, is_first=False, use_norm=self.use_norm)
        self.fc6 = SineLayer(lw, lw, use_norm=self.use_norm)
        self.fc7 = SineLayer(lw, lw, use_norm=self.use_norm)
        self.fc8 = SineLayer(lw, lw, use_norm=self.use_norm)
        self.fc9 = SineLayer(lw, lw2, use_norm=self.use_norm)
        self.fc10Col = t.nn.Linear(lw2, num_out_channels)
        self.fc10Sigma = t.nn.Linear(lw2, 1)
        self._inv_delta = 1#128

        self.fc_solar_1 = SineLayer(input_size_solar+lw2, lw2, is_first=True)
        self.fc_solar_2 = SineLayer(lw2, lw2)
        self.fc_solar_3 = SineLayer(lw2, lw2)

        # self.fc_solar_35 = SineLayer(lw, lw)

        self.fc_solar_4 = t.nn.Linear(lw2, 1)

        self.fc_sky_color_1 = SineLayer(input_size_solar, lw4, is_first=True)
        self.fc_sky_color_2 = t.nn.Linear(lw4, 3)


        self.num_out_channels = num_out_channels
        self.sig = t.nn.Sigmoid()#Sine_Last()#
        # self.non_lin = t.nn.ReLU()#t.nn.LeakyReLU(negative_slope=0.2)
        # self.ReLU = t.nn.ReLU()
        self.SoftPlus = t.nn.Softplus()


    def forward_Sigma_Only(self, X):
        X1 = self._encode_X(X)
        Sigma = self.SoftPlus(self.fc10Sigma(X1))
        return Sigma * self._inv_delta


    def _encode_X(self, X):
        X_pose = self.PE_encoder(X)
        X = self.fc1(X_pose)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        X = self.fc5(t.cat([X, X_pose], 1))
        X = self.fc6(X)
        X = self.fc7(X)
        X = self.fc8(X)
        X1 = self.fc9(X)
        return X1

    def forward_Position(self, X):
        X1 = self._encode_X(X)
        Col = self.fc10Col(X1)
        rho = self.fc10Sigma(X1)

        return X1, rho, Col

    def forward_Solar(self, X_Encoded, X_Solar):
        X_Solar_e = self.PE_encoder_solar(X_Solar)
        A = self.fc_solar_1(t.cat([X_Encoded, X_Solar_e], 1))
        A = self.fc_solar_2(A)
        A = self.fc_solar_3(A)

        # A = self.fc_solar_35(A)

        Solar_Vis = self.fc_solar_4(A)
        # print("D")
        Sky_Col = self.fc_sky_color_1(X_Solar_e)
        Sky_Col = self.fc_sky_color_2(Sky_Col)
        if self.ignore_hue:
            K = t.ones([Sky_Col.shape[0], 3])
            K[:,0] = 0
            K = K.to(X_Encoded.device)
            K2 = t.zeros([Sky_Col.shape[0], 3])
            K2[:, 0] = 1000.
            K2 = K2.to(X_Encoded.device)
            Sky_Col = Sky_Col * K + K2
        # Solar_compents = t.cat([A, Sky_Col], 1)

        return Solar_Vis, Sky_Col

    def forward_Solar_Training(self, X, X_Solar):
        with t.no_grad():
            X1, rho, Col = self.forward_Position(X)
        Solar_Vis, Sky_Col = self.forward_Solar(X1, X_Solar)
        return self.SoftPlus(rho) * self._inv_delta, self.sig(Col), self.sig(Solar_Vis), self.sig(Sky_Col)

    def forward_link_mode(self, X, X_Solar):
        X_Encode, Rho, Col = self.forward_Position(X)
        Solar_Vis, Sky_Col = self.forward_Solar(X_Encode, X_Solar)
        return Rho, Col, Solar_Vis, Sky_Col, X_Encode

    def forward_link_mode_Solar_Training(self, X, X_Solar):
        with t.no_grad():
            X_Encode, Rho, Col = self.forward_Position(X)
        Solar_Vis, Sky_Col = self.forward_Solar(X_Encode, X_Solar)
        return Rho, Solar_Vis, X_Encode

    def forward_link_mode_Solar_Training_with_sky_color(self, X, X_Solar):
        with t.no_grad():
            X_Encode, Rho, Col = self.forward_Position(X)
        Solar_Vis, Sky_Col = self.forward_Solar(X_Encode, X_Solar)
        return Rho, Solar_Vis, X_Encode, Sky_Col


    def forward(self, X, X_Solar):
        X_Encode, Rho, Col = self.forward_Position(X)
        Solar_Vis, Sky_Col = self.forward_Solar(X_Encode, X_Solar)

        return self.SoftPlus(Rho) * self._inv_delta, self.sig(Col), self.sig(Solar_Vis), self.sig(Sky_Col)

    def forward_color_only(self, X):
        X1 = self._encode_X(X)
        Col = self.fc10Col(X1)
        return self.sig(Col)