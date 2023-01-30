import torch as t
from numpy import pi as PI
from numpy import sqrt
from all_NeRF.basic_functions import eval_sigma
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import math
import scipy.stats
import json


from collections import namedtuple
def ArgDecoder(args_dict):
    return namedtuple('X', args_dict.keys())(*args_dict.values())

def load_args_from_json(json_file_loc):
    fin = open(json_file_loc, "r")
    args = json.load(fin, object_hook=ArgDecoder)
    fin.close()
    return args

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000 # meters



def get_output_loc(n_steps, n_outputs):
    if n_outputs > 0:
        alpha = np.log(n_steps) / np.log(n_outputs)
        ans = (np.arange(1, n_outputs + 1) ** alpha).astype(int)
        ans[-1] = n_steps
    else:
        ans = np.array([n_steps])
    return ans


def get_output_loc_lin_first(n_steps, n_outputs, min_gap):
    if n_outputs * min_gap >= n_steps:
        ans = np.linspace(1, n_steps, n_outputs+1, dtype=int)[1::]
    else:
        ans = get_output_loc(n_steps, n_outputs)
        lin = np.arange(1,n_outputs+1) * min_gap
        ans = np.maximum(ans, lin)

    return ans


class General_Loss():
    def __init__(self, c = .1):
        self.c = c

    def __call__(self, X, alpha):
        X_scaled = X/self.c
        if alpha  == 2:
            ans = X_scaled**2 / 2
        elif alpha == 0:
            ans = t.log(X_scaled**2 / 2 + 1)
        elif alpha < 10:
            ans = 1-t.exp(-X_scaled**2)
        else:
            ans = t.abs(alpha - 2)/alpha * (((X_scaled**2/(t.abs(alpha-2)) + 1))**(alpha/2) - 1)
        return ans

def alpha_sample(pts, delta, dense_dsm, eps = 1e-8):
    pts_scaled = ((pts.reshape([-1, 3])+1)/2 * t.tensor([[dense_dsm.shape[0]-1, dense_dsm.shape[1]-1, dense_dsm.shape[2]-1]], device=pts.device)).type(t.long)
    P_E = dense_dsm[pts_scaled[:,0], pts_scaled[:,1], pts_scaled[:,2]].reshape([-1, pts.shape[1], 1]) * (1-eps)
    sigma = -t.log(1-P_E)/delta.to(P_E.device)
    return sigma.to(pts.device)


def alpha_merge(Sigma, Sigma_alpha, alpha_percent, safe_mode = True):
    ans = (Sigma_alpha * alpha_percent + Sigma * (1-alpha_percent)).float()
    if safe_mode:
        idx = ans.isnan()
        ans[idx] = Sigma[idx]
    return ans

class Encode_Pose():
    def __init__(self):
        pass

    def forward(self, *X, **optional_args):
        return X

    def __call__(self, *X, **optional_args):
        return self.forward(*X, **optional_args)



class dumb_Encode():
    def __init__(self, **X):
        pass

    def __call__(self, X):
        return X

class PE_Encode(Encode_Pose):
    def __init__(self, n, use_Extend_encoding, scale = PI/2):
        super(PE_Encode, self).__init__()
        self.n = n
        self.k = 2 ** t.arange(0, self.n).reshape([1, 1, 1, self.n]) * scale
        self.use_Extended = use_Extend_encoding
        # min_freq = 1e-4
        # self.k =  min_freq**(2*(t.arange(n)//2)/n)

    def forward(self, X):
        X_save = X
        n_input = X.shape[0]
        X = self.forward_no_reshape(X)
        ans = X.reshape([n_input, -1])
        if self.use_Extended:
            ans = t.cat([X_save, ans],1)
        return ans

    def forward_no_reshape(self, X):
        # RG = X.requires_grad
        if self.k.device != X.device:
            self.k = self.k.to(X.device)
        X = self.k * t.unsqueeze(t.unsqueeze(X, 2).repeat_interleave(2, 2), 3).repeat_interleave(self.n, 3)
        # X[:, :, 0] = t.cos(X[:, :, 0])
        # X[:, :, 1] = t.sin(X[:, :, 1])
        A = t.cos(X[:, :, 0])
        B = t.sin(X[:, :, 1])
        X = t.stack([A, B], 2)
        # if X.requires_grad != RG:
        #     print("?")
        #     exit()

        # Y = Y.detach()
        # Y.requires_grad = RG
        return X

class Sine_Last(t.nn.Module):
    # def __init__(self):
    #     super(Sine_Last, self).__init__()
    def forward(self, input):
        return (t.sin(input)+1)/2

#Code from Implicit Neural Representations with Periodic Activation Functions paper's GitHub
class SineLayer(t.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, use_norm = False):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = t.nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

        if is_first == False and use_norm:
            self.norm = t.nn.BatchNorm1d(out_features, momentum=0.01)
        # self.norm.weight = t.nn.Parameter(t.Tensor(out_features))
        # with t.no_grad():
        #     t.fill_(self.norm.weight, self.omega_0)
        else:
            self.norm = t.nn.Identity()
        # self.omega_0 = 1.

    def init_weights(self):
        with t.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
                # self.omega_0 = 1
            else:
                self.linear.weight.uniform_(-sqrt(6 / self.in_features) / self.omega_0,
                                            sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return t.sin(self.norm(self.omega_0 * self.linear(input)))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return t.sin(intermediate), intermediate

class SineLayer2(t.nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_var=16*8, use_norm = False):
        super(SineLayer2, self).__init__()

        X = np.linspace(0,1,out_features+2)[1:-1]
        X = scipy.stats.norm.ppf(X,0)*omega_var
        X[np.abs(X) < .001] = .001 #Ensures |omega| > 0
        self.omega_0 = t.nn.Parameter(t.unsqueeze(t.tensor(X, dtype=t.float32), 0), requires_grad=False)
        self.is_first = is_first

        self.in_features = in_features
        self.linear = t.nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

        # self.omega_0 = 1.

    def init_weights(self):
        with t.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
                # self.omega_0 = 1
            else:
                self.linear.weight.uniform_(-sqrt(6 / self.in_features),
                                            sqrt(6 / self.in_features))
                self.linear.weight /= self.omega_0.T

    def forward(self, input):
        # print(input.shape, self.linear(input).shape, self.omega_0.shape)
        return t.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return t.sin(intermediate), intermediate

def sample_pt_coarse(pt_tops, pt_bots, n_course, eval_mode, include_end_pt = False):
    with t.no_grad():
        if include_end_pt == False or eval_mode == False:
            ts = t.linspace(0, 1, n_course+1)[0:-1]
        else:
            ts = t.linspace(0, 1, n_course)
        if eval_mode == False:
            ts += 1/n_course * t.rand(n_course)
        ts = ts.reshape([1,-1,1]).to(pt_tops.device)
        deltas = t.sqrt(t.sum((pt_tops - pt_bots)**2, 1)) / n_course
        pts = pt_tops.unsqueeze(1) * (1-ts) + pt_bots.unsqueeze(1) * ts
        deltas = deltas.reshape([-1,1,1]) * t.ones([deltas.shape[0], n_course, 1], device=deltas.device, dtype=deltas.dtype)

    return pts, deltas

class zero_invalid_pts():
    def __init__(self, X = (-1,1), Y=(-1,1), Z=(-1,1)):
        self.X = X
        self.Y = Y
        self.Z = Z

    def __call__(self, Xs):
        Xs = Xs.transpose(-1,0)
        good = (Xs[0] <= 1) * (Xs[1] <= 1) * (Xs[2] <= 1) * (Xs[0] >= -1) * (Xs[1] >= -1) * (Xs[2] >= -1)
        good = good.transpose(0,1)
        bad = ~good
        Xs = Xs.transpose(-1,0)
        return bad

def sample_ray_weighted_stratified(start_rays, end_rays, world_pts_base, weights):
    with t.no_grad():
        # print(weights)
        sample_idx = t.tensor(list(WeightedRandomSampler(weights+10e-5, world_pts_base.shape[1], replacement=True)))
        # print(sample_idx)
        # print(sample_idx.shape)
        # exit()


        mid_pts = (world_pts_base[:, 1::] + world_pts_base[:, 0:-1])/2
        delta_start = t.cat([start_rays.unsqueeze(1).to(mid_pts.device), mid_pts], 1)
        delta_len = delta_start[:,1::] - delta_start[:,0:-1]
        delta_len = t.cat([delta_len, (end_rays.unsqueeze(1).to(delta_start.device) - delta_start[:, -2:-1, :]).to(delta_len.device)], 1)


        shift_term = t.rand(sample_idx.shape)

        X = t.arange(0, start_rays.shape[0]).repeat_interleave(world_pts_base.shape[1],0)
        sampled_points = delta_start[X, sample_idx.reshape(-1)].reshape(delta_start.shape) + \
                         delta_len[X, sample_idx.reshape(-1)].reshape(delta_start.shape) * shift_term.unsqueeze(2).to(delta_len.device)

        all_points = t.cat([world_pts_base, sampled_points], 1)

        new_order = t.argsort(t.sum((start_rays.unsqueeze(1).to(all_points.device) - all_points)**2, 2), 1)

        X = t.arange(0, start_rays.shape[0]).repeat_interleave(world_pts_base.shape[1]*2, 0)
        all_points = all_points[X, new_order.reshape(-1)].reshape([world_pts_base.shape[0], world_pts_base.shape[1]*2, 3])


        end_pts = t.cat([start_rays.unsqueeze(1).to(all_points.device), (all_points[:, 0:-1] + all_points[:, 1::])/2, end_rays.unsqueeze(1).to(all_points.device)], 1)
        deltas = t.sqrt(t.sum((end_pts[:, 1::] - end_pts[:, 0:-1])**2, 2))

    return all_points, deltas

def sample_ray_weighted_stratified_v2(start_rays, end_rays, world_pts_base, weights, num_fine, eval_mode):
    with t.no_grad():
        sample_idx = t.tensor(list(WeightedRandomSampler(weights[:,:,0]+10e-5, num_fine, replacement=True)))
        # print(sample_idx)
        # print(sample_idx.shape)
        # exit()


        mid_pts = (world_pts_base[:, 1::] + world_pts_base[:, 0:-1])/2
        delta_start = t.cat([start_rays.unsqueeze(1).to(mid_pts.device), mid_pts], 1)
        delta_len = delta_start[:,1::] - delta_start[:,0:-1]
        delta_len = t.cat([delta_len, (end_rays.unsqueeze(1).to(delta_start.device) - delta_start[:, -2:-1, :]).to(delta_len.device)], 1)


        shift_term = t.rand(sample_idx.shape)

        X = t.arange(0, start_rays.shape[0]).repeat_interleave(world_pts_base.shape[1],0)
        sampled_points = delta_start[X, sample_idx.reshape(-1)].reshape(delta_start.shape) + \
                         delta_len[X, sample_idx.reshape(-1)].reshape(delta_start.shape) * shift_term.unsqueeze(2).to(delta_len.device)

        all_points = t.cat([world_pts_base, sampled_points], 1)

        new_order = t.argsort(t.sum((start_rays.unsqueeze(1).to(all_points.device) - all_points)**2, 2), 1)

        X = t.arange(0, start_rays.shape[0]).repeat_interleave(world_pts_base.shape[1]*2, 0)
        all_points = all_points[X, new_order.reshape(-1)].reshape([world_pts_base.shape[0], world_pts_base.shape[1]*2, 3])


        end_pts = t.cat([start_rays.unsqueeze(1).to(all_points.device), (all_points[:, 0:-1] + all_points[:, 1::])/2, end_rays.unsqueeze(1).to(all_points.device)], 1)
        deltas = t.unsqueeze(t.sqrt(t.sum((end_pts[:, 1::] - end_pts[:, 0:-1])**2, 2)), 2)

    return all_points, deltas

def sample_pt_fine(weights, pt_top, pt_bot, base_pts, num_fine, eval_mode):
    return sample_ray_weighted_stratified_v2(pt_top, pt_bot, base_pts, weights, num_fine, eval_mode)
    # with t.no_grad():
    #     n_coarse = weights.shape[1]
    #     weights_n = t.cumsum((weights + 1e-8) / t.sum(weights + 1e-8, 1, keepdim=True), 1)
    #     weights_n[:,-1] = 1.
    #     if eval_mode == False:
    #         noise = t.rand([weights.shape[0], num_fine, 1])
    #     else:
    #         noise = t.linspace(0,1,num_fine+2)[1:-1].reshape([1,-1, 1]) * t.ones([weights.shape[0], 1, 1])
    #
    #     print(weights_n.shape, noise.shape)
    #     diff = weights_n - noise.transpose(1,2).to(weights_n.device)
    #     diff[diff < 0] = 1
    #     sample_idx = t.argmin(diff, 1)#t.histc(t.argmin(diff, 1), bins=n_coarse, min=0, max=n_coarse)
    #     sample_idx = t.cat([sample_idx, (t.arange(0, weights.shape[1]).reshape([1,-1]) * t.ones([weights.shape[0], 1])).int().to(sample_idx.device)], 1)
    #     sample_idx, _ = t.sort(sample_idx, 1)
    #     print(sample_idx)
    #     print(t.histc(sample_idx, bins=64))
    #     print(sample_idx.shape)
    #
    #     counts = t.zeros([weights.shape[0], weights.shape[1]]).int()
    #     print("Not yet implemented: ID 11916")
    #     exit()
    #
    #
    #     exit()
    #
    #     print(weights_n[:,:,0])
    #     bins_filled = weights_n.int()
    #     remaining_weights = weights_n - weights_n.int()
    #     print(bins_filled[:,:,0])
    #     print(t.sum(bins_filled, 1)[:,0])
    #     print(remaining_weights[:,:,0])
    #
    #
    #     # n_pts = weights_n.int()
    #     # weights_n = weights_n - weights_n.int()
    #     # print(weights_n[:,:,0])
    #     # print(n_pts)
    #     # print(t.sum(n_pts, 1))
    #     # print(n_pts.shape)
    #     exit()
    #
    # exit()

def main_test_P_NeRF():
    an_encode = PE_Encode(3, use_Extend_encoding=True)
    test_in = t.tensor([[1,], [.5], [-1], [np.pi]])
    test_out = an_encode(test_in)
    print(test_out)
    print(test_in.shape)
    print(test_out.shape)

if __name__ == '__main__':
    main_test_P_NeRF()
