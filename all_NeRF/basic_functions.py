import torch as t
import cv2 as cv
import numpy as np
import datetime

def eval_sigma(sigma, delta):
    P_not_E = t.exp(-sigma * delta)
    P_vis = t.exp(-t.cumsum(t.cat([t.zeros([sigma.shape[0], 1, 1], dtype=t.float, device=sigma.device), sigma * delta], 1), 1)[:, 0:-1])
    P_E = 1-P_not_E
    return P_E, P_vis

#X should be prob visibile [N, S, V] number rays, samples per ray, prov vis.
def get_Surface_Strength_Reg_loss(X:t.tensor, percent_remaining = 0.):
    with t.no_grad():
        est_midpoint = t.argmin(t.abs(.5-X), 1)
        Ws = t.unsqueeze(t.stack([t.arange(0, X.shape[1], device=X.device, requires_grad=False, dtype=t.float)] * X.shape[0], 0),  2)
        Targets = Ws.clone()
        Ws = Ws - est_midpoint.reshape([-1,1,1])
        Targets[Ws < 0] = 0.
        Targets[Ws > 0] = 1.
        Ws = 1 - t.exp(-1 / 2 * (Ws / (X.shape[1] * percent_remaining + 1e-8)) ** 2)

    Error = ((1-X) - Targets)**2 * Ws
    return Error

def general_loss(error, alpha = 2., c = 1.):
    return t.mean(1/2 * error ** 2)

def show_dict_struc(the_obj, max_r_level = -1, r_level = 0):
    if isinstance(the_obj, dict):
        Sep = " "
        for i in range(r_level):
            Sep += "   "
        for a_key in the_obj.keys():
            print(r_level, a_key, sep=Sep)
            if max_r_level == -1  or r_level < max_r_level:
                show_dict_struc(the_obj[a_key], max_r_level, r_level+1)

def CV_downsize(img, DS_factor):
    img2 = (img*255).astype(np.uint8)
    out_img = cv.cvtColor(cv.resize(cv.cvtColor(img2, cv.COLOR_RGB2BGR), (img.shape[0]//DS_factor, img.shape[1]//DS_factor), fx=1/DS_factor, fy=1/DS_factor, interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
    out_img = out_img * 1. / 255.
    return out_img

def CV_reshape(img, new_shape):
    img2 = (img*255).astype(np.uint8)
    out_img = cv.cvtColor(cv.resize(cv.cvtColor(img2, cv.COLOR_RGB2BGR), new_shape, interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
    out_img = out_img * 1. / 255.
    return out_img

def time_frac_2_day(time_frac, use_leap_year = False):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    if use_leap_year:
        date_1 = datetime.datetime.strptime("01/01/20", "%m/%d/%y")
        days_per_year = 366
    else:
        date_1 = datetime.datetime.strptime("01/01/21", "%m/%d/%y")
        days_per_year = 365

    end_date = date_1 + datetime.timedelta(days=days_per_year * time_frac)

    ans = months[end_date.month - 1] + ". " + str(end_date.day)
    return ans

def day_2_time_frac(month:int, day:int, use_leap_year = False):
    if use_leap_year:
        date_1 = datetime.datetime.strptime("01/01/40", "%m/%d/%y")
        date_2 = datetime.datetime.strptime(str(month) + "/" + str(day) + "/40", "%m/%d/%y")
        days_per_year = 366.
    else:
        date_1 = datetime.datetime.strptime("01/01/41", "%m/%d/%y")
        date_2 = datetime.datetime.strptime(str(month) + "/" + str(day) + "/41", "%m/%d/%y")
        days_per_year = 365.

    return (date_2-date_1).days / days_per_year

