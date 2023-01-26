# coding=utf-8
import pre_NeRF


def pan_set_image_views(region_args):
    all_region_imgs = []
    for args in region_args:
        sat_img_list = pre_NeRF.load_sat_imgs(args.site_name, args.root_dir, args.rpc_dir)