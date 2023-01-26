import numpy as np
from Space_Carving import get_DSM_SC
from all_NeRF import get_GT_DSM
from matplotlib import pyplot as plt
import pickle

def get_DSM(args, device = "cpu", force_build = False):
    DSM_Mode = args.DSM_Mode
    training_DSM = np.array([[np.NaN, np.NaN], [np.NaN, np.NaN]])
    if DSM_Mode == "Space_Carve":
        print("Getting DSM via Space Carving.")
        training_DSM = get_DSM_SC(args, device, force_build)
    elif DSM_Mode == "Stereo":
        print("Getting DSM via Stereo.")
        print("WARNING: NOT YET IMPLENTED")
        print("NOT USING DSM!")
        exit()
    elif DSM_Mode == "LiDAR":
        print("Using LiDAR as DSM for training.")
        if args.use_Bundle_Adjust:
            print("Warning: Training DSM may not be aligned!")
        training_DSM = get_GT_DSM(args, [500, 500])
    else:
        print("Not using any DSM for training.")

    if DSM_Mode == "Space_Carve" or DSM_Mode == "Stereo" or DSM_Mode == "LiDAR":
        GT_DSM = get_GT_DSM(args, training_DSM.shape)
    else:
        GT_DSM = get_GT_DSM(args, [500,500])

    return training_DSM, GT_DSM