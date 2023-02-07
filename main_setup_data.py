import argparse
import os
import zipfile
import shutil
import json

def parse_args():
    parser = argparse.ArgumentParser()

    # Required Args for file locs.
    parser.add_argument('--path_to_zip', type=str, required=True,
                        help='Folder containint downloaded zips ex: /Downloads')
    parser.add_argument('--Season_NeRF_Outputs', type=str, required=True,
                        help='Name of Folder to contain assests for and outputs of main.py ex: /Desktop/Code_IO/Season_NeRF')
    parser.add_argument('--code_data_path', type=str, required=True,
                        help='Location of Data folder from code, ex: /Githubcode/NeRFs/Season_NeRF')

    args = parser.parse_args()

    return args

def _main():
    args = parse_args()

    print("Path to Zip:  " + args.path_to_zip)
    print("Path to Season NeRF IO:  " + args.Season_NeRF_Outputs)
    print("Path to to Season_NeRF:  " + args.code_data_path)

    if os.path.exists(args.Season_NeRF_Outputs) == False:
        os.mkdir(args.Season_NeRF_Outputs)

    if os.path.exists(args.Season_NeRF_Outputs + "/IEEE_Data") == False:
        os.mkdir(args.Season_NeRF_Outputs + "/IEEE_Data")
    if os.path.exists(args.Season_NeRF_Outputs + "/IEEE_Data/Images") == False:
        os.mkdir(args.Season_NeRF_Outputs + "/IEEE_Data/Images")

    with zipfile.ZipFile(args.path_to_zip + "/Track3-Metadata.zip", "r") as zip_ref:
        zip_ref.extractall(args.Season_NeRF_Outputs + "/IEEE_Data")

    with zipfile.ZipFile(args.path_to_zip + "/Train-Track3-RGB-1.zip", "r") as zip_ref:
        print("Unzipping JAX Images...")
        zip_ref.extractall(args.Season_NeRF_Outputs + "/IEEE_Data")
        print("Done.")
    files = os.listdir(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-RGB-1")
    for a_file in files:
        shutil.move(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-RGB-1/" + a_file, args.Season_NeRF_Outputs + "/IEEE_Data/Images/" + a_file)

    with zipfile.ZipFile(args.path_to_zip + "/Train-Track3-RGB-2.zip", "r") as zip_ref:
        print("Unzipping OMA Images...")
        zip_ref.extractall(args.Season_NeRF_Outputs + "/IEEE_Data")
        print("Done.")
    files = os.listdir(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-RGB-2")
    for a_file in files:
        shutil.move(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-RGB-2/" + a_file,
                    args.Season_NeRF_Outputs + "/IEEE_Data/Images/" + a_file)

    with zipfile.ZipFile(args.path_to_zip + "/Train-Track3-Truth.zip", "r") as zip_ref:
        print("Unzipping DSM Data...")
        zip_ref.extractall(args.Season_NeRF_Outputs + "/IEEE_Data")
        print("Done.")
    files = os.listdir(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-Truth")
    for a_file in files:
        shutil.move(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-Truth/" + a_file,
                    args.Season_NeRF_Outputs + "/IEEE_Data/Images/" + a_file)

    os.rmdir(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-RGB-1")
    os.rmdir(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-RGB-2")
    os.rmdir(args.Season_NeRF_Outputs + "/IEEE_Data/Track3-Truth")

    with zipfile.ZipFile(args.code_data_path + "/Data.zip", "r") as zip_ref:
        print("Unzipping Data...")
        zip_ref.extractall(args.code_data_path)
        print("Done.")

    shutil.copytree(args.code_data_path + "/Data", args.Season_NeRF_Outputs + "/Cache")

    shutil.move(args.Season_NeRF_Outputs + "/Cache/Testing_Regions", args.Season_NeRF_Outputs + "/Testing_Imgs")

    shutil.rmtree(args.code_data_path + "/Data")


    print("Finished setting up data!")



if __name__ == '__main__':
    _main()