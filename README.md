# Season-NeRF

## Quick Setup
### Environment Setup
To set up 'Season_NeRF` environment:

    conda env create --name Season_NeRF --file Sesaon.yml
    conda activate Season_NeRF
    pip install git+https://github.com/jonbarron/robust_loss_pytorch

### Data Setup Lite
The full dataset requires ~20 GB of space.
This form of setup contains only NUMBER regions and requires ~NUMBER MB.

To download and prepare the lite version of the data,

1. ToDo

The regions contained in the lite dataset are

1. ToDO

### Data Setup
The data used by Season-NeRF is available to download at https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019.

To download and prepare the data for Season-NeRF,

1. Download the following files from https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
   - Track 3 / Metadata	(138.5 KB)
   - Track 3 / Training data / RGB images 1/2	(7.6 GB)
   - Track 3 / Training data / RGB images 2/2	(12.49 GB)
   - Track 3 / Training data / Reference	(37.46 MB)
2. Confirm the following files have been downloaded:
   - Track3-Metadata.zip
   - Train-Track3-RGB-1.zip
   - Train-Track3-RGB-2.zip
   - Train-Track3-Truth.zip
3. Run main_setup_data.py  --path_to_zip arg1 --Season_NeRF_Outputs arg2 --code_data_path arg3
   - arg1: Folder containing downloaded zips ex: /Downloads
   - arg2: Name of Folder to contain assets for and outputs of main.py ex: /Desktop/Code_IO/Season_NeRF
   - arg3: Location of Data folder from code, ex: /Githubcode/NeRFs/Season_NeRF
   
If the data folder has been set up correctly, you should see the following message ``Finished setting up data!''.

### Quick Run
After running the data setup, the following command can be used to train a Season NeRF.

``
python main.py  --exp_name arg1 --IO_Location arg2 --site_name arg3
``
- arg1: Name of Experiment, data for output will be stored in arg2/Logs
- arg2: Same as arg2 from running main_setup_data.py
- arg3: Name of the region to build Season NeRF on ex: OMA_042, OMA_132

Once training starts, progress can be viewed by running
``
tensorboard --logdir arg2/Logs --port 6006
``
and following the link http://localhost:6006/#scalars

### Quick Model Rendering
Given a trained model, the following command will start a GUI for rendering novel views and allow testing of the model.

ToDo

## Advanced Running
### Useful Arguments
- ``--skip_Bundle_Adjust``: Does not use Bundle-Adjusted RPCs for training
- ``--Use_MSE_loss``: Use MSE loss instead of Barron's Loss for training
- ``--Solar_Type_2``: Use Solar loss described in S-NeRF

### Cache Information
The cache stores non-NeRF-related information.
When running, main.py will check the cache for relevant files before creating them.
Any file deleted in the cache will be rebuilt if needed, EXCEPT FOR .ikono files.
These files were generated via the Lego pipeline, and that code is not included.
If the .ikono files are not available, it is necessary to include ``--skip_Bundle_Adjust`` when running main.py
Most files in the cache can be deleted after completion of the program with minimal impact on future test run times.

Overview of files in cache:
- .bounds_LLA_*.npy: Contains Lat, Lon, and Height bounds for the region
- *.pickle: Contains information for the affine approximation of the RPC
- .ikono file: Contains original or corrected RPCs for the region (do not delete as they cannot be recovered without Lego pipeline)
- Full_Scores_*.npy: Contains a summary of the accuracy of affine approximation to RPCs
- SC_*.npy: Contains height map used to guide training (Takes a long time to build)

### Log Information
ToDo
