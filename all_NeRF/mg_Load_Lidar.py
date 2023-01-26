import numpy as np
from .mg_unit_converter import wgs84_to_utm
import gdal

def get_GT_DSM(args, output_resolution):
    DEM_folder = args.gt_dir
    region_ID = args.site_name
    DEM_file2 = "/" + region_ID + "_DSM.tif"
    DEM_file = DEM_folder + DEM_file2
    UTM_file = DEM_file[0:-3] + "txt"

    filler = "_Refined" if args.use_Bundle_Adjust else ""

    world_bounds = np.load(args.cache_dir + "/bounds_LLA" + filler + ".npy")
    GT = build_ground_truth_UTM(DEM_file, output_resolution, world_bounds, UTM_file)
    GT = (GT - world_bounds[2,0]) / (world_bounds[2,1] - world_bounds[2,0]) * 2 - 1
    return GT

def read_DSM(geo_tiff_file_name):
    geotiff = gdal.Open(geo_tiff_file_name)
    img = geotiff.ReadAsArray()
    img[img == -9999.] = np.NaN
    proj = geotiff.GetGeoTransform()

    return img, proj

def LLA_to_geotif_pixel_UTM(voxel_lat, voxel_lon, easting, northing, pixels, gsd):

    eastings, northings, zone_numbers, zone_letters = np.zeros(voxel_lat.shape[0]), np.zeros(voxel_lat.shape[0]), np.zeros(voxel_lat.shape[0], dtype=int), np.zeros(voxel_lat.shape[0], dtype=str)
    for i in range(voxel_lat.shape[0]):
        eastings[i], northings[i], zone_numbers[i], zone_letters[i] = wgs84_to_utm(voxel_lat[i], voxel_lon[i])


    Y = (eastings - easting)/gsd
    X = (northings - northing)/gsd
    return X, Y

def build_ground_truth_UTM(geo_tiff_file_name, model_size_voxels, area_bounds_LLA, UTM_file_location):
    img, _ = read_DSM(geo_tiff_file_name)
    easting, northing, pixels, gsd = np.loadtxt(UTM_file_location)


    # plt.imshow(img)
    # plt.show()

    model_size = model_size_voxels
    bounds = area_bounds_LLA
    voxel_x = np.tile(np.arange(model_size[0]), model_size[1])
    voxel_y = np.repeat(np.arange(model_size[1]), model_size[0])
    voxel_lat = (voxel_x + .0) / (voxel_x[-1] + .0) * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    voxel_lon = (voxel_y + .0) / (voxel_y[-1] + .0) * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
    geotif_x, geotif_y = LLA_to_geotif_pixel_UTM(voxel_lat, voxel_lon, easting, northing, pixels, gsd)
    # print(geotif_x)
    # print(geotif_y)
    geotif_x = np.array(np.round(geotif_x), dtype=int)
    geotif_y = np.array(np.round(geotif_y), dtype=int)

    good = geotif_x >= 0
    good = good * (geotif_x < img.shape[0])
    good = good * (geotif_y >= 0)
    good = good * (geotif_y < img.shape[1])

    geotif_x, geotif_y = geotif_x[good], geotif_y[good]
    voxel_x, voxel_y = voxel_x[good], voxel_y[good]
    height = img[geotif_x, geotif_y]

    out_img = np.zeros([model_size[0], model_size[1]])-1
    out_img[voxel_x, voxel_y] = height
    out_img[out_img == -1] = np.NaN

    # out_img[out_img != out_img] = -1


    return np.flip(out_img, 0)