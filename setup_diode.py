import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import PIL
from PIL import Image
import tarfile
import io

dataset_path = "/local/sandbox/DIODE_dataset/"

target_res = (192, 256)

datasets = ("train", "val")
places = ("indoors", "outdoor")
datatypes = ("rgb", "depth", "depth_mask")

data = {ds : {pl : {dt : None for dt in datatypes} for pl in places} for ds in datasets}
dnms = {ds : {pl : {dt : None for dt in datatypes} for pl in places} for ds in datasets}

print("setting up empty data, reading CSVs")

for ds in datasets:
    for pl in places:
        names = pd.read_csv(dataset_path + "data_list/" + ds + "_" + pl + ".csv", names=("rgb", "depth", "depth_mask", "normals")).applymap(lambda s: s[2:])
        dnms[ds][pl] = {dt : pd.Index(names[dt]) for dt in datatypes}
        n_names = len(dnms[ds][pl][datatypes[0]])
        data[ds][pl]["rgb"] = np.zeros((n_names, *target_res, 3), dtype=np.uint8)
        data[ds][pl]["depth"] = np.zeros((n_names, *target_res, 1), dtype=np.float32)
        data[ds][pl]["depth_mask"] = np.zeros((n_names, *target_res), dtype=np.bool)

print("starting to read data")

def tar_to_npy(f):
    array_file = io.BytesIO()
    array_file.write(f.read())
    array_file.seek(0)
    return np.load(array_file)

for ds in datasets:
    print("reading", ds)
    tarf = tarfile.open(dataset_path + ds + ".tar.gz", "r:gz")
    fileinfo = tarf.next()

    count = 0
    while fileinfo is not None:

        if not fileinfo.isfile():
            fileinfo = tarf.next()
            continue
        
        count += 1
        print("\r" + str(count), end="")

        file = tarf.extractfile(fileinfo)

        place = places[0] if places[0] in fileinfo.name else places[1]

        if fileinfo.name.endswith(".png"): # RGB image
            index = dnms[ds][place]["rgb"].get_loc(fileinfo.name)
            img = Image.open(file)
            img = img.resize(target_res[::-1], resample=PIL.Image.BOX)
            data[ds][place]["rgb"][index] = np.asarray(img)

        elif fileinfo.name.endswith("depth.npy"): # depth
            index = dnms[ds][place]["depth"].get_loc(fileinfo.name)
            data[ds][place]["depth"][index] = tar_to_npy(file)[::4, ::4]

        elif fileinfo.name.endswith("mask.npy"): # depth mask
            index = dnms[ds][place]["depth_mask"].get_loc(fileinfo.name)
            data[ds][place]["depth_mask"][index] = tar_to_npy(file)[::4, ::4] != 0

        else:
            print("error, unexpected filename: ", fileinfo.name)

        fileinfo = tarf.next()
    print("done")

out_dict = {}

for ds in datasets:
    for pl in places:
        for dt in datatypes:
            out_dict[f"{ds}_{pl}_{dt}"] = data[ds][pl][dt]

np.savez_compressed(dataset_path + "DIODE_dataset.npz", **out_dict)
