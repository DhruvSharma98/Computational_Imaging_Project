import os
import h5py
import numpy as np
import pandas as pd
import sigpy as sp
from sigpy.mri import app, poisson

metadata_csv_path = "SSDU/SSDU/fastmri_test/fastmri_test.csv" # change for train, val, test
df = pd.read_csv(metadata_csv_path)
file_names = df["hdf5_file_name"]

fast_mri_data_path = "cip_ws2023_data/brain/multicoil_test_full" # change for train, val, test
op_path = "SSDU/SSDU/fastmri_test"
# input image will be resized to this height and width
img_height = 400
img_width = 400

target_coils = 16 # only kspaces with this #coils will be used
ignore_slices = 4 # removes given slices from the end

kspace_final = None
csm_final = None

mask_poisson = poisson([400, 400], accel=4) # used to subsample fastmri kspace required by ssdu
mask_poisson = (abs(mask_poisson)).astype(np.int8) #int8

count = 1
for h5_file in file_names:
    h5_path = os.path.join(fast_mri_data_path, h5_file)
    Csms = []

    with h5py.File(h5_path, "r") as hdf:
        kspace = hdf.get("kspace")
        kspace = np.array(kspace) # kspace is (n_slice, n_coil, n_y, n_x)
        n_slice, n_coil, n_y, n_x = kspace.shape

        if n_coil != target_coils:
            print("skipped\n")
            continue # skip the h5 file

        if n_slice > ignore_slices:
            n_slice = n_slice - ignore_slices

        img_space = sp.ifft(kspace, axes=[-2, -1])
        img_space_resize = sp.resize(img_space, oshape=[n_slice, n_coil, img_height, img_width])
        img_space_resize = img_space_resize[..., ::-1, ::-1] # required for proper format in cropped kspace
        cropped_kspace = sp.ifft(img_space_resize, axes=[-2, -1])

        cropped_kspace = cropped_kspace * mask_poisson # subsampling
        cropped_kspace = np.moveaxis(cropped_kspace, 1, -1) # move n_coil to last dimension

        # get "Csm" for each slice
        for slice in range(n_slice):
            # get the coil sensitivity maps
            device = sp.Device(0)
            kspace_dev = sp.to_device(kspace, device=device)
            csm_dev = app.EspiritCalib(kspace_dev[slice, :, :, :], device=device, show_pbar=False).run()
            csm_resized = sp.resize(csm_dev, oshape=[n_coil, img_height, img_width]) # complex64
            csm = sp.to_device(csm_resized, -1)
            csm = np.moveaxis(csm, 0, -1) # move n_coil to last dimension
            # print(csm.shape)
            Csms.append(csm)

    Csms = np.array(Csms) # n_slice, img_width, img_height, n_coil

    # print(kspace_slices.shape, Csms.shape)

    if kspace_final is None:
        kspace_final = cropped_kspace

    else:
        kspace_final = np.vstack([kspace_final, cropped_kspace])

    if csm_final is None:
        csm_final = Csms
        
    else:
        csm_final = np.vstack([csm_final, Csms])
    
    print(kspace_final.shape, csm_final.shape)

    if count % 5 == 0:
        print(f"{count} files done")

    # if count == 2:
    #     break

    count += 1


kspace_path = os.path.join(op_path, "kspace.h5")
with h5py.File(kspace_path, "w") as op_hdf:
    op_hdf.create_dataset("kspace", data=kspace_final)


csm_path = os.path.join(op_path, "sens_maps.h5")
with h5py.File(csm_path, "w") as op_hdf:
    op_hdf.create_dataset("sens_maps", data=csm_final)


print("----------done-----------")