import os
import h5py
import cv2
import numpy as np
import pandas as pd
import sigpy as sp
from sigpy import mri
from sigpy.mri import app

metadata_csv_path = "MoDL/MoDL_PyTorch/data/fastmri_test.csv"
df = pd.read_csv(metadata_csv_path)
file_names = df["hdf5_file_name"]

fast_mri_data_path = "cip_ws2023_data/brain/multicoil_test_full" # change for train, val, test
prefix = "tst" # "trn" if train path, else "tst"
op_path = "MoDL/MoDL_PyTorch/data/fastmri_test"
# input image will be resized to this height and width
img_height = 400
img_width = 400

ignore_slices = 4 # removes given slices from the end

for count, h5_file in enumerate(file_names):
    h5_path = os.path.join(fast_mri_data_path, h5_file)
    Orgs = []
    Csms = []
    Masks = []

    with h5py.File(h5_path, "r") as hdf:
        kspace = hdf.get("kspace")
        kspace = np.array(kspace) # kspace is (n_slice, n_coil, n_y, n_x)
        n_slice, n_coil, n_y, n_x = kspace.shape

        recon_rss = sp.rss(sp.ifft(kspace, axes=[-2, -1]), axes=(-3)) # (n_slice, n_y, n_x)
        # resize Org i.e. ground truth images to img_width x img_height
        recon_resized = sp.resize(recon_rss, oshape=[n_slice, img_width, img_height]) # float32

        if n_slice > ignore_slices:
            n_slice = n_slice - ignore_slices
        # get "Org", "Csm", "Mask" for each slice
        for slice in range(n_slice):
            Org = recon_resized[slice, :, :]
            cv2.normalize(Org, Org, 0, 50, cv2.NORM_MINMAX)
            Orgs.append(Org)

            # get the mask
            mask_poisson = mri.poisson([img_width, img_height], accel=4) # accel is target acceleration factor
            mask_poisson_shift = np.fft.fftshift(mask_poisson) # complex128
            
            #convert complex to int
            mask_poisson_shift = (abs(mask_poisson_shift)).astype(np.int8) #int8
            Masks.append(mask_poisson_shift)

            # get the coil sensitivity maps
            device = sp.Device(0)
            kspace_dev = sp.to_device(kspace, device=device)
            csm_dev = app.EspiritCalib(kspace_dev[slice, :, :, :], device=device, show_pbar=False).run()
            csm_resized = sp.resize(csm_dev, oshape=[n_coil, img_width, img_height]) # complex64
            csm = sp.to_device(csm_resized, -1)
            Csms.append(csm)
    
    Orgs = np.array(Orgs)
    Masks = np.array(Masks)
    Csms = np.array(Csms)

    new_hdf5_data_path = op_path
    new_hdf5_file_path = os.path.join(new_hdf5_data_path, h5_file)

    with h5py.File(new_hdf5_file_path, "w") as op_hdf:
        op_hdf.create_dataset(prefix+"Org", data=Orgs)
        op_hdf.create_dataset(prefix+"Mask", data=Masks)
        op_hdf.create_dataset(prefix+"Csm", data=Csms)

    if count % 50 == 0:
        print(f"{count} files done")

print("----------done-----------")