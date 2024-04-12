import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
import pandas as pd
import os

from utils import c2r
from models import mri

class modl_dataset(Dataset):
    def __init__(self, mode, dataset_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst'
        self.dataset_path = dataset_path
        self.sigma = sigma

        #aw51ylaq_DS
        if self.prefix == 'trn':
            self.dataset_metadata = pd.read_csv("data/fastmri_train.csv")
            self.fastmri_dataset_path = "data/fastmri_train"

        else:
            self.dataset_metadata = pd.read_csv("data/fastmri_test.csv")
            self.fastmri_dataset_path = "data/fastmri_test"


    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (2 x nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        # with h5.File(self.dataset_path, 'r') as f:
        #     gt, csm, mask = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Mask'][index]

        # x0 = undersample_(gt, csm, mask, self.sigma)

        # return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)
    

        #aw51ylaq_DS
        index = index + 1 #index range is [0, len(dataset)), now [1, len(dataset)+1)
        # print(f"index is {index}")
        cur_row = self.dataset_metadata["nslice_cumulative"].searchsorted(index, side="left")
        prev_row = cur_row - 1
        file_to_open = self.dataset_metadata.loc[cur_row, ["hdf5_file_name"]].values[0]

        if prev_row >= 0:
            slice_to_get = index - self.dataset_metadata.loc[prev_row, ["nslice_cumulative"]].values[0]
        else:
            slice_to_get = index

        slice_to_get = int(slice_to_get) - 1 # if slice_to_get is 1 i.e. get first slice = 0th element
        # print(file_to_open, "from row", cur_row)
        # print(f"getting element from index {slice_to_get}")

        file_path = os.path.join(self.fastmri_dataset_path, file_to_open)
        with h5.File(file_path, 'r') as f:
            gt, csm, mask = f[self.prefix+'Org'][slice_to_get], f[self.prefix+'Csm'][slice_to_get], f[self.prefix+'Mask'][slice_to_get]

        x0 = undersample_(gt, csm, mask, self.sigma)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(c2r(gt)), torch.from_numpy(csm), torch.from_numpy(mask)



    def __len__(self):
        # with h5.File(self.dataset_path, 'r') as f:
        #     num_data = len(f[self.prefix+'Mask'])
        # return num_data

        #aw51ylaq_DS
        x = self.dataset_metadata[-1:]["nslice_cumulative"].values[0]
        return x


def undersample_(gt, csm, mask, sigma):

    ncoil, nrow, ncol = csm.shape
    csm = csm[None, ...]  # 4dim

    # shift sampling mask to k-space center
    mask = np.fft.ifftshift(mask, axes=(-2, -1))

    SenseOp = mri.SenseOp(csm, mask)

    b = SenseOp.fwd(gt)

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = noise * sigma / (2.**0.5)

    atb = SenseOp.adj(b + noise).squeeze(0).detach().numpy()

    return atb


def undersample(gt, csm, mask, sigma):
    """
    :get fully-sampled image, undersample in k-space and convert back to image domain
    """
    ncoil, nrow, ncol = csm.shape
    sample_idx = np.where(mask.flatten()!=0)[0]
    noise = np.random.randn(len(sample_idx)*ncoil) + 1j*np.random.randn(len(sample_idx)*ncoil)
    noise = noise * (sigma / np.sqrt(2.))
    b = piA(gt, csm, mask, nrow, ncol, ncoil) + noise #forward model
    atb = piAt(b, csm, mask, nrow, ncol, ncoil)
    return atb

def piA(im, csm, mask, nrow, ncol, ncoil):
    """
    fully-sampled image -> undersampled k-space
    """
    im = np.reshape(im, (nrow, ncol))
    im_coil = np.tile(im, [ncoil, 1, 1]) * csm #split coil images
    k_full = np.fft.fft2(im_coil, norm='ortho') #fft
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    k_u = k_full[mask!=0]
    return k_u

def piAt(b, csm, mask, nrow, ncol, ncoil):
    """
    k-space -> zero-filled reconstruction
    """
    if len(mask.shape) == 2:
        mask = np.tile(mask, (ncoil, 1, 1))
    zero_filled = np.zeros((ncoil, nrow, ncol), dtype=np.complex64)
    zero_filled[mask!=0] = b #zero-filling
    img = np.fft.ifft2(zero_filled, norm='ortho') #ifft
    coil_combine = np.sum(img*csm.conj(), axis=0).astype(np.complex64) #coil combine
    return coil_combine