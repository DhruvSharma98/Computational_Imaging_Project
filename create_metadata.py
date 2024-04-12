import os
import h5py
import pandas as pd

# create metadata for the new dataset
fast_mri_data_path = "cip_ws2023_data/brain/multicoil_train"
csv_op_path = "SSDU/SSDU/fastmri_train/fastmri_train.csv"
subset_percent = 0.1 # percentage of each constrast to select

file_names = sorted(os.listdir(fast_mri_data_path))
kspace_shapes = []
ignore_slices = 4 # remove the last n slices, should be same as in create_modl_dataset.py

print("processing files")
for count, h5_file in enumerate(file_names):
    if count % 500 == 0:
        print(f"{count} files done")
        
    h5_path = os.path.join(fast_mri_data_path, h5_file)

    with h5py.File(h5_path, "r") as hdf:
        kspace = hdf.get("kspace")

        kspace_shape = list(kspace.shape)
        n_slice = kspace_shape[0]
        if n_slice > ignore_slices:
            n_slice = n_slice - ignore_slices
        kspace_shape[0] = n_slice
        
        kspace_shapes.append(kspace_shape)

d = {"hdf5_file_name" : file_names,
     "kspace_shape" : kspace_shapes}

df = pd.DataFrame(data=d)
nslices = df["kspace_shape"].apply(lambda x: x[0])
df["nslices"] = nslices


# select equally distributed files in different contrasts
str_contrasts = ["_AXFLAIR_", "_AXT1POST_", "_AXT1PRE_", "_AXT1_", "_AXT2_"]

file_name_series = df["hdf5_file_name"]
output_df = pd.DataFrame(columns=df.columns)

for str_contrast in str_contrasts:
    contrast_subset = df[file_name_series.str.contains(str_contrast)]
    contrast_percent = int(contrast_subset.shape[0] * subset_percent)
    temp = contrast_subset.iloc[0:contrast_percent, :]
    output_df = pd.concat([output_df, temp])

output_df.index = range(output_df.shape[0])
output_df["nslice_cumulative"] = output_df["nslices"].cumsum()
output_df.to_csv(csv_op_path)
