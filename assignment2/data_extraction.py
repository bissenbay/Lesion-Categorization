import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage

lesion_img = './input/test_lesion_mask.nii.gz'
lesion_load = nib.load(lesion_img)
lesion_data = lesion_load.get_fdata()

brain_img = './input/test_brain_image.nii.gz'
brain_load = nib.load(brain_img)
brain_data = brain_load.get_fdata()

k = np.ones((3,3,3), dtype=int)
k[1,1,1] = 0

# X, Y, Z coordinates of lesions are filled
output_df = pd.DataFrame(np.argwhere(lesion_data > 0.), columns=['x', 'y', 'z'])

# sum of the lesion neighbors
num_of_nei_lesions = ndimage.convolve(lesion_data, k, mode='constant', cval=0.0)
num_of_nei_lesions = num_of_nei_lesions * lesion_data

# NUMBER_OF_NEIGHBOR_LESIONS are filled
output_df['num_of_neighbor_lesions'] = pd.Series(num_of_nei_lesions[num_of_nei_lesions > 0.])

# calculating the average of number of neighbor lesions
avg_of_num_of_nei_lesions = ndimage.convolve(num_of_nei_lesions, k, mode='constant', cval=0.0)
avg_of_num_of_nei_lesions = avg_of_num_of_nei_lesions * lesion_data

# AVERAGE OF NUMBER OF NEIGHBOR LESIONS are filled
output_df['avg_of_num_of_neighbor_lesions'] = pd.Series(avg_of_num_of_nei_lesions[avg_of_num_of_nei_lesions > 0.] / num_of_nei_lesions[num_of_nei_lesions > 0.])

# DIFFERENCE OF NUMBER OF NEIGHBOR LESIONS AND AVERAGE OF NUMBER OF NEIGHBOR LESIONS
# num_of_neighbor_lesions - avg_of_num_of_neighbor_lesions
output_df['diff_of_num_of_neighbor_lesions_and_avg_of_num_of_neighbor_lesions'] = output_df['num_of_neighbor_lesions'].subtract(output_df['avg_of_num_of_neighbor_lesions'])


# multiplying by lesion_data to get only lesion intensities
brain_data = brain_data * lesion_data

# VOXEL INTENSITY is filled
output_df['voxel_intensity'] = pd.Series(brain_data[brain_data > 0.])

# convolve brain_data to get the sum of the intensities of neighbor voxels
brain_data = ndimage.convolve(brain_data, k, mode='constant', cval=0.0)

# multiplying by lesion_data to get only the sum of lesion intensities
brain_data = brain_data * lesion_data

# AVERAGE INTENSITY OF NEIGHBOR VOXELS
output_df['average_intensity_of_neighbor_voxels'] = pd.Series(brain_data[brain_data > 0.] / num_of_nei_lesions[num_of_nei_lesions > 0.])

# DIFFERENCE OF VOXEL INTENSITY AND AVERAGE INTENSITY OF NEIGHBOR VOXELS
# voxel_intensity - average_intensity_of_neighbor_voxels
output_df['diff_of_vox_int_and_avg_int_of_nei_vox'] = output_df['voxel_intensity'].subtract(output_df['average_intensity_of_neighbor_voxels'])

# saving to the file
output_df.to_csv('output/output.csv')
