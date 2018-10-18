import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
import argparse # Parse arguments

# Parse the input arguments
parser = argparse.ArgumentParser(description='Extract lesion information')

parser.add_argument('origImg', help='File path to the original image')
parser.add_argument('lesImg', help='File path to the lesion image')
parser.add_argument('lesLabel', type=int, help='Intensity value for lesions in the lesion\
 image. This is because the lesion image might have other intensities.')
parser.add_argument('output', help='File path for the output file.')

args = parser.parse_args()
print 'origImg:' + args.origImg
print 'lesImg:' + args.lesImg
print 'lesLabel:' + str(args.lesLabel)

# Inputs to be set by the user
brain_img = args.origImg
lesion_img = args.lesImg
lesion_label = args.lesLabel
output_filepat = args.output

lesion_load = nib.load(lesion_img)
lesion_data = lesion_load.get_fdata()

brain_load = nib.load(brain_img)
brain_data = brain_load.get_fdata()

k = np.ones((3,3,3), dtype=int)
k[1,1,1] = 0

# Extract the lesion voxels given the lesion label = thresholding
lesion_data = np.where(lesion_data==lesion_label,1,0)

# X, Y, Z coordinates of lesions are filled
output_df = pd.DataFrame(np.argwhere(lesion_data == 1), columns=['x', 'y', 'z'])

# Get the positions to evaluated using three arrays: x, y, and z
x_y_z_matrix = output_df.get_values()
x_vector = x_y_z_matrix[:,0]
y_vector = x_y_z_matrix[:,1]
z_vector = x_y_z_matrix[:,2]

# sum of the lesion neighbors
num_of_nei_lesions = ndimage.convolve(lesion_data, k, mode='constant', cval=0.0)
num_of_nei_lesions = num_of_nei_lesions * lesion_data

# NUMBER_OF_NEIGHBOR_LESIONS are filled
output_df['num_of_neighbor_lesions'] = pd.Series(num_of_nei_lesions[x_vector, y_vector, z_vector])

# calculating the average of number of neighbor lesions
avg_of_num_of_nei_lesions = ndimage.convolve(num_of_nei_lesions, k, mode='constant', cval=0.0)
avg_of_num_of_nei_lesions = avg_of_num_of_nei_lesions * lesion_data

# AVERAGE OF NUMBER OF NEIGHBOR LESIONS are filled
output_df['avg_of_num_of_neighbor_lesions'] = pd.Series(np.float64(avg_of_num_of_nei_lesions[x_vector, y_vector, z_vector]) / np.float64(num_of_nei_lesions[x_vector, y_vector, z_vector]))

# DIFFERENCE OF NUMBER OF NEIGHBOR LESIONS AND AVERAGE OF NUMBER OF NEIGHBOR LESIONS
output_df['diff_of_num_of_neighbor_lesions_and_avg_of_num_of_neighbor_lesions'] = output_df['num_of_neighbor_lesions'].subtract(output_df['avg_of_num_of_neighbor_lesions'])

# multiplying by lesion_data to get only lesion intensities
brain_data = brain_data * lesion_data

# VOXEL INTENSITY is filled
output_df['voxel_intensity'] = pd.Series(brain_data[x_vector, y_vector, z_vector])

# convolve brain_data to get the sum of the intensities of neighbor voxels
brain_data = ndimage.convolve(brain_data, k, mode='constant', cval=0.0)

# multiplying by lesion_data to get only the sum of lesion intensities
brain_data = brain_data * lesion_data

# AVERAGE INTENSITY OF NEIGHBOR VOXELS
output_df['average_intensity_of_neighbor_voxels'] = pd.Series(np.float64(brain_data[x_vector, y_vector, z_vector]) / np.float64(num_of_nei_lesions[x_vector, y_vector, z_vector]))

# DIFFERENCE OF VOXEL INTENSITY AND AVERAGE INTENSITY OF NEIGHBOR VOXELS
output_df['diff_of_vox_int_and_avg_int_of_nei_vox'] = output_df['voxel_intensity'].subtract(output_df['average_intensity_of_neighbor_voxels'])

# saving to the file
output_df.to_csv(output_filepat)
