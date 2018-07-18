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

lesion_neighbors = ndimage.convolve(lesion_data, k, mode='constant', cval=0.0)

out_dict = {'x': [], 'y': [], 'z': [], 'brain_image': [], 'lesion_mask': []}

sum_of_neighbor_voxels_for_brain = 0
sum_of_neighbor_voxels_for_lesion = 0
number_of_neighbor_voxels = 0

def calculate(x, y, z):
    global sum_of_neighbor_voxels_for_brain, sum_of_neighbor_voxels_for_lesion, number_of_neighbor_voxels, brain_data, lesion_neighbors
    
    sum_of_neighbor_voxels_for_brain += brain_data[x, y, z]
    sum_of_neighbor_voxels_for_lesion += lesion_neighbors[x, y, z]
    number_of_neighbor_voxels += 1

for x in range(512):
    for y in range(512):
        for z in range(512):
            if lesion_data[x, y, z]:
                
                out_dict['x'].append(x)
                out_dict['y'].append(y)
                out_dict['z'].append(z)
                
                sum_of_neighbor_voxels_for_brain = 0
                sum_of_neighbor_voxels_for_lesion = 0
                number_of_neighbor_voxels = 0
                
                # y-1 slice
                if lesion_data[x-1, y-1, z+1]: calculate(x-1, y-1, z+1)

                if lesion_data[x, y-1, z+1]: calculate(x, y-1, z+1)

                if lesion_data[x+1, y-1, z+1]: calculate(x+1, y-1, z+1)
                    
                if lesion_data[x-1, y-1, z]: calculate(x-1, y-1, z)

                if lesion_data[x, y-1, z]: calculate(x, y-1, z)

                if lesion_data[x+1, y-1, z]: calculate(x+1, y-1, z)
                    
                if lesion_data[x-1, y-1, z-1]: calculate(x-1, y-1, z-1)

                if lesion_data[x, y-1, z-1]: calculate(x, y-1, z-1)

                if lesion_data[x+1, y-1, z-1]: calculate(x+1, y-1, z-1)
                    
                # y slice
                if lesion_data[x-1, y, z+1]: calculate(x-1, y, z+1)

                if lesion_data[x, y, z+1]: calculate(x, y, z+1)

                if lesion_data[x+1, y, z+1]: calculate(x+1, y, z+1)
                    
                if lesion_data[x-1, y, z]: calculate(x-1, y, z)

                # lesion_data[x, y, z] excluding the central voxel
                
                if lesion_data[x+1, y, z]: calculate(x+1, y, z)
                    
                if lesion_data[x-1, y, z-1]: calculate(x-1, y, z-1)
                
                if lesion_data[x, y, z-1]: calculate(x, y, z-1)
                
                if lesion_data[x+1, y, z-1]: calculate(x+1, y, z-1)
                
                # y+1 slice
                if lesion_data[x-1, y+1, z+1]: calculate(x-1, y+1, z+1)
                
                if lesion_data[x, y+1, z+1]: calculate(x, y+1, z+1)
                
                if lesion_data[x+1, y+1, z+1]: calculate(x+1, y+1, z+1)
                    
                if lesion_data[x-1, y+1, z]: calculate(x-1, y+1, z)
                
                if lesion_data[x, y+1, z]: calculate(x, y+1, z)
                
                if lesion_data[x+1, y+1, z]: calculate(x+1, y+1, z)
                    
                if lesion_data[x-1, y+1, z-1]: calculate(x-1, y+1, z-1)
                
                if lesion_data[x, y+1, z-1]: calculate(x, y+1, z-1)
                
                if lesion_data[x+1, y+1, z-1]: calculate(x+1, y+1, z-1)
                
                intensity_of_voxel_of_brain = brain_data[x, y, z]
                average_for_brain = sum_of_neighbor_voxels_for_brain / number_of_neighbor_voxels
                difference_for_brain = intensity_of_voxel_of_brain - average_for_brain
                
                number_of_lesion_neighbors = lesion_neighbors[x, y, z]
                average_for_lesion = sum_of_neighbor_voxels_for_lesion / number_of_neighbor_voxels
                difference_for_lesion = number_of_lesion_neighbors - average_for_lesion
                
                out_dict['brain_image'].append([intensity_of_voxel_of_brain, average_for_brain, difference_for_brain])
                out_dict['lesion_mask'].append([number_of_lesion_neighbors, average_for_lesion, difference_for_lesion])

df = pd.DataFrame(data=out_dict)
writer = pd.ExcelWriter('./output/output.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()
