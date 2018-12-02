import os, sys, glob
import pandas as pd
import numpy as np
import shutil
from scipy import ndimage
from scipy.ndimage import label

# set to print full array content in jupyter notebook
# np.set_printoptions(threshold=np.inf)

# set path for images
path_for_images = 'test_dataset/'

s = [[[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]],

     [[0, 1, 0],
      [1, 1, 1],
      [0, 1, 0]],

     [[0, 0, 0],
      [0, 1, 0],
      [0, 0, 0]]]
        
def find_components(file):
    
    global s
    
    path_to_save = 'result_improved_v2' + file[-22:-14] + file[-7:-4] + '/'
    name_of_file = file[-14:-4]
    
    data = pd.read_csv(file)
    data.columns.values[0] = 'voxel_id'
    
    img = np.zeros((512, 512, 512), dtype='<i2')
    
    coords_to_ids = {}
    
#     is this the proper way of converting to int
    for row in data.iterrows():
        x = int(row[1]['x'])
        y = int(row[1]['y'])
        z = int(row[1]['z'])
        voxel_id = int(row[1]['Unnamed: 0'])
        img[x, y, z] = 1
        coords_to_ids[(x, y, z)] = voxel_id
    
    num_components = label(img, structure=s, output=img)
    
    component_slices = ndimage.find_objects(img)
    
    result_data = pd.DataFrame(columns=['component_id', 'total_voxels', 'internal_voxels', 'external_voxels', '%_of_internal_voxels', '%_of_external_voxels', 'voxels'])
    
    for component_id in range(1, num_components + 1):
        
        component = np.copy(img[component_slices[component_id - 1]])
        
        component[component != component_id] = 0        
        component[component == component_id] = 1
        
        total_voxels = np.sum(component)
        
        internal_voxels = ndimage.convolve(component, s, mode='constant', cval=0.0)
        internal_voxels = len(internal_voxels[internal_voxels == 7])
        
        external_voxels = total_voxels - internal_voxels
        
        result_data.at[component_id, 'component_id'] = component_id
        result_data.at[component_id, 'total_voxels'] = total_voxels
        result_data.at[component_id, 'internal_voxels'] = internal_voxels
        result_data.at[component_id, 'external_voxels'] = external_voxels
        result_data.at[component_id, '%_of_internal_voxels'] = round((internal_voxels / total_voxels) * 100, 2)
        result_data.at[component_id, '%_of_external_voxels'] = round((external_voxels / total_voxels) * 100, 2)
        voxels = []
        for voxel_coords in np.argwhere(img == component_id):
            voxels.append(coords_to_ids[tuple(voxel_coords)])
        result_data.at[component_id, 'voxels'] = voxels
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    data.to_csv(path_to_save + name_of_file + '.csv', index=False)
    
    result_data.to_csv(path_to_save + name_of_file + '_connected_components.csv', index=False)
    
for file in sorted(glob.glob(os.path.join(path_for_images, '**/*.csv'), recursive=True)):
    find_components(file)