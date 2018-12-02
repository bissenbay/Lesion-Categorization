import os, glob
import pandas as pd
import shutil

# if not to set recursion limit there will be RuntimeError: maximum recursion depth exceeded
# the number 100000 is chosen by random
# what number should be right?
import sys
sys.setrecursionlimit(100000)

# set path for images
path_for_images = 'test_dataset/'
        
def find_components(file):
    path_to_save = 'result_efficient_v1' + file[-22:-14] + file[-7:-4] + '/'
    name_of_file = file[-14:-4]
    data = pd.read_csv(file)
    data.columns.values[0] = 'voxel_id'
#    data = data[['voxel_id', 'x', 'y', 'z']]
    data['visited'] = False
    component_id = 1
    result_data = pd.DataFrame(columns=['component_id', 'total_voxels', 'internal_voxels', 'external_voxels', '%_of_internal_voxels', '%_of_external_voxels', 'voxels'])
    
    def label_components(voxel_id):
        if data.at[voxel_id, 'visited'] == False:
            data.at[voxel_id, 'visited'] = True
            result_data.at[component_id, 'voxels'].append(voxel_id)
            voxel = data.loc[voxel_id]
            neighbor_voxels = data[(abs(voxel['x'] - data['x']) + abs(voxel['y'] - data['y']) + abs(voxel['z'] - data['z'])) == 1].index
            if len(neighbor_voxels) == 6:
                nonlocal internal_voxels
                internal_voxels += 1
            for neighbor_voxel_id in neighbor_voxels:
                label_components(neighbor_voxel_id)
    
    for voxel_id in data.index:
        if data.at[voxel_id, 'visited'] == False:
            result_data.at[component_id, 'voxels'] = []
            internal_voxels = 0
            label_components(voxel_id)
            result_data.at[component_id, 'voxels'].sort()
            total_voxels = len(result_data.at[component_id, 'voxels'])
            result_data.at[component_id, 'total_voxels'] = total_voxels
            external_voxels = total_voxels - internal_voxels
            result_data.at[component_id, 'component_id'] = component_id
            result_data.at[component_id, 'internal_voxels'] = internal_voxels
            result_data.at[component_id, 'external_voxels'] = external_voxels
            result_data.at[component_id, '%_of_internal_voxels'] = round((internal_voxels / total_voxels) * 100, 2)
            result_data.at[component_id, '%_of_external_voxels'] = round((external_voxels / total_voxels) * 100, 2)
#             if internal_voxels > 0: to find non_zero_percentage_of_interior_voxels
            component_id += 1
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save) 
        
    data = data.drop(columns=['visited'])
    data.to_csv(path_to_save + name_of_file + '.csv', index=False)
    
    result_data.to_csv(path_to_save + name_of_file + '_connected_components.csv', index=False)
            
for file in sorted(glob.glob(os.path.join(path_for_images, '**/*.csv'), recursive=True)):
    find_components(file)