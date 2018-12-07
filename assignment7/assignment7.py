import os, glob
import pandas as pd

# set path
path = '../test_data/'

def fill_db(file):
    
    path_to_save = 'result' + file[-22:-14] + file[-7:-4] + '/'
    name_of_file = file[-14:-4]
    
    # preprocessing
    data = pd.read_csv(file)
    data.columns.values[0] = 'voxel_id'
    
    # save original image
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save) 
        
    data.to_csv(path_to_save + name_of_file + '.csv', index=False)
    
    data = data[['voxel_id', 'x', 'y', 'z']]
    
    data_all_neighbors = pd.DataFrame(columns=['voxel_id', 'num_of_all_neighbors', 'all_neighbors'])
    data_face_neighbors = pd.DataFrame(columns=['voxel_id', 'num_of_face_neighbors', 'face_neighbors'])
    data_edge_neighbors = pd.DataFrame(columns=['voxel_id', 'num_of_edge_neighbors', 'edge_neighbors'])
    data_vertex_neighbors = pd.DataFrame(columns=['voxel_id', 'num_of_vertex_neighbors', 'vertex_neighbors'])
    data_id_to_xyz = pd.DataFrame(columns=['voxel_id', 'xyz'])
        
    for voxel_id in data.index:
        
        voxel = data.loc[voxel_id]
        
        data_id_to_xyz = data_id_to_xyz.append({'voxel_id': voxel['voxel_id'], 'xyz': (voxel['x'], voxel['y'], voxel['z'])}, ignore_index=True)
        
        part_of_data = data[(voxel['voxel_id'] != data['voxel_id']) & (data['x'].between(voxel['x']-1, voxel['x']+1)) & (data['y'].between(voxel['y']-1, voxel['y']+1)) & (data['z'].between(voxel['z']-1, voxel['z']+1))]
        
        # find all_neighbors
        neighbor_voxels = []
        for neighbor_voxel_id in part_of_data.index:
            neighbor_voxels.append(neighbor_voxel_id)
        data_all_neighbors.at[voxel_id, 'voxel_id'] = voxel_id
        data_all_neighbors.at[voxel_id, 'num_of_all_neighbors'] = len(neighbor_voxels)
        data_all_neighbors.at[voxel_id, 'all_neighbors'] = neighbor_voxels
        
        # find face_neighbors
        neighbor_voxels = []
        for neighbor_voxel_id in part_of_data[(abs(voxel['x'] - part_of_data['x']) + abs(voxel['y'] - part_of_data['y']) + abs(voxel['z'] - part_of_data['z'])) == 1].index:
            neighbor_voxels.append(neighbor_voxel_id)
        data_face_neighbors.at[voxel_id, 'voxel_id'] = voxel_id
        data_face_neighbors.at[voxel_id, 'num_of_face_neighbors'] = len(neighbor_voxels)
        data_face_neighbors.at[voxel_id, 'face_neighbors'] = neighbor_voxels
        
        # find edge_neighbors
        neighbor_voxels = []
        for neighbor_voxel_id in part_of_data[(abs(voxel['x']-part_of_data['x']) + abs(voxel['y']-part_of_data['y']) + abs(voxel['z']-part_of_data['z'])) == 2].index:
            neighbor_voxels.append(neighbor_voxel_id)
        data_edge_neighbors.at[voxel_id, 'voxel_id'] = voxel_id
        data_edge_neighbors.at[voxel_id, 'num_of_edge_neighbors'] = len(neighbor_voxels)
        data_edge_neighbors.at[voxel_id, 'edge_neighbors'] = neighbor_voxels
        
        # find vertex_neighbors
        neighbor_voxels = []
        for neighbor_voxel_id in part_of_data[(abs(voxel['x']-part_of_data['x']) + abs(voxel['y']-part_of_data['y']) + abs(voxel['z']-part_of_data['z'])) == 3].index:
            neighbor_voxels.append(neighbor_voxel_id)
        data_vertex_neighbors.at[voxel_id, 'voxel_id'] = voxel_id
        data_vertex_neighbors.at[voxel_id, 'num_of_vertex_neighbors'] = len(neighbor_voxels)
        data_vertex_neighbors.at[voxel_id, 'vertex_neighbors'] = neighbor_voxels
    
    # save image neighbor csvs
    data_all_neighbors.to_csv(path_to_save + name_of_file + '_all_neighbors.csv', index=False)
    data_face_neighbors.to_csv(path_to_save + name_of_file + '_face_neighbors.csv', index=False)
    data_edge_neighbors.to_csv(path_to_save + name_of_file + '_edge_neighbors.csv', index=False)
    data_vertex_neighbors.to_csv(path_to_save + name_of_file + '_vertex_neighbors.csv', index=False)   
    data_id_to_xyz.to_csv(path_to_save + name_of_file + '_id_to_xyz.csv', index=False) 
    
for file in sorted(glob.glob(os.path.join(path, '**/*.csv'), recursive=True)):
    fill_db(file)