import os, glob
import pandas as pd
from ast import literal_eval
from google.colab import drive
drive.mount('/content/gdrive')

# set path
path = '/content/gdrive/My Drive/projects/spine/result/'

def fill_db(file):    
    path_to_save = file[:68] + '_features.csv'
    
    if os.path.isfile(path_to_save):
        return
    
    data_face_neighbors = pd.read_csv(file[:-17] + 'face_neighbors.csv')
    data_edge_neighbors = pd.read_csv(file[:-17] + 'edge_neighbors.csv')
    data_vertex_neighbors = pd.read_csv(file[:-17] + 'vertex_neighbors.csv')
    
    component_data_face_neighbors = pd.read_csv(file[:-17] + 'face_nbrs_ccs.csv')
    component_data_edge_neighbors = pd.read_csv(file[:-17] + 'edge_nbrs_ccs.csv')
    component_data_vertex_neighbors = pd.read_csv(file[:-17] + 'vertex_nbrs_ccs.csv')
    
#     v_id = voxel_id

#     face_nbrs = number of face neighbors of the current voxel
#     edge_nbrs = number of edge neighbors of the current voxel
#     vertex_nbrs = number of vertex neighbors of the current voxel
    
#     tot_vxls_in_face_cmpnt = total number of voxels in the face component to which the current voxel relates
#     tot_vxls_in_edge_cmpnt = total number of voxels in the edge component to which the current voxel relates
#     tot_vxls_in_vertex_cmpnt = total number of voxels in the vertex component to which the current voxel relates
    
#     int_vxls_in_face_cmpnt = total number of internal voxels in the face component to which the current voxel relates
#     int_vxls_in_edge_cmpnt = total number of internal voxels in the edge component to which the current voxel relates
#     int_vxls_in_vertex_cmpnt = total number of internal voxels in the vertex component to which the current voxel relates
    
    data_features = pd.DataFrame(columns=['v_id', 'face_nbrs', 'edge_nbrs', 'vertex_nbrs', 
                                          'tot_vxls_in_face_cmpnt', 'tot_vxls_in_edge_cmpnt', 'tot_vxls_in_vertex_cmpnt', 
                                          'int_vxls_in_face_cmpnt', 'int_vxls_in_edge_cmpnt', 'int_vxls_in_vertex_cmpnt'])
    
    def createVoxelToComponentMapping(component_data):
        voxel_to_component_mapping = {}
        for component_id in component_data.index:
            voxels = literal_eval(component_data.at[component_id, 'voxels'])
            for neighbor_voxel_id in voxels:
                voxel_to_component_mapping[neighbor_voxel_id] = (component_id + 1)
        return voxel_to_component_mapping
      
    voxel_to_component_of_face_nbrs = createVoxelToComponentMapping(component_data_face_neighbors)
    voxel_to_component_of_edge_nbrs = createVoxelToComponentMapping(component_data_edge_neighbors)
    voxel_to_component_of_vertex_nbrs = createVoxelToComponentMapping(component_data_vertex_neighbors)
    
    for voxel_id in data_face_neighbors.index:
        data_features.at[voxel_id, 'v_id'] = voxel_id
        
        data_features.at[voxel_id, 'face_nbrs'] = data_face_neighbors.at[voxel_id, 'num_of_face_neighbors']
        data_features.at[voxel_id, 'edge_nbrs'] = data_edge_neighbors.at[voxel_id, 'num_of_edge_neighbors']
        data_features.at[voxel_id, 'vertex_nbrs'] = data_vertex_neighbors.at[voxel_id, 'num_of_vertex_neighbors']
        
        data_features.at[voxel_id, 'tot_vxls_in_face_cmpnt'] = component_data_face_neighbors.at[(voxel_to_component_of_face_nbrs[voxel_id] - 1), 'total_voxels']
        data_features.at[voxel_id, 'tot_vxls_in_edge_cmpnt'] = component_data_edge_neighbors.at[(voxel_to_component_of_edge_nbrs[voxel_id] - 1), 'total_voxels']
        data_features.at[voxel_id, 'tot_vxls_in_vertex_cmpnt'] = component_data_vertex_neighbors.at[(voxel_to_component_of_vertex_nbrs[voxel_id] - 1), 'total_voxels']
        
        data_features.at[voxel_id, 'int_vxls_in_face_cmpnt'] = component_data_face_neighbors.at[(voxel_to_component_of_face_nbrs[voxel_id] - 1), 'internal_voxels']
        data_features.at[voxel_id, 'int_vxls_in_edge_cmpnt'] = component_data_edge_neighbors.at[(voxel_to_component_of_edge_nbrs[voxel_id] - 1), 'internal_voxels']
        data_features.at[voxel_id, 'int_vxls_in_vertex_cmpnt'] = component_data_vertex_neighbors.at[(voxel_to_component_of_vertex_nbrs[voxel_id] - 1), 'internal_voxels']
        
    data_features.to_csv(path_to_save, index=False)
    print(path_to_save)
        
for file in sorted(glob.glob(os.path.join(path, '**/*_all_neighbors.csv'), recursive=True)):
    fill_db(file)