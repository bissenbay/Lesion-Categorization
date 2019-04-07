import os, glob
import pandas as pd
import numpy as np
from ast import literal_eval
from google.colab import drive
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
drive.mount('/content/gdrive')

path = '/content/gdrive/My Drive/projects/spine/result/'

created_kmeans = True
scaler = None
centroids = None

def fill_db(file):    
    global created_kmeans
    global scaler
    global centroids
    
    path_to_save = file[:68] + '_clusters.csv'
    
    patient_image = pd.read_csv(file)
    X = patient_image[['face_nbrs', 'edge_nbrs', 'vertex_nbrs', 'tot_vxls_in_face_cmpnt', 'tot_vxls_in_edge_cmpnt', 'tot_vxls_in_vertex_cmpnt', 'int_vxls_in_face_cmpnt', 'int_vxls_in_edge_cmpnt', 'int_vxls_in_vertex_cmpnt']].values
    
    if created_kmeans:
      scaler = StandardScaler()
      scaler.fit(X)
      kmeans = KMeans(n_clusters=10).fit(scaler.transform(X))
      centroids = kmeans.cluster_centers_
      created_kmeans = False
    
    X = scaler.transform(X)
    
    distances = [[distance.euclidean(c, x) for c in centroids] for x in X]
    clusters = [np.argmin(d) for d in distances]

    cluster_data = pd.DataFrame(columns=['voxel_id', 'cluster_id'])
    cluster_data['voxel_id'] = patient_image['v_id'].values
    cluster_data['cluster_id'] = clusters
    cluster_data.to_csv(path_to_save, index=False)
    print(path_to_save)
        
for file in sorted(glob.glob(os.path.join(path, '**/*_features.csv'), recursive=True)):
    fill_db(file)