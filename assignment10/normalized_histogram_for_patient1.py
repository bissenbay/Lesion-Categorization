import os, glob
import pandas as pd
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy
drive.mount('/content/gdrive')

path = '/content/gdrive/My Drive/projects/spine/result/S01/T2/'

def fill_db(file):
    patient_id = file[48:50]
    image_id = file[55:57]
    filename = file[58:68]
    data = pd.read_csv(file)
    height = [len(data[data['cluster_id'] == i]) for i in np.arange(10)]
    height = [i/scipy.linalg.norm(height) for i in height]
    plt.figure()
    plt.bar(np.arange(10), height, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(0, .9, .1))
    plt.title('Patient ' + patient_id + ', Image ' + image_id)
    plt.savefig('/content/gdrive/My Drive/projects/spine/images/S01/' + filename + '.png')

for file in sorted(glob.glob(os.path.join(path, '**/*_clusters.csv'), recursive=True)):
    fill_db(file)