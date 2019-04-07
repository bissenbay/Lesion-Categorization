import os, glob
import pandas as pd
import numpy as np
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy
drive.mount('/content/gdrive')

path = '/content/gdrive/My Drive/projects/spine/result/'

patient_id = '01'
height = [0] * 10

def fill_db(file):
    global patient_id
    global height
    t_patient_id = file[48:50]
    if t_patient_id != patient_id:
      plt.figure()
      plt.bar(np.arange(10), height, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
      plt.xticks(np.arange(10))
      plt.yticks(np.arange(0, 410000, 50000))
      plt.title('Patient ' + patient_id)
      plt.savefig('/content/gdrive/My Drive/projects/spine/images/' + patient_id + '.png')
      patient_id = t_patient_id
      height = [0] * 10
    data = pd.read_csv(file)
    height = [(x + y) for x, y in zip(height, [len(data[data['cluster_id'] == i]) for i in np.arange(10)])]

for file in sorted(glob.glob(os.path.join(path, '**/*_clusters.csv'), recursive=True)):
    fill_db(file)

plt.figure()
plt.bar(np.arange(10), height, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
plt.xticks(np.arange(10))
plt.yticks(np.arange(0, 410000, 50000))
plt.title('Patient ' + patient_id)
plt.savefig('/content/gdrive/My Drive/projects/spine/images/' + patient_id + '.png')