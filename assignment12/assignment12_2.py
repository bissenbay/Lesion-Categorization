import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('patients_run_test.pdf') as pdf:
    
#    def draw_fig(hist, p, num):
#    plt.figure()
#    plt.bar(np.arange(10), hist, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .8, .1))
#    plt.title('Patient '+p+', '+str(num)+' images')  
#    plt.show()
    
    def draw_fig(coords, hist, num_imgs, label='', visible=False):
        plt.subplot(coords)
        plt.bar(np.arange(10), hist, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(0, .8, .1))
        plt.gca().axes.get_yaxis().set_visible(visible)
        plt.title(str(num_imgs)+' images')  
        plt.xlabel(label)
#        plt.show()
        
        
#        plt.figure()
#    
#    plt.bar(np.arange(10), start_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.title(str(start_imgs)+' images')

    patients = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46']
    images = [['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't21', 't22', 't23', 't24', 't25'], ['t02', 't03', 't04', 't05', 't06', 't07', 't08', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't15', 't20', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't10', 't11', 't12', 't13'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't21', 't22', 't23']]

#patients = ['S01']
#images = [['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24']]

    cluster_size = 10
    
    for p in range(len(patients)):
        clusters = []
        for i in images[p]:
            file = '../clusters/'+patients[p]+'_T2_'+i+'_clusters.csv'
            data = pd.read_csv(file)
            cluster = [len(data[data['cluster_id'] == i]) for i in range(cluster_size)]
            total = np.sum(cluster)
            cluster = [i/total for i in cluster]
            clusters.append(cluster)
            
    #  min_start = []
    #  min_mid = []
    #  min_end = []
    #  num_imgs_in_start = 0
    #  num_imgs_in_mid = 0
    #  num_imgs_in_end = 0

        dist_min = 1000
        c = len(clusters)
        for i in range(c):
            for j in range(i+2, c):
    #      print(list(range(i+1)), list(range(i+1, j)), list(range(j, c)))  
            
                start = [clusters[idx] for idx in range(i+1)]
                mid = [clusters[idx] for idx in range(i+1, j)]
                end = [clusters[idx] for idx in range(j, c)]
          
    #      start = [clusters[idx] for idx in np.arange(i+1)]
    #      start_mean = np.mean(start, axis=0)
    #      
    #      mid = [clusters[idx] for idx in np.arange(i+1, j)]
    #      mid_mean = np.mean(mid, axis=0)
    #      
    #      end = [clusters[idx] for idx in np.arange(j, c)]
    #      end_mean = np.mean(end, axis=0)
          
                start = np.asarray(start)
                mid = np.asarray(mid)
                end = np.asarray(end)
            
                start_mean = np.mean(start, axis=0)
                mid_mean = np.mean(mid, axis=0)
                end_mean = np.mean(end, axis=0)
          
    #      KMEANS TESTING
#                kmeans1 = KMeans(n_clusters=1).fit(start)
#                kmeans2 = KMeans(n_clusters=1).fit(mid)
#                kmeans3 = KMeans(n_clusters=1).fit(end)
#                print('kmeans.inertia_ = ', kmeans1.inertia_, kmeans2.inertia_, kmeans3.inertia_)
#                print('kmeans1.centroids = ', kmeans1.cluster_centers_)
#                print('kmeans2.centroids = ', kmeans2.cluster_centers_)
#                print('kmeans3.centroids = ', kmeans3.cluster_centers_)
                e1 = np.sum(np.square(np.linalg.norm(start - start_mean, ord=2, axis=1)))
                e2 = np.sum(np.square(np.linalg.norm(mid - mid_mean, ord=2, axis=1)))
                e3 = np.sum(np.square(np.linalg.norm(end - end_mean, ord=2, axis=1)))
#                print('e = ', e1, e2, e3)
#                print('mean = ', start_mean, mid_mean, end_mean)
          
    #      np.linalg.norm(start - start_mean, ord=2, axis=1)
    #      is equal to
    #      for i in start: print(np.linalg.norm(i - start_mean, ord=2))
          
    #      e1 = np.sum(np.square(np.linalg.norm(start - start_mean, ord=2)))
    #      e2 = np.sum(np.square(np.linalg.norm(mid - mid_mean, ord=2)))
    #      e3 = np.sum(np.square(np.linalg.norm(end - end_mean, ord=2)))
                dist = (e1 + e2 + e3)
          
                if dist < dist_min:
                    dist_min = dist
                    start_min = start_mean
                    mid_min = mid_mean
                    end_min = end_mean
                    start_imgs = len(start)
                    mid_imgs = len(mid)
                    end_imgs = len(end)
        
#      draw_fig(start_min, patients[p], start_imgs)
#      draw_fig(mid_min, patients[p], mid_imgs)
#      draw_fig(end_min, patients[p], end_imgs)
                    
#                    coords, hist, num_imgs, label='', visible=True
                    
        plt.figure()
        draw_fig(131, start_min, start_imgs, visible=True)
        draw_fig(132, mid_min, mid_imgs, 'Patient '+patients[p])
        draw_fig(133, end_min, end_imgs)
        pdf.savefig()
    
    plt.close()
    
#    plt.figure()
#    plt.subplot(131)
#    plt.bar(np.arange(10), start_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.title(str(start_imgs)+' images')
  
#    plt.subplot(132)
#    plt.bar(np.arange(10), mid_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.gca().axes.get_yaxis().set_visible(False)
#    plt.title(str(mid_imgs)+' images')  
#    plt.xlabel('Patient '+patients[p])
#      plt.annotate('Patient '+patients[p])
  
#    plt.subplot(133)
#    plt.bar(np.arange(10), end_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.gca().axes.get_yaxis().set_visible(False)
#    plt.title(str(end_imgs)+' images')  
  
#    plt.text(0.5, 0.5, 'asdas')
  
#    plt.figure()
#    plt.subplot(131)
#    plt.bar(np.arange(10), start_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.title(str(start_imgs)+' images')
#  
#    plt.subplot(132)
#    plt.bar(np.arange(10), mid_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.gca().axes.get_yaxis().set_visible(False)
#    plt.title(str(mid_imgs)+' images')  
#    plt.xlabel('Patient '+patients[p])
##      plt.annotate('Patient '+patients[p])
#  
#    plt.subplot(133)
#    plt.bar(np.arange(10), end_min, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#    plt.xticks(np.arange(10))
#    plt.yticks(np.arange(0, .5, .1))
#    plt.gca().axes.get_yaxis().set_visible(False)
#    plt.title(str(end_imgs)+' images')  
#  
#    plt.text(0.5, 0.5, 'asdas')
#  
#    pdf.savefig()
#    plt.close()

#  plt.figure()
#  plt.bar(np.arange(10), my_start, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#  plt.xticks(np.arange(10))
#  plt.yticks(np.arange(0, .8, .1))
#  plt.title('Patient '+patients[p]+', '+str(num_imgs_in_start)+' images')  
#  plt.show()
#  
#  plt.figure()
#  plt.bar(np.arange(10), my_mid, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#  plt.xticks(np.arange(10))
#  plt.yticks(np.arange(0, .8, .1))
#  plt.title('Patient '+patients[p]+', '+str(num_imgs_in_mid)+' images')  
#  plt.show()
#  
#  plt.figure()
#  plt.bar(np.arange(10), my_end, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
#  plt.xticks(np.arange(10))
#  plt.yticks(np.arange(0, .8, .1))
#  plt.title('Patient '+patients[p]+', '+str(num_imgs_in_end)+' images')  
#  plt.show()