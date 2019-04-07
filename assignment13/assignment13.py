import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('../data_run_4/patients_run_4.pdf') as pdf:
    
    def draw_sep_fig(hist, title, fname):
        plt.figure()
        plt.grid(True, axis='y', color='black', linestyle=':', linewidth=.3)
        plt.bar(np.arange(10), hist, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(0, .8, .1))
        plt.title(title) 
        plt.savefig(fname)
        
    def draw_fig(coords, stg, hist, num_imgs):
        plt.subplot(coords)
        plt.grid(True, axis='y', color='black', linestyle=':', linewidth=.3)
        plt.bar(np.arange(10), hist, color=['lightcoral', 'lightsalmon', 'sandybrown', 'darkkhaki', 'darkseagreen', 'mediumturquoise', 'skyblue', 'cadetblue', 'slategray', 'mediumorchid'])
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(0, .8, .1))
        plt.title(str(stg)+' stage, '+str(num_imgs)+' images') 
        
    def draw_table(coords, tbl):
        plt.subplot(coords)
        plt.axis('off')
        plt.axis('tight')
        table = plt.table(cellText=tbl, cellLoc='center', rowLabels=['s1', 's2', 's3'], colLabels=list('0123456789')+['imgs'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(5)
        
    writer = pd.ExcelWriter('../data_run_4/patients_run_4.xlsx', engine='xlsxwriter')

    patients = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S43', 'S44', 'S45', 'S46']
    images = [['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't21', 't22', 't23', 't24', 't25'], ['t02', 't03', 't04', 't05', 't06', 't07', 't08', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't15', 't20', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25', 't26'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24', 't25'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't10', 't11', 't12', 't13'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08'], ['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't21', 't22', 't23']]
    
#    patients = ['S01']
#    images = [['t01', 't02', 't03', 't04', 't05', 't06', 't07', 't08', 't09', 't10', 't11', 't12', 't13', 't14', 't15', 't16', 't17', 't18', 't19', 't20', 't21', 't22', 't23', 't24']]

    cluster_size = 10
    
    for p in range(len(patients)):
        clusters = []
        for i in images[p]:
            file = '../clusters_run_4/'+patients[p]+'_T2_'+i+'_clusters.csv'
            data = pd.read_csv(file)
            cluster = [len(data[data['cluster_id'] == i]) for i in range(cluster_size)]
            total = np.sum(cluster)
            cluster = [i/total for i in cluster]
#            generating a png
            draw_sep_fig(cluster, 'Patient '+patients[p]+', Image '+i, '../data_run_4/images/'+patients[p]+'_T2_'+i+'.png')
            clusters.append(cluster)
            
        dist_min = 1000
        c = len(clusters)
        for i in range(c):
            for j in range(i+2, c):
            
                start = [clusters[idx] for idx in range(i+1)]
                mid = [clusters[idx] for idx in range(i+1, j)]
                end = [clusters[idx] for idx in range(j, c)]
          
                start = np.asarray(start)
                mid = np.asarray(mid)
                end = np.asarray(end)
            
                start_mean = np.mean(start, axis=0)
                mid_mean = np.mean(mid, axis=0)
                end_mean = np.mean(end, axis=0)
          
                e1 = np.sum(np.square(np.linalg.norm(start - start_mean, ord=2, axis=1)))
                e2 = np.sum(np.square(np.linalg.norm(mid - mid_mean, ord=2, axis=1)))
                e3 = np.sum(np.square(np.linalg.norm(end - end_mean, ord=2, axis=1)))
                
                dist = (e1 + e2 + e3)
          
                if dist < dist_min:
                    dist_min = dist
                    start_min = start_mean
                    mid_min = mid_mean
                    end_min = end_mean
                    start_imgs = len(start)
                    mid_imgs = len(mid)
                    end_imgs = len(end)
                    
#        generating separate png files
        draw_sep_fig(start_min, 'Patient '+patients[p]+', Stage 1, '+str(start_imgs)+' Images', '../data_run_4/stages/'+patients[p]+'_T2_s1.png')
        draw_sep_fig(mid_min, 'Patient '+patients[p]+', Stage 2, '+str(mid_imgs)+' Images', '../data_run_4/stages/'+patients[p]+'_T2_s2.png')
        draw_sep_fig(end_min, 'Patient '+patients[p]+', Stage 3, '+str(end_imgs)+' Images', '../data_run_4/stages/'+patients[p]+'_T2_s3.png')
        
#        generating pdf
        '''
        plt.figure()
        figure = plt.gcf()
        figure.set_size_inches([7, 10])
        plt.suptitle('Patient '+patients[p])
        plt.rc('xtick', labelsize=5)
        plt.rc('ytick', labelsize=5) 
        plt.rc('axes', titlesize=8)
        draw_fig(231, 1, start_min, start_imgs)
        draw_fig(232, 2, mid_min, mid_imgs)
        draw_fig(233, 3, end_min, end_imgs)
        '''
        
#        generating tables
        '''
        tbl_1 = np.vstack((start_min, mid_min, end_min)).round(2).tolist()
        tbl_1[0].append(start_imgs)
        tbl_1[1].append(mid_imgs)
        tbl_1[2].append(end_imgs)
        
        df1 = pd.DataFrame(tbl_1)
        df1.rename(columns={10: '# images'}, inplace=True)
        df1.index = ['stage1', 'stage2', 'stage3']
        df1.to_excel(writer, sheet_name=patients[p]+'_tbl1')
        '''
        
#        drawing tables
        '''
        draw_table(223, tbl_1)
        
        for i in range(3):
            for j in range(10):
                if tbl_1[i][j] <= 0.02: tbl_1[i][j] = '*'
                elif tbl_1[i][j] > 0.02 and tbl_1[i][j] <= 0.2: tbl_1[i][j] = '-'
                else: tbl_1[i][j] = '+'
                
        df2 = pd.DataFrame(tbl_1)
        df2.rename(columns={10: '# images'}, inplace=True)
        df2.index = ['stage1', 'stage2', 'stage3']
        df2.to_excel(writer, sheet_name=patients[p]+'_tbl2')
                
        draw_table(224, tbl_1)
        '''
        
        pdf.savefig()
    
    plt.close()
    writer.save()    