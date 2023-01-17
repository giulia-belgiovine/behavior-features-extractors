import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from analysis_utils import AnalysisUtils

analysis_utils = AnalysisUtils()

path_to_em_csv = "/home/icub/Desktop/social_exclusion/output_emotion"
path_to_head_csv = "/home/icub/Desktop/social_exclusion/output_head"
all_files_em = glob.glob(os.path.join(path_to_em_csv , "*.csv"))
all_files_head = glob.glob(os.path.join(path_to_head_csv , "*.csv"))


# Create emotions dataframe
em_data_dict = {}
for file in all_files_em:
    emotions_data = (pd.read_csv(file, index_col=None, header=0, usecols=['Frame', 'Arousal', 'Valence', 'Face Confidence']))
    emotions_data[['Arousal', 'Valence', 'Face Confidence']] = emotions_data[['Arousal', 'Valence', 'Face Confidence']].replace([-99.0], 0) #todo: sostituire con Nan
    file_name = file.split('/')[-1].split('.')[0]

    group_name = file_name.split('_')[0]
    if group_name not in em_data_dict:
        em_data_dict[group_name] = {}

    player = file_name.split('_')[-1]
    em_data_dict[group_name][player] = emotions_data


#Create head dataframe
head_data_dict = {}
for file in all_files_head:
    head_data = (pd.read_csv(file, index_col=None, header=0, usecols=['Frame', 'Player', 'Angles', 'Face_heading_label']))
    file_name = file.split('/')[-1].split('.')[0]

    group_name = file_name.split('_')[0]
    if group_name not in head_data_dict:
        head_data_dict[group_name] = {}

    player = file_name.split('_')[-1]
    #todo: aggiungere la colonna target cn una funzione
    head_data_dict[group_name][player] = head_data



#todo: Read annotations and add the column phase to each dataframe



mean_ar_blue = []
mean_val_blue= []
mean_ar_green = []
mean_val_green = []


#Plot Arousal, Valence
for key, val in em_data_dict.items():
    for player, values in em_data_dict[key].items():
        arousal = list(values['Arousal'])
        valence = list(values['Valence'])

        if player == 'blue':
            mean_ar_blue.append(np.mean(arousal))
            mean_val_blue.append(np.mean(valence))
        else:
            mean_ar_green.append(np.mean(arousal))
            mean_val_green.append(np.mean(valence))


analysis_utils.draw_cross_plot(mean_val_green, mean_ar_green, 'g')
analysis_utils.draw_cross_plot(mean_val_blue, mean_ar_blue, 'b')
plt.show()


