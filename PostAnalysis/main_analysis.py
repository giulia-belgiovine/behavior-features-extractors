import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import json

from analysis_utils import AnalysisUtils

analysis_utils = AnalysisUtils()

# Paths to GiuliaB's PC
# path_to_em_csv = "/home/icub/Desktop/social_exclusion/output_emotion"
# path_to_head_csv = "/home/icub/Desktop/social_exclusion/output_head"
# path_to_annotation_txt = add path to annotations in GiuliaB's PC

# Paths to GiuliaP's PC
path_to_em_csv = "/home/gpusceddu/Documents/SocialExclusion/Outputs/output_emotion"
path_to_head_csv = "/home/gpusceddu/Documents/SocialExclusion/Outputs/output_head"
path_to_annotation_txt = "/home/gpusceddu/Documents/SocialExclusion/Annotation"

all_files_em = glob.glob(os.path.join(path_to_em_csv, "*.csv"))
all_files_head = glob.glob(os.path.join(path_to_head_csv, "*.csv"))
all_files_annotation = glob.glob(os.path.join(path_to_annotation_txt, "*.txt"))

# Fixed values
frames_per_second = 30

# Create annotation dictionary
annotation_dict = {}
for file in all_files_annotation:
    # Pre-process each annotation file
    annotation_dataframe_header = ['time_start', 'time_stop', 'label']
    annotation_dataframe = analysis_utils.annotation_preprocessing(file, annotation_dataframe_header, frames_per_second)

    file_name = file.split('/')[-1].split('.')[0]
    group_name = file_name.split('_')[0]
    if group_name not in annotation_dict:
        annotation_dict[group_name] = {}

    annotation_dict[group_name] = annotation_dataframe

# Create emotions dataframe
emotions_data_dict = {}
for file in all_files_em:
    emotions_data = (
        pd.read_csv(file, index_col=None, header=0, usecols=['Frame', 'Arousal', 'Valence', 'Face Confidence']))
    emotions_data[['Arousal', 'Valence', 'Face Confidence']] = emotions_data[
        ['Arousal', 'Valence', 'Face Confidence']].replace([-99.0], 0)  # todo: sostituire con Nan
    file_name = file.split('/')[-1].split('.')[0]

    group_name = file_name.split('_')[0]
    if group_name not in emotions_data_dict:
        emotions_data_dict[group_name] = {}

    player = file_name.split('_')[-1]

    # Add a 'target' column indicating what the player is looking at.
    analysis_utils.add_target_column(emotions_data, "target", player)

    # Merge the annotations with the current head dataframe
    analysis_utils.add_annotations(annotation_dict, emotions_data, group_name)

    emotions_data_dict[group_name][player] = emotions_data

# Create head dataframe
head_data_dict = {}
for file in all_files_head:
    head_data = (
        pd.read_csv(file, index_col=None, header=0, usecols=['Frame', 'Player', 'Angles', 'Face_heading_label']))
    file_name = file.split('/')[-1].split('.')[0]

    group_name = file_name.split('_')[0]
    if group_name not in head_data_dict:
        head_data_dict[group_name] = {}

    player = file_name.split('_')[-1]

    # Remove blank spaces from heading label column
    head_data[head_data.columns[3]] = head_data[head_data.columns[3]].str.replace(" ", "")

    # Pre-process each annotation file
    annotation_dataframe_header = ['time_start', 'time_stop', 'label']
    annotation_dataframe = analysis_utils.annotation_preprocessing(path_to_annotation_txt + "/" + group_name + ".txt",
                                                                   annotation_dataframe_header,
                                                                   frames_per_second)

    # Add a 'target' column indicating what the player is looking at.
    analysis_utils.add_target_column(head_data, "target", player)

    # Merge the annotations with the current head dataframe
    analysis_utils.add_annotations(annotation_dict, head_data, group_name)

    # Fill the field Group in the dictionary with the data from the head_data dataframe
    head_data_dict[group_name][player] = head_data

head_data_dict = dict(sorted(head_data_dict.items()))


mean_ar_blue = []
mean_val_blue = []
mean_ar_green = []
mean_val_green = []


# Plot Arousal, Valence
for key, val in emotions_data_dict.items():
    for player, values in emotions_data_dict[key].items():
        arousal = list(values['Arousal'])
        valence = list(values['Valence'])

        if player == 'blue':
            mean_ar_blue.append(np.mean(arousal))
            mean_val_blue.append(np.mean(valence))
        else:
            mean_ar_green.append(np.mean(arousal))
            mean_val_green.append(np.mean(valence))

# analysis_utils.draw_cross_plot(mean_val_green, mean_ar_green, 'g')
# analysis_utils.draw_cross_plot(mean_val_blue, mean_ar_blue, 'b')
# plt.show()
