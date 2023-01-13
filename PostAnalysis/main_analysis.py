import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

from analysis_utils import AnalysisUtils

analysis_utils = AnalysisUtils()

path_to_em_csv = "/home/icub/Desktop/social_exclusion/output_emotion"
path_to_head_csv = "/home/icub/Desktop/social_exclusion/output_head"
all_files_em = glob.glob(os.path.join(path_to_em_csv , "*.csv"))
all_files_head = glob.glob(os.path.join(path_to_head_csv , "*.csv"))

# Create emotions dataframe
all_em_data = []
for file in all_files_em:
    emotions_data = (pd.read_csv(file, index_col=None, header=0, usecols=['Frame', 'Arousal', 'Valence', 'Face Confidence']))
    file_name = file.split('/')[-1].split('.')[0]
    emotions_data['Group'] = file_name.split('_')[0]
    emotions_data['Player'] = file_name.split('_')[-1]
    emotions_data[['Arousal', 'Valence']] = emotions_data[['Arousal', 'Valence']].replace([-99.0], 0)
    all_em_data.append(emotions_data)

em_df = pd.concat(all_em_data, axis=0, ignore_index=True)

#Create head dataframe
all_head_data = []
for file in all_files_head:
    head_data = (pd.read_csv(file, index_col=None, header=0, usecols=['Frame', 'Player', 'Angles', 'Face_heading_label']))
    file_name = file.split('/')[-1].split('.')[0]
    head_data['Group'] = file_name.split('_')[0]
    # head_data['Player'] = file_name.split('_')[-1]
    all_head_data.append(head_data)

head_df = pd.concat(all_head_data, axis=0, ignore_index=True)

# Plot Arousal and Valence for blue and green players
ar_green_players = em_df[["Arousal", "Group"]][em_df['Player'] == "green"]
val_green_players = em_df[["Valence", "Group"]][em_df['Player'] == "green"]

ar_blue_players = em_df[["Arousal", "Group"]][em_df['Player'] == "blue"]
val_blue_players = em_df[["Valence", "Group"]][em_df['Player'] == "green"]

colors = {'Group01':'tab:blue', 'Group02':'tab:orange', 'Group03':'tab:green'}

#cross plots
analysis_utils.draw_cross_plot(val_green_players['Valence'], ar_green_players['Arousal'])

#todo: check the colors
#ax.scatter(val_green_players['Valence'], ar_green_players['Arousal'], c=df['group'].map(colors))

# add a legend
# handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in colors.items()]
# ax.legend(title='color', handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
#
# plt.show()
