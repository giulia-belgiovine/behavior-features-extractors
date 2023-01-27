import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


class AnalysisUtils:

    def draw_cross_plot(self, x, y, c):
        plt.plot()
        ax = plt.gca()

        # cmap = plt.get_cmap('viridis')
        # color = cmap(np.linspace(0, 1, len(label)))

        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.scatter(x, y, color=c)

    # todo: create dictionary to add target column
    def add_target_column(self, heading_dataframe, new_column_label, player):

        # Create new NaN column for targets
        heading_dataframe[new_column_label] = "unknown"

        if player == "blue":

            # Dictionaries that link heading labels to specific targets for each player
            target_dict = {
                "DownLeft": "tablet",
                "ForwardLeft": "human",
                "ForwardDown": "robot",
                "Forward": "robot"
            }

        elif player == "green":

            # Dictionaries that link heading labels to specific targets for each player
            target_dict = {
                "DownRight": "tablet",
                "ForwardRight": "human",
                "ForwardDown": "robot",
                "Forward": "robot"
            }

        else:
            raise Exception("Invalid player. Only Blue and Green are allowed. Please check your CSV file.")

        for key in target_dict:
            mask = heading_dataframe[heading_dataframe.columns[3]].__eq__(key)
            heading_dataframe.loc[mask, new_column_label] = target_dict[key]


    def annotation_preprocessing(self, filename, header, frames_per_second):
        # Removes unnecessary columns and adds a first row with labels
        annotation_dataframe = pd.read_csv(filename, delimiter='\t', header=None)
        annotation_dataframe = annotation_dataframe.drop(columns=[0, 1, 2, 4])
        annotation_dataframe.columns = header
    
        # Adds two columns with the frame number in which the annotation (i) started, (ii) stopped
        annotation_dataframe['start_frame'] = (annotation_dataframe.time_start * frames_per_second).astype(int)
        annotation_dataframe['stop_frame'] = (annotation_dataframe.time_stop * frames_per_second).astype(int)

        annotation_dataframe = annotation_dataframe.drop(columns="time_start")
        annotation_dataframe = annotation_dataframe.drop(columns="time_stop")
    
        return annotation_dataframe

    def add_annotations(self, annotation_dict, final_dataframe, group_name):
        # Add a column called 'phase' in the head_dataframe and fill it with annotations
        final_dataframe['phase'] = "unknown"
        final_dataframe['exp_speaking'] = "false"  # boolean column

        # Create two sub-dataframe of the annotations
        tmp_dataframe = annotation_dict[group_name]
        expspeak_dataframe = tmp_dataframe[tmp_dataframe['label'] == "ExpSpeak"]
        phases_dataframe = tmp_dataframe[tmp_dataframe['label'] != "ExpSpeak"]
        # Re-adjust indexes
        phases_dataframe.index = list(range(0, phases_dataframe.shape[0]))

        # Fill the exp_speaking column in head_data dataframe
        if not expspeak_dataframe.empty:
            for index, row in expspeak_dataframe.iterrows():
                final_dataframe.loc[row.start_frame:row.stop_frame - 1, 'exp_speaking'] \
                    = 'true'

        # Fill the phase column in head_data dataframe
        if not phases_dataframe.empty:
            for index, row in phases_dataframe.iterrows():
                if index != phases_dataframe.shape[0] - 1:
                    final_dataframe.loc[row.start_frame:phases_dataframe.start_frame[index + 1] - 1, 'phase'] \
                        = phases_dataframe.loc[index, 'label']
                else:
                    final_dataframe.loc[row.start_frame, 'phase'] = phases_dataframe.loc[index, 'label']