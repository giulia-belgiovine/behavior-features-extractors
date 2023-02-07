import numpy as np
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, font_size):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=font_size)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def prepare_data_for_spiderplot(head_data_dict, labels, players, phases):
    # Initialize data structure to plot: list of objects with the following structure:
    # data = [list_of_labels, ('Group00', [player_data], [blue_data])]
    # data = [['human', 'robot', 'tablet', 'unknown']]
    # players = {'green', 'blue'}
    data = [labels]

    # Loop on groups
    for key in list(head_data_dict.keys()):
        # Loop on players
        players_data = []
        for index, player in enumerate(players):
            # Select current player's data
            player_data = head_data_dict[key][player]
            # Select only Interactive phases, labeled with a finishing "I"
            mask = player_data['phase'].str.endswith(phases[index])
            player_data = player_data['target'].loc[mask]
            # Count occurrences for each target
            player_data = player_data.value_counts()
            # Normalize on the length of the Interactive phases (total number of frames labeled with 'phase')
            total_frames = player_data.sum()
            player_data = player_data / total_frames
            # If a label is missing add it and value it with 0
            if player_data.index.size < len(labels):
                missing_labels = set(labels) - set(player_data.index)
                # missing_labels = {'human', 'robot', 'tablet', 'unknown'} - set(player_data.index)
                for label in missing_labels:
                    player_data = pd.concat([player_data, pd.Series(data=[0], index=[label])])
                player_data = player_data.sort_index()
            player_values = list(player_data.values)
            players_data.append(player_values)
        # Append the data from current group to the total data structure
        data_tuple = (key, players_data)
        data.append(data_tuple)

    return data


def normalize_in_time(head_data_dict, players):
    target_quantile_dict = {}
    # Select data from annotation label "iCubStartsMoving" to "endGame". Delete the rest.
    for key in head_data_dict:
        target_quantile_dict[key] = {}
        start_index = list(head_data_dict[key]['green']['phase']).index("IcubStartsMoving")
        end_index = list(head_data_dict[key]['green']['phase']).index("EndGame")
        for player in players:
            target_quantile_dict[key][player] = {}
            head_data_dict[key][player] = head_data_dict[key][player].loc[int(start_index):int(end_index)]
            # Split data in 10% slices (10-quantiles)
            quantiles = round(head_data_dict[key][player].quantile([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]))
            target_dataframe = pd.DataFrame(columns={'human', 'robot', 'tablet', 'unknown'})

            for start_index, end_index in zip(quantiles.values[:-1], quantiles.values[1:]):
                target_count = head_data_dict[key][player]['target'].loc[int(start_index):int(end_index)].value_counts()

                if 'human' in target_count:
                    human_count = target_count['human']
                else:
                    human_count = 0
                if 'robot' in target_count:
                    robot_count = target_count['robot']
                else:
                    robot_count = 0
                if 'tablet' in target_count:
                    tablet_count = target_count['tablet']
                else:
                    tablet_count = 0
                if 'unknown' in target_count:
                    unknown_count = target_count['unknown']
                else:
                    unknown_count = 0

                new_row = pd.DataFrame({'human': [human_count], 'robot': [robot_count],
                                        'tablet': [tablet_count], 'unknown': [unknown_count]})
                target_dataframe = pd.concat([target_dataframe, new_row])
            target_quantile_dict[key][player] = target_dataframe
    return target_quantile_dict
