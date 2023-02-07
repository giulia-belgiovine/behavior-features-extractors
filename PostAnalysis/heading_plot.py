import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import main_analysis
import plot_utils as pu


if __name__ == '__main__':

    quantile_target_dict = pu.normalize_in_time(main_analysis.head_data_dict, {'blue', 'green'})

    plt.style.use('_mpl-gallery')
    x = np.arange(1,11)
    # size and color:
    sizes = np.random.uniform(15, 80, len(x))
    colors = np.random.uniform(15, 80, len(x))

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x, quantile_target_dict['Group01']['blue']['robot'], s=sizes, c=colors, vmin=0, vmax=100)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()


    spoke_labels = ['human', 'robot', 'tablet', 'unknown']
    theta = pu.radar_factory(len(spoke_labels), frame='circle')

    ### PLOT PLAYERS IN INTERACTION PHASE IN GENERAL ###
    data = pu.prepare_data_for_spiderplot(main_analysis.head_data_dict, spoke_labels, ['blue', 'green'], ['I', 'I'])
    # Remove label list from data
    data.pop(0)

    # Number of rows and columns of the subplot
    subplot_cols = 3
    subplot_rows = math.ceil(len(data) / subplot_cols)

    # Plot subplot's grids
    fig, axs = plt.subplots(figsize=(15, 15), nrows=subplot_rows, ncols=subplot_cols,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.90, bottom=0.05)

    # Delete unnecessary subplots
    if subplot_rows*subplot_cols - len(data) > 0:
        for ax in axs.flat[len(data):]:
            fig.delaxes(ax)

    colors = ['b', 'g']

    # Plot data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        # Ticks on axes
        ax.set_rgrids([0.2, 0.4, 0.6])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels, font_size='xx-small')
        ax.set_ylim(0, 0.75)

    # add legend relative to top-left plot
    labels = ('Blue player', 'Green player')
    legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Targets for each group',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    plt.show()

    ### PLOT PLAYERS IN THEIR TURN ###
    data = pu.prepare_data_for_spiderplot(main_analysis.head_data_dict, spoke_labels, ['blue', 'green'], ['BI', 'GI'])
    # Remove label list from data
    data.pop(0)

    # Number of rows and columns of the subplot
    subplot_cols = 3
    subplot_rows = math.ceil(len(data) / subplot_cols)

    fig, axs = plt.subplots(figsize=(15, 15), nrows=subplot_rows, ncols=subplot_cols,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.90, bottom=0.05)

    # Delete unnecessary subplots
    if subplot_rows*subplot_cols - len(data) > 0:
        for ax in axs.flat[len(data):]:
            fig.delaxes(ax)

    colors = ['b', 'g']
    # Plot data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        # Ticks on axes
        ax.set_rgrids([0.2, 0.4, 0.6])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels, font_size='xx-small')
        ax.set_ylim(0, 0.75)

    # add legend relative to top-left plot
    labels = ('Blue player', 'Green player')
    legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Targets for each group - own phase',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    plt.show()

    ### PLOT PLAYERS IN OTHER'S TURN ###
    data = pu.prepare_data_for_spiderplot(main_analysis.head_data_dict, spoke_labels, ['blue', 'green'], ['GI', 'BI'])
    # Remove label list from data
    data.pop(0)

    # Number of rows and columns of the subplot
    subplot_cols = 3
    subplot_rows = math.ceil(len(data) / subplot_cols)

    fig, axs = plt.subplots(figsize=(15, 15), nrows=subplot_rows, ncols=subplot_cols,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.90, bottom=0.05)

    # Delete unnecessary subplots
    if subplot_rows*subplot_cols - len(data) > 0:
        for ax in axs.flat[len(data):]:
            fig.delaxes(ax)

    colors = ['b', 'g']
    # Plot data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        # Ticks on axes
        ax.set_rgrids([0.2, 0.4, 0.6])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels, font_size='xx-small')
        ax.set_ylim(0, 0.75)

    # add legend relative to top-left plot
    labels = ('Blue player', 'Green player')
    legend = axs[0, 0].legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Targets for each group - phase of other player',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    plt.show()