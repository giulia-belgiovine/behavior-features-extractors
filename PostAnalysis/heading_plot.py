import numpy as np
import matplotlib.pyplot as plt
import math
import main_analysis
import plot_utils as pu

if __name__ == '__main__':

    # --- SELECT IN THIS SECTION THE PLOTS YOU WANT TO BE GENERATED ---
    # Evolution of heading in time
    quantile_stem_plot = False
    turn_stem_plot = True
    line_plot = False
    overall_spider_plots = False
    own_turn_spider_plots = False
    other_turn_spider_plot = False

    # --- PARAMETERS ---
    players = ['blue', 'green']
    percentile = 25

    # --- DATA PREPARATION ---
    quantile_target_dict, x_axis_ticks, x_labels = pu.heading_in_time(main_analysis.head_data_dict, players, percentile)

    turn_target_dict = pu.heading_in_turns(main_analysis.head_data_dict, main_analysis.annotation_dict, players,
                                           ['BI', 'GI'])

    spoke_labels = ['human', 'robot', 'tablet', 'unknown']
    theta = pu.radar_factory(len(spoke_labels), frame='circle')

    if turn_stem_plot:
        for group in turn_target_dict.keys():
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 9))
            fig.subplots_adjust(top=0.90, bottom=0.1)
            # For each player
            for player, ax in zip(players, axs.flat, ):
                # For each target
                for target, color, x_slide in zip(['robot', 'human', 'tablet'], ['red', 'navy', 'dimgray'],
                                                  [-0.2, 0, 0.2]):
                    x = turn_target_dict[group][player].index
                    markerline, stemline, baseline = ax.stem(x + x_slide, turn_target_dict[group][player][target],
                                                             linefmt=color, label=target)
                    plt.setp(markerline, markersize=4.5)
                    plt.setp(stemline, linewidth=1)
                    plt.setp(baseline, visible=False)  # Make baseline invisible

                # Plot setup
                ax.set(xticks=x, ylim=(0, 1.05), yticks=np.arange(0, 1.1, 0.1),
                       xlabel='Rounds', ylabel='% [time heading target / total round time]',
                       title='Targets headed during the interaction - ' + group + ' - ' + player)

                ax.set_yticklabels(np.arange(0, 101, 10))

                grid_xticks = x[:-1] + 0.5
                ax.set_xticks(grid_xticks, minor=True)
                ax.grid(axis='x', which='minor')

                ax.legend()

            plt.savefig('/home/gpusceddu/Documents/SocialExclusion/Plots/turn_heading_plots/' + group + '.png')
            plt.show()

    if quantile_stem_plot:
        x = quantile_target_dict['Group01']['blue'].index
        # For each group
        for group in quantile_target_dict.keys():
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 9))
            fig.subplots_adjust(top=0.90, bottom=0.1)
            # For each player
            for player, ax in zip(players, axs.flat, ):
                # For each target
                for target, color, x_slide in zip(['robot', 'human', 'tablet'], ['red', 'navy', 'dimgray'],
                                                  [-percentile / 5, 0, percentile / 5]):
                    markerline, stemline, baseline = ax.stem(x + x_slide,
                                                             quantile_target_dict[group][player][target],
                                                             linefmt=color, label=target)
                    plt.setp(markerline, markersize=4.5)
                    plt.setp(stemline, linewidth=1)
                    plt.setp(baseline, visible=False)  # Make baseline invisible

                # Plot setup
                ax.set(xlim=(0, 100), xticks=x_axis_ticks, ylim=(0, 1.05), yticks=np.arange(0, 1.1, 0.1),
                       xlabel='10 %-Quantiles', ylabel='% [time heading target / total quantile time]',
                       title='Targets headed during the interaction - ' + group + ' - ' + player)

                ax.set_xticklabels(x_labels[1:])
                ax.set_yticklabels(np.arange(0, 110, 10))

                grid_xticks = x_labels[1:]
                ax.set_xticks(grid_xticks, minor=True)
                ax.grid(axis='x', which='minor')

                plt.axhline(y=0.0, color='k', linestyle='-', linewidth=.5)  # Horizontal line to split the plot
                ax.legend()

            plt.savefig('/home/gpusceddu/Documents/SocialExclusion/Plots/quantile_heading_plots/' + group + '.png')

    if line_plot:
        fig, ax = plt.subplots()
        x = quantile_target_dict['Group01']['blue'].index

        # Plot blue player's targets
        # Robot
        ax.plot(x, quantile_target_dict['Group01']['blue']['robot'], color='red', label='Robot', marker='o')
        # Human
        ax.plot(x, quantile_target_dict['Group01']['blue']['human'], color='navy', label='Human', marker='o')
        # Tablet
        ax.plot(x, quantile_target_dict['Group01']['blue']['tablet'], color='dimgray', label='Tablet', marker='o')
        # Plot green player's targets (upside-down)
        # Robot
        ax.plot(x, -quantile_target_dict['Group01']['green']['robot'], color='red', marker='o')
        # Human
        ax.plot(x, -quantile_target_dict['Group01']['green']['human'], color='navy', marker='o')
        # Tablet
        ax.plot(x, -quantile_target_dict['Group01']['green']['tablet'], color='dimgray', marker='o')

        # Plot setup
        ax.set(xlim=(0, 100), xticks=np.arange(5, 96, 10), ylim=(-1.05, 1.05), yticks=np.arange(-1, 1.1, 0.1),
               xlabel='10 %-Quantiles', ylabel='% [time heading target / total quantile time]',
               title='Targets headed during the interaction')
        ax.set_xticklabels(np.arange(10, 101, 10))
        ax.set_yticklabels(np.concatenate((np.arange(100, 0, -10), np.arange(0, 110, 10))))

        grid_xticks = np.arange(10, 101, 10)
        ax.set_xticks(grid_xticks, minor=True)
        ax.grid(axis='x', which='minor')

        plt.axhline(y=0.0, color='k', linestyle='-', linewidth=.5)  # Horizontal line to split the plot
        ax.legend()

        plt.show()

    # --- PLOT PLAYERS IN INTERACTION PHASE IN GENERAL ---
    if overall_spider_plots:
        data = pu.prepare_data_for_spiderplot(main_analysis.head_data_dict, spoke_labels, ['blue', 'green'], ['I', 'I'])
        # Remove label list from data
        data.pop(0)

        # Number of rows and columns of the subplot
        subplot_cols = 5
        subplot_rows = math.ceil(len(data) / subplot_cols)

        # Plot subplot's grids
        fig, axs = plt.subplots(figsize=(15, 10), nrows=subplot_rows, ncols=subplot_cols,
                                subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.90, bottom=0.05)

        # Delete unnecessary subplots
        if subplot_rows * subplot_cols - len(data) > 0:
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

    # --- PLOT PLAYERS IN THEIR TURN ---
    if own_turn_spider_plots:
        data = pu.prepare_data_for_spiderplot(main_analysis.head_data_dict, spoke_labels, ['blue', 'green'],
                                              ['BI', 'GI'])
        # Remove label list from data
        data.pop(0)

        # Number of rows and columns of the subplot
        subplot_cols = 5
        subplot_rows = math.ceil(len(data) / subplot_cols)

        fig, axs = plt.subplots(figsize=(15, 10), nrows=subplot_rows, ncols=subplot_cols,
                                subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.90, bottom=0.05)

        # Delete unnecessary subplots
        if subplot_rows * subplot_cols - len(data) > 0:
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

    # --- PLOT PLAYERS IN OTHER'S TURN ---
    if other_turn_spider_plot:
        data = pu.prepare_data_for_spiderplot(main_analysis.head_data_dict, spoke_labels, ['blue', 'green'],
                                              ['GI', 'BI'])
        # Remove label list from data
        data.pop(0)

        # Number of rows and columns of the subplot
        subplot_cols = 5
        subplot_rows = math.ceil(len(data) / subplot_cols)

        fig, axs = plt.subplots(figsize=(15, 10), nrows=subplot_rows, ncols=subplot_cols,
                                subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.5, hspace=0.8, top=0.90, bottom=0.05)

        # Delete unnecessary subplots
        if subplot_rows * subplot_cols - len(data) > 0:
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
