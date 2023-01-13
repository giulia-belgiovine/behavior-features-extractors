import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class AnalysisUtils:

    def draw_cross_plot(self, x, y):
        plt.plot()
        ax = plt.gca()


        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.scatter(x, y)
        plt.show()
