# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap


class Plots:
    @staticmethod
    def values_heat_map(data, title, size):
        data = np.around(np.array(data).reshape(size), 2)
        df = pd.DataFrame(data=data)
        sns.heatmap(df, annot=True).set_title(title)
        plt.show()

    @staticmethod
    def v_iters_plot(data, title):
        df = pd.DataFrame(data=data)
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=df, legend=None).set_title(title)
        plt.show()

    #modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def get_policy_map(pi, val_max, actions, map_size):
        """Map the best learned action to arrows."""
        #convert pi to numpy array
        best_action = np.zeros(val_max.shape[0], dtype=np.int32)
        for idx, val in enumerate(val_max):
            best_action[idx] = pi[idx]
        policy_map = np.empty(best_action.flatten().shape, dtype=str)
        for idx, val in enumerate(best_action.flatten()):
            policy_map[idx] = actions[val]
        policy_map = policy_map.reshape(map_size[0], map_size[1])
        val_max = val_max.reshape(map_size[0], map_size[1])
        return val_max, policy_map

    #modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def plot_policy(val_max, directions, map_size, title):
        """Plot the policy learned."""
        sns.heatmap(
            val_max,
            annot=directions,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title=title)
        img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
        plt.show()
