# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Plots:
    """Static helpers for visualizing values, policies, and training traces."""

    @staticmethod
    def values_heat_map(data, title, size, show=True):
        """Plot state values as an annotated heat map."""
        data = np.around(np.array(data).reshape(size), 2)
        df = pd.DataFrame(data=data)
        sns.heatmap(df, annot=True).set_title(title)

        if show:
            plt.show()

    @staticmethod
    def v_iters_plot(data, title, show=True):
        """Plot one or more value traces across iterations."""
        df = pd.DataFrame(data=data)
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=df, legend=None).set_title(title)

        if show:
            plt.show()

    @staticmethod
    def get_values_agg_axis_means(pi, val_max, map_size, agg_axes):
        """Aggregate multidimensional maps over axes before visualization."""
        val_max, policy_map = Plots.get_policy_map(pi, val_max, None, map_size)
        for ax in agg_axes:
            policy_map = np.mean(policy_map, axis=ax)
        return policy_map

    # modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def get_policy_map(pi, val_max, actions, map_size):
        """Map the best learned action to arrows."""
        # convert pi to numpy array
        best_action = np.zeros(val_max.shape[0], dtype=np.int32)
        for idx, val in enumerate(val_max):
            best_action[idx] = pi[idx]
        policy_map = np.empty(best_action.flatten().shape, dtype=str)
        for idx, val in enumerate(best_action.flatten()):
            policy_map[idx] = actions[val] if actions is not None else val
        policy_map = policy_map.reshape(*map_size)
        val_max = val_max.reshape(*map_size)
        return val_max, policy_map

    # modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def plot_policy(val_max, directions, map_size, title, show=True):
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

        if show:
            plt.show()
