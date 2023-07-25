# -*- coding: utf-8 -*-
import math

import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap


class Plots:
    @staticmethod
    def grid_world_policy_plot(data, label):
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            data = np.around(np.array(data).reshape((8, 8)), 2)
            df = pd.DataFrame(data=data)
            my_colors = ((0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
            cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
            ax = sns.heatmap(df, cmap=cmap, linewidths=1.0)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([.4, 1.1, 1.9, 2.6])
            colorbar.set_ticklabels(['Left', 'Down', 'Right', 'Up'])
            plt.title(label)
            plt.show()

    @staticmethod
    def grid_values_heat_map(data, label):
        if not math.modf(math.sqrt(len(data)))[0] == 0.0:
            warnings.warn("Grid map expected.  Check data length")
        else:
            data = np.around(np.array(data).reshape((8, 8)), 2)
            df = pd.DataFrame(data=data)
            sns.heatmap(df, annot=True).set_title(label)
            plt.show()

    @staticmethod
    def v_iters_plot(data, label):
        df = pd.DataFrame(data=data)
        df.columns = [label]
        sns.set_theme(style="whitegrid")
        title = label + " v Iterations"
        sns.lineplot(x=df.index, y=label, data=df).set_title(title)
        plt.show()

if __name__ == "__main__":
    frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

    # VI/PI grid_world_policy_plot
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy")

    # Q-learning grid_world_policy_plot
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # n_states = frozen_lake.env.observation_space.n
    # new_pi = list(map(lambda x: pi(x), range(n_states)))
    # s = int(math.sqrt(n_states))
    # Plots.grid_world_policy_plot(np.array(new_pi), "Grid World Policy")

    # Q-learning v_iters_plot
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # max_q_value_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    # Plots.v_iters_plot(max_q_value_per_iter, "Max Q-Values")

    # VI/PI v_iters_plot
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Max State Values")

    # Q-learning grid_values_heat_map
    # Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()
    # Plots.grid_values_heat_map(V, "State Values")

    # VI/PI grid_values_heat_map
    # V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
    V, V_track, pi = Planner(frozen_lake.env.P).policy_iteration()
    Plots.grid_values_heat_map(V, "State Values")
