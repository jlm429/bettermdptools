# -*- coding: utf-8 -*-
import math

import gym
import pygame
from algorithms.rl import QLearner as QL
from algorithms.planning import ValueIteration as VI
from algorithms.planning import PolicyIteration as PI
from examples.test_env import TestEnv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


class Plots:
    @staticmethod
    def basic_grid_policy_plot():
        pass

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

    # Q-learning v_iters_plot
    # QL = QL(frozen_lake.env)
    # Q, V, pi, Q_track, pi_track = QL.q_learning()
    # max_reward_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    # Plots.v_iters_plot(max_reward_per_iter, "Reward")

    # VI/PI v_iters_plot
    # V, V_track, pi = VI(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = PI(frozen_lake.env.P).policy_iteration()
    # max_value_per_iter = np.amax(V_track, axis=1)
    # Plots.v_iters_plot(max_value_per_iter, "Value")

    # Q-learning grid_values_heat_map
    # QL = QL(frozen_lake.env)
    # Q, V, pi, Q_track, pi_track = QL.q_learning()
    # Plots.grid_values_heat_map(V, "State Values")

    # VI/PI grid_values_heat_map
    V, V_track, pi = VI(frozen_lake.env.P).value_iteration()
    # V, V_track, pi = PI(frozen_lake.env.P).policy_iteration()
    Plots.grid_values_heat_map(V, "State Values")
