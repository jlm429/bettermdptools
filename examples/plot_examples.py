# -*- coding: utf-8 -*-

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


class PlotExamples:
    @staticmethod
    def policy_plot():
        pass

    @staticmethod
    def state_values_plot():
        pass

    @staticmethod
    def max_value_v_iters_plot():
        pass

    @staticmethod
    def reward_v_iters_plot(rewards):
        df = pd.DataFrame(data=rewards)
        df.columns = ["Reward"]
        sns.set_theme(style="whitegrid")
        sns.lineplot(x=df.index, y="Reward", data=df).set_title('Reward v Iterations')
        plt.show()


if __name__ == "__main__":

    frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

    # VI/PI
    #V, pi = VI(frozen_lake.env.P).value_iteration()
    #V, pi = PI(frozen_lake.env.P).policy_iteration()

    #Q-learning
    QL = QL(frozen_lake.env)
    Q, V, pi, Q_track, pi_track = QL.q_learning()

    max_reward_per_iter = np.amax(np.amax(Q_track, axis=2), axis=1)
    PlotExamples.reward_v_iters_plot(max_reward_per_iter)
