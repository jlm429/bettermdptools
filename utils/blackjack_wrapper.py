"""
Author: John Mansfield

Blackjack wrapper that modifies the observation space and creates a transition/reward matrix P.

# Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
#    Observations:
#   There are 29 * 10 = 290 discrete observable states:
#    29 player hands: H4-H21, S12-S21, BJ (0-28)
#     H4   =  0
#     H5   =  1
#     H6   =  2
#     H7   =  3
#     H8   =  4
#     H9   =  5
#     H10  =  6
#     H11  =  7
#     H12  =  8
#     H13  =  9
#     H14  = 10
#     H15  = 11
#     H16  = 12
#     H17  = 13
#     H18  = 14
#     H19  = 15
#     H20  = 16
#     H21  = 17
#     S12  = 18
#     S13  = 19
#     S14  = 20
#     S15  = 21
#     S16  = 22
#     S17  = 23
#     S18  = 24
#     S19  = 25
#     S20  = 26
#     S21  = 27
#     BJ   = 28
#    Concatenated with 10 dealer cards: 2-9, T, A (0-9)
#     _2 = 0
#     _3 = 1
#     _4 = 2
#     _5 = 3
#     _6 = 4
#     _7 = 5
#     _8 = 6
#     _9 = 7
#     _T = 8 # 10, J, Q, K are all denoted as T
#     _A = 9
"""

import gymnasium as gym
import os
import pickle

class CustomTransformObservation(gym.ObservationWrapper):
    def __init__(self, env, func, observation_space):
        """
        Parameters
        ----------------------------
        env {gymnasium.Env}:
            Blackjack base environment to be wrapped

        func {lambda}:
            Function that converts the observation

        observation_space {gymnasium.spaces.Space}:
            New observation space
        """
        super().__init__(env)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def observation(self, observation):
        """
        Parameters
        ----------------------------
        observation {Tuple}:
            Blackjack base environment observation tuple

        Returns
        ----------------------------
        func(observation) {int}
        """
        return self.func(observation)

class BlackjackWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Parameters
        ----------------------------
        env {gymnasium.Env}:
            Blackjack base environment

        Explanation of convert_state_obs lambda:
        Lambda function assigned to the variable `self._convert_state_obs` takes parameter, `state` and
        converts the input into a compact single integer value by concatenating player hand with dealer card.
        See comments above for further information.

        """
        self._transform_obs = lambda obs: (
            int(f"{28}{(obs[1] - 2) % 10}") if (obs[0] == 21 and obs[2])
            else int(f"{27}{(obs[1] - 2) % 10}") if (obs[0] == 21 and not obs[2])
            else int(f"{obs[0] + 6}{(obs[1] - 2) % 10}") if obs[2]
            else int(f"{obs[0] - 4}{(obs[1] - 2) % 10}"))
        env = CustomTransformObservation(env, self._transform_obs, gym.spaces.Discrete(290))
        super().__init__(env)
        current_dir = os.path.dirname(__file__)
        file_name = 'blackjack-envP.pickle'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)

    @property
    def P(self):
        """
        Returns
        ----------------------------
        _P {dict}
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns
        ----------------------------
        _transform_obs {lambda}
        """
        return self._transform_obs
