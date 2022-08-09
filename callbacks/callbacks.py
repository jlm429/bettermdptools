# -*- coding: utf-8 -*-
"""
Author: John Mansfield
"""


class Callbacks:
    def __init__(self):
        pass

    def on_episode_begin(self, caller):
        pass

    def on_episode_end(self, caller):
        pass

    def on_episode(self, caller, episode):
        pass

    def on_env_step(self, caller):
        pass


class MyCallbacks(Callbacks):
    def __init__(self):
        pass

    def on_episode(self, caller, episode):
        # do things on specific episodes
        pass

    def on_episode_begin(self, caller):
        # do things on episode begin
        pass

    def on_episode_end(self, caller):
        # do things on episode end
        pass

    def on_env_step(self, caller):
        # do things on env. step
        pass
    