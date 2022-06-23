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


class MyCallbacks(Callbacks):
    def __init__(self):
        pass

    def on_episode(self, caller, episode):
        if episode % 1000 == 0:
            print(" episode=", episode)
            # caller.render = True
