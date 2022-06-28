<h2>Reinforcement Learning (RL) Algorithms</h2>

The RL algorithms work out of the box with any [OpenAI Gym environment](https://www.gymlibrary.ml/)  that has a single discrete valued state space, like [frozen lake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/#observation-space). 
A lambda function is required to convert the state space if not in this format.  For example, the [blackjack state space](https://www.gymlibrary.ml/environments/toy_text/blackjack/#observation-space) is "a 3-tuple containing: the player’s current sum, the value of the dealer’s one showing card (1-10 where 1 is ace), and whether the player holds a usable ace (0 or 1)." 

Here, blackjack.convert_state_obs converts the 3-tuple into a discrete space with 290 states by concatenating player states 0-28 (hard 4-21 & soft 12-21) with dealer states 0-9 (2-9, ten, ace).   

```
self.convert_state_obs = lambda state, done: ( -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(f"{state[0] - 4}{(state[1] - 2) % 10}"))
```

The number of states and actions are taken from the environment by default.  Since the number of states for blackjack is modified, this is passed in as well. 
```
# Q-learning
QL = QL(blackjack.env)
Q, V, pi, Q_track, pi_track = QL.q_learning(n_states, n_actions, blackjack.convert_state_obs)
```
Q-learning and SARSA return the final action-value function Q, final state-value function V, final policy pi, and action-values Q_track and policies pi_track as a function of episodes.  

<h3> Callbacks </h3>

SARSA and Q-learning have callback hooks for episode number, begin and end.   To create a callback, override one of the parent class methods in the child class MyCallbacks.  Here, on_episode prints the episode number and sets render to True every 1000 episodes.

```
class MyCallbacks(Callbacks):
    def __init__(self):
        pass

    def on_episode(self, caller, episode):
        if episode % 1000 == 0:
            print(" episode=", episode)
            caller.render = True
```


<h2> Planning Algorithms </h2>

The planning algorithms, policy iteration (PI) and value iteration (VI), require an [OpenAI Gym](https://www.gymlibrary.ml/) discrete environment style transition and reward matrix (i.e., P[s][a]=[(prob, next, reward, done), ...]).  

Frozen Lake VI example:
```
env = gym.make('FrozenLake8x8-v1')
V, pi = VI().value_iteration(env.P)
```
PI and VI return the final state-value function V and final policy pi.  
