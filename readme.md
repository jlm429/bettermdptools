<h2>Reinforcement Learning (RL) Algorithms</h2>

The RL algorithms expect the number of states and actions to be passed in.   They will work out of the box with any [OpenAI Gym environment](https://www.gymlibrary.ml/)  that has a single discrete valued state space, like [frozen lake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/#observation-space).  
If the state space is not in this format, a lambda function is required to convert it.  For example, the [blackjack state space](https://www.gymlibrary.ml/environments/toy_text/blackjack/#observation-space) is "a 3-tuple containing: the player’s current sum, the value of the dealer’s one showing card (1-10 where 1 is ace), and whether the player holds a usable ace (0 or 1)." 

Here, blackjack.convert_state_obs is converting the 3-tuple into a single discrete space with 290 states by concatenating player states 0-28 (hard 4-21 & soft 12-21) with dealer states 0-9 (2-9, ten, ace).   

```
self.convert_state_obs = lambda state, done: ( -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(f"{state[0] - 4}{(state[1] - 2) % 10}"))
```

Q-learning blackjack example:
```
# Q-learning
QL = QL(blackjack.env)
Q, V, pi, Q_track, pi_track = QL.q_learning(n_states, n_actions, blackjack.convert_state_obs)
```
Q-learning and SARSA return the final action-value function Q, final state-value function V, final policy pi, and action-values Q_track and policies pi_track as a function of episodes.  

The default parameters for Q-learning and SARSA are: 


```
convert_state_obs=lambda state, done: state
gamma=.99
init_alpha=0.5
min_alpha=0.01
alpha_decay_ratio=0.5
init_epsilon=1.0
min_epsilon=0.1
epsilon_decay_ratio=0.9
n_episodes=10000 
```

<h2> Planning Algorithms </h2>

The planning algorithms, policy iteration (PI) and value iteration (VI), require an [OpenAI Gym](https://www.gymlibrary.ml/) discrete environment style transition and reward matrix (i.e., P[s][a]=[(prob, next, reward, done), ...]).  

Frozen Lake VI example:
```
env = gym.make('FrozenLake8x8-v1')
V, pi = VI().value_iteration(env.P)
```
PI and VI return the final state-value function V and final policy pi.  

The default parameters for VI and PI are: 
```
gamma=1.0 
theta=1e-10
```