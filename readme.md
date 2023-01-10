# bettermdptools

1. [Getting Started](#getting-started)
2. [API](#api)
3. [Contributing](#contributing)

## Getting Started
pip install or git clone bettermdptools.   

```
pip3 install bettermdptools
```

```
git clone https://github.com/jlm429/bettermdptools
```

Here's a quick Q-learning example using OpenAI's frozen lake environment. See bettermdptools/examples for more.  

```
import gym
import pygame
from algorithms.rl import QLearner as QL
from examples.test_env import TestEnv

frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

# Q-learning
QL = QL(frozen_lake.env)
Q, V, pi, Q_track, pi_track = QL.q_learning()

test_scores = TestEnv.test_env(env=frozen_lake.env, render=True, user_input=False, pi=pi)
```

#### Planning Algorithms

The planning algorithms, policy iteration (PI) and value iteration (VI), require an [OpenAI Gym](https://www.gymlibrary.ml/) discrete environment style transition and reward matrix (i.e., P[s][a]=[(prob, next, reward, done), ...]).  

Frozen Lake VI example:
```
env = gym.make('FrozenLake8x8-v1')
V, V_track, pi = VI(env.P).value_iteration()
```

#### Reinforcement Learning (RL) Algorithms

The RL algorithms (Q-learning, SARSA) work out of the box with any [OpenAI Gym environment](https://www.gymlibrary.ml/)  that have single discrete valued state spaces, like [frozen lake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/#observation-space). 
A lambda function is required to convert state spaces not in this format.  For example, [blackjack](https://www.gymlibrary.ml/environments/toy_text/blackjack/#observation-space) is "a 3-tuple containing: the player’s current sum, the value of the dealer’s one showing card (1-10 where 1 is ace), and whether the player holds a usable ace (0 or 1)." 

Here, blackjack.convert_state_obs changes the 3-tuple into a discrete space with 280 states by concatenating player states 0-27 (hard 4-21 & soft 12-21) with dealer states 0-9 (2-9, ten, ace).   

```
self.convert_state_obs = lambda state, done: ( -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(f"{state[0] - 4}{(state[1] - 2) % 10}"))
```
 
Since n_states is modified by the state conversion, this new value is passed in along with n_actions, and convert_state_obs.    
  
```
# Q-learning
QL = QL(blackjack.env)
Q, V, pi, Q_track, pi_track = QL.q_learning(blackjack.n_states, blackjack.n_actions, blackjack.convert_state_obs)
```

#### Callbacks 

SARSA and Q-learning have callback hooks for episode number, begin, end, and env. step.   To create a callback, override one of the parent class methods in the child class MyCallbacks.  Here, on_episode prints the episode number every 1000 episodes.

```
class MyCallbacks(Callbacks):
    def __init__(self):
        pass

    def on_episode(self, caller, episode):
        if episode % 1000 == 0:
            print(" episode=", episode)
```

Or, you can use the add_to decorator and define the override outside of the class definition. 

```
from decorators.decorators import add_to
from callbacks.callbacks import MyCallbacks

@add_to(MyCallbacks)
def on_episode_end(self, caller):
	print("toasty!")
```

## API

1. [Planning (*class*)](#planning)
	1. [Value Iteration (*class*)](#value-iteration)
		1. [value_iteration (*function*)](#value_iteration)
	2. [Policy Iteration (*class*)](#policy-iteration)
		1. [policy_iteration (*function*)](#policy_iteration)
2. [RL](#rl)
	1. [decay_schedule (*function*)](#decay_schedule)
	2. [QLearner (*class*)](#QLearner)
		1. [q_learning (*function*)](#q_learning)	 
	3. [SARSA (*class*)](#sarsa)
		1. [sarsa (*function*)](#sarsa)
		
		
### Planning 

#### Value Iteration 

```
class bettermdptools.algorithms.planning.ValueIteration(P) 
```

about class 

##### value_iteration  
```
function bettermdptools.algorithms.planning.ValueIteration.value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10) ->  V, V_track, pi
```

PARAMETERS:

gamma {float}:
	Discount factor

n_iters {int}:
	Number of iterations

theta {float}:
	Convergence criterion for value iteration.  State values are considered to be converged when the maximum difference between new and previous state values is less than theta. Stops at n_iters or theta convergence - whichever comes first.


RETURNS:

V {numpy array}, shape(possible states):
	State values array 

V_track {numpy array}, shape(n_episodes, nS):
	Log of V(s) for each iteration
	
pi {lambda}, input state value, output action value:
	Policy which maps state action value

#### Policy Iteration

```
class algorithms.planning.PolicyIteration(P) 
```

about class

##### policy_iteration
```
function bettermdptools.algorithms.planning.PolicyIteration.policy_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10) ->  V, V_track, pi
```

PARAMETERS:

gamma {float}:
	Discount factor

n_iters {int}:
	Number of iterations

theta {float}:
	Convergence criterion for policy evaluation.  State values are considered to be converged when the maximum difference between new and previous state values is less than theta.  


RETURNS:

V {numpy array}, shape(possible states):
	State values array 

V_track {numpy array}, shape(n_episodes, nS):
	Log of V(s) for each iteration
	
pi {lambda}, input state value, output action value:
	Policy which maps state action value
	
	
2. [RL](#rl)
	1. [decay_schedule (*function*)](#decay_schedule)
	2. [QLearner (*class*)](#QLearner)
		1. [q_learning (*function*)](#q_learning)	 
	3. [SARSA (*class*)](#sarsa)
		1. [sarsa (*function*)](#sarsa)
### RL 

##### decay_schedule

#### QLearner

##### q_learning

#### SARSA

##### SARSA


	
## Contributing

Pull requests are welcome.  

* Fork bettermdptools.
* Create a branch (`git checkout -b branch_name`)
* Commit changes (`git commit -m "Comments"`)
* Push to branch (`git push origin branch_name`)
* Open a pull request