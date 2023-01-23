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

Starter code to get up and running on OpenAI's frozen lake environment. See bettermdptools/examples for more.  

```
import gym
import pygame
from algorithms.rl import RL
from examples.test_env import TestEnv

frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

# Q-learning
Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning()

test_scores = TestEnv.test_env(env=frozen_lake.env, render=True, user_input=False, pi=pi)
```

#### Plotting and Grid Search

```
#grid search
epsilon_decay = [.4, .7, .9]
iters = [500, 5000, 50000]
GridSearch.Q_learning_grid_search(frozen_lake.env, epsilon_decay, iters)


#plot state values
frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)
V, V_track, pi = Planner(frozen_lake.env.P).value_iteration()
Plots.grid_values_heat_map(V, "State Values")
```

![grid_state_values](https://user-images.githubusercontent.com/10093986/211906047-bc13956b-b8e6-411d-ae68-7a3eb5f2ad32.PNG)


## API

1. [Planner](#planner)
   1. [value_iteration](#value_iteration)
   2. [policy_iteration](#policy_iteration)
2. [RL](#rl)
   1. [q_learning](#q_learning)
   2. [sarsa](#sarsa)
3. [Callbacks](#callbacks)		
	1. [MyCallbacks](#mycallbacks)	
		1. [on_episode](#on_episode)
		2. [on_episode_begin](#on_episode_begin)
		3. [on_episode_end](#on_episode_end)
		4. [on_env_step](#on_env_step)
		
### Planner 

```
class bettermdptools.algorithms.planner.Planner(P)
```

Class that contains functions related to planning algorithms (Value Iteration, Policy Iteration).  Planner __init__ expects a reward and transitions matrix P, which is nested dictionary 
[OpenAI Gym](https://www.gymlibrary.ml/) style discrete environment where 
P[state][action] is a list of tuples (probability, next state, reward, terminal).

Frozen Lake VI example:
```
env = gym.make('FrozenLake8x8-v1')
V, V_track, pi = Planner(env.P).value_iteration()
```


##### value_iteration  
```
function bettermdptools.algorithms.planner.Planner.value_iteration(self, 
	gamma=1.0, n_iters=1000, theta=1e-10) ->  V, V_track, pi
```

**PARAMETERS**:

gamma {float}:
	Discount factor

n_iters {int}:
	Number of iterations

theta {float}:
	Convergence criterion for value iteration.  State values are considered to be converged when the maximum difference between new and previous state values is less than theta. Stops at n_iters or theta convergence - whichever comes first.


**RETURNS**:

V {numpy array}, shape(possible states):
	State values array 

V_track {numpy array}, shape(n_episodes, nS):
	Log of V(s) for each iteration
	
pi {lambda}, input state value, output action value:
	Policy mapping states to actions.  

##### policy_iteration
```
function bettermdptools.algorithms.planner.Planner.policy_iteration(self, 
	gamma=1.0, n_iters=1000, theta=1e-10) ->  V, V_track, pi
```

**PARAMETERS**:

gamma {float}:
	Discount factor

n_iters {int}:
	Number of iterations

theta {float}:
	Convergence criterion for policy evaluation.  State values are considered to be converged when the maximum difference between new and previous state values is less than theta.  


**RETURNS**:

V {numpy array}, shape(possible states):
	State values array 

V_track {numpy array}, shape(n_episodes, nS):
	Log of V(s) for each iteration
	
pi {lambda}, input state value, output action value:
	Policy mapping states to actions.  
	
	
### RL 

```
class bettermdptools.algorithms.rl.RL(env) 
```

Class that contains functions related to reinforcement learning algorithms. RL __init__ expects an OpenAI environment (env). 

The RL algorithms (Q-learning, SARSA) work out of the box with any [OpenAI Gym environment](https://www.gymlibrary.ml/)  that have single discrete valued state spaces, like [frozen lake](https://www.gymlibrary.ml/environments/toy_text/frozen_lake/#observation-space). 
A lambda function is required to convert state spaces not in this format.  For example, [blackjack](https://www.gymlibrary.ml/environments/toy_text/blackjack/#observation-space) is "a 3-tuple containing: the player’s current sum, the value of the dealer’s one showing card (1-10 where 1 is ace), and whether the player holds a usable ace (0 or 1)." 

Here, blackjack.convert_state_obs changes the 3-tuple into a discrete space with 280 states by concatenating player states 0-27 (hard 4-21 & soft 12-21) with dealer states 0-9 (2-9, ten, ace).   

```
self.convert_state_obs = lambda state, done: ( -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(f"{state[0] - 4}{(state[1] - 2) % 10}"))
```
 
Since n_states is modified by the state conversion, this new value is passed in along with n_actions, and convert_state_obs.    
  
```
# Q-learning
Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions, blackjack.convert_state_obs)
```

##### q_learning

```
function bettermdptools.algorithms.rl.RL.q_learning(self, nS=None, nA=None, 
	convert_state_obs=lambda state, done: state, 
	gamma=.99, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, 
	init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, n_episodes=10000)  
	->  Q, V, pi, Q_track, pi_track
```

**PARAMETERS**: 

nS {int}:
	Number of states

nA {int}:
	Number of available actions
	
convert_state_obs {lambda}:
	The state conversion utilized in BlackJack ToyText problem.
	Returns three state tuple as one of the 280 converted states.

gamma {float}, default = 0.99:
	Discount factor

init_alpha {float}, default = 0.5:
	Learning rate

min_alpha {float}, default = 0.01:
	Minimum learning rate

alpha_decay_ratio {float}, default = 0.5:
	Decay schedule of learing rate for future iterations

init_epsilon {float}, default = 0.1:
	Initial epsilon value for epsilon greedy strategy.
	Chooses max(Q) over available actions with probability 1-epsilon.

min_epsilon {float}, default = 0.1:
	Minimum epsilon. Used to balance exploration in later stages.

epsilon_decay_ratio {float}, default = 0.9:
	Decay schedule of epsilon for future iterations
	
n_episodes {int}, default = 10000:
	Number of episodes for the agent


**RETURNS**: 

Q {numpy array}, shape(nS, nA):
	Final action-value function Q(s,a)

pi {lambda}, input state value, output action value:
	Policy mapping states to actions.  

V {numpy array}, shape(nS):
	State values array 

Q_track {numpy array}, shape(n_episodes, nS, nA):
	Log of Q(s,a) for each episode

pi_track {list}, len(n_episodes):
	Log of complete policy for each episode

##### SARSA

```
function bettermdptools.algorithms.rl.RL.sarsa(self, nS=None, nA=None, 
	convert_state_obs=lambda state, done: state, 
	gamma=.99, init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5, 
	init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9, n_episodes=10000)
	-> Q, V, pi, Q_track, pi_track
```

**PARAMETERS**:

nS {int}:
	Number of states

nA {int}:
	Number of available actions
	
convert_state_obs {lambda}:
	The state conversion utilized in BlackJack ToyText problem.
	Returns three state tuple as one of the 280 converted states.

gamma {float}, default = 0.99:
	Discount factor

init_alpha {float}, default = 0.5:
	Learning rate

min_alpha {float}, default = 0.01:
	Minimum learning rate

alpha_decay_ratio {float}, default = 0.5:
	Decay schedule of learing rate for future iterations

init_epsilon {float}, default = 0.1:
	Initial epsilon value for epsilon greedy strategy.
	Chooses max(Q) over available actions with probability 1-epsilon.

min_epsilon {float}, default = 0.1:
	Minimum epsilon. Used to balance exploration in later stages.

epsilon_decay_ratio {float}, default = 0.9:
	Decay schedule of epsilon for future iterations
	
n_episodes {int}, default = 10000:
	Number of episodes for the agent


**RETURNS**:

Q {numpy array}, shape(nS, nA):
	Final action-value function Q(s,a)

pi {lambda}, input state value, output action value:
	Policy mapping states to actions. 

V {numpy array}, shape(nS):
	State values array 

Q_track {numpy array}, shape(n_episodes, nS, nA):
	Log of Q(s,a) for each episode

pi_track {list}, len(n_episodes):
	Log of complete policy for each episode
	
	
### Callbacks 

Base class. 

```
class bettermdptools.callbacks.Callbacks():
```
RL algorithms SARSA and Q-learning have callback hooks for episode number, begin, end, and env. step.   

##### MyCallbacks 

```
class bettermdptools.callbacks.MyCallbacks(Callbacks):
```

To create a callback, override one of the callback functions in the child class MyCallbacks.  Here, on_episode prints the episode number every 1000 episodes.

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

##### on_episode

```
function on_episode(self, caller, episode):
```

**PARAMETERS**:

caller (RL type):
	Calling object 

episode {int}:
	Current episode from caller 

##### on_episode_begin

```
function on_episode_begin(self, caller):
```

**PARAMETERS**:

caller (RL type):
	Calling object

##### on_episode_end

```
function on_episode_end(self, caller):
```

**PARAMETERS**:

caller (RL type):
	Calling object

##### on_env_step

```
function on_env_step(self, caller):
```

**PARAMETERS**:

caller (RL type):
	Calling object

## Contributing

Pull requests are welcome.  

* Fork bettermdptools.
* Create a branch (`git checkout -b branch_name`)
* Commit changes (`git commit -m "Comments"`)
* Push to branch (`git push origin branch_name`)
* Open a pull request