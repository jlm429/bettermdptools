## API

1. [Planner](#planner)
   1. [value_iteration](#value_iteration)
   2. [policy_iteration](#policy_iteration)
2. [RL](#rl)
   1. [q_learning](#q_learning)
   2. [sarsa](#sarsa)

### Planner 

```
class bettermdptools.algorithms.planner.Planner(P)
```

Class that contains functions related to planning algorithms (Value Iteration, Policy Iteration).  Planner __init__ expects a reward and transitions matrix P, which is nested dictionary 
[gym](https://gymnasium.farama.org/) style discrete environment where 
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

The RL algorithms (Q-learning, SARSA) work out of the box with any [gymnasium environments](https://gymnasium.farama.org/)  that have single discrete valued state spaces, like [frozen lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/#observation-space). 
A lambda function is required to convert state spaces not in this format. 

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
	Converts state into an integer.  

gamma {float}, default = 0.99:
	Discount factor

init_alpha {float}, default = 0.5:
	Learning rate

min_alpha {float}, default = 0.01:
	Minimum learning rate

alpha_decay_ratio {float}, default = 0.5:
	Decay schedule of learning rate for future iterations

init_epsilon {float}, default = 1.0:
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
	Converts state into an integer.  

gamma {float}, default = 0.99:
	Discount factor

init_alpha {float}, default = 0.5:
	Learning rate

min_alpha {float}, default = 0.01:
	Minimum learning rate

alpha_decay_ratio {float}, default = 0.5:
	Decay schedule of learning rate for future iterations

init_epsilon {float}, default = 1.0:
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
	
	