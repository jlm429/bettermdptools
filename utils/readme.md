## API

1. [CustomTransformObservation](#CustomTransformObservation)
	1. [observation](#observation)
2. [BlackjackWrapper](#BlackjackWrapper)
   1. [P](#P)
   2. [transform_obs](#transform_obs)
3. [Callbacks](#callbacks)		
	1. [MyCallbacks](#mycallbacks)	
		1. [on_episode](#on_episode)
		2. [on_episode_begin](#on_episode_begin)
		3. [on_episode_end](#on_episode_end)
		4. [on_env_step](#on_env_step)
4. [TestEnv](#TestEnv)
   1. [test_env](#test_env)


### CustomTransformObservation 

```
class bettermdptools.utils.blackjack_wrapper.CustomTransformObservation(gym.ObservationWrapper)
```

Helper class that modifies the observation space.  The v26 gymnasium TransformObservation wrapper does not accept an observation_space 
parameter, which is needed in order to match the lambda conversion (tuple->int).  Instead, we subclass gym.ObservationWrapper (parent class of gym.TransformObservation) to set both the conversion function and new observation space.

```
base_env = gym.make('Blackjack-v1', render_mode=None)
blackjack = BlackjackWrapper(base_env)
```
##### observation

Applies a function to the observation received from the environment's step function, which is passed back to the user.

```
function bettermdptools.utils.blackjack_wrapper.BlackjackWrapper.observation(self, observation) ->  self.func(observation)
```

**PARAMETERS**: 

observation {Tuple}:
	Blackjack base environment observation tuple

**RETURNS**:

func(observation) {int}:
	The converted observation (290 discrete observable states)


### BlackjackWrapper 

```
class bettermdptools.utils.blackjack_wrapper.BlackjackWrapper(gym.Wrapper)
```

Blackjack wrapper class that modifies the observation space and creates a transition/reward matrix P.

```
base_env = gym.make('Blackjack-v1', render_mode=None)
blackjack = BlackjackWrapper(base_env)
```


##### P
```
function bettermdptools.utils.blackjack_wrapper.BlackjackWrapper.P(self) -> P
```

**RETURNS**:

P {dict}:
	Transition and Reward Matrix

##### transform_obs
```
function bettermdptools.utils.blackjack_wrapper.BlackjackWrapper.transform_obs(self) ->  _tranform_obs
```

**RETURNS**:

transform_obs {lambda}, input tuple, output int:
	Lambda function assigned to the variable `self._convert_state_obs` takes parameter, `state` and
	converts the input into a compact single integer value by concatenating player hand with dealer card.

### Callbacks 

Base class. 

```
class bettermdptools.utils.Callbacks():
```
RL algorithms SARSA and Q-learning have callback hooks for episode number, begin, end, and env. step.   

##### MyCallbacks 

```
class bettermdptools.utils.MyCallbacks(Callbacks):
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
from utils.decorators import add_to
from utils.callbacks import MyCallbacks

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

### TestEnv 

```
class bettermdptools.utils.TestEnv() 
```

Simulation of the agent's decision process after it has learned a policy.

##### test_env

```
function bettermdptools.utils.TestEnv.test_env(env, desc=None, render=False, 
n_iters=10, pi=None, user_input=False, convert_state_obs=lambda state: state)
	->  test_scores
```

**PARAMETERS**: 

env {OpenAI Gym Environment}:
	MDP problem

desc {numpy array}:
	description of the environment (for custom environments)

render {Boolean}, default = False:
	openAI human render mode

n_iters {int}, default = 10:
	Number of iterations to simulate the agent for

pi {lambda}:
	Policy used to calculate action value at a given state

user_input {Boolean}, default = False:
	Prompt for letting user decide which action to take at a given state

convert_state_obs {lambda}:
	Optionally used in environments where state observation is transformed.

**RETURNS**: 

test_scores {numpy array}:
	Log of rewards from each episode.  
