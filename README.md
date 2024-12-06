# bettermdptools

## Getting Started

### About

Bettermdptools is a package designed to help users get started with [gymnasium](https://gymnasium.farama.org/), a maintained fork of OpenAI’s Gym library. 
Bettermdptools includes planning and reinforcement learning algorithms, useful utilities and plots, environment models for blackjack and cartpole, and starter code for working with gymnasium.

### Install 

pip install or git clone bettermdptools.   

```
pip install bettermdptools
```

```
git clone https://github.com/jlm429/bettermdptools
```

Starter code to get up and running on the gymnasium frozen lake environment. See [bettermdptools/notebooks](notebooks/) for more.  

```
import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots

# make gym environment 
frozen_lake = gym.make('FrozenLake8x8-v1', render_mode=None)

# run VI
V, V_track, pi, V_diff_max, t_elapsed = Planner(frozen_lake.P).value_iteration()

#plot state values
size=(8,8)
Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", size)
```

![grid_state_values](https://user-images.githubusercontent.com/10093986/211906047-bc13956b-b8e6-411d-ae68-7a3eb5f2ad32.PNG)

A custom version of grid search was created to run pendulum and acrobot problems that caches the results and runs iterations of the grid search in parallel. Note: Acrobot has very large state space sizes (4D continuous problem). Once you get to n_bins of around 30, the RAM requirements might get to be around 10GB per process, so n_jobs = 1 may be adviseable for large problems unless you have lots of RAM.

```
from bettermdptools.utils.grid_search_large_domain import GridSearch

gammas = [0.7, 0.9]
epsilon_decays = [0.8, 0.9]
n_bins_list = [11]

wrapper_params_list = [{'angle_bins': n_bins, 'angular_velocity_bins': n_bins, 'torque_bins': 11} for n_bins in n_bins_list]

n_jobs = 10
n_jobs_pi = 8

vi_grid_search_results = GridSearch.vi_grid_search(
    env_name='pendulum',
    gammas=gammas,
    wrapper_params_list=wrapper_params_list,
    n_jobs=n_jobs
)

pi_grid_search_results = GridSearch.pi_grid_search(
    env_name='pendulum',
    gammas=gammas,
    wrapper_params_list=wrapper_params_list,
    n_jobs=n_jobs_pi
)

ql_grid_search_results = GridSearch.ql_grid_search(
    env_name='pendulum',
    gammas=gammas,
    epsilon_decays=epsilon_decays,
    wrapper_params_list=wrapper_params_list,
    n_jobs=n_jobs
)
```

## Contributing

Pull requests are welcome.  

* Fork bettermdptools.
* Create a branch (`git checkout -b branch_name`)
* Commit changes (`git commit -m "Comments"`)
* Push to branch (`git push origin branch_name`)
* Open a pull request

Setting up an environment (assuming you have conda):

```
conda create -n bettermdptools_dev python=3.10 -y
conda activate bettermdptools_dev
pip install -r requirements.txt
```

Running unit tests:
```
python -m unittest discover -s tests
```