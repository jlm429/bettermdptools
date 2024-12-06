"""
Author: Aleksandr Spiridonov
BSD 3-Clause License

This is a variation of the grid search utils, but modifier to work with the large domain of the pendulum and acrobot environments.
"""

from bettermdptools.algorithms.rl import RL
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from bettermdptools.envs.acrobot_wrapper import init_wrapper_env as init_acrobot_wrapper_env, get_env_str as get_acrobot_env_str
from bettermdptools.envs.pendulum_wrapper import init_wrapper_env as init_pendulum_wrapper_env, get_env_str as get_pendulum_env_str
import os
import pickle
import gzip
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.rl import RL
import gymnasium as gym

def init_wrapper_env(env_name, wrapper_params):
    if env_name == 'acrobot':
        return init_acrobot_wrapper_env(**wrapper_params), get_acrobot_env_str(**wrapper_params)
    elif env_name == 'pendulum':
        return init_pendulum_wrapper_env(**wrapper_params), get_pendulum_env_str(**wrapper_params)
    else:
        return None, None
    
def get_env_str(env_name, wrapper_params):
    if env_name == 'acrobot':
        return get_acrobot_env_str(**wrapper_params)
    elif env_name == 'pendulum':
        return get_pendulum_env_str(**wrapper_params)
    else:
        return None
    
def get_gym_env(env_name):
    if env_name == 'acrobot':
        return gym.make('Acrobot-v1')
    elif env_name == 'pendulum':
        return gym.make('Pendulum-v1')
    else:
        return None
    
def get_env_state_and_action_space(env_name, wrapper_params):
    if env_name == 'acrobot':
        angle_bins = wrapper_params['angle_bins']
        angular_velocity_bins = wrapper_params['angular_velocity_bins']
        nS = angle_bins**2 * angular_velocity_bins**2
        nA = 3
        return nS, nA
    elif env_name == 'pendulum':
        angle_bins = wrapper_params['angle_bins']
        angular_velocity_bins = wrapper_params['angular_velocity_bins']
        nS = angle_bins * angular_velocity_bins
        nA = 11
        return nS, nA
    else:
        return None, None

class GridSearch:

    @staticmethod
    def get_vi_grid_search_results_filepath(env_str, gamma):
        gamma_str = str(gamma).replace('.', 'd')
        filepath = f'./cached/{env_str}__{gamma_str}_vi_gs_partial.pkl.gz'
        return filepath

    @staticmethod
    def get_pi_grid_search_results_filepath(env_str, gamma):
        gamma_str = str(gamma).replace('.', 'd')
        filepath = f'./cached/{env_str}__{gamma_str}_pi_gs_partial.pkl.gz'
        return filepath

    @staticmethod
    def get_ql_grid_search_results_filepath(env_str, gamma, epsilon_decay):
        gamma_str = str(gamma).replace('.', 'd')
        epsilon_decay_str = str(epsilon_decay).replace('.', 'd')
        filepath = f'./cached/{env_str}__{gamma_str}_{epsilon_decay_str}_ql_gs_partial.pkl.gz'
        return filepath

    @staticmethod
    def vi_grid_search_single_results(gamma, env_str):
        filepath = GridSearch.get_vi_grid_search_results_filepath(env_str, gamma)

        if os.path.exists(filepath):
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
                return filepath, results
        return filepath, None
        
    @staticmethod
    def pi_grid_search_single_results(gamma, env_str):
        filepath = GridSearch.get_pi_grid_search_results_filepath(env_str, gamma)

        if os.path.exists(filepath):
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
                return filepath, results
        return filepath, None

    @staticmethod
    def ql_grid_search_single_results(gamma, epsilon_decay, env_str):
        filepath = GridSearch.get_ql_grid_search_results_filepath(env_str, gamma, epsilon_decay)

        if os.path.exists(filepath):
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
                return filepath, results
        return filepath, None

    @staticmethod
    def vi_grid_search_single(args):
        gamma, wrapper_params, env_name = args

        env, env_str = init_wrapper_env(env_name, wrapper_params)

        _, nA = get_env_state_and_action_space(env_name, wrapper_params)
        
        filepath = GridSearch.get_vi_grid_search_results_filepath(env_str, gamma)
        p = Planner(env.P)

        V_vi, _, pi_vi, V_diff_max_vi, t_elapsed_vi = p.value_iteration(gamma=gamma, n_iters=10000, output_V_track=False)

        gym_env = get_gym_env(env_name)

        iteration_scores_vi = TestEnv.test_env(
                # env=acrobot_genv_test,
                env=gym_env,
                n_iters=100,
                render=False,
                pi=pi_vi,
                user_input=False,
                convert_state_obs=env.transform_obs,
                n_actions=nA,
                convert_action=env.get_action_value
            )
        
        results = {
            'V': V_vi,
            'V_diff_max': V_diff_max_vi,
            'pi': pi_vi,
            'iteration_scores': iteration_scores_vi,
            't_elapsed': t_elapsed_vi
        }

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(results, f)

        return f'VI grid search for {env_str} with gamma {gamma} complete'

    @staticmethod
    def pi_grid_search_single(args):
        gamma, wrapper_params, env_name = args

        env, env_str = init_wrapper_env(env_name, wrapper_params)

        _, nA = get_env_state_and_action_space(env_name, wrapper_params)
        
        filepath = GridSearch.get_pi_grid_search_results_filepath(env_str, gamma)
        p = Planner(env.P)

        V_pi, _, pi_pi, V_diff_max_pi, t_elapsed_pi = p.policy_iteration(gamma=gamma, n_iters=10000, output_V_track=False)

        gym_env = get_gym_env(env_name)

        iteration_scores_pi = TestEnv.test_env(
                # env=acrobot_genv_test,
                env=gym_env,
                n_iters=100,
                render=False,
                pi=pi_pi,
                user_input=False,
                convert_state_obs=env.transform_obs,
                n_actions=nA,
                convert_action=env.get_action_value
            )
        
        results = {
            'V': V_pi,
            'V_diff_max': V_diff_max_pi,
            'pi': pi_pi,
            'iteration_scores': iteration_scores_pi,
            't_elapsed': t_elapsed_pi
        }

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(results, f)

        return f'PI grid search for {env_str} with gamma {gamma} complete'

    @staticmethod
    def ql_grid_search_single(args):
        gamma, epsilon_decay, wrapper_params, env_name = args

        env, env_str = init_wrapper_env(env_name, wrapper_params)

        nS, nA = get_env_state_and_action_space(env_name, wrapper_params)

        gym_env = get_gym_env(env_name)

        filepath = GridSearch.get_ql_grid_search_results_filepath(env_str, gamma, epsilon_decay)

        rl = RL(gym_env)

        Q_ql, V_ql, pi_ql, _, _, rewards_ql, epsilons_ql, visits_ql, V_diff_max_ql, t_ql = rl.q_learning(
                gamma=gamma,
                epsilon_decay_ratio=epsilon_decay,
                nS=nS,
                nA=nA,
                convert_state_obs=env.transform_obs,
                convert_action=env.get_action_value,
                n_episodes=50000,
                output_pi_Q_track=False,
                use_tqdm=False
            )
        
        iteration_scores_ql = TestEnv.test_env(
                # env=acrobot_genv_test,
                env=gym_env,
                n_iters=100,
                render=False,
                pi=pi_ql,
                user_input=False,
                convert_state_obs=env.transform_obs,
                n_actions=nA,
                convert_action=env.get_action_value
            )
        
        results = {
            'Q': Q_ql,
            'V': V_ql,
            'pi': pi_ql,
            'rewards': rewards_ql,
            'epsilons': epsilons_ql,
            'visits': visits_ql,
            'V_diff_max': V_diff_max_ql,
            't': t_ql,
            'iteration_scores': iteration_scores_ql
        }

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(results, f)

        return f'QL grid search for {env_str} with gamma {gamma} and epsilon decay {epsilon_decay} complete'
        

    @staticmethod
    def vi_grid_search(env_name, gammas, wrapper_params_list, verbose=True, n_jobs=1):
        # combinations of gammas and wrapper_params
        tasks = [(gamma, wrapper_params) for gamma in gammas for wrapper_params in wrapper_params_list]

        tasks = [(gamma, wrapper_params, get_env_str(env_name, wrapper_params)) for gamma, wrapper_params in tasks]

        tasks_to_run = [(gamma, wrapper_params, env_str) for gamma, wrapper_params, env_str in tasks if not os.path.exists(GridSearch.get_vi_grid_search_results_filepath(env_str, gamma))]
        
        initialized_envs = set()

        for _, wrapper_params, env_str in tasks_to_run:
            if env_str not in initialized_envs:
                init_wrapper_env(env_name, wrapper_params)
                initialized_envs.add(env_str)

        tasks_to_run = [(gamma, wrapper_params, env_name) for gamma, wrapper_params, _ in tasks_to_run]
        # gamma, wrapper_params, env_name = args

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(GridSearch.vi_grid_search_single, args) for args in tasks_to_run]

            for future in as_completed(futures):
                print(future.result())

        # for args in tasks_to_run:
        #     print(GridSearch.vi_grid_search_single(args))

        results = {}

        for gamma, wrapper_params, env_str in tasks:
            _, result = GridSearch.vi_grid_search_single_results(gamma, env_str)
            results[(gamma, env_str)] = (result, wrapper_params)

        return results

    @staticmethod
    def pi_grid_search(env_name, gammas, wrapper_params_list, verbose=True, n_jobs=1):
        # combinations of gammas and wrapper_params
        tasks = [(gamma, wrapper_params) for gamma in gammas for wrapper_params in wrapper_params_list]

        tasks = [(gamma, wrapper_params, get_env_str(env_name, wrapper_params)) for gamma, wrapper_params in tasks]

        tasks_to_run = [(gamma, wrapper_params, env_str) for gamma, wrapper_params, env_str in tasks if not os.path.exists(GridSearch.get_pi_grid_search_results_filepath(env_str, gamma))]
        
        initialized_envs = set()

        for _, wrapper_params, env_str in tasks_to_run:
            if env_str not in initialized_envs:
                init_wrapper_env(env_name, wrapper_params)
                initialized_envs.add(env_str)

        tasks_to_run = [(gamma, wrapper_params, env_name) for gamma, wrapper_params, _ in tasks_to_run]
        # gamma, wrapper_params, env_name = args

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(GridSearch.pi_grid_search_single, args): args for args in tasks_to_run}

            for future in as_completed(futures):
                try:
                    print(future.result())
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'Failed task: {futures[future]}')

        results = {}

        for gamma, wrapper_params, env_str in tasks:
            _, result = GridSearch.pi_grid_search_single_results(gamma, env_str)
            results[(gamma, env_str)] = (result, wrapper_params)

        return results
    
    @staticmethod
    def ql_grid_search(env_name, gammas, epsilon_decays, wrapper_params_list, verbose=True, n_jobs=1):
        # combinations of gammas and wrapper_params
        tasks = [(gamma, epsilon_decay, wrapper_params) for gamma in gammas for epsilon_decay in epsilon_decays for wrapper_params in wrapper_params_list]

        tasks = [(gamma, epsilon_decay, wrapper_params, get_env_str(env_name, wrapper_params)) for gamma, epsilon_decay, wrapper_params in tasks]

        tasks_to_run = [(gamma, epsilon_decay, wrapper_params, env_str) for gamma, epsilon_decay, wrapper_params, env_str in tasks if not os.path.exists(GridSearch.get_ql_grid_search_results_filepath(env_str, gamma, epsilon_decay))]
        
        initialized_envs = set()

        for _, _, wrapper_params, env_str in tasks_to_run:
            if env_str not in initialized_envs:
                init_wrapper_env(env_name, wrapper_params)
                initialized_envs.add(env_str)

        tasks_to_run = [(gamma, epsilon_decay, wrapper_params, env_name) for gamma, epsilon_decay, wrapper_params, _ in tasks_to_run]
        # gamma, epsilon_decay, wrapper_params, env_name = args

        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(GridSearch.ql_grid_search_single, args) for args in tasks_to_run]

                for future in as_completed(futures):
                    print(future.result())
        else:
            for args in tasks_to_run:
                print(GridSearch.ql_grid_search_single(args))

        results = {}

        for gamma, epsilon_decay, wrapper_params, env_str in tasks:
            _, result = GridSearch.ql_grid_search_single_results(gamma, epsilon_decay, env_str)
            results[(gamma, epsilon_decay, env_str)] = (result, wrapper_params)

        return results

if __name__ == '__main__':
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