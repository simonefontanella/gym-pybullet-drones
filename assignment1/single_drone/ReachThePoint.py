"""Learning script for multi-agent problems.
Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:

    http://127.0.0.1:8265



ActionType:
    RPM = "rpm"                 # RPMS
    DYN = "dyn"                 # Desired thrust and torques
    PID = "pid"                 # PID control
    VEL = "vel"                 # Velocity input (using PID control)
    TUN = "tun"                 # Tune the coefficients of a PID controller
    ONE_D_RPM = "one_d_rpm"     # 1D (identical input to all motors) with RPMs
    ONE_D_DYN = "one_d_dyn"     # 1D (identical input to all motors) with desired thrust and torques
    ONE_D_PID = "one_d_pid"     # 1D (identical input to all motors) with PID control

ObservationType(Enum):
    KIN = "kin"     # Kinematic information (pose, linear and angular velocities)
    RGB = "rgb"     # RGB camera capture in each drone's POV

"""
import os
import time
import argparse
from datetime import datetime
from sys import platform
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env, CLIReporter
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from utils import build_env_by_name, from_env_name_to_class
import shared_constants
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from env_builder import EnvBuilder
from gym_pybullet_drones.utils.utils import str2bool, sync
from ray.rllib.examples.models.shared_weights_model import (

    TorchSharedWeightsModel,
)

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones', default=2, type=int, help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env', default='ReachThePointAviary', type=str, choices=['ReachThePointAviary'],
                        help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs', default='kin', type=ObservationType, help='Observation space (default: kin)',
                        metavar='')
    parser.add_argument('--act', default='pid', type=ActionType, help='Action space (default: one_d_rpm)',
                        metavar='')
    parser.add_argument('--algo', default='cc', type=str, choices=['cc'], help='MARL approach (default: cc)',
                        metavar='')
    parser.add_argument('--workers', default=1, type=int, help='Number of RLlib workers (default: 0)', metavar='')
    parser.add_argument('--debug', default=False, type=str2bool,
                        help='Run in one Thread if true, for debugger to work properly', metavar='')
    parser.add_argument('--gui', default=False, type=str2bool,
                        help='Enable gui rendering', metavar='')
    parser.add_argument('--exp', type=str,
                        help='The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>',
                        metavar='')

    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__)) + '/results/save-' + ARGS.env + '-' + str(
        ARGS.num_drones) + '-' + ARGS.algo + '-' + ARGS.obs.value + '-' + ARGS.act.value + '-' + datetime.now().strftime(
        "%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename + '/')

    #
    # #### Print out current git commit hash #####################
    # if platform == "linux" or platform == "darwin":
    #     git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    #     with open(filename + '/git_commit.txt', 'w+') as f:
    #         f.write(str(git_commit))

    #### Constants, and errors #################################
    if ARGS.obs == ObservationType.KIN:
        OWN_OBS_VEC_SIZE = 52
    elif ARGS.obs == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit(2)
    else:
        print("[ERROR] unknown ObservationType")
        exit(3)
    if ARGS.act in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VEC_SIZE = 1
    elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VEC_SIZE = 4
    elif ARGS.act == ActionType.PID:
        ACTION_VEC_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit(4)

    #### Uncomment to debug slurm scripts ######################
    # exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True, local_mode=ARGS.debug)
    from ray import tune
    #
    INIT_XYZS = np.vstack([np.array([0, -5]), \
                                        np.array([0, 0]), \
                                        np.ones(2)]).transpose().reshape(2, 3)

    # INIT_XYZS = np.vstack([np.array([9.2, -5]), \
    #                        np.array([3.4508020977360783, 0]), \
    #                        np.array([5.722600605763271, 1])]).transpose().reshape(2, 3)

    #9.468482773404116, 3.4508020977360783, 5.722600605763271

    env_callable, obs_space, act_space, temp_env = build_env_by_name(env_class=from_env_name_to_class(ARGS.env),
                                                                     num_drones=ARGS.num_drones,
                                                                     aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                                     obs=ARGS.obs,
                                                                     act=ARGS.act,
                                                                     gui=ARGS.gui,
                                                                     initial_xyzs = INIT_XYZS
                                                                     )
    #### Register the environment ##############################
    register_env(ARGS.env, env_callable)


    config = {
        "env": ARGS.env,
        "gamma":0.9999,  #0.999
        "num_workers": 0 + ARGS.workers,
        "num_gpus": torch.cuda.device_count(),
        "batch_mode": "complete_episodes",
        "no_done_at_end": True,
        "framework": "torch",
        "lr": 5e-3, #0.003
        "optimizer": "adam",
        "num_envs_per_worker": 8,
        #"lambda" : 0.95,
        "multiagent": {
            # We only have one policy (calling it "shared").
            # Class, obs/act-spaces, and config will be derived
            # automatically.
            "policies": {
                "pol0": (None, obs_space[0], act_space[0], {"agent_id": 0, }),
                "pol1": (None, obs_space[1], act_space[1], {"agent_id": 1, }),
            },
            "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",
            # Always use "shared" policy.

        }
    }
    stop = {
        "timesteps_total": 300000,  # 100000 ~= 10'
        # "episode_reward_mean": 0,
        # "training_iteration": 100,
    }

    if not ARGS.exp:

        results = tune.run(
            "PPO",
            stop=stop,
            config=config,
            verbose=True,
            progress_reporter=CLIReporter(max_progress_rows=10),
            # checkpoint_freq=50000,
            checkpoint_at_end=True,
            local_dir=filename
        )

        # check_learning_achieved(results, 1.0)

        #### Sa/results/save-ReachThePointAviary-2-cc-kin-pid-11.23.2022_14.34.23ve agent ############################################
        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                       mode='max'
                                                                                       ),
                                                          metric='episode_reward_mean'
                                                          )
        with open(filename + '/checkpoint.txt', 'w+') as f:
            f.write(checkpoints[0][0])

        print(checkpoints)

    else:

        OBS = ObservationType.KIN if ARGS.exp.split("-")[4] == 'kin' else ObservationType.RGB
        action_name = ARGS.exp.split("-")[5]
        NUM_DRONES = int(ARGS.exp.split("-")[2])
        ACT = [action for action in ActionType if action.value == action_name][0]
        #### Restore agent #########################################
        agent = ppo.PPOTrainer(config=config)
        with open(ARGS.exp + '/checkpoint.txt', 'r+') as f:
            checkpoint = f.read()
        agent.restore(checkpoint)
        print(checkpoint)

        #### Extract and print policies ############################
        policy0 = agent.get_policy("pol0")
        policy1 = agent.get_policy("pol1")

        #### Show, record a video, and log the model's performance #
        obs = temp_env.reset()
        logger = Logger(logging_freq_hz=int(temp_env.SIM_FREQ / temp_env.AGGR_PHY_STEPS),
                        num_drones=NUM_DRONES
                        )
        if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            action = {i: np.array([0]) for i in range(NUM_DRONES)}
        elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            action = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
        elif ACT == ActionType.PID:
            action = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
        else:
            print("[ERROR] unknown ActionType")
            exit()
        start = time.time()
        duration = 200
        for i in range(duration * int(temp_env.SIM_FREQ / temp_env.AGGR_PHY_STEPS)):  # Up to 6''
            #### Deploy the policies ###################################
            temp = {}
            temp[0] = policy0.compute_single_action(
                np.hstack(obs[0]))  # Counterintuitive order, check params.json
            temp[1] = policy1.compute_single_action(np.hstack(obs[1]))
            action = {0: temp[0][0], 1: temp[1][0]}
            obs, reward, done, info = temp_env.step(action)
            temp_env.render()
            if OBS == ObservationType.KIN:
                for j in range(NUM_DRONES):
                    logger.log(drone=j,
                               timestamp=i / temp_env.SIM_FREQ,
                               state=np.hstack([obs[j][0:3], np.zeros(4), obs[j][3:15], np.resize(action[j], (4))]),
                               control=np.zeros(12)
                               )
            sync(np.floor(i * temp_env.AGGR_PHY_STEPS), start, temp_env.TIMESTEP)
            # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
        temp_env.close()
        logger.save_as_csv("ma")  # Optional CSV save
        logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()
