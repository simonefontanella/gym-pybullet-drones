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
import argparse
import os
import sys
import time
from datetime import datetime

sys.path.append('../')
import numpy as np
import ray
import torch
from ray.rllib.agents import ppo
from ray.tune import register_env, CLIReporter
from ray.tune.logger import pretty_print
import shared_constants
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import str2bool, sync
from utils import build_env_by_name, from_env_name_to_class

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones', default=2, type=int, help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env', default='ReachThePointAviary_sparse', type=str, choices=['ReachThePointAviary_sparse'],
                        help='Task (default: leaderfollower)', metavar='')
    parser.add_argument('--obs', default='kin', type=ObservationType, help='Observation space (default: kin)',
                        metavar='')
    ####NON CAMBIRE TIPO DI AZIONE, IMPLEMENTAZIONE DIPENDENTE, CLIPPING DA RIFARE!
    parser.add_argument('--act', default='vel', type=ActionType, help='Action space (default: one_d_rpm)',
                        metavar='')
    parser.add_argument('--algo', default='cc', type=str, choices=['cc'], help='MARL approach (default: cc)',
                        metavar='')
    parser.add_argument('--workers', default=0, type=int, help='Number of RLlib workers (default: 0)', metavar='')
    parser.add_argument('--debug', default=False, type=str2bool,
                        help='Run in one Thread if true, for debugger to work properly', metavar='')
    parser.add_argument('--gui', default=False, type=str2bool,
                        help='Enable gui rendering', metavar='')
    parser.add_argument('--train', default=False, type=str2bool,
                        help='If enabled is in training mode, if not is in tuning mode, need exp to be not defined',
                        metavar='')
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
        OWN_OBS_VEC_SIZE = 42
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

    # shperes can spawn from 0 THIS POSITION ARE RANDOMIZED FROM RESET METHOD
    # IT IS USELESS TO CHANGE THIS
    INIT_XYZS = np.vstack([np.array([0, 0]), \
                           np.array([0, 0]), \
                           np.ones(2)]).transpose().reshape(2, 3)
    # INIT_XYZS = np.vstack([np.array([9.2, -5]), \
    #                        np.array([3.4508020977360783, 0]), \
    #                        np.array([5.722600605763271, 1])]).transpose().reshape(2, 3)

    # 9.468482773404116, 3.4508020977360783, 5.722600605763271

    env_callable, obs_space, act_space, temp_env = build_env_by_name(env_class=from_env_name_to_class(ARGS.env),
                                                                     exp=ARGS.exp,
                                                                     num_drones=ARGS.num_drones,
                                                                     aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                                     obs=ARGS.obs,
                                                                     act=ARGS.act,
                                                                     gui=ARGS.gui,
                                                                     initial_xyzs=INIT_XYZS
                                                                     )
    #### Register the environment ##############################
    register_env(ARGS.env, env_callable)

    config = ppo.DEFAULT_CONFIG.copy()  # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py

    config = {  # **config,
        "env": ARGS.env,
        "gamma": 0.9997,  # 0.999
        "num_workers": 0 + ARGS.workers,
        "num_gpus": torch.cuda.device_count(),
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "sgd_minibatch_size": 4000,
        "lr": 5e-5,  # 0.003
        "optimizer": "RAdam",
        "train_batch_size": 8000,
        "kl_target": 0.04,
        # "num_envs_per_worker": 4,
        "lambda": 0.95,
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
        "timesteps_total": 1000000,  # 100000 ~= 10'
        # "episode_reward_mean": 0,
        # "training_iteration": 100,
    }

    if not ARGS.exp:
        # logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "./logging/timings.json")

        if ARGS.train:

            agent = ppo.PPOTrainer(config=config)
            for i in range(10000):
                result = agent.train()
                print(pretty_print(result))

                if i % 5 == 0:
                    checkpoint_dir = agent.save()
                    print(f"Checkpoint saved in directory {checkpoint_dir}")

        else:
            results = tune.run(
                "PPO",
                stop=stop,
                config=config,
                verbose=True,
                progress_reporter=CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"],
                                              max_progress_rows=10),
                # checkpoint_freq=50000,
                checkpoint_at_end=True,
                local_dir=filename,
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
        duration = 20000
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
            if done["__all__"]: obs = temp_env.reset()  # OPTIONAL EPISODE HALT
        temp_env.close()
        logger.save_as_csv("ma")  # Optional CSV save
        logger.plot()

    #### Shut down Ray #########################################
    print("-------------------Ended------------------")
    ray.shutdown()
