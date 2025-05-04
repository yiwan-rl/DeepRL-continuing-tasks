# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import argparse
import copy
import glob
import logging
import os
import pickle
import random

import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Type

# pyre-fixme
import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# pyre-fixme
from alphaex.sweeper import Sweeper
from pearl import action_representation_modules
from pearl.action_representation_modules import IdentityActionRepresentationModule
from pearl.action_representation_modules.action_representation_module import (
    ActionRepresentationModule,
)
from pearl.api.observation import Observation
from pearl.policy_learners.policy_learner import PolicyLearner
from pearl.neural_networks.common import value_networks
from pearl.neural_networks.common.utils import (
    kaiming_normal_init_weights,
    orthogonal_init_weights,
    xavier_init_weights,
)
from pearl.neural_networks.common.value_networks import ValueNetwork
from pearl.neural_networks.sequential_decision_making import (
    actor_networks,
    q_value_networks,
)
from pearl.neural_networks.sequential_decision_making.actor_networks import ActorNetwork
from pearl.neural_networks.sequential_decision_making.q_value_networks import (
    QValueNetwork,
)
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners import sequential_decision_making as policy_learners
from pearl.policy_learners.exploration_modules import (
    common as exploration_modules,
    wrappers as exploration_wrappers,
)
from pearl.policy_learners.exploration_modules.common.epsilon_greedy_exploration import (
    EGreedyExploration,
)
from pearl.policy_learners.exploration_modules.common.no_exploration import (
    NoExploration,
)
from pearl.policy_learners.exploration_modules.exploration_module import (
    ExplorationModule,
)
from pearl.policy_learners.exploration_modules.exploration_module_wrapper import (
    ExplorationModuleWrapper,
)
from pearl.policy_learners.exploration_modules.wrappers.warmup import Warmup
from pearl.policy_learners.sequential_decision_making.ddpg import (
    DeepDeterministicPolicyGradient,
)
from pearl.policy_learners.sequential_decision_making.deep_q_learning import (
    DeepQLearning,
)
from pearl.policy_learners.sequential_decision_making.ppo import (
    ProximalPolicyOptimization,
)
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import (
    SoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.soft_actor_critic_continuous import (
    ContinuousSoftActorCritic,
)
from pearl.policy_learners.sequential_decision_making.td3 import TD3
from pearl.replay_buffers import (
    ReplayBuffer,
    sequential_decision_making as replay_buffers,
)

from pearl.user_envs.wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    HalfCheetahWrapper,
    MaxAndSkipEnv,
    NoopResetEnv,
    PusherWrapper,
    ReacherWrapper,
    SwimmerWrapper,
    AdditionalActionWrapper,
    AgentResetWrapper,
    EpisodicTaskAddCostWrapper,
    EpisodicToContinuingWrapper,
    RandomResetWrapper,
)
from pearl.utils.functional_utils.learning.preprocessing import (
    ObservationNormalization,
    Preprocessor,
    RewardClipping,
)
from pearl.utils.functional_utils.learning.reward_centering import MA_RC, RVI_RC, TD_RC

from pearl.utils.instantiations.environments.gym_environment import GymEnvironment
from pearl.utils.instantiations.spaces import BoxActionSpace, DiscreteActionSpace


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def get_env(env_config: Dict[str, Any]) -> GymEnvironment:
    """
    attach a versatility wrapper to the environment
    """
    env_config = env_config[0]

    if env_config.get("random_reset_wrapper", False):
        # create an evironment that randomly resets with a probability of reset_prob
        return GymEnvironment(
            RandomResetWrapper(
                env=get_gym_env(env_config),
                reset_prob=env_config.get("reset_prob", None),
            )
        )
    else:
        env = get_gym_env(env_config)

    if env_config.get("additional_action_wrapper", False):
        # this additional action dimension/action can be used to control resetting
        env = AdditionalActionWrapper(
            env=env,
        )

    if env_config.get("agent_reset_wrapper", False):
        # this wrapper is used for learning resetting in continuing tasks. This wrapper should be used only when AdditionalActionWrapper is used.
        return GymEnvironment(
            AgentResetWrapper(
                env=env,
                reset_cost=env_config.get("reset_cost", None),
            )
        )
    if env_config.get("episodic_to_continuing_wrapper", False):
        # converting an episodic task to a continuing task. 
        # An action that leads to a termination will immediately reset the environment and incur a cost.
        return GymEnvironment(
            EpisodicToContinuingWrapper(
                env=env,
                reset_cost=env_config.get("reset_cost", None),
            )
        )
    
    if env_config.get("episodic_task_add_cost_wrapper", False):
        # adding a cost to the reward when the episode terminates. Only used for episodic tasks.
        return GymEnvironment(
            EpisodicTaskAddCostWrapper(
                env=env,
            )
        )

    return GymEnvironment(
        env_or_env_name=env,
    )


def env_supports_termination_when_unhealthy(env_name: str) -> bool:
    return (
        "Ant-" in env_name
        or "AntNew-" in env_name
        or "SpecialAnt-" in env_name
        or "Hopper-" in env_name
        or "Walker2d-" in env_name
        or "Humanoid-" in env_name
    )


def get_gym_env(env_config: Dict[str, Any]) -> gym.Env:
    env_name = env_config.get("env_name", None)
    if env_name is None:
        raise ValueError("env_name is not specified")
    render_mode = env_config.get("render_mode", None)
    terminate_when_unhealthy = env_config.get("terminate_when_unhealthy", None)
    if env_supports_termination_when_unhealthy(env_name) is True:
        assert (
            terminate_when_unhealthy is not None
        ), f"terminate_when_unhealthy should be specified for {env_name}"

    if "AntNew-" in env_name:
        return gym.make(
            env_name.replace("AntNew-", "Ant-"),
            render_mode=render_mode,
            healthy_z_range=(0.3, 1.0),
            terminate_when_unhealthy=terminate_when_unhealthy,
        )
    elif "HalfCheetahNew-" in env_name:
        return HalfCheetahWrapper(
            gym.make(
                env_name.replace("HalfCheetahNew-", "HalfCheetah-"),
                render_mode=render_mode,
            )
        )
    elif "ReacherNew-" in env_name:
        return ReacherWrapper(
            gym.make(
                env_name.replace("ReacherNew-", "Reacher-"),
                render_mode=render_mode,
            )
        )
    elif "PusherNew-" in env_name:
        return PusherWrapper(
            gym.make(
                env_name.replace("PusherNew-", "Pusher-"),
                render_mode=render_mode,
            )
        )
    elif "SwimmerNew-" in env_name:
        return SwimmerWrapper(
            gym.make(
                env_name.replace("SwimmerNew-", "Swimmer-"),
                render_mode=render_mode,
            )
        )
    elif "SpecialAnt-" in env_name:
        return gym.make(
            env_name.replace("SpecialAnt-", "Ant-"),
            xml_file="special_ant.xml",
            render_mode=render_mode,
            terminate_when_unhealthy=terminate_when_unhealthy,
        )
    elif "ALE/" in env_name or "NoFrameskip" in env_name:
        # Atari envs
        max_num_frames_per_episode = env_config.get("max_num_frames_per_episode", None)
        if max_num_frames_per_episode is not None:
            max_num_frames_per_episode = int(max_num_frames_per_episode)
        env = gym.make(
            env_name,
            render_mode=render_mode,
            max_num_frames_per_episode=max_num_frames_per_episode,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env
    else:
        if env_supports_termination_when_unhealthy(env_name) is True:
            return gym.make(
                env_name,
                render_mode=render_mode,
                terminate_when_unhealthy=terminate_when_unhealthy,
            )
        else:
            return gym.make(
                env_name,
                render_mode=render_mode,
            )


def generate_video(imgs: List[np.ndarray], save_folder: str, video_name: str) -> None:
    """
    create a video using images from imgs.
    imgs: a list of np arrays representing a list of images
    save_folder: the folder to save the video and the generated images
    env_name: used as file name
    agent_name: used_as file name
    """
    # First create and save png files for imgs.
    for i in range(len(imgs)):
        logger.info(f"save image {i}")
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.savefig(f"{save_folder}/{video_name}%02d.png" % i)

    # Create a video using the generated png files.
    subprocess.call(
        [
            "ffmpeg",
            "-framerate",
            "16",
            "-i",
            f"{save_folder}/{video_name}%02d.png",
            # "-r",
            # "60",
            "-pix_fmt",
            "yuv420p",
            f"{save_folder}/{video_name}.mp4",
        ]
    )

    # Remove the generated png files.
    for file_name in glob.glob(f"{save_folder}/{video_name}*.png"):
        os.remove(file_name)


def set_test_time_true(policy_learner: PolicyLearner, preprocessors: List[Preprocessor]) -> None:
    if hasattr(policy_learner.exploration_module, "set_test_time_true"):
        # Do not change counter in the exploration module during evaluation
        # pyre-fixme
        policy_learner.exploration_module.set_test_time_true()
    if hasattr(policy_learner, "_test_time"):
        policy_learner._test_time = True

    # preprocessors switch from training to evaluation. All agents share the same preprocessors.
    for p in preprocessors:
        if hasattr(p, "_test_time"):
            # pyre-fixme
            p._test_time = True


def set_test_time_false(policy_learner: PolicyLearner, preprocessors: List[Preprocessor]) -> None:
    if hasattr(policy_learner.exploration_module, "set_test_time_false"):
        # pyre-fixme
        policy_learner.exploration_module.set_test_time_false()
    if hasattr(policy_learner, "_test_time"):
        policy_learner._test_time = False
    for p in preprocessors:
        if hasattr(p, "_test_time"):
            p._test_time = False


def eval_continuing(
    eval_continuing_agent: PearlAgent,
    eval_continuing_env: GymEnvironment,
    eval_max_steps: int,
    preprocessors: List[Preprocessor],
    render: bool = False,
    video_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    run_idx: Optional[int] = None,
    qpos: Optional[np.ndarray] = None,
    qvel: Optional[np.ndarray] = None,
) -> Tuple[
    float, float, float
]:
    set_test_time_true(eval_continuing_agent.policy_learner, preprocessors)

    # evaluate the agent in the continuing version of the environment
    # pyre-fixme
    assert eval_continuing_env is not None
    observation, action_space = eval_continuing_env.reset()
    if qpos is not None and qvel is not None:
        # for mujoco env, if qpos and qvel are not None, we set the state of the environment
        eval_continuing_env.env.set_state(qpos, qvel)
    eval_continuing_agent.reset(observation, action_space)
    eval_cum_reward, eval_cum_reset, eval_cum_clipped_reward = 0, 0, 0
    frames = []
    for _ in range(1, eval_max_steps + 1):
        if render is True:
            frames.append(eval_continuing_env.env.render())
        # the agent takes an action
        action = eval_continuing_agent.act(exploit=False)

        # the environment receives the action and returns the result
        # pyre-fixme
        action_result = eval_continuing_env.step(action)

        # update stats
        eval_cum_reward += action_result.reward
        eval_cum_reset += int(action_result.info.get("reset", False))

        for preprocessor in preprocessors:
            preprocessor.process(action_result)
        eval_cum_clipped_reward += action_result.reward

        # the agent receives the action result
        eval_continuing_agent.observe(action_result)
    avg_reward = eval_cum_reward / eval_max_steps
    avg_reset = eval_cum_reset / eval_max_steps
    avg_clipped_reward = eval_cum_clipped_reward / eval_max_steps

    # recover from testing to training
    logger.info(
        f"eval continuing, avg reward {avg_reward}, avg clipped reward {avg_clipped_reward}, avg reset {avg_reset}"
    )

    # recover from evaluation to training
    set_test_time_false(eval_continuing_agent.policy_learner, preprocessors)
    if render is True:
        assert video_dir is not None and video_name is not None and run_idx is not None
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        generate_video(frames, video_dir, video_name=video_name + "_" + str(run_idx))
    return avg_reward, avg_clipped_reward, avg_reset


def eval_episodic(
    eval_episodic_agent: PearlAgent,
    eval_episodic_env: GymEnvironment,
    eval_max_steps: int,
    preprocessors: List[Preprocessor],
    render: bool = False,
    video_dir: Optional[str] = None,
    video_name: Optional[str] = None,
    run_idx: Optional[int] = None,
) -> Tuple[
    float, float
]:
    # policy learner switches from training to evaluation
    set_test_time_true(eval_episodic_agent.policy_learner, preprocessors)

    assert eval_episodic_env is not None
    # evaluate the agent in the episodic version of the environment
    eps_return_list = []
    eps_clipped_return_list = []
    episode_steps = 0
    total_steps = 0
    eps_return = 0
    frames = []
    while total_steps < eval_max_steps:
        info, episode_steps = run_episode(
            agent=eval_episodic_agent,
            # pyre-fixme
            env=eval_episodic_env,
            exploit=False,
            learn=False,
            preprocessors=preprocessors,
            total_steps=total_steps,
            number_of_steps=eval_max_steps,
            render=render,
        )
        if render:
            frames.extend(info["frames"])
        total_steps += episode_steps
        eps_return_list.append(info["return"])
        if "clipped_return" in info:
            eps_clipped_return_list.append(info["clipped_return"])
    eps_return = np.mean(eps_return_list)
    if len(eps_clipped_return_list) > 0:
        eps_clipped_return = np.mean(eps_clipped_return_list)
    else:
        eps_clipped_return = None
    # recover from testing to training
    logger.info(
        f"eval episodic, epsodic return {eps_return}, episodic clipped return {eps_clipped_return}"
    )
    set_test_time_false(eval_episodic_agent.policy_learner, preprocessors)
    if render:
        assert video_dir is not None and video_name is not None and run_idx is not None
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        generate_video(frames, video_dir, video_name=video_name + "_" + str(run_idx))
    return eps_return, eps_clipped_return


def run_episode(
    agent: PearlAgent,
    env: GymEnvironment,
    exploit: bool = True,
    learn_after_episode: bool = False,
    learn_every_k_steps: int = 1,
    total_steps: int = 0,
    learn: bool = True,
    learning_start: int = 0,
    preprocessors: Optional[List[Preprocessor]] = None,
    learning_report_cache: Optional[Dict[str, List[float]]] = None,
    number_of_steps: Optional[int] = None,
    visited_observations: Optional[List[Any]] = None,
    record_visited_observations: bool = False,
    observation_record_period: int = 1000,
    render: bool = False,
) -> Tuple[Dict[str, Any], int]:
    """
    Runs one episode and returns an info dict and number of steps taken.

    Args:
        agent (Agent): the agent.
        env (Environment): the environment.
        learn (bool, optional): Runs `agent.learn()` after every step. Defaults to False.
        exploit (bool, optional): asks the agent to only exploit. Defaults to False.
        learn_after_episode (bool, optional): asks the agent to only learn at
                                              the end of the episode. Defaults to False.
        learn_every_k_steps (int, optional): asks the agent to learn every k steps.
        total_steps (int, optional): the total number of steps taken so far. Defaults to 0.
    Returns:
        Tuple[Dict[str, Any], int]: the return of the episode and the number of steps taken.
    """
    observation, action_space = env.reset()
    agent.reset(observation, action_space)
    cum_reward = 0
    cum_clipped_reward = 0

    done = False
    episode_steps = 0
    info = {}
    frames = []
    while not done:
        if render:
            frames.append(env.render())
        # the agent takes an action
        action = agent.act(exploit=exploit)

        # the environment receives the action and returns the result
        action_result = env.step(action)
        original_reward = action_result.reward
        for preprocessor in preprocessors:
            preprocessor.process(action_result)
        clipped_reward = action_result.reward

        # the agent observes the result
        agent.observe(action_result)

        done = action_result.truncated or action_result.terminated
        episode_steps += 1
        
        # learn
        if learn and total_steps + episode_steps >= learning_start:
            if learn_after_episode:
                # when learn_after_episode is True, we learn only at the end of the episode,
                # regardless of the value of learn_every_k_steps.
                if done:
                    report = agent.learn()
                else:
                    report = {}
            else:
                assert learn_every_k_steps > 0, "learn_every_k_steps must be positive"
                if (total_steps + episode_steps) % learn_every_k_steps == 0:
                    report = agent.learn()
                else:
                    report = {}
        else:
            report = {}

        # update stats
        cum_reward += original_reward
        cum_clipped_reward += clipped_reward

        # record stats
        for key in report:
            learning_report_cache.setdefault(key, []).append(report[key])
        if record_visited_observations and (total_steps + episode_steps) % observation_record_period == 0:
            visited_observations.append(action_result.observation)

        if (
            number_of_steps is not None
            and total_steps + episode_steps >= number_of_steps
        ):
            break

    info["return"] = cum_reward
    info["clipped_return"] = cum_clipped_reward
    if "episode" in action_result.info:
        # in Atari games, we terminate the episode when the agent loses a life
        # Each game has multiple lives. Sometimes we care about the total return accumulated over all lives.
        # This is saved in info["full_return"]. 
        logger.info(action_result.info["episode"]["r"])
        info["full_return"] = action_result.info["episode"]["r"]
    if render:
        info["frames"] = frames
    return info, episode_steps


def train_episodic(
    train_agent: PearlAgent,
    eval_continuing_agent: PearlAgent,
    eval_episodic_agent: PearlAgent,
    train_env: GymEnvironment,
    eval_continuing_env: Optional[GymEnvironment],
    eval_episodic_env: Optional[GymEnvironment],
    param_sweeper_dict: Dict[str, Any],
) -> None:
    print_every_x_steps = param_sweeper_dict["print_every_x_steps"]
    learn_every_k_steps = param_sweeper_dict["learn_every_k_steps"]
    record_period = param_sweeper_dict["record_period"]
    run_idx = param_sweeper_dict["id"]
    number_of_steps = param_sweeper_dict["max_steps"]
    eval_max_steps = param_sweeper_dict["eval_max_steps"]
    record_visited_observations = param_sweeper_dict.get("record_visited_observations", False)
    observation_record_period = param_sweeper_dict.get("observation_record_period", 1000)
    total_steps = 0
    total_episodes = 0
    info = {}
    info_period = {}
    eval_episodic_return_list, eval_average_reward_list, eval_episodic_clipped_return_list, eval_average_clipped_reward_list, eval_average_reset_list = [], [], [], [], []  # noqa
    learning_report = {}
    learning_report_cache = {}
    start_time = time.time()
    last_timed_steps = 0
    visited_observations = []

    while total_steps < number_of_steps:
        old_total_steps = total_steps
        episode_info, episode_total_steps = run_episode(
            train_agent,
            train_env,
            exploit=False,
            learn_after_episode=False,  # not for this project
            learn_every_k_steps=learn_every_k_steps,
            total_steps=old_total_steps,
            learning_start=param_sweeper_dict["learning_starts"],
            learn=True,
            preprocessors=param_sweeper_dict["preprocessors"],
            learning_report_cache=learning_report_cache,
            number_of_steps=number_of_steps,
            visited_observations=visited_observations,
            record_visited_observations=record_visited_observations,
            observation_record_period=observation_record_period,
        )

        total_steps += episode_total_steps
        total_episodes += 1

        # print stats
        if old_total_steps // print_every_x_steps < total_steps // print_every_x_steps:
            logger.info(
                f"episode {total_episodes}, steps {total_steps}, agent={train_agent}, env={train_env}",
            )
            end_time = time.time()
            SPS = int((total_steps - last_timed_steps) / (end_time - start_time))
            logger.info(f"samples per second: {SPS}")
            start_time = end_time
            last_timed_steps = total_steps
            for key in episode_info:
                logger.info(f"{key}: {episode_info[key]}")

        for key in episode_info:
            info_period.setdefault(key, []).append(episode_info[key])
        if old_total_steps // record_period < total_steps // record_period:
            # multiple record_periods may pass between old_total_steps and total_steps
            # duplicate the recording to simulate recording every record_period steps
            num_repeating_recordings = (total_steps // record_period) - (
                old_total_steps // record_period
            )
            for _ in range(num_repeating_recordings):
                for key in info_period:
                    info.setdefault(key, []).append(np.mean(info_period[key]))
            info_period = {}
            # evaluate the learned policy in the episodic and continuing versions of the environment
            if param_sweeper_dict.get("eval_in_episodic_env", False):
                eval_episodic_return, eval_episodic_clipped_return = eval_episodic(
                    eval_episodic_agent=eval_episodic_agent,
                    eval_episodic_env=eval_episodic_env,
                    eval_max_steps=eval_max_steps,
                    preprocessors=param_sweeper_dict["preprocessors"],
                )
            else:
                eval_episodic_return, eval_episodic_clipped_return = None, None

            if param_sweeper_dict.get("eval_in_continuing_env", False):
                eval_average_reward, eval_average_clipped_reward, eval_average_reset = eval_continuing(
                    eval_continuing_agent=eval_continuing_agent,
                    eval_continuing_env=eval_continuing_env,
                    eval_max_steps=eval_max_steps,
                    preprocessors=param_sweeper_dict["preprocessors"],
                )
            else:
                eval_average_reward, eval_average_clipped_reward, eval_average_reset = None, None, None

            for _ in range(num_repeating_recordings):
                if eval_episodic_return is not None:
                    eval_episodic_return_list.append(eval_episodic_return)
                if eval_average_reward is not None:
                    eval_average_reward_list.append(eval_average_reward)
                if eval_episodic_clipped_return is not None:
                    eval_episodic_clipped_return_list.append(eval_episodic_clipped_return)
                if eval_average_clipped_reward is not None:
                    eval_average_clipped_reward_list.append(eval_average_clipped_reward)
                if eval_average_reset is not None:
                    eval_average_reset_list.append(eval_average_reset)
                for key in learning_report_cache:
                    learning_report.setdefault(key, []).append(
                        np.mean(learning_report_cache[key])
                    )

    output_dir = param_sweeper_dict["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # save stats
    save_data_dict = {
        "eval_episodic_return": eval_episodic_return_list,
        "eval_average_reward": eval_average_reward_list,
        "eval_episodic_clipped_return": eval_episodic_clipped_return_list,
        "eval_average_clipped_reward": eval_average_clipped_reward_list,
        "eval_average_reset": eval_average_reset_list,
        "visited_observations": visited_observations,
    }
    save_data_dict.update(info)
    save_data_dict.update(learning_report)
    save_as_npy(data=save_data_dict, output_dir=output_dir, run_idx=run_idx)


def train_continuing(
    train_agent: PearlAgent,
    eval_continuing_agent: PearlAgent,  # an agent sharing the same policy as train_agent, and will be evaluated in a continuing environment
    eval_episodic_agent: PearlAgent,  # an agent sharing the same policy as train_agent, and will be evaluated in an episodic environment
    train_env: GymEnvironment,
    eval_continuing_env: Optional[GymEnvironment],
    eval_episodic_env: Optional[GymEnvironment],
    param_sweeper_dict: Dict[str, Any],
) -> None:
    # get the parameters
    print_every_x_steps = param_sweeper_dict["print_every_x_steps"]
    learn_every_k_steps = param_sweeper_dict["learn_every_k_steps"]
    assert learn_every_k_steps > 0, "learn_every_k_steps must be positive"
    learning_starts = param_sweeper_dict["learning_starts"]
    record_period = param_sweeper_dict["record_period"]
    run_idx = param_sweeper_dict["id"]
    max_steps = param_sweeper_dict["max_steps"]
    eval_max_steps = param_sweeper_dict["eval_max_steps"]
    record_visited_observations = param_sweeper_dict.get("record_visited_observations", False)
    observation_record_period = param_sweeper_dict.get("observation_record_period", 1000)
    preprocessors = param_sweeper_dict["preprocessors"]
    eval_in_episodic_env = param_sweeper_dict.get("eval_in_episodic_env", False)
    eval_in_continuing_env = param_sweeper_dict.get("eval_in_continuing_env", False)
    output_dir = param_sweeper_dict["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # record stats initialization
    experiment_stats = {
        "avg_reward_list": [],
        "eval_average_reward_list": [],
        "eval_episodic_return_list": [],
        "avg_reset_list": [],
        "eval_average_reset_list": [],
        "avg_clipped_reward_list": [],
        "eval_average_clipped_reward_list": [],
        "eval_episodic_clipped_return_list": [],
        "learning_report": {},
        "visited_observations": [],
    }

    # variables used in the training loop but not recorded
    cum_reward = 0
    cum_reset = 0
    cum_clipped_reward = 0
    last_cum_reward_print = 0
    last_cum_reward_record = 0
    last_cum_reset_print = 0
    last_cum_reset_record = 0
    last_cum_clipped_reward_print = 0
    last_cum_clipped_reward_record = 0
    learning_report_cache = {}
    last_timed_steps = 0
    steps = 0

    # initialize the environment and the agent
    assert train_env is not None
    observation, action_space = train_env.reset()
    train_agent.reset(observation, action_space)

    start_time = time.time()

    # starts the training loop
    while steps < max_steps:
        # agent takes an action
        action = train_agent.act(exploit=False)

        # environment receives the action and returns the result
        action_result = train_env.step(action)
        original_reward = action_result.reward
        for preprocessor in preprocessors:
            preprocessor.process(action_result)
        clipped_reward = action_result.reward
        num_resets = action_result.info.get("num_resets", 0)

        steps += 1

        # agent observes the new result of the action
        train_agent.observe(action_result)

        # agent learns
        if (
            steps >= learning_starts
            and steps % learn_every_k_steps == 0
        ):
            report = train_agent.learn()
        else:
            report = {}
        
        # update stats
        cum_reward += original_reward
        cum_reset += num_resets
        cum_clipped_reward += clipped_reward
        for key in report:
            learning_report_cache.setdefault(key, []).append(report[key])

        # print stats
        if steps % print_every_x_steps == 0:
            actor_loss = (
                np.mean(learning_report_cache["actor_loss"])
                if "actor_loss" in learning_report_cache
                else None
            )
            critic_loss = (
                np.mean(learning_report_cache["critic_loss"])
                if "critic_loss" in learning_report_cache
                else None
            )
            message = f"steps {steps}, agent={train_agent}, env={train_env}, average_reward={(cum_reward - last_cum_reward_print) / print_every_x_steps}, average_clipped_reward={(cum_clipped_reward - last_cum_clipped_reward_print) / print_every_x_steps}, average_reset={(cum_reset - last_cum_reset_print) / print_every_x_steps}, actor_loss = {actor_loss}, critic_loss = {critic_loss}"
            last_cum_clipped_reward_print = cum_clipped_reward
            last_cum_reward_print = cum_reward
            last_cum_reset_print = cum_reset
            end_time = time.time()
            SPS = int((steps - last_timed_steps) / (end_time - start_time))
            logger.info(f"samples per second: {SPS}")
            logger.info(message)
            start_time = end_time
            last_timed_steps = steps

        # record visited observations
        if record_visited_observations is True and steps % observation_record_period == 0:
            experiment_stats["visited_observations"].append(observation)

        # record stats
        if steps % record_period == 0:
            # record the average reward over the last record_period time steps
            experiment_stats["avg_reward_list"].append(
                (cum_reward - last_cum_reward_record) / record_period
            )
            last_cum_reward_record = cum_reward

            # record the average reset over the last record_period time steps
            experiment_stats["avg_reset_list"].append((
                cum_reset - last_cum_reset_record) / record_period
            )
            last_cum_reset_record = cum_reset

            # record the average clipped reward over the last record_period time steps
            experiment_stats["avg_clipped_reward_list"].append(
                (cum_clipped_reward - last_cum_clipped_reward_record) / record_period
            )
            last_cum_clipped_reward_record = cum_clipped_reward

            # evaluate the learned policy in an episodic and a continuing versions of the environment
            if eval_in_episodic_env:
                (
                    eval_episodic_return,
                    eval_episodic_clipped_return,
                ) = eval_episodic(
                    eval_episodic_agent=eval_episodic_agent,
                    eval_episodic_env=eval_episodic_env,
                    eval_max_steps=eval_max_steps,
                    preprocessors=preprocessors,
                )
            else:
                eval_episodic_return = None
                eval_episodic_clipped_return = None

            if eval_in_continuing_env:
                (
                    eval_average_reward,
                    eval_average_clipped_reward,
                    eval_average_reset,
                ) = eval_continuing(
                    eval_continuing_agent=eval_continuing_agent,
                    eval_continuing_env=eval_continuing_env,
                    eval_max_steps=eval_max_steps,
                    preprocessors=preprocessors,
                )
            else:
                eval_average_reward = None
                eval_average_clipped_reward = None
                eval_average_reset = None

            if eval_episodic_return is not None:
                experiment_stats["eval_episodic_return_list"].append(eval_episodic_return)
            if eval_average_reward is not None:
                experiment_stats["eval_average_reward_list"].append(eval_average_reward)
            if eval_episodic_clipped_return is not None:
                experiment_stats["eval_episodic_clipped_return_list"].append(eval_episodic_clipped_return)
            if eval_average_clipped_reward is not None:
                experiment_stats["eval_average_clipped_reward_list"].append(eval_average_clipped_reward)
            if eval_average_reset is not None:
                experiment_stats["eval_average_reset_list"].append(eval_average_reset)

            # record stats in learning report
            for key in learning_report_cache:
                experiment_stats["learning_report"].setdefault(key, []).append(
                    np.mean(learning_report_cache[key])
                )

    # save all the recorded stats
    save_data_dict = {
        "average_reward": experiment_stats["avg_reward_list"],
        "average_reset": experiment_stats["avg_reset_list"],
        "average_clipped_reward": experiment_stats["avg_clipped_reward_list"],
        "eval_episodic_return": experiment_stats["eval_episodic_return_list"],
        "eval_average_reward": experiment_stats["eval_average_reward_list"],
        "eval_average_reset": experiment_stats["eval_average_reset_list"],
        "eval_episodic_clipped_return": experiment_stats["eval_episodic_clipped_return_list"],
        "eval_average_clipped_reward": experiment_stats["eval_average_clipped_reward_list"],
        "visited_observations": experiment_stats["visited_observations"],
    }
    save_data_dict.update(experiment_stats["learning_report"])  # assume no overlap in keys

    save_as_npy(
        data=save_data_dict,
        output_dir=output_dir,
        run_idx=run_idx,
    )

    # save game states so that we can visualize the behavior of the agent after training
    # by running the learned policy starting from the saved game state
    if (
        "save_last_state" in param_sweeper_dict
        and param_sweeper_dict["save_last_state"]
    ):
        if (
            hasattr(env.env, "data")
            # pyre-fixme
            and hasattr(env.env.data, "qpos")
            and hasattr(env.env.data, "qvel")
        ):
            # This is the case for mujoco games
            np.save(f"{output_dir}/{run_idx}_qpos.npy", np.array(env.env.data.qpos))
            np.save(f"{output_dir}/{run_idx}_qvel.npy", np.array(env.env.data.qvel))
        else:
            raise NotImplementedError("Can not obtain last environment state!")
    return

def save_as_npy(data: Dict[str, Any], output_dir: str, run_idx: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for key in data:
        if len(data[key]) > 0:
            np.save(f"{output_dir}/{run_idx}_{key}.npy", np.array(data[key]))


def init_class(
    # pyre-fixme
    module_class,
    module_name: str,
    param_sweeper_dict: Dict[str, Any],
) -> None:
    filtered_dict = {}
    for key, value in param_sweeper_dict.items():
        prefix = module_name + ":"
        if (
            len(key) > len(prefix)
            and prefix == key[0 : len(prefix)]
            and key[len(prefix) :] != "type"
        ):
            filtered_dict[key[len(prefix) :]] = value
    # logger.info(filtered_dict)
    param_sweeper_dict[module_name] = module_class(**filtered_dict)


def ppo_init_network_continuous(param_sweeper_dict: Dict[str, Any]) -> None:
    param_sweeper_dict["actor_network_instance"].apply(orthogonal_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(orthogonal_init_weights)
    if hasattr(param_sweeper_dict["actor_network_instance"], "fc_mu"):
        param_sweeper_dict["actor_network_instance"].fc_mu.weight.data.copy_(
            0.01 * param_sweeper_dict["actor_network_instance"].fc_mu.weight.data
        )


def ppo_init_network_discrete(param_sweeper_dict: Dict[str, Any]) -> None:
    param_sweeper_dict["actor_network_instance"].apply(orthogonal_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(orthogonal_init_weights)
    if hasattr(param_sweeper_dict["actor_network_instance"], "_model_fc"):
        param_sweeper_dict["actor_network_instance"]._model_fc[-1][0].weight.data.copy_(
            0.01
            / 1.4142
            * param_sweeper_dict["actor_network_instance"]._model_fc[-1][0].weight.data
        )
    if hasattr(param_sweeper_dict["critic_network_instance"], "_model_fc"):
        param_sweeper_dict["critic_network_instance"]._model_fc[-1][
            0
        ].weight.data.copy_(
            1.0
            / 1.4142
            * param_sweeper_dict["critic_network_instance"]._model_fc[-1][0].weight.data
        )


def sac_atari_init_network(param_sweeper_dict: Dict[str, Any]) -> None:
    param_sweeper_dict["actor_network_instance"].apply(kaiming_normal_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(kaiming_normal_init_weights)


def ac_init_network(param_sweeper_dict: Dict[str, Any]) -> None:
    param_sweeper_dict["actor_network_instance"].apply(xavier_init_weights)
    param_sweeper_dict["critic_network_instance"].apply(xavier_init_weights)


def q_init_network(param_sweeper_dict: Dict[str, Any]) -> None:
    param_sweeper_dict["network_instance"].apply(xavier_init_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--gpu-id", default=-1)
    parser.add_argument("--base-id", default=0)
    parser.add_argument(
        "--config-file", default="experiments/new_exps/mujoco_no_resets/inputs.json"
    )
    parser.add_argument("--out-dir", default="/tmp/pearl")
    parser.add_argument("--eval-agent", action="store_true")
    parser.add_argument("--render", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    exp_name: str = args.config_file.split("/")[1]
    project_root: str = os.path.abspath(os.path.dirname(__file__))
    # pyre-fixme
    param_sweeper = Sweeper(os.path.join(project_root, args.config_file))
    agents: List[PearlAgent] = []
    envs: List[Optional[GymEnvironment]] = []
    run_id: int = int(args.base_id)
    param_sweeper_dict: Dict[str, Any] = param_sweeper.parse(run_id)
    param_sweeper_dict["id"] = run_id
    param_sweeper_dict["device_id"] = args.gpu_id
    param_sweeper_dict["output_dir"] = args.out_dir

    """
    log hyper parameters
    """

    for item in param_sweeper_dict:
        logger.info(f"{item}: {param_sweeper_dict[item]}")

    """
    set random seed
    """

    set_seed(run_id)

    """
    Initialize the environment
    """

    envs_configs = ["env", "eval_env_episodic", "eval_env_continuing"]
    for i in range(3):
        # envs[0]: training agent and env
        # envs[1]: evaluated in episodic env
        # envs[2]: evaluated in continuing env
        if envs_configs[i] not in param_sweeper_dict:
            envs.append(None)
            continue
        if args.render:
            param_sweeper_dict[envs_configs[i]][0]["render_mode"] = "rgb_array"
            env = get_env(param_sweeper_dict[envs_configs[i]])
        else:
            env = get_env(param_sweeper_dict[envs_configs[i]])
        env.action_space._gym_space.seed(seed=run_id)
        env.reset(seed=run_id)
        envs.append(env)
        if i == 1 or i == 2:
            # make sure the three environments have the same size of state and action spaces
            assert envs[0].observation_space.shape == envs[i].observation_space.shape
            assert envs[0].action_space.shape == envs[i].action_space.shape

    # pyre-fixme
    train_env: GymEnvironment = envs[0]  # training environment
    eval_episodic_env: Optional[GymEnvironment] = envs[1]  # evaluated in episodic env
    eval_continuing_env: Optional[GymEnvironment] = envs[2]  # evaluated in continuing env

    param_sweeper_dict["action_space"] = env.action_space
    param_sweeper_dict["preprocessors"] = []
    if isinstance(env.action_space, DiscreteActionSpace):
        max_number_actions: int = env.action_space.n
        action_dim: int = env.action_space.action_dim
    elif isinstance(env.action_space, BoxActionSpace):
        max_number_actions = -1
        action_dim = env.action_space.action_dim
    else:
        raise NotImplementedError

    """
    Initialize preprocessors
    """
    param_sweeper_dict["preprocessors"] = []
    training_env_name: str = param_sweeper_dict["env"][0]["env_name"]
    if "ALE/" in training_env_name or "NoFrameskip" in training_env_name:
        param_sweeper_dict["preprocessors"].append(RewardClipping())

    if (
        param_sweeper_dict["policy_learner:type"] == "ProximalPolicyOptimization"
        and param_sweeper_dict["is_action_continuous"] is True
    ):
        param_sweeper_dict["preprocessors"].append(
            # pyre-fixme
            ObservationNormalization(train_env.observation_space.shape)
        )

    """
    Initialize action representation module
    """

    if "action_representation_module:type" in param_sweeper_dict:
        # if action representation module name is specified, initialize a module
        if param_sweeper_dict["action_representation_module:type"] in [
            "OneHotActionTensorRepresentationModule",
        ]:
            param_sweeper_dict["action_representation_module:max_number_actions"] = (
                max_number_actions
            )
        elif param_sweeper_dict["action_representation_module:type"] in [
            "IdentityActionRepresentationModule"
        ]:
            param_sweeper_dict["action_representation_module:representation_dim"] = (
                action_dim
            )
        else:
            raise NotImplementedError
        action_representation_module_class: Type[ActionRepresentationModule] = getattr(
            action_representation_modules,
            param_sweeper_dict["action_representation_module:type"],
        )
        init_class(
            action_representation_module_class,
            "action_representation_module",
            param_sweeper_dict,
        )
    else:
        param_sweeper_dict["action_representation_module"] = (
            IdentityActionRepresentationModule(
                max_number_actions=max_number_actions,
                representation_dim=action_dim,
            )
        )

    action_representation_dim: int = param_sweeper_dict[
        "action_representation_module"
    ].representation_dim

    """
    Initialize exploration module
    """

    if "exploration_module:type" in param_sweeper_dict:
        # if exploration module name is specified, initialize an exploration module
        exploration_module_class: Type[ExplorationModule] = getattr(
            exploration_modules, param_sweeper_dict["exploration_module:type"]
        )
        if "exploration_module:std_dev" in param_sweeper_dict and isinstance(
            param_sweeper_dict["exploration_module:std_dev"], list
        ):
            assert isinstance(env.action_space, BoxActionSpace)
            assert len(param_sweeper_dict["exploration_module:std_dev"]) == 2
            tmp: torch.Tensor = (
                torch.ones(
                    param_sweeper_dict[
                        "action_representation_module:representation_dim"
                    ]
                )
                * param_sweeper_dict["exploration_module:std_dev"][0]
            )
            tmp[-1] = param_sweeper_dict["exploration_module:std_dev"][1]
            param_sweeper_dict["exploration_module:std_dev"] = tmp
        init_class(exploration_module_class, "exploration_module", param_sweeper_dict)
        if "exploration_module_wrapper:type" in param_sweeper_dict:
            # if exploration wrapper module name is specified, initialize an exploration module
            exploration_wrapper_class: Type[ExplorationModuleWrapper] = getattr(
                exploration_wrappers,
                param_sweeper_dict["exploration_module_wrapper:type"],
            )
            param_sweeper_dict["exploration_module_wrapper:exploration_module"] = (
                param_sweeper_dict["exploration_module"]
            )
            init_class(
                exploration_wrapper_class,
                "exploration_module_wrapper",
                param_sweeper_dict,
            )
            param_sweeper_dict["exploration_module"] = param_sweeper_dict[
                "exploration_module_wrapper"
            ]
    else:
        param_sweeper_dict["exploration_module"] = NoExploration()

    if "actor_update_noise" in param_sweeper_dict and isinstance(
        param_sweeper_dict["actor_update_noise"], list
    ):
        assert isinstance(env.action_space, BoxActionSpace)
        assert len(param_sweeper_dict["actor_update_noise"]) == 2
        tmp = (
            torch.ones(
                param_sweeper_dict["action_representation_module:representation_dim"]
            )
            * param_sweeper_dict["actor_update_noise"][0]
        )
        tmp[-1] = param_sweeper_dict["actor_update_noise"][1]
        param_sweeper_dict["actor_update_noise"] = tmp

    """
    Initialize replay buffer
    """

    if "replay_buffer:type" in param_sweeper_dict:
        # if replay buffer is specified, intialize one
        replay_buffer_class: Type[ReplayBuffer] = getattr(
            replay_buffers, param_sweeper_dict["replay_buffer:type"]
        )
        init_class(replay_buffer_class, "replay_buffer", param_sweeper_dict)
    else:
        print("Replay buffer must be specified")
        raise NotImplementedError

    """
    Initialize networks
    """

    if "network_instance:type" in param_sweeper_dict:
        # for q-learning methods
        if param_sweeper_dict["network_instance:type"] in [
            "CNNQValueNetwork",
            "CNNQValueMultiHeadNetwork",
        ]:
            # image based inputs
            assert len(env.observation_space.shape) == 3
            param_sweeper_dict["network_instance:input_width"] = 84
            param_sweeper_dict["network_instance:input_height"] = 84
            param_sweeper_dict["network_instance:input_channels_count"] = 4
        elif param_sweeper_dict["network_instance:type"] in [
            "VanillaQValueNetwork",
            "VanillaQValueMultiHeadNetwork",
        ]:
            # vector based inputs
            assert len(env.observation_space.shape) == 1
            param_sweeper_dict["network_instance:state_dim"] = (
                env.observation_space.shape[0]
            )
        else:
            raise NotImplementedError
        param_sweeper_dict["network_instance:action_dim"] = action_representation_dim
        network_class: Type[QValueNetwork] = getattr(
            q_value_networks, param_sweeper_dict["network_instance:type"]
        )
        init_class(network_class, "network_instance", param_sweeper_dict)

    if "actor_network_instance:type" in param_sweeper_dict:
        # if actor network is specified, intialize one
        if param_sweeper_dict["actor_network_instance:type"] in [
            "CNNActorNetwork",
        ]:
            # image based inputs
            assert len(env.observation_space.shape) == 3
            param_sweeper_dict["actor_network_instance:input_width"] = 84
            param_sweeper_dict["actor_network_instance:input_height"] = 84
            param_sweeper_dict["actor_network_instance:input_channels_count"] = 4
        elif param_sweeper_dict["actor_network_instance:type"] in [
            "VanillaActorNetwork",
            "VanillaContinuousActorNetwork",
            "GaussianActorNetwork",
            "ClipGaussianActorNetwork",
        ]:
            param_sweeper_dict["actor_network_instance:input_dim"] = (
                env.observation_space.shape[0]
            )
        else:
            raise NotImplementedError
        param_sweeper_dict["actor_network_instance:output_dim"] = (
            action_representation_dim
            if max_number_actions == -1  # continuous actions
            else max_number_actions  # discrete actions
        )
        param_sweeper_dict["actor_network_instance:action_space"] = env.action_space
        actor_class: Type[ActorNetwork] = getattr(
            actor_networks, param_sweeper_dict["actor_network_instance:type"]
        )
        init_class(actor_class, "actor_network_instance", param_sweeper_dict)

    if "critic_network_instance:type" in param_sweeper_dict:
        # if critic network is specified, intialize one
        if param_sweeper_dict["critic_network_instance:type"] in [
            "VanillaValueNetwork",
        ]:
            critic_class: Type[ValueNetwork] = getattr(
                value_networks, param_sweeper_dict["critic_network_instance:type"]
            )
            param_sweeper_dict["critic_network_instance:input_dim"] = (
                env.observation_space.shape[0]
            )
            init_class(critic_class, "critic_network_instance", param_sweeper_dict)
        elif param_sweeper_dict["critic_network_instance:type"] in [
            "CNNValueNetwork",
        ]:
            critic_class = getattr(
                value_networks, param_sweeper_dict["critic_network_instance:type"]
            )
            param_sweeper_dict["critic_network_instance:input_width"] = 84
            param_sweeper_dict["critic_network_instance:input_height"] = 84
            param_sweeper_dict["critic_network_instance:input_channels_count"] = 4
            init_class(critic_class, "critic_network_instance", param_sweeper_dict)
        elif param_sweeper_dict["critic_network_instance:type"] in [
            "EnsembleQValueNetwork",
        ]:
            list_of_member_networks: List[QValueNetwork] = []
            ensemble_size: int = param_sweeper_dict[
                "critic_network_instance:ensemble_size"
            ]
            for _ in range(ensemble_size):
                if param_sweeper_dict["critic_member_network:type"] in [
                    "CNNQValueNetwork",
                    "CNNQValueMultiHeadNetwork",
                ]:
                    # image based inputs
                    assert len(env.observation_space.shape) == 3
                    param_sweeper_dict["critic_member_network:input_width"] = 84
                    param_sweeper_dict["critic_member_network:input_height"] = 84
                    param_sweeper_dict["critic_member_network:input_channels_count"] = 4
                elif param_sweeper_dict["critic_member_network:type"] in [
                    "VanillaQValueNetwork",
                    "VanillaQValueMultiHeadNetwork",
                ]:
                    # vector based inputs
                    assert len(env.observation_space.shape) == 1
                    param_sweeper_dict["critic_member_network:state_dim"] = (
                        env.observation_space.shape[0]
                    )
                else:
                    raise NotImplementedError
                param_sweeper_dict["critic_member_network:action_dim"] = (
                    action_representation_dim
                )
                member_network_class = getattr(
                    q_value_networks,
                    param_sweeper_dict["critic_member_network:type"],
                )
                filtered_dict = {}
                for key, value in param_sweeper_dict.items():
                    prefix = "critic_member_network:"
                    if (
                        len(key) > len(prefix)
                        and prefix == key[0 : len(prefix)]
                        and key[len(prefix) :] != "type"
                    ):
                        filtered_dict[key[len(prefix) :]] = value
                list_of_member_networks.append(member_network_class(**filtered_dict))
            models: nn.ModuleList = nn.ModuleList(list_of_member_networks)
            critic_class: Type[QValueNetwork] = getattr(
                q_value_networks, param_sweeper_dict["critic_network_instance:type"]
            )
            # pyre-fixme
            param_sweeper_dict["critic_network_instance"] = critic_class(
                models=models, ensemble_size=ensemble_size
            )
        else:
            raise NotImplementedError

    if param_sweeper_dict.get("reward_centering:type", None) is not None:
        if param_sweeper_dict["reward_centering:type"] == "TD":
            param_sweeper_dict["reward_rate"] = nn.Parameter(torch.zeros(1))
        elif param_sweeper_dict["reward_centering:type"] == "MA":
            param_sweeper_dict["reward_rate"] = torch.zeros(1)
            param_sweeper_dict["reward_centering:ma_rate"] = param_sweeper_dict.get(
                "ma_rate", 0.99
            )
        elif param_sweeper_dict["reward_centering:type"] == "RVI":
            param_sweeper_dict["reward_rate"] = torch.zeros(1)
            param_sweeper_dict["reward_centering:ref_states_update_freq"] = (
                param_sweeper_dict.get(
                    "ref_states_update_freq", param_sweeper_dict["max_steps"]
                )
            )  # 0 means no update
        else:
            raise NotImplementedError
    else:
        param_sweeper_dict["reward_rate"] = torch.zeros(1)
    """
    network initialization
    """

    if (
        param_sweeper_dict["policy_learner:type"] == "ProximalPolicyOptimization"
        and param_sweeper_dict["is_action_continuous"] is True
    ):
        ppo_init_network_continuous(param_sweeper_dict)

    if (
        param_sweeper_dict["policy_learner:type"] == "ProximalPolicyOptimization"
        and param_sweeper_dict["is_action_continuous"] is False
    ):
        ppo_init_network_discrete(param_sweeper_dict)

    if param_sweeper_dict["policy_learner:type"] == "SoftActorCritic" and (
        "ALE/" in training_env_name or "NoFrameskip" in training_env_name
    ):
        sac_atari_init_network(param_sweeper_dict)

    """
    Initialize optimizers
    """

    if "optimizer:type" in param_sweeper_dict:
        assert "network_instance" in param_sweeper_dict
        optimizer_class: Type[torch.optim.Optimizer] = getattr(
            torch.optim, param_sweeper_dict["optimizer:type"]
        )
        param_sweeper_dict["optimizer:params"] = param_sweeper_dict[
            "network_instance"
        ].parameters()
        init_class(optimizer_class, "optimizer", param_sweeper_dict)

    if "actor_optimizer:type" in param_sweeper_dict:
        assert "actor_network_instance" in param_sweeper_dict
        actor_optimizer_class: Type[torch.optim.Optimizer] = getattr(
            torch.optim, param_sweeper_dict["actor_optimizer:type"]
        )
        param_sweeper_dict["actor_optimizer:params"] = param_sweeper_dict[
            "actor_network_instance"
        ].parameters()
        init_class(actor_optimizer_class, "actor_optimizer", param_sweeper_dict)

    if "critic_optimizer:type" in param_sweeper_dict:
        assert "critic_network_instance" in param_sweeper_dict
        critic_optimizer_class: Type[torch.optim.Optimizer] = getattr(
            torch.optim, param_sweeper_dict["critic_optimizer:type"]
        )
        param_sweeper_dict["critic_optimizer:params"] = param_sweeper_dict[
            "critic_network_instance"
        ].parameters()
        init_class(critic_optimizer_class, "critic_optimizer", param_sweeper_dict)

    if param_sweeper_dict.get("reward_centering:type", None) is not None:
        if param_sweeper_dict["reward_centering:type"] == "TD":
            reward_rate_optimizer_class: Type[torch.optim.Optimizer] = getattr(
                torch.optim, param_sweeper_dict["reward_rate_optimizer:type"]
            )
            param_sweeper_dict["reward_rate_optimizer:params"] = [
                param_sweeper_dict["reward_rate"]
            ]
            init_class(
                reward_rate_optimizer_class, "reward_rate_optimizer", param_sweeper_dict
            )
            param_sweeper_dict["reward_centering:optimizer"] = param_sweeper_dict[
                "reward_rate_optimizer"
            ]
            init_class(TD_RC, "reward_centering", param_sweeper_dict)
        elif param_sweeper_dict["reward_centering:type"] == "MA":
            init_class(MA_RC, "reward_centering", param_sweeper_dict)
        elif param_sweeper_dict["reward_centering:type"] == "RVI":
            init_class(RVI_RC, "reward_centering", param_sweeper_dict)
    """
    Initialize a policy learner
    """

    policy_learner_class: (
        Type[ProximalPolicyOptimization]
        | Type[DeepQLearning]
        | Type[SoftActorCritic]
        | Type[ContinuousSoftActorCritic]
        | Type[TD3]
        | Type[DeepDeterministicPolicyGradient]
    ) = getattr(policy_learners, param_sweeper_dict["policy_learner:type"])

    filtered_dict: Dict[str, Any] = {
        key: value
        for key, value in param_sweeper_dict.items()
        # pyre-fixme
        if key in policy_learner_class.__init__.__code__.co_varnames
    }
    param_sweeper_dict["policy_learner"] = policy_learner_class(**filtered_dict)

    """
    Initialize a pearl agent
    """

    filtered_dict = {
        key: value
        for key, value in param_sweeper_dict.items()
        # pyre-fixme
        if key in PearlAgent.__init__.__code__.co_varnames
    }
    train_agent = PearlAgent(**filtered_dict)

    """
    Run the experiment
    """
    def create_an_eval_agent_from_a_train_agent(a_train_agent: PearlAgent) -> PearlAgent:
        an_eval_agent: PearlAgent = copy.deepcopy(a_train_agent)
        an_eval_agent.policy_learner = a_train_agent.policy_learner
        # pyre-fixme
        an_eval_agent.replay_buffer = a_train_agent.replay_buffer.__class__(capacity=0)
        return an_eval_agent

    if args.eval_agent:
        # load and evaluate a model and generate videos
        assert (
            "model_folder" in param_sweeper_dict
        ), "model_folder not found in param_sweeper_dict"
        model_path: str = (
            args.out_dir + param_sweeper_dict["model_folder"] + str(run_id)
        )
        try:
            train_agent.policy_learner.load_model(path=model_path)

            for i in range(len(param_sweeper_dict["preprocessors"])):
                p = param_sweeper_dict["preprocessors"][i]
                if isinstance(p, ObservationNormalization):
                    with open(
                        model_path + "_observation_norm.pkl",
                        "rb",
                    ) as file:
                        param_sweeper_dict["preprocessors"][i] = pickle.load(file)
        except Exception as e:
            logger.info(f"Failed to load model from {model_path}: {e}")
            exit(0)

        if isinstance(train_agent.policy_learner.exploration_module, Warmup):
            train_agent.policy_learner.exploration_module.warmup_steps = 0
        if isinstance(
            train_agent.policy_learner.exploration_module, EGreedyExploration
        ):
            train_agent.policy_learner.exploration_module.warmup_steps = None
            # pyre-fixme
            train_agent.policy_learner.exploration_module.curr_epsilon = (
                train_agent.policy_learner.exploration_module.end_epsilon
            )
        if eval_continuing_env:
            eval_continuing_agent: PearlAgent = create_an_eval_agent_from_a_train_agent(train_agent)
            eval_avg_reward, eval_avg_clipped_reward, eval_avg_reset = eval_continuing(
                eval_continuing_agent=eval_continuing_agent,
                eval_continuing_env=eval_continuing_env,
                render=args.render,
                video_dir=args.out_dir + param_sweeper_dict["video_folder"],
                name=training_env_name + "_" + param_sweeper_dict["policy_learner:type"],
                max_steps=param_sweeper_dict["eval_max_steps"],
                run_idx=run_id,
                preprocessors=param_sweeper_dict["preprocessors"],
            )
        if eval_episodic_env:
            eval_episodic_agent: PearlAgent = create_an_eval_agent_from_a_train_agent(train_agent)
            eval_episodic_return, eval_episodic_clipped_return = eval_episodic(
                eval_episodic_agent=eval_episodic_agent,
                eval_episodic_env=eval_episodic_env,
                render=args.render,
                video_dir=args.out_dir + param_sweeper_dict["video_folder"],
                name=training_env_name + "_" + param_sweeper_dict["policy_learner:type"],
                max_steps=param_sweeper_dict["eval_max_steps"],
                run_idx=run_id,
                preprocessors=param_sweeper_dict["preprocessors"],
            )
    else:
        # create two copies of the agent, one for continuing and one for episodic evaluation
        eval_continuing_agent: PearlAgent = create_an_eval_agent_from_a_train_agent(train_agent)
        eval_episodic_agent: PearlAgent = create_an_eval_agent_from_a_train_agent(train_agent)

        if param_sweeper_dict.get(
            "train_env_is_continuing", False
        ):
            train_continuing(
                train_agent, 
                eval_continuing_agent, 
                eval_episodic_agent, 
                train_env, 
                eval_continuing_env, 
                eval_episodic_env, 
                param_sweeper_dict
            )
        else:
            train_episodic(
                train_agent, 
                eval_continuing_agent, 
                eval_episodic_agent, 
                train_env, 
                eval_continuing_env, 
                eval_episodic_env, 
                param_sweeper_dict
            )

        if param_sweeper_dict["save_model"]:
            assert (
                "model_folder" in param_sweeper_dict
            ), "model_folder not found in param_sweeper_dict"
            model_path = args.out_dir + param_sweeper_dict["model_folder"] + str(run_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            train_agent.policy_learner.save_model(path=model_path)

            for p in param_sweeper_dict["preprocessors"]:
                if isinstance(p, ObservationNormalization):
                    with open(
                        model_path + "_observation_norm.pkl",
                        "wb",
                    ) as file:
                        pickle.dump(p, file)
