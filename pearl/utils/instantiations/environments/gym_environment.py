# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict

import logging
from typing import Any, Dict, Optional, Union, List, Callable, Tuple
import numpy as np
from pearl.api.action import Action

from pearl.api.observation import Observation
from pearl.utils.instantiations.spaces import VectorDiscreteSpace, VectorBoxSpace
import torch
import gymnasium as gym
import multiprocessing as mp

GYM_TO_VECTOR_SPACE = {
    "Discrete": VectorDiscreteSpace,
    "Box": VectorBoxSpace,
    # Add more here as needed
}

def _get_vector_space(
    gym_spaces: List[gym.Space], gym_to_vector_map: Dict[str, Any]
):
    gym_space_name = gym_spaces[0].__class__.__name__
    try:
        vector_space_cls = gym_to_vector_map[gym_space_name]
    except KeyError:
        raise NotImplementedError(
            f"The Gym space '{gym_space_name}' is not yet supported in Pearl."
        )
    return vector_space_cls.from_gym(gym_spaces)


def batched_worker(remote, parent_remote, env_fns: List[Callable[[], gym.Env]]):
    parent_remote.close()
    envs = [fn() for fn in env_fns]
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            result = [env.reset(seed=data) for env in envs]
            remote.send(result)
        elif cmd == "step":
            actions = data
            # remove the padding and send to the envs
            result = [env.step(act) for env, act in zip(envs, actions)]
            remote.send(result)
        elif cmd == "close":
            for env in envs:
                env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command {cmd}")


class GymEnvironment:
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        batched: bool = True,
        num_processes: int = 10,
    ) -> None:
        self.batched = batched
        self.num_envs = len(env_fns)

        if batched:
            print(f"Batched mode: {self.num_envs} envs, {num_processes} processes")
            self.num_processes = num_processes
            self.env_indices_per_proc_list = np.array_split(np.arange(self.num_envs), self.num_processes)
            self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_processes)])
            self.processes = []

            for i in range(self.num_processes):
                sub_fns = [env_fns[j] for j in self.env_indices_per_proc_list[i]]
                p = mp.Process(target=batched_worker, args=(self.work_remotes[i], self.remotes[i], sub_fns))
                p.daemon = True
                p.start()
                self.processes.append(p)
                self.work_remotes[i].close()
        else:
            self.env = [fn() for fn in env_fns]

        # Use one sample env to extract space info
        sample_envs = [fn() for fn in env_fns]
        self._action_space = _get_vector_space(
            gym_spaces=[env.action_space for env in sample_envs],
            gym_to_vector_map=GYM_TO_VECTOR_SPACE,
        )
        self._observation_space = _get_vector_space(
            gym_spaces=[env.observation_space for env in sample_envs],
            gym_to_vector_map=GYM_TO_VECTOR_SPACE,
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self, seed: Optional[int] = None) -> Observation:
        if self.batched:
            for remote in self.remotes:
                remote.send(("reset", seed))
            results = sum([remote.recv() for remote in self.remotes], [])  # flatten
        else:
            results = [env.reset(seed=seed) for env in self.env]

        observations, infos = zip(*results)

        # pad the observations to the same dimension
        observations = np.array([
            np.pad(obs, (0, self._observation_space.element_dim() - len(obs)), mode='constant')
            for obs in observations
        ])
        if observations.dtype == np.float64:
            observations = observations.astype(np.float32)
        return observations, infos


    def step(self, action: Action) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        unpadded_actions = [action[i, :self._action_space.actual_sizes[i]] for i in range(self.num_envs)]
        if self.batched:
            # remove the padding
            for i in range(self.num_processes):
                remote = self.remotes[i]
                act_chunk = [unpadded_actions[j] for j in self.env_indices_per_proc_list[i]]
                remote.send(("step", act_chunk))
            results = sum([remote.recv() for remote in self.remotes], [])  # flatten
        else:
            results = [env.step(unpadded_actions[i]) for i, env in enumerate(self.env)]

        observations, rewards, terminations, truncations, infos = zip(*results)

        # pad the observations to the same dimension
        observations = np.array([
            np.pad(obs, (0, self._observation_space.element_dim() - len(obs)), mode='constant')
            for obs in observations
        ])
        if observations.dtype == np.float64:
            observations = observations.astype(np.float32)
        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminations),
            np.array(truncations),
            infos,
        )
    

    def render(self) -> None:
        if self.batched:
            raise NotImplementedError("Rendering is not supported in batched mode")
        for env in self.env:
            env.render()

    def close(self) -> None:
        if self.batched:
            for remote in self.remotes:
                remote.send(("close", None))
            for p in self.processes:
                p.join()
        else:
            for env in self.env:
                env.close()

    def __str__(self) -> str:
        if not self.batched:
            rtn_str = ""
            for env in self.env:
                if env.spec is not None:
                    rtn_str += env.spec.id + "_"
                else:
                    rtn_str += "CustomGymEnvironment_"
            return rtn_str[:-1]
        else:
            return f"BatchedGymEnvironment_{self.num_envs}_envs_{self.num_processes}_procs"
