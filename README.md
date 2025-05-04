# Deep Reinforcement Learning in Continuing Tasks (Under development)

This repository provides the codebase for the paper *An Empirical Study of Deep Reinforcement Learning in Continuing Tasks*. The paper explores challenges that continuing tasks present to current deep reinforcement learning (RL) algorithms using a suite of continuing task testbeds.

## Testbeds
Our testbeds are modified from Mujoco environments. They are based on Swimmer, HumanoidStandup, Reacher, Pusher, HalfCheetah, Ant, Hopper, Humanoid, and Walker2d. We are working on adding more testbeds (maybe Atari).

## Tested algorithms
Continuous control: DDPG, TD3, SAC, PPO

Discrete control: DQN, SAC, PPO

## How to use the codebase

### Setup conda environment and dependencies
- Create a conda environment ```conda create --name pearl python==3.10```
- ```conda activate pearl```
- Install pearl dependencies using ```./setup.sh```
- For mujoco games, copy ```user_envs/special_ant.xml``` to ```CONDA_DIR_PATH/envs/pearl/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/```. This is the xml file of the Ant task with a wider range of the angles at which its legs can move. Replace ```CONDA_DIR_PATH``` by the path to the conda directory in your machine.
- For Atari games, we have to manually increase the default maximum episode length in ```CONDA_DIR_PATH/envs/pearl/lib/python3.10/site-packages/ale_py/registration.py```. The default is 108000. You may change it to any number that is larger than the training steps so that the maximum episode length is not reached during training.

### Experiment configurations
The codebase has several experiment folders, each of which includes a file ```inputs.json```, which specifies a set of experiment configurations. This configuration file is compatible with AlphaEx's sweeper for configuration sweeping. https://github.com/AmiiThinks/AlphaEx?tab=readme-ov-file#sweeper explains how to understand the configuration file. Running experiments given these configurations gives experiment results. The table below shows the correspondence between these folders and the figures/tables in the paper summarizing the experiment results.

| Experiment folder | Description |
|---------------|---------------------------|
| ```experiments/mujoco_no_resets/``` | mujoco tasks without resets |
| ```experiments/mujoco_predefined_resets/``` | mujoco tasks with predefined resets |
| ```experiments/mujoco_agent_resets/``` | mujoco tasks with agent resets |
| ```experiments/mujoco_episodic_tasks/``` | episodic mujoco tasks  |

### Running experiments
- Suppose we want to perform all experiments specified in ```experiments/mujoco_no_resets/inputs.json``` for five runs. Note that there are overall 40 experiment configurations in ```experiments/mujoco_no_resets/inputs.json```. Therefore, overall there will be 40 * 5 = 400 experiments. One could run these experiments sequentially using ```for i in {0..399}; do ./run.sh run.py --config-file experiments/mujoco_no_resets/inputs.json --out-dir=experiments/mujoco_no_resets --base-id=i; done --gpu-id 0``` (if you want to not use gpus, simply remove ```--gpu-id 0```). Alternatively, one could run them in parallel using ```run_batch_gpu.sh``` or ```run_batch_cpu.sh``` on a single machine with multiple GPUs/CPUs. Or, if you have cluster access, you may request multiple machines to run your code.

## License
Pearl is MIT licensed, as found in the LICENSE file.
