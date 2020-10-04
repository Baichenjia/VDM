# Variational Dynamic for Self-Supervised Exploration in Deep Reinforcement Learning

[Website](https://sites.google.com/view/exploration-vdm)  

## Introduction

This is a TensorFlow based implementation for our paper on 
"Variational Dynamic for Self-Supervised Exploration in Deep Reinforcement Learning".

## Prerequisites

VDM requires python3.6,
tensorflow-gpu 1.13 or 1.14,
tensorflow-probability 0.6.0,
openAI [baselines](https://github.com/openai/baselines),
openAI [Gym](http://gym.openai.com/),
openAI [Retro](https://github.com/openai/retro)

## Installation and Usage

### Noise mnist

The following command should train the variational dynamic model (VDM) for "noise mnist" MDP.

```
cd dvae_model/noise_mnist
python noise_mnist_model.py
```

This command will train VDM for 500 epochs. Actually, 200 epochs is enough to get good results. The weights of VDM saved in `model/`. Then use following command to perform the conditional generation process to reproduce the figure in our paper.
```
python noise_mnist_test.py
```

### Atari games

The following command should train a pure exploration 
agent on "Breakout" with default experiment parameters.

```
python run.py --env BreakoutNoFrameskip-v4
```

### Atari games with sticky actions

The following command should train a pure exploration 
agent on "sticky Breakout" with a probability of 0.25

```
python run.py --env BreakoutNoFrameskip-v4 --stickyAtari
```

### Super Mario 
Download the ROM of Super Mario at [Google Drive](https://drive.google.com/file/d/1EtfCS4UqDoC4jLSHtFGcYd5blX6IHBib/view?usp=sharing), 
unzip it, and run the following command to import the ROM of Mario. 
```
cd mario 
python -m retro.import .
``` 
There are several levels in Super Mario. The level is 
specified in function `make_mario_env` of `wrappers.py`.

The following command should train a pure exploration 
agent in level 1 of Super Mario with default experiment parameters.
```
python run.py --env mario --env_kind mario
```

### Two-player Pong

Download the ROM of Two-player Pong at [Google Drive](https://drive.google.com/file/d/1DwqOXwUEcRJ5-UBLJ1saLwXYaGGBz19h/view?usp=sharing), 
unzip it, and run the following command to import the ROM of Two-player Pong. 
```
cd multi-pong 
python -m retro.import .
``` 

### Running on a Real Robot (UR5)

We use VDM to train a real [UR5](https://www.universal-robots.com/products/ur5-robot/) robot arm.
We develop a [robot environment](https://drive.google.com/file/d/1hZS61dqEOsP1IlFMjRQHBjlcqWg6Tgy9/view?usp=sharing) 
based on `gym.Env` to provide the interface like Gym. 

#### Setting up Camera System

Our system uses the RGB-D image taken by an 
[Intel® RealSense™ D435 Camera](https://click.intel.com/intelr-realsensetm-depth-camera-d435.html). 
We use a lightweight C++ executable package from [librealsense SDK 2.0](https://github.com/IntelRealSense/librealsense).
The camera configuration process shows as follows. 

1. Download and install [librealsense SDK 2.0](https://github.com/IntelRealSense/librealsense)

2. Navigate to `realsense` and compile `realsense.cpp`:

   ```shell
   cd realsense
   cmake .
   make
   ```

3. Connect your RealSense camera with a USB 3.0 compliant cable 

4. To start the TCP server and RGB-D streaming, run the following:

   ```shell
   ./realsense
   ```
#### Setting up Robot-Gym

We develop a robot environment and package it in [Gym](http://gym.openai.com/) 
through the following command:

1. Download and Install [OpenAI Gym](http://gym.openai.com/docs/)

2. Clone [user](https://drive.google.com/file/d/1hZS61dqEOsP1IlFMjRQHBjlcqWg6Tgy9/view?usp=sharing) into `gym/envs`

3. Add the following code in `__init__.py`

   ```python
    register(
        id = 'GymRobot-v1',
        entry_point='gym.envs.user:GymRobotPushEnv', 
        max_episode_steps=1000,
   )
   ```

4. After configuring the TCP, setting the action space, and connecting the camera, 
you can test the environment through following command,

   ```python
   import gym, baseline
   env = gym.make('GymRobot-v1')
   obs = env.reset()
   action = env.action_space.sample()
   next_obs, rew, done, info = env.step(action)
   ```

#### Run VDM

The training code for the robot arm is slightly different from this repository because of the
action type and gym wrapper. The code can be downloaded [here](https://drive.google.com/file/d/1qeUiDwXxRGytqUqRNBevgBdAqL1PXu1P/view?usp=sharing). 

The following command should train a pure exploration 
agent on UR5 robot arm.
```
python run.py --env GymRobot-v1 --env_kind GymRobot-v1
```

In every run, the robot starts with 3 objects placed in front of it. If either the robot 
completes 100 interactions or there are no objects in front of it, the objects are replaced manually. 
We save the model every 1000 interactions.

We use [Self-Supervised Exploration via Disagreement, ICML 2019](https://arxiv.org/abs/1906.04161) as a baseline. The official [code](https://github.com/pathak22/exploration-by-disagreement) has been slightly [modified](https://drive.google.com/file/d/1T2oBme8YyKpmaPfLZB3W4h-daWt-JgE7/view?usp=sharing) to run on our robot arm.

### Baselines

- **ICM**: We use the official [code](https://github.com/openai/large-scale-curiosity) of "Curiosity-driven Exploration by Self-supervised Prediction, ICML 2017" and "Large-Scale Study of Curiosity-Driven Learning, ICLR 2019".   
- **RFM**: We use the official [code](https://github.com/openai/large-scale-curiosity) of "Large-Scale Study of Curiosity-Driven Learning, ICLR 2019".    
- **Disagreement**: We use the official [code](https://github.com/pathak22/exploration-by-disagreement) of "Self-Supervised Exploration via Disagreement, ICML 2019".    
