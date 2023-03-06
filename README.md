# RL for airborne wind power generation

This repository contains the code used to control by means of model-free Reinforcement Learning an airborne wind energy system used for electric energy production. The physical model is based on the "yo-yo configuration" proposed by [Canale et al., 2009](https://ieeexplore.ieee.org/abstract/document/5152910).

First of all, the `C++` environment `libkite.so` has to be compiled, by executing the command:

```makefile
make x86 #to compile on standard x86 systems

make m1 #to compile on Apple silicon macs

make parallel #to enable parallel loading of turbulent flow data (on x86 only)
```

Constant and linear wind patterns are generated algorithmically on-the-go, while the turbulent flow data (~14GB) can be downloaded by typing the following command in the `env` folder:

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15u4cvvwuiFLsNw6VlbYSLADoMQReszOn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15u4cvvwuiFLsNw6VlbYSLADoMQReszOn" -O flow.tgz && rm -rf /tmp/cookies.txt
```

Learning can be performed by running the command:

```
python3 main.py
```

Execution specifications:

- `episodes`: number of training episodes;
- `wind_type`: kind of wind pattern to use, either `const` or `lin` or `turbo`;
- `step`: duration in time of each algorithmic iteration;
- `critic_lr`: learning rate of the critic neural network;
- `actor_lr`:  leatning rate of the actor neural network;
- `save_dir`: main folder where results and weight of trained actor and critic are saved; 
- `test_episodes`: number of test episodes;
- `range_actions`: actions variation in each step, for instance --range_actions=2,1 means that the attack angle can vary only of 2 in each step, and the bank angle can vary only of 1 

Suggested parameters for execution: 

Constant Wind:
```
python3 main.py --episodes=2000 --wind_type=const --step=0.1 --critic_lr=0.0001 --actor_lr=0.0001 
```
Linear Wind:
```
python3 main.py --episodes=2000 --wind_type=lin --step=0.1 --critic_lr=0.0007 --actor_lr=0.0007 
```
Turbulent Wind:
```
python3 main.py --episodes=2000 --wind_type=turbo --step=0.1 --critic_lr=0.0008 --actor_lr=0.0008 
```

After the learning preocess, a test is automatically performed; it can be perfored afeter using the execution specification: 

- `wind_type`: same used in the training;
- `step`: same value used in the training;
- `save_dir`: main folder where the "net" folder is saved; 
- `test_episodes`: number of test episodes;
- `range_actions`:same value used in the training;

To use the pretrained models: 

Linear Wind: 
```
python3 test.py --wind_type=lin --save_dir=trained_linear 

```

Turbulent Wind: 
```
python3 test.py --wind_type=turbo --save_dir=trained_turbo 

```









kite Library and Interface Authors:

- [Lorenzo Basile](https://github.com/lorenzobasile)
- [Claudio Leone](https://github.com/LionClaude)

TD3 Implementation:

- [Maria Grazia Berni](https://github.com/mariagraziaberni)

Project developed in the group of prof. Antonio Celani (QLS@ICTP, Trieste)
