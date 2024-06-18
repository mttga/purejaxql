# Exploring Q-Learning in pure-GPU setting

The goal of this project is to provide very simple and light scripts for Q-Learning baselines in a number of single-agent and multi-agent settings. 

Follows cleanrl and purejaxrl philosophy of single-file scripts.

## Quick Stats

With a single NVIDIA A40 (similar performances in a RTX3090) you can:
- Train a CartPole agent in 15 seconds (10 seeds in 25 seconds)
- Train Minatar agent for 40M frames in 1 minute (10 seeds in 5 minutes)
- Train an Atari agent for 200M frames in one hour (in this case the environments are running in a single CPU, tested in AMD EPYC 7513 32-Core Processor)
- 

## 🚀 Usage (highly reccomended with Docker)

Steps:

1. Ensure you have Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) properly installed. 
2. (Optional) Set your WANDB key in the [Dockerfile](docker/Dockerfile).
3. Build with `bash docker/build.sh`
4. (Optional) build also the specific image for Atari (which uses different gym requirements): `bash docker/build_atari.sh`
5. Run a container: `bash docker/run.sh` (for Atari: `bash docker/run_atari.sh`)
6. Test a training script: `python purejaxql/pqn_minatar.py +alg=pqn_minatar`


#### Useful commands:

```bash
# run atari training with a specific game
python purejaxql/pqn_atari.py +alg=pqn_atari alg.ENV_NAME=NameThisGame-v5
```


## Experiment Configuration

Check [```purejaxql/config/config.yaml```](purejaxql/config/config.yaml) for default configuration. It allows to setup WANDB, seed and choose the number of parallel seeds per experiment.

The alg-env specific configuration files are in [```purejaxql/config/alg```]((purejaxql/config/alg)).

Most of scripts include a ```tune``` function to perform hyperparameter tuning. You'll need to set ```HYP_TUNE=True``` in the default config file to use it. 