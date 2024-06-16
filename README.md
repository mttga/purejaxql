### Exploring efficient ways to do $Q$-Learning in a pure-gpu setting

Follows cleanrl and purejaxrl philosophy of single-file scripts.

Build with:

```
docker build -t purejaxql .
```

Uses hydra for managing configuration:

```bash
python dqn.py +alg=dqn_minatar
```

Check ```purejaxql/config/config.yaml``` for default configuration and ```purejaxql/config/alg``` for alg-env specific configurations.

Most of scripts include a ```tune``` function to perform hyperparameter tuning. You'll need to set ```HYP_TUNE=True``` in the default config file to use it. 

Setup WANDB, seed and number of parallel seeds per experiment in the default config. 