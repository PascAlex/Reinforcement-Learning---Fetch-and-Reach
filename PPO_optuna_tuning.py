import gym
import optuna
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MultiInputPolicy

env = gym.make("FetchReach-v1")


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """

    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward


# objective functions used by optuna for evaluating the best parameters of PPO

# tuning of clip_range parameter

def objective(trial):
    x = trial.suggest_uniform('clip_range', 0.1, 0.4)
    model = PPO("MultiInputPolicy", env, verbose=0,
                clip_range=x)
    model.learn(total_timesteps=300000)
    return evaluate(model)


study = optuna.create_study(direction="maximize")
study.optimize(objective)

print(study.best_params)  # E.g. {'x': 2.002108042}


# tuning of gamma parameter

def objective(trial):
    x = trial.suggest_uniform('gamma', 0.8, 0.9997)
    model = PPO("MultiInputPolicy", env, verbose=0,
                gamma=x)
    model.learn(total_timesteps=300000)
    return evaluate(model)


study = optuna.create_study(direction="maximize")
study.optimize(objective)

print(study.best_params)  # E.g. {'x': 2.002108042}


# tuning of gae_lambda parameter

def objective(trial):
    x = trial.suggest_uniform('gae_lambda', 0.9, 1)
    model = PPO("MultiInputPolicy", env, verbose=0,
                gae_lambda=x)
    model.learn(total_timesteps=50000)
    return evaluate(model)


study = optuna.create_study(direction="maximize")
study.optimize(objective)

print(study.best_params)  # E.g. {'x': 2.002108042}


# tuning of learning_rate parameter


def objective(trial):
    x = trial.suggest_uniform('learning_rate', 0.0001, 0.0005)
    model = PPO("MultiInputPolicy", env, verbose=0,
                learning_rate=x)
    model.learn(total_timesteps=50000)
    return evaluate(model)


study = optuna.create_study(direction="maximize")
study.optimize(objective)

print(study.best_params)  # E.g. {'x': 2.002108042}