import os
from datetime import datetime
import pandas as pd

import numpy as np
from reward import reward_function

import gymnasium as gym
import gym_trading_env # shows not used but it's used in gymnasium. pip install gym-trading-env
import matplotlib.pyplot as plt
import pickle


from PPO import PPO


def get_save_path(dir_path="data/weights") -> str:
    # create a new directory if does not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    return "data/weights/PPO_{}.pth".format(num_files)


def get_dataset(split=0.99, sin=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sin:
        train_df = pd.read_pickle("dataset/out/simple_dataset.pkl")
        test_df = pd.read_pickle("dataset/out/simple_dataset_val.pkl")
    else:
        with open("dataset/out/EURUSD.pkl", "rb") as f:
            df = pickle.load(f)
        train_df = df.iloc[: int(len(df) * split)]
        test_df = df.iloc[int(len(df) * split) :]
    return train_df, test_df


class matplotlib_renderer:
    # class that starts a thread that continuously updates the window
    def __init__(self):
        self.data = []
        plt.ion()

    def render(self):
        if len(self.data) < 25:
            return
        plt.clf()
        smoothed_data = np.convolve(self.data, np.ones(25) / 25, mode="valid")
        plt.plot(smoothed_data)
        plt.show(block=False)
        plt.pause(0.1)

    def push(self, point):
        self.data.append(point)


def train():
    checkpoint_path = get_save_path()
    train_df, _ = get_dataset(sin=True)

    episode_length = 1000  # length of one episode

    update_timestep = 3  # update policy every n timesteps
    K_epochs = 1  # udate policy for K epochs in one PPO update

    eps_clip = 0.1  # clip parameter for PPO (the higher it is, the more aggressive is the policy update)
    gamma = 0.98  # discount factor
    entropy_coefficient = 0.01  # entropy term coefficient

    lr_actor = 0.00003       # learning rate for actor network
    lr_critic = 0.0001

    renderer = matplotlib_renderer()

    env = gym.make(
        "TradingEnv",
        df=train_df,
        positions=[-2, 2],
        reward_function=reward_function,
        # trading_fees=0.00025,
        portfolio_initial_value=100000,
        windows=45,
        max_episode_duration=episode_length,
    )

    state_dim = env.observation_space.shape[1]  # state space dimension
    action_dim = env.action_space.n  # action space dimension (positions the agent can take)

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        entropy_coefficient,
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    time_step = 0

    max_reward = 0

    # training loop
    state, info = env.reset()

    portfolios = []

    for i_episode in range(1, len(train_df) // episode_length):
        current_ep_reward = 0  # reward accumulated within one episode
        current_positions = np.zeros(action_dim)  # current positions of the agent
        print("\n\n\n============================================================================================")
        print(f"Episode : {i_episode}/{len(train_df) // episode_length + 1}")
        print(f"Start date: {info['date']}")
        print("Initial portfolio value : ", info["portfolio_valuation"])

        while True:
            # print(f"Time step: {time_step}")
            # select action with policy
            state = np.expand_dims(state, axis=1)
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)

            # update current positions at action index
            current_positions[action] += 1

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent after every update_timestep
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # break; if the episode is over
            if truncated or done:
                if max_reward < current_ep_reward:
                    env.unwrapped.save_for_render(dir="wip_render_logs")
                    max_reward = current_ep_reward
                portfolios.append(info["portfolio_valuation"])
                renderer.push(current_ep_reward)
                renderer.render()
                print("Final portfolio value : ", info["portfolio_valuation"])
                print("Positions: ", current_positions)
                print("End date: ", info["date"])
                print("Reward (cum): ", current_ep_reward)
                # print("Portfolio value moving average 25: ", np.mean(portfolios[-25:]))
                # print("Portfolio value moving average 50: ", np.mean(portfolios[-50:]))
                # print("Portfolio value moving average 100: ", np.mean(portfolios[-100:]))
                # print("Portfolio change (ma25-ma50): ", np.mean(portfolios[-25:]) - np.mean(portfolios[-50:]))
                print("============================================================================================\n\n\n")
                current_ep_reward = 0
                state, info = env.reset()
                break

    # save final model
    ppo_agent.save(checkpoint_path)

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    train()
