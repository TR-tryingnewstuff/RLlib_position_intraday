import numpy as np
import pandas as pd
from get_data_stock_trading_mbfa import *
from helpers_mbfa import *

import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict

from ray.rllib.algorithms import ppo


from custom_model import KerasModel
from ray.rllib.models.catalog import ModelCatalog

import matplotlib.pyplot as plt
import PIL


SEE_PROGRESS = True
VISUALIZE = True

toml = get_toml_data('config.toml')
image_path = toml['file']['image']

start = 30000
TRADING_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def backtest_agent(checkpoint_path):
    df = data_main(0, 50000)
    
    class Market(gym.Env):
        """A class to represent a trading environment"""
        def __init__(self, env_config):
            
            # Initializing the attributes
            self.list_capital = []
            self.position = np.array([0.0])
            self.commission = 0.75 # per contract 
            self.capital = 1000
        
            self.done = False 
            self.n_step = start
            self.episode = 0
            
            self.df = df.iloc[self.n_step:self.n_step+WINDOW]
            
            
            image = PIL.Image.open(f'{image_path}graph_image_{self.df.index.values[1]}.png')
            image = np.asarray(image)
            self.observation = {'image': image, 'position': self.position}
            
            print(self.position, self.position.shape, self.position.dtype)
        
            self.observation_space = Dict({
                'image': Box(-np.inf, np.inf, shape=image.shape), 
                'position': Box(-1, 1)     })
            
            self.action_space = Dict({'enter': Discrete(3), 'size': Box(low=0, high=1)})
            
        
        def reset(self):
            """Resets the environment"""
            
            self.list_capital = []
            self.position = np.array([0.0])
            self.capital = 1000
            self.done = False
            self.n_step = start
            self.episode +=1
            
            self.df = df.iloc[self.n_step:self.n_step+WINDOW]
            
            image = PIL.Image.open(f'{image_path}graph_image_{self.df.index.values[1]}.png')
            image = np.asarray(image)
            self.observation = {'image': image, 'position': self.position}
            
            return self.observation
            
            
        def step(self, action):
            """Performs one iteration given the action choosen by the agent \\
            returns the following observation, the reward, whether the episode is done and additional info"""
            
            self.n_step += 1  
            self.df = df.iloc[self.n_step:self.n_step+WINDOW]
            
            if action['enter'] != 1:
                reward = self.get_reward(action)
                
            else:
                reward = (self.df['close'].values[-1] - self.df['close'].values[-2]) * self.position
            
            self.capital = self.capital + reward
            
            reward = self.adjust_reward(reward)
            

            if self.df['hour'].values[-1] not in TRADING_HOURS:        # Filter for taking trades only during certain hours
                reward -= self.commission * action['size']                   # close position at the end of the day
                self.position = np.array([0.0])
                
                while not self.df['hour'].values[-1] in TRADING_HOURS:
                    self.n_step += 1
                    self.df = df.iloc[self.n_step:self.n_step+WINDOW]       
                    
                
            info = {'capital': self.capital}     
            
            if VISUALIZE:
                self.view(action)
                
            if SEE_PROGRESS and (self.n_step % 10 == 0):
                self.list_capital.append(round(float(self.capital)))
                
                
            if (self.n_step > len(df) - 300) or (self.capital < 500): 
                self.done = True
                print(self.list_capital)
                
            image = PIL.Image.open(f'{image_path}graph_image_{self.df.index.values[1]}.png')
            image = np.asarray(image)
            self.observation = {'image': image, 'position': self.position}
                
            return self.observation, float(reward), self.done, info
        
        
        def get_reward(self, action):
            """Compute reward from the chosen action, it returns the price change multiplied by the size of the position"""
            reward = 0
            
            # you can disable the following line if you want 
            #action = self.rule_of_thumbs_check(action)
            
            if action['enter'] == 2:
                self.position =  min(action['size'] + self.position, np.asarray([1.0]))

                
            elif action['enter'] == 0:
                self.position =  max([-action['size'] + self.position, np.asarray([-1.0])])             
                    
            num_of_contracts = abs(self.position * (self.capital / self.df['close'].values[-1]))
            
            reward -= self.commission * num_of_contracts
            reward += (self.df['close'].values[-1] - self.df['close'].values[-2]) * num_of_contracts
                
            return reward
        
        def rule_of_thumbs_check(self, action):
            """Basic rule of thumbs to prevent the bot from taking low probability actions \\
                Returns the action with 'enter' set to 1 if a rule was broken"""
                
            if action['enter'] == 2:
                
                if (self.df['close'].values[-1]  > self.df['fib_0.75'].values[-1]) and (self.position > 0) : # Check no buy above the fib 0.75
                    action['enter'] = 1

            elif action['enter'] == 0:
                
                if (self.df['close'].values[-1]  < self.df['fib_0.25'].values[-1]) and (self.position < 0) : # Check no short below the fib 0.25
                    action['enter'] = 1
                    
            return action

        def adjust_reward(self, reward):
            """reward is adjusted to risk based on personal preferences \n
            In our case, we divide the reward by a downside risk \\ 
            measure that is adjusted to risk-aversion preferences
            """
            
            if self.position > 0: 
                index = (self.df['close'].values - self.df['open'].values) < 0
            
            else :
                index = (self.df['close'].values - self.df['open'].values) > 0
            
            downside_risk = self.df['close'].values[index].std()
            adjusted_dev = downside_risk * np.exp(-0.2)

                
            adjusted_reward = reward / adjusted_dev
            
            return adjusted_reward

        def view(self, action):
            """ Information printing and plotting to visualize the process. \n
                significantly slows down training so it is not meant for extensive training. \\
                Set VISUALIZE=False to disable it"""
            enter = action['enter']
            size = action['size']

            image = PIL.Image.open(f'{image_path}graph_image_{self.df.index.values[1]}.png')
            
            plt.title(f' capital : {round(float(self.capital))}  position : {round(float(self.position), 2)}  action : {enter} size : {round(float(size), 2)} ')
            plt.imshow(image)
            plt.axis('off')
            plt.pause(0.1)
            plt.clf()

    

    ModelCatalog.register_custom_model("keras_model", KerasModel)

    config = ppo.PPOConfig().environment(Market).rollouts(num_rollout_workers=1).resources(num_cpus_for_local_worker=2)
    config = config.training(lr=0.00001, clip_param=0.25, gamma=0.95, use_critic=True, use_gae=True, model={'custom_model': 'keras_model'}, train_batch_size=200)

    #checkpoint_path = pyperclip.paste()
    algo = config.build()
    algo.restore(checkpoint_path)


    episodes_len = []
    for i in range(50):
        train_res = algo.train()
        print(train_res['episode_len_mean'])

        episodes_len.append(train_res['episode_len_mean'])

    print(train_res, '\n'*3,episodes_len)
    
backtest_agent("/Users/thomasrigou/ray_results/PPO/PPO_Market_34dbc_00000_0_2023-05-19_15-58-18/checkpoint_000150")
