from typing import Dict
import numpy as np
import pandas as pd
from get_data_stock_trading_mbfa import *
from helpers_mbfa import *

import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict

from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune
from ray import air
import PIL

import matplotlib.pyplot as plt

VISUALIZE = False
SEE_PROGRESS = True

TRADING_HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
DF_SPLIT = 10000

toml = get_toml_data('config.toml')

DF_SIZE = toml['config_data']['size'] 
WINDOW = toml['config_data']['window']

image_path = toml['file']['image']

# --------------------------------------------

df_base = data_main(DF_SIZE)
df = df_base.iloc[0:-DF_SPLIT]
    
print(df.columns, '\n'*2, df.info(), '\n'*3, df.head(), df.tail())

class Market(gym.Env):
    """A class to represent financial Markets"""
    def __init__(self, env_config):
        
        # Initializing the attributes
        self.list_capital = []
        self.position = 0
        self.commission = 0.75 # per contract 
        self.capital = 1000
       
        self.done = False 
        self.n_step = 0
        self.episode = 0
        
        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        
        
        image = PIL.Image.open(f'{image_path}graph_image_{self.df.index.values[1]}.png')
        image = np.asarray(image)
        self.observation = image
       
        self.observation_space = Box(-np.inf, np.inf, shape=self.observation.shape)
        self.action_space = Dict({'enter': Discrete(3), 'size': Box(low=0, high=1)})
        
    
    def reset(self):
        """Resets the environment"""
        
        self.list_capital = []
        self.position = 0
        self.capital = 1000
        self.done = False
        self.n_step = 0
        self.episode +=1
        
        self.df = df.iloc[self.n_step:self.n_step+WINDOW]
        
        image = PIL.Image.open(f'{image_path}graph_image_{self.df.index.values[1]}.png')
        image = np.asarray(image)
        self.observation = image
        
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
            self.position = 0
            
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
        self.observation = image
              
        return self.observation, float(reward), self.done, info
       
    def get_reward(self, action):
        """Compute reward from the chosen action, it returns the price change multiplied by the size of the position"""
        reward = 0
        
        action = self.rule_of_thumbs_check(action)
        
        if action['enter'] == 2:
            self.position =  min(action['size'] + self.position, 1)

            
        elif action['enter'] == 0:
            self.position =  max([-action['size'] + self.position, -1])             
                
        num_of_contracts = abs(self.position * (self.capital / self.df['close'].values[-1]))
        
        reward -= self.commission * num_of_contracts
        reward += (self.df['close'].values[-1] - self.df['close'].values[-2]) * self.position
            
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

        i = self.n_step
        color_index = np.where(df.open[i:i+WINDOW] < df.close[i:i+WINDOW], 'gray', 'black')
        date_index = np.array(df[i:i+WINDOW].index)
        
        bars = np.array(df.close[i:i+WINDOW])-np.array(df.open[i:i+WINDOW])
        wicks = np.array(df.high[i:i+WINDOW])-np.array(df.low[i:i+WINDOW])
        plt.bar(date_index, bars, width=0.7, bottom=df.open[i:i+WINDOW], color=color_index)
        plt.bar(date_index, wicks, width=0.1, bottom=df.low[i:i+WINDOW], color=color_index)
        
        plt.title(f' capital : {round(float(self.capital))}  position : {round(float(self.position), 2)}  action : {enter} size : {round(float(size), 2)} ')
            
        plt.pause(0.1)
        plt.clf()



neural_network =  {
                  'conv_filters': tune.grid_search([  
                                                      [ [8, [2, 2], 3], 
                                                        [8, [2, 2], 3], 
                                                        [8, [2, 2], 3],
                                                        [8, [2, 2], 2]], 
                                                                                                                                                                                                      
                                                      [ [10, [2, 2], 5], 
                                                        [10, [2, 2], 4], 
                                                        [10, [2, 2], 3] ], 
                                                    
                                                      [ [16, [2, 2], 8], 
                                                        [10, [2, 2], 8] ] ]),
                  'conv_activation': 'tanh',
                  'post_fcnet_hiddens': [5],
                  'post_fcnet_activation': 'tanh',
                  'use_lstm': True,
                  'lstm_cell_size': 84}

neural_network =  {'conv_filters': 
                                                      [ [4, [3, 3], 5],  
                                                        [8, [2, 2], 4], 
                                                        [16, [2, 2], 3],
                                                        [16, [2, 2], 2],
                                                        [32, [2, 2], 2],                                                       
                                                        [64, [2, 2], 2]], 
                                                                                                                                                                                                      
                  'conv_activation': 'tanh',
                  }



config = ppo.PPOConfig().environment(Market, render_env=True).rollouts(num_rollout_workers=1).resources(num_cpus_for_local_worker=2)
config = config.training(lr_schedule=toml['model']['lr_schedule'], clip_param=0.25, gamma=0.95, use_critic=True, use_gae=True, model=neural_network, train_batch_size=200)
algo = config.build()

print(algo.get_policy().model.base_model.summary())
tuner = tune.Tuner(  
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
                    stop={"training_iteration": toml['model']['training_iteration']},
                    checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=100)

        )
)

results = tuner.fit()

# ---------------------- backtesting best parameters on unseen data ---------------------------

best_result = results.get_best_result(metric="episode_reward_max", mode="max")
# Get the best checkpoint corresponding to the best result.
checkpoint_path = best_result.checkpoint
print('\n'*5, checkpoint_path, '\n'*5)

algo = Algorithm.from_checkpoint(checkpoint_path)

df = df_base.iloc[-DF_SPLIT:]

episodes_len = []
for i in range(20):
    train_res = algo.train()
    print(train_res['episode_len_mean'])

    episodes_len.append(train_res['episode_len_mean'])

print(train_res, '\n'*3,episodes_len)

