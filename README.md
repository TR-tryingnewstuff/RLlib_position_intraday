# RLlib_position_intraday
RLlib reinforcement learning for intraday stock trading 

This is a RLlib reinforcement learning algorithms designed for stock trading. 
The bot can choose between a discrete action space for entering {0 : sell, 1: do nothing, 2: buy} and a continuous action space for the size of the order between 0 and 1 (all available capital)

The bot takes as input a graph image representing the latest 60 15min candles plus some annotations to try to mimic what a trader would look at.
The input is then processed through a convolutional neural network (CNN) as they are particularly suited for image analysis, before outputing a trading decision which will then be processed to calculate the reward from the action. 
The reward is calculated from the profits made during the following 15min candle, and adjusted to downside risk and risk aversion.

The PPO agent was chosen due to its ability to handle both dicrete and continuous action spaces as well as showing more than decent performance for stock trading apllications due to its clipping parameter that allows for stable learning. 


