# Importing Dependencies
import gym
import gym_anytrading
# Use v1 of tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt




#loading our dataset
df = pd.read_csv('/content/gmedata.csv')
#viewing first 5 columns
df.head()

#converting Date Column to DateTime Type
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes

#setting the column as index
df.set_index('Date', inplace=True)
df.head()

#passing the data and creating our environment
env = gym.make('stocks-v0', df=df, frame_bound=(5,100), window_size=5)




# running the test environment
state = env.reset()
while True:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    if done:
        print("info", info)
        break

plt.figure(figsize=(15, 6))
plt.cla()
env.render_all()
plt.show()




# setting up our environment for training
env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
env = DummyVecEnv([env_maker])

# Applying the Trading RL Algorithm
model = A2C('MlpLstmPolicy', env, verbose=1)

# setting the learning timesteps
model.learn(total_timesteps=1000)




#Setting up the Agent Environment
env = gym.make('stocks-v0', df=df, frame_bound=(90,110), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break


#Plotting our Model for Trained Trades
plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.show()