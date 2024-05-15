import time
import gym
import numpy as np 
import random
from agent_ddpg import DDPGAgent
import os
import torch
#搭建环境
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

agent = DDPGAgent(STATE_DIM,ACTION_DIM)

#超参数设置
NUM_EPISODE = 100
NUM_STEP = 200#不同环境步数不一样
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000#第10000步之后按0.02探索



REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    episode_reward = 0

    for step_i in range(NUM_STEP):
        epsilon = np.interp(x=episode_i*NUM_STEP+step_i, xp=[0, EPSILON_DECAY], fp=[EPSILON_START,EPSILON_END])
        random_sample = random.random()
        if random_sample <= epsilon:
            action = np.random.uniform(low=-2, high=2, size=ACTION_DIM)#连续动作 均匀分布(负号向左 正号向右)
        else:
            action = agent.get_action(state)
        

        next_state, reward, done, truncation, info = env.step(action)

        agent.replay_buffer.add_memo(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        agent.update()
        
        if done:
            break
    REWARD_BUFFER[episode_i] = episode_reward
    print(f"Episode: {episode_i+1}, Reward:{round(episode_reward,2)}")


current_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_path, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

timestamp = time.strftime("%Y%m%d%H%M%S")

#保存模型参数
torch.save(agent.actor.state_dict(), os.path.join(model_dir, f"ddpg_actor_{timestamp}.path"))
torch.save(agent.critic.state_dict(), os.path.join(model_dir, f"ddpg_critic_{timestamp}.path"))

env.close()