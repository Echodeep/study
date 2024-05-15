import torch
import gym
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device type: ", device)

#初始化环境
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

#加载参数
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
actor_path = model + "***********.pth"


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2
        return x


actor = Actor(STATE_DIM, ACTION_DIM).to(device)
actor.load_state_dict(torch.load(actor_path))

NUM_EPISODE = 30
NUM_STEP = 200
for episode_i in range(NUM_EPISODE):
    state, others = env.reset()
    #测试指标
    episode_reward = 0

    for step_i in range(NUM_STEP):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state, reward, done, truncation, info = env.step(action)
        state = next_state


    print(f"Episode {episode_i}.Reward: {episode_reward}")

env.close()
