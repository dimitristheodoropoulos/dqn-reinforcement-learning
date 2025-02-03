import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Q-Network (Neural Network)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.001  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

        # Copy the weights to target network
        self.update_target_network()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))  # Exploration: Random action
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()  # Exploitation: Select best action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        return states, actions, rewards, next_states, dones

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Compute target Q-values using the target network
        next_q_values = self.target_network(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Get current Q-values from the Q network
        q_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Update the Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Main Training Loop
def train():
    env = gym.make('CartPole-v1')  # CartPole-v1 with discrete action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episodes = 100  # Reduced episodes for quicker testing
    total_reward = 0

    start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset()  # Reset the environment and get the initial state
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            episode_reward += reward

        total_reward += episode_reward
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {episode_reward:.2f}, Time: {time.time() - start_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    train()
