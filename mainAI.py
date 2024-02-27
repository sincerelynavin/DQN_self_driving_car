import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque



# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define your neural network architecture
        # Example:
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Define training algorithm
def train_dqn(env, model, target_model, optimizer, replay_buffer, gamma, batch_size):
    # Sample minibatch from replay buffer
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    # Compute Q-values
    Q = model(state).gather(1, action.unsqueeze(1)).squeeze(1)
    Q_next = target_model(next_state).max(1)[0]
    target = reward + gamma * Q_next * (1 - done)

    # Compute loss
    loss = nn.MSELoss()(Q, target.detach())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize environment, DQN model, target model, optimizer, etc.
env = ... # Initialize your environment
input_size = 10  # Example: If your input state space has 10 dimensions
output_size = 4   # Example: If your output action space has 4 possible actions
model = DQN(input_size, output_size)
target_model = DQN(input_size, output_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer(capacity=10000)
gamma = 0.99
batch_size = 32
epsilon = 0.1
num_episodes = 1000  # Number of episodes for training

# Main training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                Q_values = model(torch.tensor(state, dtype=torch.float32))
                action = Q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Train the model
        if len(replay_buffer) > batch_size:
            train_dqn(env, model, target_model, optimizer, replay_buffer, gamma, batch_size)

    # Update target network every few episodes
    if episode % target_update_frequency == 0:
        target_model.load_state_dict(model.state_dict())

# Once trained, deploy the model to drive the car autonomously
def test_dqn(env, model):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            Q_values = model(torch.tensor(state, dtype=torch.float32))
            action = Q_values.argmax().item()

        state, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward

# Test the trained model
test_reward = test_dqn(env, model)
print("Test Reward:", test_reward)
