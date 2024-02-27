import pygame
import sys
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 40
CAR_SIZE_Y = 40
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit
GREEN_COLOR = (34, 177, 76)  # Color of the green spawn area
RED_COLOR = (255, 127, 39)  # Color for radar detection

# Define DQN constants
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1, 1)  # Adjusted max dimension
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

def preprocess_state(car):
    return torch.tensor([car.distance_forward, car.angle, car.speed], device=device, dtype=torch.float32)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

class Car:
    def __init__(self, game_map):
        self.sprite = pygame.image.load('car6.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.angle = 0
        self.speed = 0  # Start with 0 speed
        self.alive = True
        self.distance_forward = 0
        self.distance_backward = 0  # Add distance_backward attribute
        self.radars = []  # Store radar information here
        self.position = [WIDTH // 2 - CAR_SIZE_X / 2, HEIGHT // 2 - CAR_SIZE_Y / 2]
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

    def update(self, game_map):
        # Update car's state based on random actions for now
        state = preprocess_state(self)
        action = select_action(state)
        self.perform_action(action)

    def perform_action(self, action):
        if action == 0:  # Accelerate
            self.speed += 0.1
        elif action == 1:  # Decelerate
            self.speed -= 0.1
            # Track distance traveled backward only if the car is reversing
            if self.speed < 0:
                self.distance_backward += abs(self.speed)
        elif action == 2:  # Turn left
            self.angle += 5
        elif action == 3:  # Turn right
            self.angle -= 5


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Initialize DQN, replay memory, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = []

# Training loop
num_episodes = 1000  # Example number of episodes
steps_done = 0  # Initialize steps_done
for episode in range(num_episodes):
    # Initialize environment and car
    game_map = pygame.image.load('map5.png').convert()
    car = Car(game_map)

    # Reset episode-specific variables
    episode_reward = 0

    while car.alive:
        # Update the car based on random actions
        car.update(game_map)

        # Calculate reward based on distance traveled
        reward = car.distance_forward - car.distance_backward  # Example reward calculation

        # Optimize the DQN
        optimize_model()

        # Increment steps_done
        steps_done += 1

        # Accumulate episode reward
        episode_reward += reward

    # Update target network every few episodes
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Print training progress
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")

# Save the trained model
torch.save(policy_net.state_dict(), 'dqn_model.pth')


# Finalize Pygame
pygame.quit()
sys.exit()
