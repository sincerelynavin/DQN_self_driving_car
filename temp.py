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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

# Define the car class
class Car:
    def __init__(self, game_map):
        self.sprite = pygame.image.load('car6.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.angle = 0
        self.speed = 0  # Start with 0 speed
        self.alive = True
        self.distance_forward = 0
        self.radars = []  # Store radar information here
        self.position = [WIDTH // 2 - CAR_SIZE_X / 2, HEIGHT // 2 - CAR_SIZE_Y / 2]
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

    def update(self, game_map):
        # Update car's state based on DQN actions
        state = torch.tensor([self.distance_forward, self.angle, self.speed], device=device, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            action = select_action(state)
        self.perform_action(action)

    def perform_action(self, action):
        if action == 0:  # Accelerate
            self.speed += 0.1
        elif action == 1:  # Decelerate
            self.speed -= 0.1
        elif action == 2:  # Turn left
            self.angle += 5
        elif action == 3:  # Turn right
            self.angle -= 5

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Initialize DQN and load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load('dqn_model.pth'))
policy_net.eval()

# Main loop
game_map = pygame.image.load('map5.png').convert()
car = Car(game_map)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    car.update(game_map)

    # Render the scene
    screen.fill((0, 0, 0))
    screen.blit(game_map, (0, 0))
    # Render the car
    rotated_car = pygame.transform.rotate(car.sprite, car.angle)
    rotated_rect = rotated_car.get_rect(center=car.rotated_sprite.get_rect(topleft=(car.position[0] + CAR_SIZE_X / 2, car.position[1] + CAR_SIZE_Y / 2)).center)
    screen.blit(rotated_car, rotated_rect.topleft)
    pygame.display.flip()
    clock.tick(60)  # Adjust as needed

# Finalize Pygame
pygame.quit()
sys.exit()
