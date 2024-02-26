import pygame
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 40
CAR_SIZE_Y = 40
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit
GREEN_COLOR = (34, 177, 76)  # Color of the green spawn area
RED_COLOR = (237, 28, 36)  # Color for radar detection
INPUT_SIZE = 6  # Number of input features to the DQN agent
OUTPUT_SIZE = 4  # Number of possible actions (WASD) for the DQN agent
BATCH_SIZE = 32  # Batch size for training the DQN agent


# Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Transition:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

# DQN agent
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(input_size, output_size).to(self.device)
        self.target_model = DQN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(OUTPUT_SIZE))
        else:
            with torch.no_grad():
                return self.model(state).argmax().item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        non_none_next_states = [s for s in batch.next_state if s is not None]
        if not non_none_next_states:  # Check if there are any non-None values
            return
        
        non_final_next_states = torch.stack(non_none_next_states)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)

        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)

        # Check for collision with borders
        collision_mask = torch.tensor([1 if s else 0 for s in batch.next_state], dtype=torch.bool)
        
        # Calculate radar distances
        radar_distances = []
        for s in batch.state:
            if s is not None:
                radar_distances.append(self.calculate_radar_distances(s))
            else:
                radar_distances.append([0] * 5)  # Fill with zeros if state is None
        
        radar_distances = torch.tensor(radar_distances, dtype=torch.float32)
        
        # Track distance traveled
        distances = torch.tensor([s.distance_forward if s else 0 for s in batch.state], dtype=torch.float32)
        
        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Pygame car class
class Car:
    def __init__(self, game_map):
        self.sprite = pygame.image.load('car6.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.angle = 0
        self.speed = 0
        self.speed_set = False
        self.alive = True
        self.distance_forward = 0
        self.radars = []
        self.position = self.find_spawn_position(game_map)
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.corners = []

    def find_spawn_position(self, game_map):
        for y in range(game_map.get_height()):
            for x in range(game_map.get_width()):
                color = game_map.get_at((x, y))
                if color == GREEN_COLOR:
                    position = [x - CAR_SIZE_X / 2, y - CAR_SIZE_Y / 2]
                    return position
        default_position = [WIDTH // 2 - CAR_SIZE_X / 2, HEIGHT // 2 - CAR_SIZE_Y / 2]
        return default_position

    def draw(self, screen, respawn_counter):
        rotated_center = (self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2)
        screen.blit(self.rotated_sprite, self.position)
        self.center = rotated_center
        self.draw_radar(screen)
        font = pygame.font.Font(None, 24)
        text_color = (255, 127, 39)
        respawn_text = font.render("Respawn Counter: {}".format(respawn_counter), True, text_color)
        screen.blit(respawn_text, (10, 100))
        distance_text = font.render("Forward Distance: {:.2f}".format(self.distance_forward), True, text_color)
        screen.blit(distance_text, (10, 130))

    def draw_radar(self, screen):
        for idx, radar_info in enumerate(self.radars):
            radar_pos, _ = radar_info
            intersection_point = self.find_intersection_with_boundary(radar_pos, screen)
            pygame.draw.line(screen, RED_COLOR, self.center, intersection_point, 2)

    def find_intersection_with_boundary(self, radar_pos, screen):
        radar_x, radar_y = radar_pos
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        car_center_x, car_center_y = self.center
        intersection_point = radar_pos
        if radar_y < 0:
            intersection_point = (car_center_x + (0 - car_center_y) * (radar_x - car_center_x) / (radar_y - car_center_y), 0)
        elif radar_y > screen_height:
            intersection_point = (car_center_x + (screen_height - car_center_y) * (radar_x - car_center_x) / (radar_y - car_center_y), screen_height)
        elif radar_x < 0:
            intersection_point = (0, car_center_y + (0 - car_center_x) * (radar_y - car_center_y) / (radar_x - car_center_x))
        elif radar_x > screen_width:
            intersection_point = (screen_width, car_center_y + (screen_width - car_center_x) * (radar_y - car_center_y) / (radar_x - car_center_x))
        return intersection_point

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 0
            self.speed_set = True
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)
        self.speed = 1  # Example: always move forward
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1])]
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game_map = pygame.image.load('map5.png').convert()
    car = Car(game_map)
    agent = DQNAgent(INPUT_SIZE, OUTPUT_SIZE)

    running = True
    respawn_counter = 0

    previous_distance_forward = 0
    next_state = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        car.update(game_map)

        state = torch.tensor([car.distance_forward] + [radar[1] for radar in car.radars], dtype=torch.float32)
        action = agent.select_action(state.unsqueeze(0))
        if action == 0:  # Move forward
            # Update car's state (e.g., increase distance_forward)
            pass
        elif action == 1:  # Move backward
            # Update car's state (e.g., decrease distance_forward)
            pass
        elif action == 2:  # Turn left
            # Update car's state (e.g., increase angle)
            pass
        elif action == 3:  # Turn right
            # Update car's state (e.g., decrease angle)
            pass

        # Calculate reward based on distance traveled
        reward = car.distance_forward - previous_distance_forward
        previous_distance_forward = car.distance_forward

        # For now, set next_state to None
        next_state = None

        agent.memory.push((state, action, reward, next_state))
        agent.train(BATCH_SIZE)
        agent.update_epsilon()

        screen.fill((0, 0, 0))
        screen.blit(game_map, (0, 0))
        car.draw(screen, respawn_counter)

        pygame.display.flip()
        clock.tick(60)

        if not car.alive:
            respawn_counter += 1
            print("Car died. Respawned. Iteration:", respawn_counter)
            car = Car(game_map)
            car.speed_set = True
            car.distance_forward = 0

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()