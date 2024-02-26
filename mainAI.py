import pygame
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 40
CAR_SIZE_Y = 40
BORDER_COLOR = (255, 255, 255, 255)
GREEN_COLOR = (34, 177, 76)
RED_COLOR = (237, 28, 36)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(output_size)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class Car:
    def __init__(self, game_map):
        self.sprite = pygame.image.load('car6.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.angle = 0
        self.speed = 0  # Start with 0 speed
        self.speed_set = False
        self.alive = True
        self.distance = 0
        self.distance_forward = 0
        self.distance_backward = 0
        self.total_distance = 0
        self.time = 0
        self.radars = []  # Store radar information here

        # Find initial position within green spawn area
        self.position = self.find_spawn_position(game_map)
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.corners = []  # Store corner points of the car here

    def find_spawn_position(self, game_map):
        # Loop through the map to find a green area to spawn the car
        for y in range(game_map.get_height()):
            for x in range(game_map.get_width()):
                color = game_map.get_at((x, y))
                if color == GREEN_COLOR:
                    position = [x - CAR_SIZE_X / 2, y - CAR_SIZE_Y / 2]  # Adjust for car size
                    print("Found green at ({}, {})".format(x, y))  # Debug: Print detected green position
                    print("Setting position to:", position)  # Debug: Print position to be set
                    return position
        # If no green area is found, return a default position at the center of the map
        default_position = [WIDTH // 2 - CAR_SIZE_X / 2, HEIGHT // 2 - CAR_SIZE_Y / 2]
        print("No green found, setting default position:", default_position)  # Debug: Print default position
        return default_position

    def draw_radar(self, screen):
            for idx, radar_info in enumerate(self.radars):
                radar_pos, _ = radar_info
                
                # Find the position where radar line intersects the boundary
                intersection_point = self.find_intersection_with_boundary(radar_pos, screen)
                
                # Render radar distance near the intersection point
                font = pygame.font.Font(None, 24)
                text_color = (255, 127, 39)  # White color for text
                radar_distance_text = font.render("Distance: {}".format(radar_info[1]), True, text_color)
                screen.blit(radar_distance_text, (intersection_point[0] + 10, intersection_point[1] + 10))  # Adjust position as needed

                # Draw radar line from car center to intersection point
                pygame.draw.line(screen, RED_COLOR, self.center, intersection_point, 2)
                
                # Print the position and distance of the radar line along with its direction
                direction = self.get_radar_direction(idx)
                # print("Radar Line {} ({}) Position: {}".format(idx, direction, radar_pos))
                # print("Radar Line {} ({}) Intersection Point: {}".format(idx, direction, intersection_point))
                print("Radar Line {} ({}) Distance: {}".format(idx, direction, radar_info[1]))

    def get_radar_direction(self, idx):
            directions = {
                0: 'Front',
                1: 'Front-Left',
                2: 'Left',
                3: 'Back-Left',
                4: 'Back'
            }
            return directions.get(idx, 'Unknown')

    def find_intersection_with_boundary(self, radar_pos, screen):
        # Extract radar position coordinates
        radar_x, radar_y = radar_pos
        
        # Extract screen dimensions
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Car center coordinates
        car_center_x, car_center_y = self.center
        
        # Initialize intersection point to radar position
        intersection_point = radar_pos
        
        # Check for intersection with top boundary
        if radar_y < 0:
            intersection_point = (car_center_x + (0 - car_center_y) * (radar_x - car_center_x) / (radar_y - car_center_y), 0)
        # Check for intersection with bottom boundary
        elif radar_y > screen_height:
            intersection_point = (car_center_x + (screen_height - car_center_y) * (radar_x - car_center_x) / (radar_y - car_center_y), screen_height)
        # Check for intersection with left boundary
        elif radar_x < 0:
            intersection_point = (0, car_center_y + (0 - car_center_x) * (radar_y - car_center_y) / (radar_x - car_center_x))
        # Check for intersection with right boundary
        elif radar_x > screen_width:
            intersection_point = (screen_width, car_center_y + (screen_width - car_center_x) * (radar_y - car_center_y) / (radar_x - car_center_x))
        
        return intersection_point


    def draw(self, screen):
        rotated_center = (self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2)
        screen.blit(self.rotated_sprite, self.position)
        self.center = rotated_center  # Update center based on rotated position
        self.draw_radar(screen)
        
        # Render distance traveled on the screen
        font = pygame.font.Font(None, 24)
        text_color = (255, 127, 39)  # White color for text

        distance_text = font.render("Forward Distance: {:.2f}".format(self.distance_forward), True, text_color)
        screen.blit(distance_text, (10, 10))

        distance_text = font.render("Backward Distance: {:.2f}".format(self.distance_backward), True, text_color)
        screen.blit(distance_text, (10, 40))

        distance_text = font.render("Total Distance: {:.2f}".format(self.total_distance), True, text_color)
        screen.blit(distance_text, (10, 70))


    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR or \
                    game_map.get_at((int(point[0]), int(point[1]))) == (255, 255, 255):  # Modified condition to check for white pixel as well
                self.alive = False
                break

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

    def update(self, game_map, keys):
        if not self.speed_set:
            self.speed = 0  # Ensure speed starts at 0
            self.speed_set = True

        # Update position based on speed and angle
        if keys[pygame.K_w]:
            self.speed += 0.05
            self.distance_forward += abs(self.speed)  # Track distance traveled forward
        if keys[pygame.K_s]:
            self.speed -= 0.05
            self.distance_backward += abs(self.speed)  # Track distance traveled backward

        # Additional code to calculate total distance
        self.total_distance = self.distance_forward - self.distance_backward

        # Rotation based on key presses
        if keys[pygame.K_a]:
            self.angle += 1.5
        if keys[pygame.K_d]:
            self.angle -= 1.5

        # Apply friction
        if not keys[pygame.K_w] and not keys[pygame.K_s]:
            # If neither acceleration key is pressed, apply friction
            if self.speed > 0:
                self.speed -= 0.05  # Adjust the friction coefficient as needed
            elif self.speed < 0:
                self.speed += 0.05

        # Update position based on speed and angle
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += abs(self.speed)

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

        self.check_collision(game_map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        # Check distances to obstacles in specific directions
        self.check_radar_distances(game_map)

        # Check if any radar distance is less than 20
        for dist in self.radars:
            if dist[1] < 15:
                self.alive = False
                break
        
    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def check_radar_distances(self, game_map):
        radar_distances = {
            'front': None,
            'back': None,
            'left': None,
            'right': None,
            'center': None
        }

        for radar_angle in [-45, 0, 45, 90, 135]:
            length = 0
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + radar_angle))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + radar_angle))) * length)

            while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
                length += 1
                x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + radar_angle))) * length)
                y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + radar_angle))) * length)

            dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
            radar_distances[self.get_radar_name(radar_angle)] = dist

        # print("Radar Distances:", radar_distances)


    def get_radar_name(self, angle):
        if angle == -45 or angle == 45:
            return 'front'
        elif angle == 0:
            return 'center'
        elif angle == 90 or angle == 135:
            return 'left'
        elif angle == -90 or angle == -135:
            return 'right'
        elif angle == 180:
            return 'back'


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game_map = pygame.image.load('map5.png').convert()
    car = Car(game_map)

    # Initialize DQN agent
    input_size = 5  # Adjust according to state representation
    output_size = 3  # Adjust according to the number of actions
    agent = DQNAgent(input_size, output_size)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get current state
        state = get_state()  # Implement this function to return the current state

        # Select action
        action = agent.select_action(state)

        # Perform action
        perform_action(action)

        # Get next state and reward
        next_state = get_state()
        reward = calculate_reward()

        # Store transition in replay memory
        agent.memory.append((state, action, next_state, reward))

        # Optimize the model
        agent.optimize_model()

        # Update the screen
        screen.fill((0, 0, 0))
        screen.blit(game_map, (0, 0))
        car.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
