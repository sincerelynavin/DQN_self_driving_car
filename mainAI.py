import pygame
import sys
import math
import time
import numpy as np
from paramaters import *



class CarAI:
    def __init__(self, game_map):
        self.sprite = pygame.image.load('car6.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.angle = 0
        self.speed = 0  # Start with 0 speed
        self.speed_set = False
        self.alive = True
        self.distance = 0
        self.reward = 0
        self.time = 0
        self.last_death_time = time.time()  # Initialize time of last death
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

    def draw(self, screen, respawn_counter, speed, acceleration, turning):
        rotated_center = (self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2)
        screen.blit(self.rotated_sprite, self.position)
        self.center = rotated_center  # Update center based on rotated position
        self.draw_radar(screen)
        # Render respawn counter on the screen
        font = pygame.font.Font(None, 24)
        text_color = (255, 127, 39)  # White color for text
        # Render respawn counter
        respawn_text = font.render("Respawn Counter: {}".format(respawn_counter), True, text_color)
        screen.blit(respawn_text, (10, 910))
        # Render forward distance on the screen
        distance_text = font.render("Distance Travelled: {:.2f}".format(self.distance), True, text_color)
        screen.blit(distance_text, (10, 940))  # Adjust position as needed
        # Render speed, acceleration, and turning angle on the screen
        speed_text = font.render("Speed: {:.2f}".format(speed), True, text_color)
        screen.blit(speed_text, (10, 970))
        acceleration_text = font.render("Acceleration: {:.2f}".format(acceleration), True, text_color)
        screen.blit(acceleration_text, (10, 1000))
        angle_text = font.render("Turning: {}".format(turning), True, text_color)
        screen.blit(angle_text, (10, 1030))
        # Calculate and render time alive with at least 3 decimal places
        time_alive = time.time() - self.last_death_time
        time_alive_text = font.render("Time Alive: {:.3f}s".format(time_alive), True, text_color)
        screen.blit(time_alive_text, (10, 1060))


    def check_collision(self, game_map, distance, reward):
        self.alive = True
        
        # Calculate distance traveled since last check
        prev_distance = self.distance_traveled
        self.distance_traveled = distance
        
        # Check if the distance hasn't increased and the game has been running for more than 30 seconds
        if (self.distance_traveled - prev_distance) <= 0 and time.time() - self.start_time > 30:
            self.alive = False
            reward -= 10  # Decrease reward by 10
            self.last_death_time = time.time()  # Update time of last death
            return reward
        
        # Check for collision with borders or specific color
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR or \
            game_map.get_at((int(point[0]), int(point[1]))) == (55, 126, 71):  # Modified condition to check for white pixel as well
                self.alive = False
                reward -= 10  # Decrease reward by 10
                self.last_death_time = time.time()  # Update time of last death
                return reward
        
        return reward


    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        while 0 <= x < game_map.get_width() and 0 <= y < game_map.get_height() \
                and not game_map.get_at((x, y)) == BORDER_COLOR and not game_map.get_at((x, y)) == (55, 126, 71):
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update_reward(self, reward):
        # Increase reward based on distance traveled
        reward += self.distance_traveled * 0.1  # Assuming 0.1 reward per unit distance traveled

        # Check if any distance in radar falls below 20
        for dist in self.radar:
            if dist[1] < 20:
                reward -= 5  # Decrease reward by 5 if distance falls below 20

        return reward

    def update(self, game_map, keys, action):
        if not self.speed_set:
            self.speed = 0  # Ensure speed starts at 0
            self.speed_set = True

        if np.array_equal(action, [1, 0, 0, 0]):
            self.speed += 0.08
            self.distance += abs(self.speed)
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.speed -= 0.1
            
        if np.array_equal(action, [0, 0, 1, 0]):
            self.angle += 1.5
        elif np.array_equal(action, [0, 0, 0, 1]):
            self.angle -= 1.5
            
        if not np.array_equal(action, [1, 0, 0, 0]) and not np.array_equal(action, [0, 1, 0, 0]):
            if self.speed > 0:
                self.speed -= 0.05
            elif self.speed < 0:
                self.speed += 0.05
        
        
        # [acceleration, decceleration, left turn, right turn]
        # # Update position based on speed and angle
        # if keys[pygame.K_w]:
        #     self.speed += 0.08
        #     self.distance += abs(self.speed)  # Track distance traveled forward
        # elif keys[pygame.K_s]:
        #     self.speed -= 0.1

        # # Rotation based on key presses
        # if keys[pygame.K_a]:
        #     self.angle += 1.5
        # if keys[pygame.K_d]:
        #     self.angle -= 1.5

        # # Apply friction
        # if not keys[pygame.K_w] and not keys[pygame.K_s]:
        #     # If neither acceleration key is pressed, apply friction
        #     if self.speed > 0:
        #         self.speed -= 0.05  # Adjust the friction coefficient as needed
        #     elif self.speed < 0:
        #         self.speed += 0.05

        # Update position based on speed and angle
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
        
        self.check_collision(game_map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)
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


def main(action):  # Pass action as a parameter to the main function
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    game_map = pygame.image.load('map5.png').convert()
    car = CarAI(game_map)

    running = True
    respawn_counter = 0  # Counter to keep track of respawn iterations
    prev_speed = 0
    turning = 0  # Counter for turning

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if action[2] == 1:  # Turn left
                turning += 1
            elif action[3] == 1:  # Turn right
                turning -= 1
            else:
                turning = 0


        keys = pygame.key.get_pressed()
        car.update(game_map, keys, action)  # Pass action to the update method

        speed = car.speed
        acceleration = speed - prev_speed

        screen.fill((0, 0, 0))
        screen.blit(game_map, (0, 0))
        car.draw(screen, respawn_counter, speed, acceleration, turning)  # Pass turning counter to draw method
        pygame.display.flip()
        clock.tick(165)

        prev_speed = speed

        if not car.alive:
            respawn_counter += 1
            print("Car died. Respawned. Iteration:", respawn_counter)
            car = CarAI(game_map)  # Respawn the car
            car.speed_set = True  # Ensure speed is reset
            car.distance = 0  # Reset forward distance

    pygame.quit()
    sys.exit()


