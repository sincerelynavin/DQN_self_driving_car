import pygame
import sys
import math
import time

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 40
CAR_SIZE_Y = 40
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit
GREEN_COLOR = (34, 177, 76)  # Color of the green spawn area
RED_COLOR = (255, 127, 39)  # Color for radar detection


class Car:
    def __init__(self, game_map, initial_position, automated=False):
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
        self.last_death_time = time.time()  # Initialize time of last death
        self.radars = []  # Store radar information here
        self.automated = automated

        # Find initial position within green spawn area
        self.position = initial_position
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.corners = []  # Store corner points of the car here

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

    def draw(self, screen, respawn_counter, speed, acceleration, turning_counter):
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
        distance_text = font.render("Forward Distance: {:.2f}".format(self.distance_forward), True, text_color)
        screen.blit(distance_text, (10, 940))  # Adjust position as needed

        # Render speed, acceleration, and turning angle on the screen
        speed_text = font.render("Speed: {:.2f}".format(speed), True, text_color)
        screen.blit(speed_text, (10, 970))

        acceleration_text = font.render("Acceleration: {:.2f}".format(acceleration), True, text_color)
        screen.blit(acceleration_text, (10, 1000))

        angle_text = font.render("Turning Counter: {}".format(turning_counter), True, text_color)
        screen.blit(angle_text, (10, 1030))

        # Calculate and render time alive with at least 3 decimal places
        time_alive = time.time() - self.last_death_time
        time_alive_text = font.render("Time Alive: {:.3f}s".format(time_alive), True, text_color)
        screen.blit(time_alive_text, (10, 1060))

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR or \
                    game_map.get_at((int(point[0]), int(point[1]))) == (55, 126, 71):  # Modified condition to check for white pixel as well
                self.alive = False
                self.last_death_time = time.time()  # Update time of last death
                break

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

    def update(self, game_map, keys=None):
        if not self.speed_set:
            self.speed = 0  # Ensure speed starts at 0
            self.speed_set = True

        # Update position based on speed and angle
        if not self.automated and keys:
            if keys[pygame.K_w]:
                self.speed += 0.05
                self.distance_forward += abs(self.speed)  # Track distance traveled forward
            elif keys[pygame.K_s]:
                self.speed -= 0.05
                # Track distance traveled backward only if the car is reversing
                if self.speed < 0:
                    self.distance_backward += abs(self.speed)

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

        else:  # Automated movement
            # For automated car, let's make it move forward continuously
            self.speed += 0.05
            self.distance_forward += abs(self.speed)  # Track distance traveled forward

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


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game_map = pygame.image.load('map5.png').convert()

    # Player controlled car
    player_car = Car(game_map, initial_position=[WIDTH // 2 - CAR_SIZE_X / 2, HEIGHT // 2 - CAR_SIZE_Y / 2])

    # Automated car
    automated_car = Car(game_map, initial_position=[100, 100], automated=True)

    running = True
    respawn_counter = 0  # Counter to keep track of respawn iterations
    prev_speed = 0
    turning_counter = 0  # Counter for turning

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a or event.key == pygame.K_d:
                    turning_counter += 1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a or event.key == pygame.K_d:
                    turning_counter = 0

        keys = pygame.key.get_pressed()
        player_car.update(game_map, keys)
        automated_car.update(game_map)

        speed = player_car.speed
        acceleration = speed - prev_speed
        turning_angle = player_car.angle

        screen.fill((0, 0, 0))
        screen.blit(game_map, (0, 0))
        player_car.draw(screen, respawn_counter, speed, acceleration, turning_counter)  # Pass turning counter to draw method
        automated_car.draw(screen, respawn_counter, automated_car.speed, 0, 0)  # No acceleration or turning for automated car

        pygame.display.flip()
        clock.tick(60)  # Limit frame rate

        prev_speed = speed

        if not player_car.alive:
            respawn_counter += 1
            print("Player car died. Respawned. Iteration:", respawn_counter)
            player_car = Car(game_map, initial_position=[WIDTH // 2 - CAR_SIZE_X / 2, HEIGHT // 2 - CAR_SIZE_Y / 2])  # Respawn the player car
            player_car.speed_set = True  # Ensure speed is reset
            player_car.distance = 0  # Reset distance
            player_car.distance_forward = 0  # Reset forward distance
            player_car.distance_backward = 0  # Reset backward distance

        if not automated_car.alive:
            print("Automated car died. Respawned.")
            automated_car = Car(game_map, initial_position=[100, 100], automated=True)  # Respawn the automated car

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
