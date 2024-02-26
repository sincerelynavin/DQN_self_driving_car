import pygame
import sys
import math

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 40
CAR_SIZE_Y = 40
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit
GREEN_COLOR = (34, 177, 76)  # Color of the green spawn area
RED_COLOR = (237, 28, 36)  # Color for radar detection

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
        for radar_info in self.radars:
            radar_pos, _ = radar_info
            pygame.draw.line(screen, RED_COLOR, self.center, radar_pos, 2)

    def draw(self, screen):
        rotated_center = (self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2)
        screen.blit(self.rotated_sprite, self.position)
        self.center = rotated_center  # Update center based on rotated position
        self.draw_radar(screen)

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

        # Acceleration based on key presses
        if keys[pygame.K_w]:
            self.speed += 0.05
        if keys[pygame.K_s]:
            self.speed -= 0.05

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
        
    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def check_radar_distances(self, game_map):
        radar_distances = {
            'front-left': None,
            'front-center': None,
            'front-right': None,
            'back-left': None,
            'back-center': None,
            'back-right': None
        }

        for radar_angle in [-45, 0, 45, 135, 180, 225]:
            length = 0
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + radar_angle))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + radar_angle))) * length)

            while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
                length += 1
                x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + radar_angle))) * length)
                y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + radar_angle))) * length)

            dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
            radar_distances[self.get_radar_name(radar_angle)] = dist

        print("Radar Distances:", radar_distances)

    def get_radar_name(self, angle):
        if angle == -45:
            return 'front-left'
        elif angle == 0:
            return 'front-center'
        elif angle == 45:
            return 'front-right'
        elif angle == 135:
            return 'back-left'
        elif angle == 180:
            return 'back-center'
        elif angle == 225:
            return 'back-right'


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    game_map = pygame.image.load('map5.png').convert()
    car = Car(game_map)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        car.update(game_map, keys)

        screen.fill((0, 0, 0))
        screen.blit(game_map, (0, 0))
        car.draw(screen)

        pygame.display.flip()
        clock.tick(60)

        if not car.alive:
            running = False

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
