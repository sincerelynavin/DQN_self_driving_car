import pygame
import sys
import math

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255) # Color To Crash on Hit
class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.velocity = [0, 0]  # Velocity [x, y]
        self.acceleration = 0.2  # Acceleration rate
        self.max_speed = 5  # Maximum speed
        self.friction = 0.1  # Friction coefficient
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = []
        self.drawing_radars = []
        self.alive = True
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR or game_map.get_at((int(point[0]), int(point[1]))) == (255, 255, 255):  # Modified condition to check for white pixel as well
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
        # Handle acceleration and deceleration
        if keys[pygame.K_w]:
            self.velocity[0] += math.cos(math.radians(360 - self.angle)) * self.acceleration
            self.velocity[1] += math.sin(math.radians(360 - self.angle)) * self.acceleration
        if keys[pygame.K_s]:
            self.velocity[0] -= math.cos(math.radians(360 - self.angle)) * self.acceleration
            self.velocity[1] -= math.sin(math.radians(360 - self.angle)) * self.acceleration

        # Apply friction
        self.velocity[0] *= (1 - self.friction)
        self.velocity[1] *= (1 - self.friction)

        # Limit speed
        speed = math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity[0] *= scale
            self.velocity[1] *= scale

        # Update position
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        # Ensure the car stays within the boundaries
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)

        # Update angle based on user input
        if keys[pygame.K_a]:
            self.angle += 5
        if keys[pygame.K_d]:
            self.angle -= 5

        # Keep the angle within the range of 0 to 360 degrees
        self.angle %= 360

        # Update center and corners
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check collision and radar
        self.check_collision(game_map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    car = Car()
    game_map = pygame.image.load('map.png').convert()

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
