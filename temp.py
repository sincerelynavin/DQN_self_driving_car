import pygame
import sys
import math

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60
BOUNDARY_COLOR = (255, 255, 255)  # White
MAX_SPEED = 5
ACCELERATION = 0.1
TURN_SPEED = 5

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [830, 920] # Starting Position
        self.angle = 0
        self.velocity = [0, 0]
        self.acceleration = 0
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
            # Check if any corner touches a white pixel (boundary)
            if game_map.get_at((int(point[0]), int(point[1]))) == BOUNDARY_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while game_map.get_at((x, y)) != BOUNDARY_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map, keys):
        if keys[pygame.K_w]:
            self.acceleration = ACCELERATION
        elif keys[pygame.K_s]:
            self.acceleration = -ACCELERATION
        else:
            self.acceleration = 0

        if keys[pygame.K_a]:
            self.angle += TURN_SPEED
        elif keys[pygame.K_d]:
            self.angle -= TURN_SPEED

        self.velocity[0] += self.acceleration * math.cos(math.radians(self.angle))
        self.velocity[1] += self.acceleration * math.sin(math.radians(self.angle))

        self.velocity[0] = max(-MAX_SPEED, min(self.velocity[0], MAX_SPEED))
        self.velocity[1] = max(-MAX_SPEED, min(self.velocity[1], MAX_SPEED))

        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        self.distance += math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

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

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
