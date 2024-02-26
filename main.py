import pygame
import sys
import math

# Constants
WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60
BOUNDARY_COLOR = (255, 255, 255)  # White

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite 
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.alive = True
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)

    def check_collision(self, game_map):
        # Calculate corners dynamically based on position and angle
        length = 0.5 * CAR_SIZE_X
        corner_angles = [self.angle + offset for offset in (30, 150, 210, 330)]
        corners = [(self.center[0] + math.cos(math.radians(360 - angle)) * length,
                    self.center[1] + math.sin(math.radians(360 - angle)) * length)
                   for angle in corner_angles]

        # Check if any corner touches a white pixel (boundary)
        self.alive = not any(game_map.get_at((int(point[0]), int(point[1]))) == BOUNDARY_COLOR
                             for point in corners)

    def update(self, game_map, keys):
        if keys[pygame.K_w]:
            self.speed += 1
        if keys[pygame.K_s]:
            self.speed -= 1
        if keys[pygame.K_a]:
            self.angle += 5
        if keys[pygame.K_d]:
            self.angle -= 5

        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(20, min(self.position[0], WIDTH - 120))

        self.distance += self.speed

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(20, min(self.position[1], HEIGHT - 120))

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        self.check_collision(game_map)

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
