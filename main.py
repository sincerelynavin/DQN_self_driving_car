import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
FPS = 60
BG_COLOR = (255, 255, 255)  # White

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Racing Game")

clock = pygame.time.Clock()

# Load images
car_image = pygame.image.load('car.png')
car_rect = car_image.get_rect()
map_image = pygame.image.load('map.png')

# Initial position of the car
car_rect.center = (WIDTH // 2, HEIGHT // 2)

# Movement speed of the car
speed = 5

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        car_rect.y -= speed
    if keys[pygame.K_s]:
        car_rect.y += speed
    if keys[pygame.K_a]:
        car_rect.x -= speed
    if keys[pygame.K_d]:
        car_rect.x += speed

    # Draw everything
    screen.fill(BG_COLOR)
    screen.blit(map_image, (0, 0))
    screen.blit(car_image, car_rect)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(FPS)

pygame.quit()
sys.exit()
