import pygame
import random
import math

WIDTH, HEIGHT = 800, 600

NUM_BOIDS = 100
BOID_RADIUS = 5
MAX_SPEED = 2
MAX_FORCE = 2

NEIGHBOR_RADIUS = 50
AVOID_RADIUS = 20
FOV_ANGLE = 360

ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0
SEPARATION_WEIGHT = 1.5

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Flocking Simulation")
clock = pygame.time.Clock()

class Boid:
    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))

    def update(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohere(boids)
        separation = self.separate(boids)

        steering = (alignment * ALIGNMENT_WEIGHT +
                    cohesion * COHESION_WEIGHT +
                    separation * SEPARATION_WEIGHT)

        # Limit turn rate
        if steering.length() > MAX_FORCE:
            steering.scale_to_length(MAX_FORCE)

        self.velocity += steering

        # Limit speed
        if self.velocity.length() > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)

        # Update position
        self.position += self.velocity

        if self.position.x > WIDTH:
            self.position.x = 0
        if self.position.x < 0:
            self.position.x = WIDTH
        if self.position.y > HEIGHT:
            self.position.y = 0
        if self.position.y < 0:
            self.position.y = HEIGHT

    def in_field_of_view(self, other):
        direction_to_other = (other.position - self.position).normalize()
        forward_direction = self.velocity.normalize()
        angle = forward_direction.angle_to(direction_to_other)
        return abs(angle) < (FOV_ANGLE / 2)

    def align(self, boids):
        """Calculate alignment vector."""
        avg_velocity = pygame.Vector2(0, 0)
        count = 0
        for boid in boids:
            if boid != self:
                if self.in_field_of_view(boid):
                    distance = self.position.distance_to(boid.position)
                    if distance < NEIGHBOR_RADIUS:
                        avg_velocity += boid.velocity
                        count += 1
        if count > 0:
            avg_velocity /= count
            avg_velocity = avg_velocity - self.velocity
        return avg_velocity

    def cohere(self, boids):
        """Calculate cohesion vector."""
        center_of_mass = pygame.Vector2(0, 0)
        count = 0
        for boid in boids:
            if boid != self and self.in_field_of_view(boid):
                distance = self.position.distance_to(boid.position)
                if distance < NEIGHBOR_RADIUS:
                    center_of_mass += boid.position
                    count += 1
        if count > 0:
            center_of_mass /= count
            return (center_of_mass - self.position) * 0.01
        return pygame.Vector2(0, 0)

    def separate(self, boids):
        """Calculate separation vector."""
        avoid_vector = pygame.Vector2(0, 0)
        for boid in boids:
            if boid != self and self.in_field_of_view(boid):
                distance = self.position.distance_to(boid.position)
                if distance < AVOID_RADIUS:
                    avoid_vector += (self.position - boid.position) / distance
        return avoid_vector

boids = [Boid(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) for _ in range(NUM_BOIDS)]

def draw_arrow(screen, color, position, velocity, size=10):
    angle = math.atan2(-velocity.y, velocity.x)
    points = [
        (position.x + math.cos(angle) * size, position.y - math.sin(angle) * size),
        (position.x - math.cos(angle) * size * 0.5 + math.sin(angle) * size * 0.5,
         position.y + math.sin(angle) * size * 0.5 + math.cos(angle) * size * 0.5),
        (position.x - math.cos(angle) * size * 0.5 - math.sin(angle) * size * 0.5,
         position.y + math.sin(angle) * size * 0.5 - math.cos(angle) * size * 0.5)
    ]
    pygame.draw.polygon(screen, color, points)

running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for boid in boids:
        boid.update(boids)
        draw_arrow(screen, (255, 255, 255), boid.position, boid.velocity, BOID_RADIUS)

    pygame.display.flip() 
    clock.tick(60)  

pygame.quit()
