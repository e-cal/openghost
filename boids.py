from collections import defaultdict
from math import sqrt

from py5 import *
import py5

# width and height of the sketch
WIDTH = 720
HEIGHT = 720

# overall settings
MARGIN = 200
NUM_BOIDS = 100

# boid flocking behavior constants
CENTER_FACTOR = 0.01
AVOIDANCE_FACTOR = 0.2
VELOCITY_FACTOR = 0.2
VISUAL_RANGE = 50
VISUAL_RANGE_SQ = VISUAL_RANGE * VISUAL_RANGE
MIN_DISTANCE = 20
MIN_DISTANCE_SQ = MIN_DISTANCE * MIN_DISTANCE
BOUND_FACTOR = 1
SPEED_LIMIT = 10

CELL_SIZE = VISUAL_RANGE
NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),  (0, 0),  (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def build_grid(boids):
    """Return a mapping of grid cells to boid indices."""
    grid = defaultdict(list)
    for idx, boid in enumerate(boids):
        cell = int(boid.x // CELL_SIZE), int(boid.y // CELL_SIZE)
        grid[cell].append(idx)
    return grid


def iter_neighbor_indices(boid, grid):
    """Yield indices for boids near the given boid."""
    cx, cy = int(boid.x // CELL_SIZE), int(boid.y // CELL_SIZE)
    for offset_x, offset_y in NEIGHBOR_OFFSETS:
        yield from grid.get((cx + offset_x, cy + offset_y), ())

def build_color_lookup(size=256):
    """Pre-calculate colors indexed by normalized speed."""
    table = []
    for i in range(size):
        normalized_velocity = i / (size - 1)
        if normalized_velocity < 0.5:
            red = int(normalized_velocity * 2 * 255)
            green = 255
            blue = 0
        else:
            red = 255
            green = int((1 - normalized_velocity) * 2 * 255)
            blue = 0
        table.append((red, green, blue))
    return tuple(table)


COLOR_LOOKUP = build_color_lookup()


class Boid:
    __slots__ = ("x", "y", "dx", "dy")

    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
    
    def update(self):
        self.x += self.dx
        self.y += self.dy
    
    def distance_squared(self, other):
        """Fast distance calculation without square root"""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy
    
    def show(self):
        velocity = sqrt(self.dx ** 2 + self.dy ** 2)
        
        # Use lookup table for color calculation
        normalized_velocity = min(velocity / 10.0, 1.0)
        color_index = int(normalized_velocity * 255)
        red, green, blue = COLOR_LOOKUP[color_index]
        
        fill(red, green, blue)
        circle(self.x, self.y, 10)

    def keep_in_bounds(self):
        if self.x < MARGIN:
            self.dx += BOUND_FACTOR
        elif self.x > WIDTH - MARGIN:
            self.dx -= BOUND_FACTOR
        
        if self.y < MARGIN:
            self.dy += BOUND_FACTOR
        elif self.y > HEIGHT - MARGIN:
            self.dy -= BOUND_FACTOR

    def limit_speed(self):
        speed = sqrt(self.dx ** 2 + self.dy ** 2)
        if speed > SPEED_LIMIT:
            self.dx = (self.dx / speed) * SPEED_LIMIT
            self.dy = (self.dy / speed) * SPEED_LIMIT

    def apply_flocking_behaviors(self, boids, neighbor_indices):
        """Optimized single-pass flocking behavior calculation"""
        # Cohesion variables
        center_x = 0
        center_y = 0
        cohesion_neighbors = 0
        
        # Alignment variables
        avg_dx = 0
        avg_dy = 0
        alignment_neighbors = 0
        
        # Separation variables
        avoid_dx = 0
        avoid_dy = 0
        
        for idx in neighbor_indices:
            other_boid = boids[idx]
            if other_boid is self:
                continue

            dist_sq = self.distance_squared(other_boid)

            if dist_sq < MIN_DISTANCE_SQ:
                avoid_dx += (self.x - other_boid.x)
                avoid_dy += (self.y - other_boid.y)

            if dist_sq < VISUAL_RANGE_SQ:
                center_x += other_boid.x
                center_y += other_boid.y
                cohesion_neighbors += 1

                avg_dx += other_boid.dx
                avg_dy += other_boid.dy
                alignment_neighbors += 1

        
        self.dx += avoid_dx * AVOIDANCE_FACTOR
        self.dy += avoid_dy * AVOIDANCE_FACTOR

        if cohesion_neighbors > 0:
            center_x /= cohesion_neighbors
            center_y /= cohesion_neighbors
            self.dx += (center_x - self.x) * CENTER_FACTOR
            self.dy += (center_y - self.y) * CENTER_FACTOR

        if alignment_neighbors > 0:
            avg_dx /= alignment_neighbors
            avg_dy /= alignment_neighbors
            self.dx += (avg_dx - self.dx) * VELOCITY_FACTOR


def settings():
    size(WIDTH, HEIGHT, py5.P2D)

boids: list[Boid] = []

def setup():
    rect_mode(py5.CENTER)
    background(0)

    boids.clear()
    for _ in range(NUM_BOIDS):
        x = random(MARGIN / 2, WIDTH - MARGIN / 2)
        y = random(MARGIN / 2, HEIGHT - MARGIN / 2)
        dx = random(-3, 3)
        dy = random(-3, 3)
        boids.append(Boid(x, y, dx, dy))

def draw():
    no_cursor()
    background(0)

    grid = build_grid(boids)
    for boid in boids:
        neighbor_indices = iter_neighbor_indices(boid, grid)
        boid.apply_flocking_behaviors(boids, neighbor_indices)
        boid.limit_speed()
        boid.keep_in_bounds()
        boid.update()
        boid.show()


run_sketch()