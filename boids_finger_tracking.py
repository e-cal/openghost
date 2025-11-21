import threading
import time
from collections import defaultdict
from math import sqrt

import cv2
import mediapipe as mp
from picamera2 import Picamera2
from py5 import *
import py5

# width and height of the sketch
WIDTH = 720
HEIGHT = 720

# hand tracking sensitivity and smoothing
SENSITIVITY = 1.8   # equivalent to mouse sensitivity
SMOOTHING = 0.5     # movement smoothing. higher values produce more jittery movement

# overall settings
NUM_BOIDS = 100
MARGIN = 200
CURSOR_MARGIN = 50

# boid flocking behavior constants
CENTER_FACTOR = 0.01
AVOIDANCE_FACTOR = 0.3
VELOCITY_FACTOR = 0.2
VISUAL_RANGE = 50
VISUAL_RANGE_SQ = VISUAL_RANGE * VISUAL_RANGE
MIN_DISTANCE = 20
MIN_DISTANCE_SQ = MIN_DISTANCE * MIN_DISTANCE
BOUND_FACTOR = 1
SPEED_LIMIT = 10

# cursor attraction/repulsion constants
CURSOR_ATTRACT_FACTOR = 0.05
CURSOR_REPEL_FACTOR = 0.05
CURSOR_REPEL_DISTANCE = 50

# hand tracking constants
THUMB_DISTANCE_THRESHOLD = 0.1

CELL_SIZE = VISUAL_RANGE
NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),  (0, 0),  (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def build_color_lookup(size: int = 256) -> tuple[tuple[int, int, int], ...]:
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

    def __init__(self, x: float, y: float, dx: float, dy: float) -> None:
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def update(self) -> None:
        self.x += self.dx
        self.y += self.dy

    def distance_squared(self, other: "Boid") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def show(self) -> None:
        velocity = sqrt(self.dx * self.dx + self.dy * self.dy)
        normalized_velocity = min(velocity / SPEED_LIMIT, 1.0)
        color_index = int(normalized_velocity * (len(COLOR_LOOKUP) - 1))
        red, green, blue = COLOR_LOOKUP[color_index]
        fill(red, green, blue)
        circle(self.x, self.y, 10)

    def keep_in_bounds(self) -> None:
        if self.x < MARGIN:
            self.dx += BOUND_FACTOR
        elif self.x > WIDTH - MARGIN:
            self.dx -= BOUND_FACTOR

        if self.y < MARGIN:
            self.dy += BOUND_FACTOR
        elif self.y > HEIGHT - MARGIN:
            self.dy -= BOUND_FACTOR

    def limit_speed(self) -> None:
        speed = sqrt(self.dx * self.dx + self.dy * self.dy)
        if speed > SPEED_LIMIT and speed > 0:
            scale = SPEED_LIMIT / speed
            self.dx *= scale
            self.dy *= scale

    def apply_flocking_behaviors(
        self,
        boids: list["Boid"],
        neighbor_indices,
        cursor_active: bool,
        cursor_x: float,
        cursor_y: float,
    ) -> None:
        center_x = 0.0
        center_y = 0.0
        cohesion_neighbors = 0

        avg_dx = 0.0
        avg_dy = 0.0
        alignment_neighbors = 0

        avoid_dx = 0.0
        avoid_dy = 0.0

        # add cursor attraction force
        if cursor_active:
            cursor_dx = cursor_x - self.x
            cursor_dy = cursor_y - self.y
            cursor_dist = sqrt(cursor_dx * cursor_dx + cursor_dy * cursor_dy)

            if cursor_dist < CURSOR_REPEL_DISTANCE and cursor_dist > 0:
                self.dx -= cursor_dx * CURSOR_REPEL_FACTOR
                self.dy -= cursor_dy * CURSOR_REPEL_FACTOR
            elif cursor_dist >= CURSOR_REPEL_DISTANCE:
                self.dx += cursor_dx * CURSOR_ATTRACT_FACTOR
                self.dy += cursor_dy * CURSOR_ATTRACT_FACTOR

        # regular boid flocking behavior
        for idx in neighbor_indices:
            other = boids[idx]
            if other is self:
                continue

            dist_sq = self.distance_squared(other)

            if dist_sq < MIN_DISTANCE_SQ and dist_sq > 0:
                avoid_dx += self.x - other.x
                avoid_dy += self.y - other.y

            if dist_sq < VISUAL_RANGE_SQ:
                center_x += other.x
                center_y += other.y
                cohesion_neighbors += 1

                avg_dx += other.dx
                avg_dy += other.dy
                alignment_neighbors += 1

        avoidance_scale = AVOIDANCE_FACTOR * (0.05 if cursor_active else 1.0)
        self.dx += avoid_dx * avoidance_scale
        self.dy += avoid_dy * avoidance_scale

        if cohesion_neighbors > 0:
            center_x /= cohesion_neighbors
            center_y /= cohesion_neighbors
            self.dx += (center_x - self.x) * CENTER_FACTOR
            self.dy += (center_y - self.y) * CENTER_FACTOR

        if alignment_neighbors > 0:
            avg_dx /= alignment_neighbors
            avg_dy /= alignment_neighbors
            self.dx += (avg_dx - self.dx) * VELOCITY_FACTOR


def build_grid(boids: list[Boid]) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, boid in enumerate(boids):
        cell = int(boid.x // CELL_SIZE), int(boid.y // CELL_SIZE)
        grid[cell].append(idx)
    return grid


def iter_neighbor_indices(boid: Boid, grid: dict[tuple[int, int], list[int]]):
    cx, cy = int(boid.x // CELL_SIZE), int(boid.y // CELL_SIZE)
    for offset_x, offset_y in NEIGHBOR_OFFSETS:
        yield from grid.get((cx + offset_x, cy + offset_y), ())


def cursor_is_off_screen(x: float, y: float) -> bool:
    return (
        x < CURSOR_MARGIN
        or x > WIDTH - CURSOR_MARGIN
        or y < CURSOR_MARGIN
        or y > HEIGHT - CURSOR_MARGIN
    )


picam2 = None
tracking_thread = None
stop_tracking = None
finger_lock = threading.Lock()
finger_pos = [WIDTH / 2.0, HEIGHT / 2.0]
finger_visible = False
boids: list[Boid] = []

# run hand tracking loop in a separate thread
def tracking_loop() -> None:
    global finger_pos, finger_visible
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.4)
    try:
        while stop_tracking is None or not stop_tracking.is_set():
            frame = picam2.capture_array()
            if frame.ndim != 3:
                continue
            if frame.shape[2] == 4:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb_frame)
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

                thumb_dx = thumb_tip.x - index_tip.x
                thumb_dy = thumb_tip.y - index_tip.y
                thumb_dist_sq = thumb_dx * thumb_dx + thumb_dy * thumb_dy
                thumb_near = thumb_dist_sq <= THUMB_DISTANCE_THRESHOLD * THUMB_DISTANCE_THRESHOLD

                norm_x = (index_tip.x - 0.5) * SENSITIVITY
                norm_y = (index_tip.y - 0.5) * SENSITIVITY
                x = (0.5 + norm_x) * WIDTH
                y = (0.5 + norm_y) * HEIGHT
                x = max(0.0, min(float(WIDTH), x))
                y = max(0.0, min(float(HEIGHT), y))
                with finger_lock:
                    if thumb_near:
                        if not finger_visible:
                            finger_pos[0] = float(x)
                            finger_pos[1] = float(y)
                        else:
                            finger_pos[0] += SMOOTHING * (x - finger_pos[0])
                            finger_pos[1] += SMOOTHING * (y - finger_pos[1])
                        finger_visible = True
                    else:
                        finger_visible = False
            else:
                with finger_lock:
                    finger_visible = False
            time.sleep(0.01)
    finally:
        hands.close()


def settings():
    size(WIDTH, HEIGHT, py5.P2D)

def setup():
    global picam2, tracking_thread, stop_tracking, boids
    rect_mode(py5.CENTER)
    no_stroke()
    background(0)

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration())
    picam2.start()
    time.sleep(0.2)

    stop_tracking = threading.Event()
    tracking_thread = threading.Thread(target=tracking_loop, daemon=True)
    tracking_thread.start()

    boids = []
    for _ in range(NUM_BOIDS):
        x = py5.random(MARGIN / 2, WIDTH - MARGIN / 2)
        y = py5.random(MARGIN / 2, HEIGHT - MARGIN / 2)
        dx = py5.random(-3, 3)
        dy = py5.random(-3, 3)
        boids.append(Boid(x, y, dx, dy))


def draw():
    no_cursor()
    background(0)

    with finger_lock:
        cursor_x, cursor_y = finger_pos[0], finger_pos[1]
        cursor_visible = finger_visible

    cursor_active = cursor_visible and not cursor_is_off_screen(cursor_x, cursor_y)

    grid = build_grid(boids)
    for boid in boids:
        neighbor_indices = iter_neighbor_indices(boid, grid)
        boid.apply_flocking_behaviors(boids, neighbor_indices, cursor_active, cursor_x, cursor_y)
        boid.limit_speed()
        boid.keep_in_bounds()
        boid.update()
        boid.show()


def dispose():
    global tracking_thread, stop_tracking
    if stop_tracking is not None:
        stop_tracking.set()
    if tracking_thread is not None:
        tracking_thread.join(timeout=1.0)
        tracking_thread = None
    stop_tracking = None
    if picam2 is not None:
        picam2.stop()
        picam2.close()

run_sketch()