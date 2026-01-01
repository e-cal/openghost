#!/usr/bin/env python3
"""
Boids with Finger Tracking - Pygame version
Fullscreen, black background, camera-based hand tracking
"""

import threading
import time
import random
from collections import defaultdict
from math import sqrt

import pygame
from pygame.locals import *
import cv2
import mediapipe as mp
from picamera2 import Picamera2

# Screen dimensions
WIDTH = 800
HEIGHT = 480

# Hand tracking sensitivity and smoothing
SENSITIVITY = 1.8
SMOOTHING = 0.5

# Boid settings
NUM_BOIDS = 100
MARGIN = 100
CURSOR_MARGIN = 50

# Boid flocking behavior constants
CENTER_FACTOR = 0.01
AVOIDANCE_FACTOR = 0.3
VELOCITY_FACTOR = 0.2
VISUAL_RANGE = 50
VISUAL_RANGE_SQ = VISUAL_RANGE * VISUAL_RANGE
MIN_DISTANCE = 20
MIN_DISTANCE_SQ = MIN_DISTANCE * MIN_DISTANCE
BOUND_FACTOR = 1
SPEED_LIMIT = 10

# Cursor attraction/repulsion constants
CURSOR_ATTRACT_FACTOR = 0.05
CURSOR_REPEL_FACTOR = 0.05
CURSOR_REPEL_DISTANCE = 50

# Hand tracking constants
THUMB_DISTANCE_THRESHOLD = 0.1

CELL_SIZE = VISUAL_RANGE
NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),  (0, 0),  (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def build_color_lookup(size: int = 256) -> tuple:
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

    def draw(self, screen) -> None:
        velocity = sqrt(self.dx * self.dx + self.dy * self.dy)
        normalized_velocity = min(velocity / SPEED_LIMIT, 1.0)
        color_index = int(normalized_velocity * (len(COLOR_LOOKUP) - 1))
        color = COLOR_LOOKUP[color_index]
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 5)

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
        boids: list,
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

        # Cursor attraction force
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

        # Regular boid flocking behavior
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


def build_grid(boids: list) -> dict:
    grid = defaultdict(list)
    for idx, boid in enumerate(boids):
        cell = int(boid.x // CELL_SIZE), int(boid.y // CELL_SIZE)
        grid[cell].append(idx)
    return grid


def iter_neighbor_indices(boid: Boid, grid: dict):
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


class HandTracker:
    def __init__(self):
        self.finger_lock = threading.Lock()
        self.finger_pos = [WIDTH / 2.0, HEIGHT / 2.0]
        self.finger_visible = False
        self.thumb_near = False
        self.stop_tracking = threading.Event()
        self.tracking_thread = None
        self.picam2 = None
        # Debug info
        self.debug_frame = None
        self.debug_landmarks = None
        self.hand_detected = False
    
    def start(self):
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration())
        self.picam2.start()
        time.sleep(0.2)
        
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
    
    def stop(self):
        self.stop_tracking.set()
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
    
    def get_finger_state(self):
        with self.finger_lock:
            return self.finger_pos[0], self.finger_pos[1], self.finger_visible
    
    def get_debug_info(self):
        with self.finger_lock:
            return (
                self.debug_frame.copy() if self.debug_frame is not None else None,
                self.debug_landmarks,
                self.hand_detected,
                self.thumb_near,
                self.finger_pos[0],
                self.finger_pos[1]
            )
    
    def _tracking_loop(self):
        hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.4)
        try:
            while not self.stop_tracking.is_set():
                frame = self.picam2.capture_array()
                if frame.ndim != 3:
                    continue
                if frame.shape[2] == 4:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = hands.process(rgb_frame)
                
                # Store debug frame (flipped for mirror effect)
                with self.finger_lock:
                    self.debug_frame = cv2.flip(rgb_frame, 1)
                    self.debug_landmarks = None
                    self.hand_detected = False
                
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

                    thumb_dx = thumb_tip.x - index_tip.x
                    thumb_dy = thumb_tip.y - index_tip.y
                    thumb_dist_sq = thumb_dx * thumb_dx + thumb_dy * thumb_dy
                    thumb_near = thumb_dist_sq <= THUMB_DISTANCE_THRESHOLD * THUMB_DISTANCE_THRESHOLD

                    # Mirror X axis so cursor matches mirrored camera view
                    mirrored_x = 1.0 - index_tip.x
                    norm_x = (mirrored_x - 0.5) * SENSITIVITY
                    norm_y = (index_tip.y - 0.5) * SENSITIVITY
                    x = (0.5 + norm_x) * WIDTH
                    y = (0.5 + norm_y) * HEIGHT
                    x = max(0.0, min(float(WIDTH), x))
                    y = max(0.0, min(float(HEIGHT), y))
                    
                    with self.finger_lock:
                        self.hand_detected = True
                        self.thumb_near = thumb_near
                        # Store landmarks for debug drawing (mirrored)
                        self.debug_landmarks = [(1.0 - lm.x, lm.y) for lm in hand_landmarks.landmark]
                        
                        if thumb_near:
                            if not self.finger_visible:
                                self.finger_pos[0] = float(x)
                                self.finger_pos[1] = float(y)
                            else:
                                self.finger_pos[0] += SMOOTHING * (x - self.finger_pos[0])
                                self.finger_pos[1] += SMOOTHING * (y - self.finger_pos[1])
                            self.finger_visible = True
                        else:
                            self.finger_visible = False
                else:
                    with self.finger_lock:
                        self.finger_visible = False
                        self.hand_detected = False
                        self.thumb_near = False
                time.sleep(0.01)
        finally:
            hands.close()


def draw_debug_overlay(screen, tracker, font):
    """Draw camera feed and hand tracking debug info"""
    debug_frame, landmarks, hand_detected, thumb_near, finger_x, finger_y = tracker.get_debug_info()
    
    if debug_frame is None:
        return
    
    # Scale down camera frame to fit in corner
    cam_h, cam_w = debug_frame.shape[:2]
    scale = 200 / cam_w
    new_w = int(cam_w * scale)
    new_h = int(cam_h * scale)
    small_frame = cv2.resize(debug_frame, (new_w, new_h))
    
    # Draw landmarks on the small frame if detected
    if landmarks:
        # Draw all landmarks as small dots
        for i, (lx, ly) in enumerate(landmarks):
            px = int(lx * new_w)
            py = int(ly * new_h)
            color = (0, 255, 0)  # Green for most points
            # Index finger tip = 8, Thumb tip = 4
            if i == 8:
                color = (255, 0, 0)  # Red for index finger
                cv2.circle(small_frame, (px, py), 6, color, -1)
            elif i == 4:
                color = (0, 0, 255)  # Blue for thumb
                cv2.circle(small_frame, (px, py), 6, color, -1)
            else:
                cv2.circle(small_frame, (px, py), 2, color, -1)
        
        # Draw line between thumb and index if pinching
        if thumb_near:
            idx_pos = (int(landmarks[8][0] * new_w), int(landmarks[8][1] * new_h))
            thumb_pos = (int(landmarks[4][0] * new_w), int(landmarks[4][1] * new_h))
            cv2.line(small_frame, idx_pos, thumb_pos, (255, 255, 0), 2)
    
    # Convert to pygame surface
    surf = pygame.surfarray.make_surface(small_frame.swapaxes(0, 1))
    
    # Draw camera feed in top-left corner with border
    screen.blit(surf, (10, 10))
    pygame.draw.rect(screen, (100, 100, 100), (10, 10, new_w, new_h), 2)
    
    # Draw status text
    status_y = 20 + new_h
    
    # Hand detection status
    if hand_detected:
        hand_text = font.render("Hand: DETECTED", True, (0, 255, 0))
    else:
        hand_text = font.render("Hand: NOT FOUND", True, (255, 100, 100))
    screen.blit(hand_text, (10, status_y))
    
    # Pinch status
    if thumb_near:
        pinch_text = font.render("Pinch: ACTIVE", True, (255, 255, 0))
    else:
        pinch_text = font.render("Pinch: inactive", True, (150, 150, 150))
    screen.blit(pinch_text, (10, status_y + 25))
    
    # Cursor position
    pos_text = font.render(f"Pos: ({int(finger_x)}, {int(finger_y)})", True, (200, 200, 200))
    screen.blit(pos_text, (10, status_y + 50))
    
    # Instructions
    help_text = font.render("D=toggle debug, Q/ESC=quit", True, (150, 150, 150))
    screen.blit(help_text, (10, HEIGHT - 30))


def main():
    pygame.init()
    
    screen = pygame.display.set_mode((WIDTH, HEIGHT), FULLSCREEN)
    pygame.display.set_caption("Boids with Finger Tracking")
    pygame.mouse.set_visible(False)
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Initialize boids
    boids = []
    for _ in range(NUM_BOIDS):
        x = random.uniform(MARGIN / 2, WIDTH - MARGIN / 2)
        y = random.uniform(MARGIN / 2, HEIGHT - MARGIN / 2)
        dx = random.uniform(-3, 3)
        dy = random.uniform(-3, 3)
        boids.append(Boid(x, y, dx, dy))
    
    # Start hand tracker
    tracker = HandTracker()
    tracker.start()
    
    print("Boids running. Press ESC or Q to quit, D to toggle debug view.")
    print("Pinch thumb and index finger to attract boids.")
    
    running = True
    debug_mode = False
    
    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        running = False
                    elif event.key == K_d:
                        debug_mode = not debug_mode
                        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            # Get finger state
            cursor_x, cursor_y, cursor_visible = tracker.get_finger_state()
            cursor_active = cursor_visible and not cursor_is_off_screen(cursor_x, cursor_y)
            
            # Clear screen
            screen.fill((0, 0, 0))
            
            # Update and draw boids
            grid = build_grid(boids)
            for boid in boids:
                neighbor_indices = iter_neighbor_indices(boid, grid)
                boid.apply_flocking_behaviors(boids, neighbor_indices, cursor_active, cursor_x, cursor_y)
                boid.limit_speed()
                boid.keep_in_bounds()
                boid.update()
                boid.draw(screen)
            
            # Draw cursor indicator when active
            if cursor_active:
                pygame.draw.circle(screen, (100, 100, 255), (int(cursor_x), int(cursor_y)), 10, 2)
            
            # Draw debug overlay if enabled
            if debug_mode:
                draw_debug_overlay(screen, tracker, font)
            
            pygame.display.flip()
            clock.tick(60)
    
    finally:
        tracker.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
