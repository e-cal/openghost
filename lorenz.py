#!/usr/bin/env python3
"""
Lorenz Attractor - 3D visualization using Pygame + PyOpenGL
Full screen, black background, animated in real-time
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Lorenz parameters
SIGMA = 10
RHO = 28
BETA = 8 / 3
DT = 0.01

# Initial position
pos = [0.01, 0.0, 0.0]

# Trail
points = []
MAX_POINTS = 3000

# Frame counter for rotation
frame_num = 0

def init_gl(width, height):
    """Initialize OpenGL settings"""
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
    
    # Disable depth test completely
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    
    # Enable line smoothing
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLineWidth(1.5)
    
    # Set viewport
    glViewport(0, 0, width, height)
    
    # Set up orthographic projection (no perspective clipping)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = width / height
    # Orthographic: left, right, bottom, top, near, far
    glOrtho(-100 * aspect, 100 * aspect, -100, 100, -1000, 1000)
    glMatrixMode(GL_MODELVIEW)

def update_lorenz():
    """Update Lorenz attractor position"""
    global pos, points
    
    x, y, z = pos
    
    # Calculate next point (Lorenz equations)
    dx = SIGMA * (y - x) * DT
    dy = (x * (RHO - z) - y) * DT
    dz = (x * y - BETA * z) * DT
    
    pos[0] += dx
    pos[1] += dy
    pos[2] += dz
    
    # Add to trail
    points.append((pos[0], pos[1], pos[2]))
    if len(points) > MAX_POINTS:
        points.pop(0)

def draw_scene():
    """Draw the scene"""
    global frame_num
    
    # Calculate center of mass
    if points:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        cz = sum(p[2] for p in points) / len(points)
    else:
        cx = cy = cz = 0.0
    
    # Clear buffer
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    
    # Transform order: first center the attractor, then rotate, then scale
    # This ensures rotation happens around the attractor's center
    
    # Scale to fit nicely on screen
    glScalef(2.0, 2.0, 2.0)
    
    # Rotations around origin (which will be attractor center)
    glRotatef(20, 1, 0, 0)
    glRotatef(frame_num * 0.3, 0, 1, 0)
    
    # Translate attractor to origin
    glTranslatef(-cx, -cy, -cz)
    
    # Draw as individual line segments
    if len(points) > 1:
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            glVertex3f(p1[0], p1[1], p1[2])
            glVertex3f(p2[0], p2[1], p2[2])
        glEnd()
    
    frame_num += 1

def main():
    """Main loop"""
    pygame.init()
    
    WIDTH = 800
    HEIGHT = 480
    
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), 
        DOUBLEBUF | OPENGL | FULLSCREEN
    )
    pygame.display.set_caption("Lorenz Attractor")
    pygame.mouse.set_visible(False)
    
    init_gl(WIDTH, HEIGHT)
    
    clock = pygame.time.Clock()
    running = True
    
    print("Lorenz Attractor running. Press ESC or Q to quit.")
    
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    running = False
        
        update_lorenz()
        draw_scene()
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
