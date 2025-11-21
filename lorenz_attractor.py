# inspired by https://www.youtube.com/watch?v=f0lkz2gSsIk

from py5 import *

WIDTH = 800
HEIGHT = 800

x = 0.01
y = 0.0
z = 0.0
frame_num = 0

a = 10
b = 28
c = 8 / 3
dt = 0.01

points = []
max_points = 3000

def settings():
    size(WIDTH, HEIGHT, P3D)


def setup():
    color_mode(HSB, 360, 100, 100)
    stroke_weight(0.6)
    no_fill()
    background(0)


def draw():
    no_cursor()
    global x, y, z, frame_num

    # lorenz attractor equation
    dx = a * (y - x) * dt
    dy = (x * (b - z) - y) * dt
    dz = (x * y - c * z) * dt

    x += dx
    y += dy
    z += dz

    # append the points to the list
    points.append((x, y, z))

    if len(points) > max_points:
        points.pop(0)

    if points:
        xs, ys, zs = zip(*points)
        size = len(points)
        cx = sum(xs) / size
        cy = sum(ys) / size
        cz = sum(zs) / size
    else:
        cx = cy = cz = 0.0

    # draw the lorenz attractor
    background(0)
    translate(WIDTH / 2 - WIDTH / 12, HEIGHT / 2, -120) # offset lorenz attractor a bit to the right as it tends to go off left
    rotate_x(radians(30))
    rotate_y(frame_num * 0.01)
    scale(6)
    translate(-cx, -cy, -cz)

    # white stroke
    stroke(255)

    # rainbow stroke
    # stroke((frame_num * 2) % 360, 80, 100)

    begin_shape()
    for px, py, pz in points:
        vertex(px, py, pz)
    end_shape()

    frame_num += 1

run_sketch()
