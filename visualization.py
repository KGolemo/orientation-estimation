import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import OPENGL, DOUBLEBUF



def run(sys, data, Ts):
    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((640, 480), video_flags)
    pygame.display.set_caption("Orientation visualization")
    resizewin(640, 480)
    init()
    clock = pygame.time.Clock()
    i = 0
    n = data.shape[0]
    running = True
    while running and i < n:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.K_ESCAPE:
                running = False
        
        clock.tick(100)

        data_row = data[i, :]
        w = [data_row[0], data_row[1], data_row[2]]
        a = [data_row[3], data_row[4], data_row[5]]
        m = [data_row[6], data_row[7], data_row[8]]

        sys.predict(w, Ts)
        sys.correct(a, m)
        [w, nx, ny, nz] = sys.xHat[0:4]

        draw(w, nx, ny, nz, i, Ts)

        pygame.display.flip()
        i += 1
    pygame.quit()


def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def draw(w, nx, ny, nz, i, Ts):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    # drawText((-2.6, 1.6, 2), "Module to visualize quaternion data", 16)
    drawText((-2.6, 1.8, 2), "Time of the experiment: %f" %(i*Ts), 16)

    [yaw, pitch, roll] = quaternionToYpr([w, nx, ny, nz])

    drawText((-2.6, -1.4, 2), f"Yaw:   {yaw:.4f}", 16)
    drawText((-2.6, -1.6, 2), f"Pitch: {pitch:.4f}", 16)
    drawText((-2.6, -1.8, 2), f"Roll:  {roll:.4f}", 16)
    drawText((-2.6, -2, 2), "Press Escape to exit.", 16)
    # glRotatef(2 * math.acos(w) * 180.00/math.pi, -1 * nx, nz, ny)
    glRotatef(2 * np.degrees(np.arccos(w)), -1*ny, 1*nz, -1*nx)

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.5, -1.0)
    glVertex3f(-1.0, 0.5, -1.0)
    glVertex3f(-1.0, 0.5, 1.0)
    glVertex3f(1.0, 0.5, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.5, 1.0)
    glVertex3f(-1.0, -0.5, 1.0)
    glVertex3f(-1.0, -0.5, -1.0)
    glVertex3f(1.0, -0.5, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.5, 1.0)
    glVertex3f(-1.0, 0.5, 1.0)
    glVertex3f(-1.0, -0.5, 1.0)
    glVertex3f(1.0, -0.5, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.5, -1.0)
    glVertex3f(-1.0, -0.5, -1.0)
    glVertex3f(-1.0, 0.5, -1.0)
    glVertex3f(1.0, 0.5, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.5, 1.0)
    glVertex3f(-1.0, 0.5, -1.0)
    glVertex3f(-1.0, -0.5, -1.0)
    glVertex3f(-1.0, -0.5, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.5, -1.0)
    glVertex3f(1.0, 0.5, 1.0)
    glVertex3f(1.0, -0.5, 1.0)
    glVertex3f(1.0, -0.5, -1.0)
    glEnd()


def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def quaternionToYpr(q):
    roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    pitch = np.degrees(np.arcsin(2 * (q[0] * q[2] - q[3] * q[1])))
    yaw = np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    return [yaw, pitch, roll]