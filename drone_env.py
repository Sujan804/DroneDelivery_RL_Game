"""
A small 2D game environment for deep reinforcement learning: a delivery drone
must navigate around two fixed circular obstacles to reach a delivery zone.
Built as a Gym-style NumPy environment with an optional pygame renderer.
"""

import numpy as np

# ----------------------------- game constants -------------------------------
WORLD_W = 600.0          # world width  (pixels)
WORLD_H = 400.0          # world height (pixels)

DRONE_R = 12.0           # drone radius
TARGET = (540.0, 200.0)  # delivery zone center
TARGET_R = 26.0          # delivery zone radius

# Two fixed obstacles (cx, cy, r) -- "two fixed obstacles" per the assignment.
# The first sits on the direct start->target line, so a straight dash is
# impossible: the drone must fly around it (and mind the second one).
OBSTACLES = ((300.0, 200.0, 55.0), (450.0, 290.0, 50.0))

START = (60.0, 200.0)    # episode start position
MAX_STEPS = 300          # time limit per episode

ACCEL = 300.0            # thrust acceleration (px/s^2)
DRAG = 0.92              # multiplicative velocity damping per step
MAX_SPEED = 250.0        # speed clamp (px/s)

DT = 1.0 / 30.0          # fixed physics step (30 steps per simulated second)

# Actions: 0=left, 1=right, 2=up, 3=down, 4=hover
ACTION_NAMES = ["LEFT", "RIGHT", "UP", "DOWN", "HOVER"]
N_ACTIONS = 5

# ------------------------------ reward values -------------------------------
# Simple numerical values, explained in the report:
R_SUCCESS = +100.0        # reaching the delivery zone
R_CRASH = -100.0          # colliding with an obstacle
R_STEP = -1.0             # per-step living cost (rewards finishing fast)
R_SHAPE = 0.5             # potential-based shaping scale (per potential unit)
SAFE_MARGIN = 25.0        # obstacle safety ring width used by the shaping
R_TIMEOUT = -25.0         # small extra penalty for running out of time

# ------------------------------ obs / state ---------------------------------
OBS_DIM = 6  # x, y, vx, vy, target dx (normalized), target dy (normalized)


def dist_to_target(x, y):
    return float(np.hypot(TARGET[0] - x, TARGET[1] - y))


def hits_obstacle(x, y):
    """Circle collision between drone and any obstacle."""
    for cx, cy, r in OBSTACLES:
        if np.hypot(x - cx, y - cy) < r + DRONE_R:
            return True
    return False


def in_target(x, y):
    return np.hypot(x - TARGET[0], y - TARGET[1]) <= TARGET_R - DRONE_R * 0.5


def potential(x, y):
    """Shaping potential phi(s):

    phi = -distance_to_target  -  total intrusion into obstacle safety rings.

    Potential-based shaping (Ng, Harada & Russell, 1999) adds
    R_shape = scale * (phi(s') - phi(s)) per step; it speeds up learning
    without changing the optimal policy. The safety-ring term makes the
    shaped reward prefer paths that go AROUND obstacles instead of the
    raw shortest line (which passes through the central obstacle).
    """
    phi = -dist_to_target(x, y)
    for cx, cy, r in OBSTACLES:
        gap = np.hypot(x - cx, y - cy) - (r + DRONE_R)
        if gap < SAFE_MARGIN:
            phi -= (SAFE_MARGIN - gap)  # 1 unit per pixel inside the ring
    return phi


class DroneDeliveryEnv:
    """
    Minimal 2D drone game with a NumPy-only reset()/step() interface.

    Observation (6 numbers, compact):
        0: x / WORLD_W
        1: y / WORLD_H
        2: vx / MAX_SPEED
        3: vy / MAX_SPEED
        4: (target_x - x) / WORLD_W
        5: (target_y - y) / WORLD_H
    """

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self):
        self.x, self.y = START
        self.vx = self.vy = 0.0
        self.steps = 0
        self.score = 0.0          # visible game score (rewards collected)
        self.delivered = False
        self.crashed = False
        self._prev_pot = potential(self.x, self.y)
        return self._obs()

    def step(self, action):
        assert 0 <= action < N_ACTIONS

        # --- physics: thrust, drag, clamp, integrate --------------------
        ax = ay = 0.0
        if action == 0:
            ax = -ACCEL
        elif action == 1:
            ax = +ACCEL
        elif action == 2:
            ay = -ACCEL
        elif action == 3:
            ay = +ACCEL
        # action == 4: hover (no thrust)

        self.vx = (self.vx + ax * DT) * DRAG
        self.vy = (self.vy + ay * DT) * DRAG
        sp = float(np.hypot(self.vx, self.vy))
        if sp > MAX_SPEED:
            self.vx *= MAX_SPEED / sp
            self.vy *= MAX_SPEED / sp

        self.x += self.vx * DT
        self.y += self.vy * DT
        self.x = float(min(max(self.x, DRONE_R), WORLD_W - DRONE_R))
        self.y = float(min(max(self.y, DRONE_R), WORLD_H - DRONE_R))

        self.steps += 1
        reward = R_STEP
        terminated = False
        truncated = False

        # potential-based shaping: rewards getting closer to the target
        # AND escaping the obstacle safety rings (i.e., detouring around them)
        new_pot = potential(self.x, self.y)
        reward += (new_pot - self._prev_pot) * R_SHAPE
        self._prev_pot = new_pot

        # --- terminal checks -------------------------------------------
        if in_target(self.x, self.y):
            terminated = True
            self.delivered = True
            reward += R_SUCCESS
            self.score += 100
        elif hits_obstacle(self.x, self.y):
            terminated = True
            self.crashed = True
            reward += R_CRASH
        elif self.steps >= MAX_STEPS:
            truncated = True
            reward += R_TIMEOUT

        return self._obs(), float(reward), (terminated or truncated), {
            "delivered": self.delivered,
            "crashed": self.crashed,
            "score": self.score,
            "steps": self.steps,
        }

    def _obs(self):
        dx = (TARGET[0] - self.x) / WORLD_W
        dy = (TARGET[1] - self.y) / WORLD_H
        return np.array(
            [self.x / WORLD_W, self.y / WORLD_H,
             self.vx / MAX_SPEED, self.vy / MAX_SPEED, dx, dy],
            dtype=np.float32,
        )

    # --------------------------------------------------- human play (pygame)
    def render_init(self):
        import pygame
        pygame.init()
        self._screen = pygame.display.set_mode((int(WORLD_W), int(WORLD_H)))
        pygame.display.set_caption("Drone Delivery -- human play (arrow keys thrust)")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 26)

    def render_frame(self):
        import pygame
        for event in pygame.event.get(pygame.QUIT):
            pygame.quit()
            raise SystemExit
        s = self._screen
        s.fill((18, 24, 38))
        # delivery zone
        pygame.draw.circle(s, (40, 120, 80), (int(TARGET[0]), int(TARGET[1])), int(TARGET_R))
        pygame.draw.circle(s, (70, 200, 130), (int(TARGET[0]), int(TARGET[1])), int(TARGET_R), 3)
        # obstacles
        for cx, cy, r in OBSTACLES:
            pygame.draw.circle(s, (120, 120, 130), (int(cx), int(cy)), int(r))
        # drone
        color = (90, 170, 255) if not self.crashed else (200, 60, 60)
        pygame.draw.circle(s, color, (int(self.x), int(self.y)), int(DRONE_R))
        pygame.draw.circle(s, (230, 240, 255), (int(self.x), int(self.y)), int(DRONE_R), 2)
        # HUD: visible score + step counter
        s.blit(self._font.render(
            f"Score: {self.score:.0f}   Steps: {self.steps}/{MAX_STEPS}", True, (235, 235, 235)),
            (12, 10))
        pygame.display.flip()
        self._clock.tick(30)

    def render_close(self):
        import pygame
        pygame.quit()
