"""
Human-play and trained-agent play modes for Drone Delivery (Assignment 2).

Usage:
    python play.py human     # you fly with the arrow keys (30 fps)
    python play.py agent     # the saved DQN policy flies the drone

Requires pygame (visualization only; training never needs it):
    pip install pygame
"""

import sys

import pygame

from drone_env import DroneDeliveryEnv, MAX_STEPS

FPS = 30


def run(policy_fn, label):
    env = DroneDeliveryEnv()
    env.render_init()
    info = {}
    while True:
        obs = env.reset()
        done = False
        while not done:
            a = policy_fn(obs)
            obs, r, done, info = env.step(a)
            env.render_frame()
        # show the final frame + result for a moment before auto-reset
        result = ("DELIVERED!" if info.get("delivered") else
                  "CRASHED!" if info.get("crashed") else "TIME UP!")
        print(result, f"reward so far: see score on screen  ({label})")
        pygame.time.wait(1200)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "human"
    pygame.init()

    if mode == "human":
        def human_policy(_obs):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                return 0
            if keys[pygame.K_RIGHT]:
                return 1
            if keys[pygame.K_UP]:
                return 2
            if keys[pygame.K_DOWN]:
                return 3
            return 4  # hover
        run(human_policy, "human")
    elif mode == "agent":
        from dqn_agent import DQNAgent
        agent = DQNAgent.load("dqn_drone.pth")
        run(lambda o: agent.act(o, epsilon=0.0), "agent")
    else:
        print("usage: python play.py [human|agent]")
        sys.exit(1)


if __name__ == "__main__":
    main()
