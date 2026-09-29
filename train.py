"""
Train the DQN agent on Drone Delivery (Assignment 2).

Run:  python train.py

Produces:
    dqn_drone.pth      -- trained policy checkpoint
    training_plot.png  -- smoothed training reward per episode
    training_log.csv   -- raw per-episode rewards
"""

import csv
import io
import random

import numpy as np
import torch

from drone_env import DroneDeliveryEnv
from dqn_agent import DQNAgent

# ----------------------------- hyperparameters -------------------------------
EPISODES = 1000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 700   # linear decay over episodes
GAMMA = 0.99
LR = 1e-3
BATCH = 128
BUFFER = 50_000
TARGET_SYNC = 1000        # learn steps between target-network syncs
LEARN_EVERY = 2           # gradient step every N env steps (CPU speed knob)
SEED = 0

SMOOTH = 30               # window for the smoothed plot line


def main():
    torch.set_num_threads(4)   # small nets train faster without huge thread pools
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = DroneDeliveryEnv()
    agent = DQNAgent(lr=LR, gamma=GAMMA, buffer_size=BUFFER, batch=BATCH,
                     target_sync=TARGET_SYNC)

    rewards, wins = [], []
    best_sr = -1.0

    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY)
        obs = env.reset()
        total = 0.0
        done = False
        step_i = 0
        while not done:
            a = agent.act(obs, epsilon=eps)
            nobs, r, done, info = env.step(a)
            agent.remember(obs, a, r, nobs, float(done))
            obs = nobs
            total += r
            if step_i % LEARN_EVERY == 0:
                agent.learn()   # no-op until the buffer holds at least one batch
            step_i += 1
        rewards.append(total)
        wins.append(1.0 if info["delivered"] else 0.0)

        if (ep + 1) % 50 == 0:
            m = np.mean(rewards[-50:])
            sr = np.mean(wins[-50:])
            print(f"episode {ep+1:5d} | eps {eps:.2f} | "
                  f"reward(avg50) {m:8.1f} | success(avg50) {sr:.2f}", flush=True)
            # keep the best checkpoint (DQN can drift late in training)
            if sr > best_sr:
                best_sr = sr
                agent.save("dqn_drone.pth")
                print(f"          new best policy (success {sr:.2f}) saved", flush=True)

    agent.save("dqn_drone_last.pth")
    print(f"saved final policy -> dqn_drone_last.pth (best avg50 success: {best_sr:.2f})")
    print("best policy kept in  -> dqn_drone.pth")

    # ------------------------------ plot --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.array(rewards)
    smooth = np.convolve(raw, np.ones(SMOOTH) / SMOOTH, mode="valid")
    plt.figure(figsize=(9, 5))
    plt.plot(raw, alpha=0.25, label="per-episode reward")
    plt.plot(range(SMOOTH - 1, len(raw)), smooth, label=f"{SMOOTH}-episode mean")
    plt.xlabel("Training episode")
    plt.ylabel("Episode reward")
    plt.title("DQN training reward -- Drone Delivery")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # write via BytesIO: matplotlib's PIL save can fail on OneDrive-synced folders
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    with open("training_plot.png", "wb") as f:
        f.write(buf.getvalue())
    print("saved plot -> training_plot.png")

    with open("training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards, 1):
            w.writerow([i, f"{r:.2f}"])
    print("saved log -> training_log.csv")

    # quick greedy sanity check with the saved best policy
    best = DQNAgent.load("dqn_drone.pth")
    d = 0
    for seed in range(10):
        random.seed(seed)
        obs = env.reset()
        done = False
        while not done:
            obs, r, done, info = env.step(best.act(obs, epsilon=0.0))
        d += int(info["delivered"])
    print(f"greedy check with best policy: {d}/10 deliveries")



if __name__ == "__main__":
    main()
