"""
DQN agent for the Drone Delivery environment.

A standard Deep Q-Network: MLP Q-function, experience replay, target network,
epsilon-greedy exploration with linear decay. The trained policy is saved as
a state_dict checkpoint so a play mode can reload it.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import drone_env
from drone_env import DroneDeliveryEnv, N_ACTIONS, OBS_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Small MLP mapping the 6-d observation to one Q-value per action."""

    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch):
        s, a, r, s2, d = zip(*random.sample(self.buf, batch))
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1),
            torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1),
            torch.as_tensor(np.array(s2), dtype=torch.float32, device=DEVICE),
            torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, buffer_size=50_000, batch=128,
                 target_sync=1000):
        self.q = QNetwork().to(DEVICE)
        self.target = QNetwork().to(DEVICE)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.batch = batch
        self.target_sync = target_sync
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    # ------------------------------------------------------------- policy
    def act(self, obs, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return int(self.q(t).argmax(dim=1).item())

    # ------------------------------------------------------------ learning
    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def learn(self):
        if len(self.buffer) < self.batch:
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch)
        with torch.no_grad():
            best_next = self.target(s2).max(dim=1, keepdim=True)[0]
            y = r + self.gamma * best_next * (1.0 - d)
        q_sa = self.q(s).gather(1, a)
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------ persist
    def save(self, path="dqn_drone.pth"):
        torch.save({
            "q_state": self.q.state_dict(),
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
        }, path)

    @classmethod
    def load(cls, path="dqn_drone.pth"):
        ckpt = torch.load(path, map_location=DEVICE)
        agent = cls()
        agent.q.load_state_dict(ckpt["q_state"])
        agent.target.load_state_dict(ckpt["q_state"])
        agent.q.eval()
        return agent


def random_action(_obs):
    return random.randrange(N_ACTIONS)


def run_episode(env, policy_fn, epsilon=0.0, max_steps=None, seed=None):
    """Run one full episode; returns (total_reward, info)."""
    if seed is not None:
        env.rng = np.random.default_rng(seed)
    obs = env.reset()
    total = 0.0
    steps = 0
    done = False
    while not done:
        a = policy_fn(obs) if epsilon == 0.0 else (
            random.randrange(N_ACTIONS) if random.random() < epsilon else policy_fn(obs))
        obs, r, done, info = env.step(a)
        total += r
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total, info
