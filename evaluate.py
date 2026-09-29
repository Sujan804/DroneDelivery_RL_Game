"""
Evaluate trained vs random agent on Drone Delivery (Assignment 2).

Runs 10 episodes with the trained policy and 10 with a random policy on
identical starting scenarios (same env, deterministic start; the seed makes
any stochasticity match). Reports average episode reward, deliveries out of
10, crashes out of 10 and average steps, and writes evaluation_results.csv.

Run:  python evaluate.py
"""

import csv
import random

import numpy as np

from drone_env import DroneDeliveryEnv
from dqn_agent import DQNAgent

N_EPISODES = 10
SEEDS = list(range(10))          # same starting scenarios for both agents


def run_eval(policy_fn, label):
    env = DroneDeliveryEnv()
    rewards, deliveries, crashes, steps_list = [], 0, 0, []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        env.reset()
        obs = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            a = policy_fn(obs)
            obs, r, done, info = env.step(a)
            total += r
            steps += 1
        rewards.append(total)
        deliveries += int(info["delivered"])
        crashes += int(info["crashed"])
        steps_list.append(steps)
    print(f"\n=== {label} ({N_EPISODES} episodes) ===")
    print(f"avg episode reward : {np.mean(rewards):8.1f}")
    print(f"deliveries         : {deliveries}/10")
    print(f"crashes            : {crashes}/10")
    print(f"avg steps          : {np.mean(steps_list):6.1f}")
    return {
        "agent": label,
        "avg_reward": round(float(np.mean(rewards)), 2),
        "deliveries": deliveries,
        "crashes": crashes,
        "avg_steps": round(float(np.mean(steps_list)), 1),
        "rewards": [round(r, 2) for r in rewards],
    }


def main():
    agent = DQNAgent.load("dqn_drone.pth")
    trained = run_eval(lambda o: agent.act(o, epsilon=0.0), "Trained DQN agent")
    rnd = run_eval(lambda o: random.randrange(5), "Random agent")

    with open("evaluation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "avg_reward", "deliveries_out_of_10",
                    "crashes_out_of_10", "avg_steps"])
        for row in (trained, rnd):
            w.writerow([row["agent"], row["avg_reward"], row["deliveries"],
                        row["crashes"], row["avg_steps"]])
        w.writerow([])
        w.writerow(["per_episode_rewards_trained"] + trained["rewards"])
        w.writerow(["per_episode_rewards_random"] + rnd["rewards"])
    print("\nsaved -> evaluation_results.csv")


if __name__ == "__main__":
    main()
