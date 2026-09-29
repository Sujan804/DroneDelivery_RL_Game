# 3-Minute Demonstration Script — Drone Delivery (Option 2)

Use this as your outline for the live or recorded demonstration required by
the assignment. Total ≈ 3 minutes. The assignment asks the demo to show
**human play** and **trained-agent play**.

## Option A — Live demo (recommended, easiest)

1. `python play.py human` — fly it yourself with the arrow keys (~45 s).
   Narrate: "This is the game: reach the green delivery zone, avoid the two
   obstacles. The drone has momentum, so you must counter-thrust to stop.
   Score and steps are on screen."
2. `python play.py agent` — the saved DQN policy flies (~2 episodes, auto-
   resets). Narrate: "Now the trained DQN agent. It detours around the central
   obstacle and delivers every time — 10/10 in evaluation."

## Option B — Recorded demo with this script

### 0. Setup (before recording, ~10 s on camera or cut)

- Terminal open in the project folder, `README.md` visible.
- Say: name and roll number, "Option 2 — Drone Delivery, trained with DQN."

### 1. Human play — `python play.py human` (~45 s)

- Fly with the arrow keys. Try a straight dash first — you will likely hit the
  central obstacle; that shows why the task is not trivial.
- Then deliver at least once deliberately: fly above the obstacle, drift right,
  slow down before the green zone.
- Say: "Objective is the green zone. Left/right/up/down thrust, no key = hover.
  +100 for delivery, −100 for a crash, −1 per step, 300-step limit."

### 2. Trained agent play — `python play.py agent` (~45 s)

- Let it run 2–3 episodes (auto-resets).
- Say: "The saved policy is `dqn_drone.pth`. Note it flies *around* the central
  obstacle — the reward shaping put a safety ring around obstacles, so the
  shortest-reward path is a detour, not a dash."
- Point out it stops being fragile near the target: it counter-thrusts to slow
  down instead of overshooting into the wall.

### 3. Training evidence (~40 s)

- Show `training_plot.png`: "Raw reward is noisy; the smoothed line takes off
  around episode 570 and the 50-episode success rate hits 100% near episode
  750. The best checkpoint is what we just watched."
- Show `train_console.log` tail (optional): the `new best policy saved` lines.

### 4. Evaluation comparison (~30 s)

- Run `python evaluate.py` live or show `evaluation_results.csv`.
- Say: "Same 10 starting scenarios for both agents. Trained agent: +148
  average reward, 10/10 deliveries, zero crashes. Random agent: −333, zero
  deliveries, always times out at 300 steps."

### 5. Close (~10 s)

- Say: "Code, saved policy, README, and report are in the submission. The
  agent's success is the obstacle detour; the limitation is that it was
  trained from one fixed start; the improvement would be randomized starts
  plus Double-DQN to fix the late-training drift."

## Recording the GIF (optional extra artifact)

```bash
python record_demo.py            # you fly 45 s, then 2 agent episodes
python record_demo.py --agent-only --agent-episodes 3   # agent only
```

Produces `demo.gif` (15 fps, downscaled) with "HUMAN PLAY" / "TRAINED AGENT"
captions burned in — useful to embed in the report or submit alongside it.
