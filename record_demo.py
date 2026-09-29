"""
Record the demonstration VIDEO (demo.mp4): human play first, then
trained-agent play, with on-screen captions ("HUMAN PLAY" / "TRAINED AGENT").

Run:
    python record_demo.py                                   # you fly 45 s, then 2 agent episodes
    python record_demo.py --agent-only --agent-episodes 3   # agent part only
    python record_demo.py --human-seconds 30 --out my_demo.mp4
    python record_demo.py --gif                             # small GIF instead of MP4

  * Phase 1 -- YOU fly with the arrow keys. ESC ends the human phase early.
  * Phase 2 -- the saved DQN policy (dqn_drone.pth) flies N episodes.
  * The video is then written to --out (default demo.mp4, 15 fps).

MP4 frames are streamed straight into the encoder (no RAM buildup); GIF mode
accumulates downscaled frames and encodes with Pillow at the end.

Requires: pygame, and imageio + imageio-ffmpeg for MP4 (Pillow for GIF mode).
"""

import sys

import numpy as np
import pygame
from PIL import Image

import drone_env
from drone_env import DroneDeliveryEnv

FPS = 30
CAPTURE_EVERY = 2          # keep every 2nd simulation frame -> 15 fps video
GIF_SCALE = 0.5            # GIF is downscaled; MP4 keeps full resolution
HUMAN_SECONDS = 45         # default human-play phase length
AGENT_EPISODES = 2         # default agent episodes after the human phase
OUT_PATH = "demo.mp4"

def store(img):
    """Send one frame to the active sink (video writer or frame list)."""
    global n_stored
    n_stored += 1
    if writer is not None:
        writer.append_data(np.asarray(img))
    else:
        frames.append(img)


frames = []                # GIF mode: accumulated PIL frames
writer = None              # MP4 mode: imageio writer (frames streamed)
n_stored = 0               # total frames captured (both modes)


def grab_frame(screen):
    """Copy the display surface to a full-resolution PIL image."""
    arr = pygame.surfarray.array3d(screen)                 # (w, h, 3)
    return Image.fromarray(np.transpose(arr, (1, 0, 2)))   # -> (h, w, 3)


def draw_caption(screen, font, text):
    screen.blit(font.render(text, True, (255, 255, 255), (0, 0, 0)), (10, 364))


def capture_pause(screen, font, text, ms=1200):
    """Show an end-of-episode caption for ~1.2 s while capturing frames."""
    draw_caption(screen, font, text)
    pygame.display.flip()
    end = pygame.time.get_ticks() + ms
    i = 0
    while pygame.time.get_ticks() < end:
        pygame.time.wait(100)
        i += 1
        if i % CAPTURE_EVERY == 0:
            store(grab_frame(screen))


def human_phase(screen, font, env, limit_s, frame_counter):
    """You fly. Appends frames; returns False only if the window was closed."""
    obs = env.reset()
    label = "HUMAN PLAY -- arrow keys to fly, ESC to finish"
    t0 = pygame.time.get_ticks()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
        if (pygame.time.get_ticks() - t0) > limit_s * 1000:
            running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            a = 0
        elif keys[pygame.K_RIGHT]:
            a = 1
        elif keys[pygame.K_UP]:
            a = 2
        elif keys[pygame.K_DOWN]:
            a = 3
        else:
            a = 4

        obs, r, done, info = env.step(a)
        env.render_frame()
        draw_caption(screen, font, label)
        pygame.display.flip()

        frame_counter[0] += 1
        if frame_counter[0] % CAPTURE_EVERY == 0:
            store(grab_frame(screen))

        if done:
            result = ("DELIVERED!" if info["delivered"] else
                      "CRASHED!" if info["crashed"] else "TIME UP!")
            capture_pause(screen, font, f"HUMAN PLAY -- {result}")
            obs = env.reset()
    return True


def agent_phase(screen, font, env, agent, n_eps, frame_counter):
    """The trained DQN policy flies n_eps episodes."""
    label = "TRAINED AGENT (DQN) -- greedy policy"
    for ep in range(n_eps):
        obs = env.reset()
        done = False
        info = {}
        while not done:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return False
            a = agent.act(obs, epsilon=0.0)
            obs, r, done, info = env.step(a)
            env.render_frame()
            draw_caption(screen, font, label)
            pygame.display.flip()

            frame_counter[0] += 1
            if frame_counter[0] % CAPTURE_EVERY == 0:
                store(grab_frame(screen))
        result = ("DELIVERED!" if info["delivered"] else
                  "CRASHED!" if info["crashed"] else "TIME UP!")
        capture_pause(screen, font,
                      f"AGENT PLAY ep {ep + 1}/{n_eps} -- {result}")
    return True


def finish(out):
    """Close the sink and report the output file."""
    global writer
    if writer is not None:
        writer.close()
        writer = None
    elif frames:
        small = [f.resize((int(f.width * GIF_SCALE), int(f.height * GIF_SCALE)),
                          Image.BILINEAR).convert(
                     "P", palette=Image.ADAPTIVE, colors=128)
                 for f in frames]
        small[0].save(out, save_all=True, append_images=small[1:],
                      duration=int(1000 / (FPS / CAPTURE_EVERY)), loop=0,
                      optimize=True)
    import os
    if os.path.exists(out):
        print(f"saved -> {out}  ({os.path.getsize(out) / 1024:.0f} KB, "
              f"{n_stored} frames)", flush=True)


def main():
    global writer
    argv = sys.argv[1:]
    agent_only = "--agent-only" in argv
    want_gif = "--gif" in argv
    out = OUT_PATH
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    elif want_gif:
        out = "demo.gif"
    human_s = float(argv[argv.index("--human-seconds") + 1]) \
        if "--human-seconds" in argv else HUMAN_SECONDS
    n_eps = int(argv[argv.index("--agent-episodes") + 1]) \
        if "--agent-episodes" in argv else AGENT_EPISODES

    if not out.endswith(".gif"):
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            out, fps=FPS // CAPTURE_EVERY, codec="libx264", quality=8,
            macro_block_size=1, pixelformat="yuv420p")

    pygame.init()
    env = DroneDeliveryEnv()
    env.render_init()
    screen = pygame.display.get_surface()
    font = pygame.font.SysFont(None, 30)

    counter = [0]
    try:
        ok = True
        if not agent_only:
            print(f"human phase: fly with arrow keys for up to {human_s:.0f} s "
                  f"(ESC to finish early)...", flush=True)
            ok = human_phase(screen, font, env, human_s, counter)
        if ok:
            from dqn_agent import DQNAgent
            agent = DQNAgent.load("dqn_drone.pth")
            print(f"agent phase: recording {n_eps} episodes...", flush=True)
            agent_phase(screen, font, env, agent, n_eps, counter)
    finally:
        finish(out)
        pygame.quit()


if __name__ == "__main__":
    main()
