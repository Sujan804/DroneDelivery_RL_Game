"""Build demo.gif for the README from the existing demo.mp4.

Keeps the informative part (the trained-agent delivery run), speeds it up
temporally, downscales to 300x200 and palette-optimizes so it stays small.

Run:  python make_demo_gif.py
"""

import os

import imageio.v2 as imageio
from PIL import Image

SRC = "demo.mp4"
OUT = "demo.gif"

# Timeline of demo.mp4 (74 s @ 15 fps): ~0-31 s human play, ~32-74 s agent.
# We keep agent play from 32 s to the end (2 episodes + result captions).
START_S = 32.0
STEP = 2                 # keep every 2nd source frame -> 7.5 fps
SCALE = 0.5              # 600x400 -> 300x200
MAX_COLORS = 128
MAX_KB = 5000            # warn if the GIF gets huge


def main():
    reader = imageio.get_reader(SRC)
    src_fps = reader.get_meta_data()["fps"]
    duration_ms = int(round(1000 * STEP / src_fps))     # ~133 ms per frame

    # 1) extract + downscale the frames we need
    frames = []
    for i, frame in enumerate(reader):
        if i / src_fps < START_S or i % STEP != 0:
            continue
        img = Image.fromarray(frame)
        img = img.resize((int(img.width * SCALE), int(img.height * SCALE)),
                         Image.LANCZOS)
        frames.append(img)
    reader.close()
    print(f"kept {len(frames)} frames "
          f"({len(frames) * duration_ms / 1000:.1f}s at {src_fps / STEP:.1f} fps)")

    # 2) one global palette (scene colors barely change), then dither-quantize
    pal_img = frames[0].quantize(colors=MAX_COLORS, method=Image.MEDIANCUT)
    quantized = [f.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
                 for f in frames]
    print("frames quantized")

    # 3) save with per-frame duration + infinite loop
    quantized[0].save(
        OUT,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    kb = os.path.getsize(OUT) / 1024
    print(f"saved -> {OUT}  ({kb:.0f} KB, {len(quantized)} frames, "
          f"{quantized[0].size[0]}x{quantized[0].size[1]}, {src_fps / STEP:.1f} fps)")
    if kb > MAX_KB:
        print(f"warning: GIF larger than {MAX_KB} KB — consider cutting "
              f"more frames or lowering MAX_COLORS")


if __name__ == "__main__":
    main()
