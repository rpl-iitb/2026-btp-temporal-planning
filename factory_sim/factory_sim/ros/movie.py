"""Turn the camera frames of a `record:=true` run into GIFs (one per view).

    python -m factory_sim.ros.movie ros_runs/<run> --every 2 --width 640
"""
import argparse
import os
import re

from PIL import Image


def frames(view_dir):
    key = lambda f: int(re.findall(r"(\d+)\.\w+$", f)[0]) if re.findall(r"(\d+)\.\w+$", f) else 0
    return [os.path.join(view_dir, f) for f in sorted(os.listdir(view_dir), key=key) if f.lower().endswith((".jpg", ".png"))]


def gif(paths, out, width=640, ms=120):
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((width, int(im.height * width / im.width)), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=200))
    if ims:
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--ms", type=int, default=120, help="milliseconds per frame")
    a = ap.parse_args()
    root = os.path.join(a.run, "frames")
    for view in sorted(os.listdir(root)):
        out = gif(frames(os.path.join(root, view))[::a.every], os.path.join(a.run, view + ".gif"), a.width, a.ms)
        print(out)


if __name__ == "__main__":
    main()
