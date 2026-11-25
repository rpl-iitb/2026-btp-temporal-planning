"""Top-down rendering (PIL), GIF export and Gantt charts. Needs Sim(..., record=True) for gif() and gantt()."""
from PIL import Image, ImageDraw

STATE = {"idle": (225, 228, 232), "busy": (120, 190, 120), "blocked": (240, 170, 70), "down": (220, 80, 80), "setup": (120, 150, 230), "service": (190, 140, 220)}
KIND = {"sink": (200, 215, 240), "buffer": (230, 225, 200)}
PX, M, TOP = 42, 30, 34


def _xy(F, node):
    x, y = F.nodes[node]
    ymax = max(p[1] for p in F.nodes.values())
    return M + x * PX, TOP + M + (ymax - y) * PX


def _robot_xy(F, src, dest, t0, t1, phase, t):
    if phase != "travel" or t1 <= t0 or t >= t1:
        return _xy(F, dest if (phase == "travel" and t >= t1) or phase != "travel" else src)
    pts = [_xy(F, n) for n in F.path(src, dest)]
    seg = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:])]
    goal, acc = max(0.0, min(1.0, (t - t0) / (t1 - t0))) * sum(seg), 0.0
    for (a, b), L in zip(zip(pts, pts[1:]), seg):
        if acc + L >= goal and L > 0:
            f = (goal - acc) / L
            return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
        acc += L
    return pts[-1]


def draw(F, snap, t=None, title=""):
    t = snap["t"] if t is None else t
    xs, ys = [p[0] for p in F.nodes.values()], [p[1] for p in F.nodes.values()]
    W, H = int(2 * M + max(xs) * PX + 60), int(TOP + 2 * M + max(ys) * PX + 20)
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    for a, nb in F.adj.items():
        for b in nb:
            d.line([_xy(F, a), _xy(F, b)], fill=(200, 200, 205), width=3)
    for (a, b) in F.transfers:
        d.line([_xy(F, F.stations[a]["node"]), _xy(F, F.stations[b]["node"])], fill=(150, 150, 220), width=2)
    color = {p: v["color"] for p, v in F.products.items()}
    for n, sp in F.stations.items():
        x, y = _xy(F, sp["node"])
        state, q, out, cur = snap["stations"][n]
        d.rectangle([x - 26, y - 16, x + 26, y + 16], fill=KIND.get(sp["kind"], STATE[state]), outline=(90, 90, 90))
        d.text((x - 24, y - 15), n[:9], fill=(20, 20, 20))
        if cur:
            d.rectangle([x - 6, y - 2, x + 6, y + 10], fill=color[cur], outline=(0, 0, 0))
        for i, p in enumerate(q[:6]):
            d.rectangle([x - 26 + 7 * i, y + 18, x - 21 + 7 * i, y + 23], fill=color[p], outline=(0, 0, 0))
        for i, p in enumerate(out[:6]):
            d.rectangle([x + 21 - 7 * i, y - 23, x + 26 - 7 * i, y - 18], fill=color[p], outline=(0, 0, 0))
    for n, (src, dest, t0, t1, phase, carry) in snap["robots"].items():
        x, y = _robot_xy(F, src, dest, t0, t1, phase, t)
        d.ellipse([x - 10, y - 10, x + 10, y + 10], fill=(60, 60, 70), outline=(0, 0, 0))
        d.text((x - 6, y - 6), n, fill=(255, 255, 255))
        if carry:
            d.rectangle([x + 6, y - 14, x + 16, y - 4], fill=color[carry], outline=(0, 0, 0))
    d.text((8, 8), "{}  t = {:7.1f} s   done {}   late {}".format(title or F.name, t, snap["done"], snap["late"]), fill=(20, 20, 20))
    for i, (k, c) in enumerate(STATE.items()):
        d.rectangle([W - 70, 8 + 14 * i, W - 60, 18 + 14 * i], fill=c)
        d.text((W - 56, 7 + 14 * i), k, fill=(20, 20, 20))
    return img


def frame(sim):
    """Current state as an RGB numpy array (works without record=True)."""
    import numpy as np
    rec = sim.record
    sim.record = True
    sim._snap()
    snap = sim.trace.pop()
    sim.record = rec
    return np.asarray(draw(sim.F, snap))


def gif(sim, path, dt=10.0, fps=12, title=""):
    """Animate a recorded run (Sim(..., record=True)) at one frame per dt simulated seconds."""
    tr = sim.trace
    frames, k, t = [], 0, 0.0
    while t <= tr[-1]["t"] + 1e-9:
        while k + 1 < len(tr) and tr[k + 1]["t"] <= t:
            k += 1
        frames.append(draw(sim.F, tr[k], t, title))
        t += dt
    frames.append(draw(sim.F, tr[-1], tr[-1]["t"], title))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def gantt(sim, path):
    """Station states and robot phases over time from a recorded run (needs matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr, F = sim.trace, sim.F
    rows = [n for n, s in F.stations.items() if s["kind"] not in ("sink", "buffer")] + list(sim.robots)
    rc = {"travel": (110, 150, 220), "service": (120, 190, 120), "queue": (240, 170, 70)}
    fig, ax = plt.subplots(figsize=(12, 0.32 * len(rows) + 1))
    for i, n in enumerate(rows):
        for a, b in zip(tr, tr[1:] + [tr[-1]]):
            st = a["stations"][n][0] if n in a["stations"] else a["robots"][n][4]
            c = STATE.get(st) if n in a["stations"] else rc.get(st)
            if c and st != "idle" and b["t"] > a["t"]:
                ax.broken_barh([(a["t"], b["t"] - a["t"])], (i - 0.4, 0.8), color=[x / 255 for x in c])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
