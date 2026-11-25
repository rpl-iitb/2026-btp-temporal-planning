"""Simulation-view rendering of a recorded FactorySim run (Sim(..., record=True)): a top-down plant floor with aisles,
lane markings and junctions (one robot at a time), machine cabinets with stack lights, conveyor belts, benches, the
store rack and charging pads; TurtleBot-style robots drive smoothly along the aisles carrying trays (coloured by
product). Under the plant: a clock, counters and the order list in columns (delivered / late / due / overdue).

    from factory_sim import render_sim as RS
    RS.video(sim, "run.mp4", speed=60, title="...")     # MP4 (needs imageio-ffmpeg), `speed` simulated s per video s
    RS.gif(sim, "run.gif", frames=240, title="...")      # GIF of the same view

render.py keeps the schematic view (station-state boxes); this module only reads sim.F, sim.trace and sim.jobs.
"""
import math

from PIL import Image, ImageDraw, ImageFont

STATE = {"idle": (150, 155, 160), "busy": (40, 200, 70), "blocked": (250, 170, 30), "down": (230, 50, 50),
         "setup": (70, 130, 240), "service": (180, 110, 230)}
ROBOT = [(230, 90, 40), (40, 120, 220), (30, 170, 110), (200, 60, 160), (150, 110, 40), (90, 90, 200),
         (220, 170, 20), (20, 160, 180), (170, 50, 50), (110, 150, 40), (130, 70, 180), (60, 60, 60)]
FLOOR, AISLE, LINE, WALL = (214, 216, 212), (96, 100, 106), (245, 200, 40), (70, 74, 80)
BG = (245, 244, 240)
_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        try:
            _FONTS[key] = ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            _FONTS[key] = ImageFont.load_default(size=size)
    return _FONTS[key]


def clock(t):
    t = int(round(t))
    return "{:d}:{:02d}:{:02d}".format(t // 3600, t % 3600 // 60, t % 60)


class View:
    """Static geometry of a plant (scale, transposition, station placement) and the pre-drawn floor."""

    def __init__(self, sim, width=1500, title="", cols=None, panel_text=None, panel="auto"):
        """panel_text: font size (px) of the order panel under the plant (default: small, from the plant scale);
        panel: 'list' (one line per order), 'grid' (one numbered chip per order, coloured by status) or 'auto'
        (list up to 16 orders at the default size; with a large panel_text, grid above 16 orders)."""
        self.sim, self.F, self.title = sim, sim.F, title
        F = self.F
        xs, ys = [p[0] for p in F.nodes.values()], [p[1] for p in F.nodes.values()]
        self.flip = (max(ys) - min(ys)) > 1.2 * (max(xs) - min(xs))     # tall plants are drawn on their side
        u = [self._uv(p) for p in F.nodes.values()]
        self.u0, self.v0 = min(a for a, _ in u) - 1.9, min(b for _, b in u) - 1.4
        span_u, span_v = max(a for a, _ in u) - self.u0 + 1.9, max(b for _, b in u) - self.v0 + 1.6
        self.px = width / span_u
        self.W, self.Hp = width, int(span_v * self.px)
        self.top = int(max(70, self.px * 1.3))
        self.jobs = sorted(sim.jobs.values(), key=lambda j: j.id)
        self.cols = cols or (2 if len(self.jobs) <= 16 else 4 if len(self.jobs) <= 40 else 5)
        self.f = font(max(13, int(self.px * 0.26)))
        self.fs = font(max(11, int(self.px * 0.21)))
        self.fb = font(max(16, int(self.px * 0.34)), True)
        self.fp = font(panel_text) if panel_text else self.fs
        n = len(self.jobs)
        self.mode = panel if panel != "auto" else ("grid" if panel_text and n > 16 else "list")
        self.row = int(self.fp.size * 1.55)
        if self.mode == "list":
            if panel_text and not cols:          # as many columns as the longest line allows
                need = self.fp.size + 14 + self.fp.getlength("order 00 PR0 RUSH") + 10 + self.fp.getlength("delivered 0:00:00") + 16
                self.cols = max(1, min(4, int(self.W // need)))
            self.panel = int(self.row * math.ceil(n / self.cols) + self.row * 1.0)
        else:
            self.chip_w = max(self.fp.getlength("00") + 24, (self.W - 20) / max(1, math.ceil(n / math.ceil(n / 14))))
            self.per_row = max(1, int((self.W - 20) // self.chip_w))
            self.chip_h = int(self.fp.size * 1.6)
            self.panel = int(self.row * 1.3 + (self.chip_h + 6) * math.ceil(n / self.per_row) + self.row * 0.5)
        self.H = self.top + self.Hp + self.panel
        self.H += self.H % 2
        self.W += self.W % 2
        self.body = {n: self._body(n) for n in F.stations}
        self.robots = list(sim.robots)
        self.static = self._floor()

    # ---- geometry ------------------------------------------------------------------------------------
    def _uv(self, p):
        x, y = p
        return (y, x) if self.flip else (x, -y)

    def xy(self, node):
        u, v = self._uv(self.F.nodes[node])
        return (u - self.u0) * self.px, self.top + (v - self.v0) * self.px

    def _body(self, n):
        """Centre of a station's body: behind its dock, away from the aisle it docks on."""
        F, node = self.F, self.F.stations[n]["node"]
        x, y = self.xy(node)
        nb = [m for m in F.adj.get(node, []) if m not in F.stations]
        if not nb:
            return x, y, 0.0, 0.0
        ax, ay = self.xy(nb[0])
        dx, dy = x - ax, y - ay
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        off = 0.55 * self.px
        return x + dx * off, y + dy * off, dx, dy

    # ---- static floor ----------------------------------------------------------------------------------
    def _floor(self):
        F, px = self.F, self.px
        img = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, self.top, self.W, self.top + self.Hp], fill=FLOOR)
        step = max(8, int(px))
        for x in range(0, self.W, step):
            d.line([(x, self.top), (x, self.top + self.Hp)], fill=(205, 207, 203))
        for y in range(self.top, self.top + self.Hp, step):
            d.line([(0, y), (self.W, y)], fill=(205, 207, 203))
        d.rectangle([1, self.top, self.W - 2, self.top + self.Hp], outline=WALL, width=4)
        aw = max(6, int(px * 0.62))
        seen = set()
        for a, nb in F.adj.items():
            for b in nb:
                if (b, a) in seen:
                    continue
                seen.add((a, b))
                d.line([self.xy(a), self.xy(b)], fill=AISLE, width=aw)
        for n in F.nodes:
            if F.adj.get(n):
                x, y = self.xy(n)
                r = aw / 2
                d.ellipse([x - r, y - r, x + r, y + r], fill=AISLE)
        for (a, b) in seen:                     # dashed centre line
            (x0, y0), (x1, y1) = self.xy(a), self.xy(b)
            L = math.hypot(x1 - x0, y1 - y0)
            k = int(L // max(6, px * 0.3))
            for i in range(0, k, 2):
                f0, f1 = i / max(k, 1), min(1, (i + 1) / max(k, 1))
                d.line([(x0 + f0 * (x1 - x0), y0 + f0 * (y1 - y0)), (x0 + f1 * (x1 - x0), y0 + f1 * (y1 - y0))],
                       fill=(225, 225, 225), width=max(1, int(px * 0.04)))
        for j in getattr(F, "junctions", ()):   # junctions: hatched box, one robot at a time
            x, y = self.xy(j)
            r = aw * 0.62
            d.rectangle([x - r, y - r, x + r, y + r], outline=LINE, width=max(2, int(px * 0.06)))
            d.line([(x - r, y - r), (x + r, y + r)], fill=LINE, width=max(1, int(px * 0.03)))
            d.line([(x - r, y + r), (x + r, y - r)], fill=LINE, width=max(1, int(px * 0.03)))
        for (a, b) in F.transfers:              # conveyor belts with rollers
            (x0, y0), (x1, y1) = self.body[a][:2], self.body[b][:2]
            bw = max(6, int(px * 0.34))
            d.line([(x0, y0), (x1, y1)], fill=(60, 60, 64), width=bw)
            L = math.hypot(x1 - x0, y1 - y0) or 1
            nx, ny = -(y1 - y0) / L * bw / 2, (x1 - x0) / L * bw / 2
            for i in range(1, int(L // max(5, px * 0.16))):
                f = i * max(5, px * 0.16) / L
                cx, cy = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                d.line([(cx - nx, cy - ny), (cx + nx, cy + ny)], fill=(120, 120, 126), width=1)
        for n in F.nodes:                        # charging pads
            if "charger" in n and n not in F.stations:
                x, y = self.xy(n)
                r = px * 0.32
                d.rounded_rectangle([x - r, y - r, x + r, y + r], radius=int(r / 3), fill=(60, 120, 200), outline=(30, 60, 110), width=2)
                d.text((x, y), "⚡", font=self.fb, fill=(255, 230, 60), anchor="mm")
        return img

    # ---- dynamic parts -----------------------------------------------------------------------------
    def tray(self, d, x, y, product, s):
        c = self.F.products[product]["color"]
        d.rounded_rectangle([x - s, y - s, x + s, y + s], radius=max(1, int(s / 3)), fill=c, outline=(20, 20, 20), width=1)

    def station(self, d, n, snap_st):
        state, q, out, cur = snap_st
        sp, px = self.F.stations[n], self.px
        x, y, dx, dy = self.body[n]
        kind = sp["kind"]
        w = h = px * 0.42
        if kind == "machine":
            body, edge = (176, 186, 198), (70, 80, 92)
        elif kind == "port":
            body, edge = (150, 156, 166), (60, 64, 70)
            w = h = px * 0.3
        elif kind in ("kitting", "assembly"):
            body, edge = ((190, 150, 100), (110, 80, 50)) if kind == "kitting" else ((120, 160, 210), (50, 80, 120))
            w, h = (px * 0.55, px * 0.32) if abs(dy) > abs(dx) else (px * 0.32, px * 0.55)
        elif kind == "inspection":
            body, edge = (230, 230, 236), (90, 90, 110)
        elif kind == "sink":
            body, edge = (150, 110, 70), (80, 55, 30)
            w = h = px * 0.55
        else:                                   # buffer: floor area with hazard stripes
            body, edge = (235, 220, 150), (160, 130, 40)
            w = h = px * 0.45
        d.rounded_rectangle([x - w, y - h, x + w, y + h], radius=int(px * 0.08), fill=body, outline=edge, width=max(2, int(px * 0.05)))
        if kind == "machine":
            win = px * 0.24
            d.rectangle([x - win, y - win, x + win, y + win], fill=(40, 50, 60), outline=(20, 20, 20))
            if cur:
                self.tray(d, x, y, cur, px * 0.14)
                if state == "busy":
                    d.arc([x - win * 0.9, y - win * 0.9, x + win * 0.9, y + win * 0.9], 0, 300, fill=(250, 250, 250), width=2)
        elif kind == "sink":
            for i in range(3):
                yy = y - h + (i + 1) * 2 * h / 4
                d.line([(x - w + 3, yy), (x + w - 3, yy)], fill=(90, 60, 30), width=2)
        elif kind == "buffer":
            for i in range(-3, 4):
                d.line([(x + i * w / 3 - w / 3, y - h + 2), (x + i * w / 3 + w / 3, y + h - 2)], fill=(200, 170, 50), width=2)
        elif cur:
            self.tray(d, x, y, cur, px * 0.14)
        if kind not in ("sink", "buffer"):      # stack light
            r = max(4, px * 0.1)
            lx, ly = x + w - r * 0.4, y - h + r * 0.4
            d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=STATE.get(state, STATE["idle"]), outline=(20, 20, 20))
        s = px * 0.1                             # input queue (left) / output (right) trays
        for i, p in enumerate(q[:4]):
            self.tray(d, x - w + s + i * 2.3 * s, y + h + s * 1.4, p, s)
        for i, p in enumerate(out[:4]):
            self.tray(d, x + w - s - i * 2.3 * s, y + h + s * 1.4, p, s)
        label = sp.get("tag") or n
        d.text((x, y - h - 2), label, font=self.fs, fill=(30, 30, 34), anchor="md")

    def robot_pose(self, rb, t):
        from factory_sim.render import _robot_xy
        src, dest, t0, t1, phase, carry = rb
        F = self.F
        if phase != "travel" or t1 <= t0 or t >= t1:
            node = dest if (phase == "travel" and t >= t1) or phase != "travel" else src
            return self.xy(node), None
        pts = [self.xy(n) for n in F.path(src, dest)]
        seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
        goal, acc = max(0.0, min(1.0, (t - t0) / (t1 - t0))) * sum(seg), 0.0
        for (a, b), L in zip(zip(pts, pts[1:]), seg):
            if acc + L >= goal and L > 0:
                f = (goal - acc) / L
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), math.atan2(b[1] - a[1], b[0] - a[0])
            acc += L
        return pts[-1], None

    def robot(self, d, i, name, pos, heading, carry, phase):
        (x, y), px = pos, self.px
        r = px * 0.26
        col = ROBOT[i % len(ROBOT)]
        d.ellipse([x - r - 2, y - r + 2, x + r + 2, y + r + 4], fill=(150, 150, 150))          # shadow
        d.ellipse([x - r, y - r, x + r, y + r], fill=(45, 48, 54), outline=col, width=max(3, int(px * 0.07)))
        hx, hy = (math.cos(heading), math.sin(heading)) if heading is not None else (0.0, -1.0)
        nx, ny = -hy, hx
        for sgn in (-1, 1):                     # wheels
            cx, cy = x + sgn * nx * r * 0.9, y + sgn * ny * r * 0.9
            d.line([(cx - hx * r * 0.4, cy - hy * r * 0.4), (cx + hx * r * 0.4, cy + hy * r * 0.4)], fill=(15, 15, 15), width=max(3, int(px * 0.08)))
        d.ellipse([x + hx * r * 0.55 - r * 0.2, y + hy * r * 0.55 - r * 0.2, x + hx * r * 0.55 + r * 0.2, y + hy * r * 0.55 + r * 0.2],
                  fill=(200, 200, 210))                                                  # lidar
        if carry:
            self.tray(d, x - hx * r * 0.15, y - hy * r * 0.15, carry, r * 0.5)
        if phase == "queue":
            d.text((x + r, y - r), "…", font=self.fb, fill=(240, 120, 20), anchor="lb")
        bx = d.textbbox((x, y + r + 3), name, font=self.fs, anchor="ma")
        d.rounded_rectangle([bx[0] - 3, bx[1] - 1, bx[2] + 3, bx[3] + 2], radius=3, fill=(255, 255, 255), outline=col)
        d.text((x, y + r + 3), name, font=self.fs, fill=col, anchor="ma")

    def _status(self, j, t):
        td = j.t_done if j.t_done is not None and j.t_done <= t + 1e-9 else None
        if td is not None:
            return ("late", td) if td > j.due + 1e-9 else ("delivered", td)
        if j.arrival > t:
            return "not arrived", None
        return ("overdue" if t > j.due else "due"), None

    def panel_grid(self, d, t):
        col = {"delivered": (40, 160, 70), "late": (210, 50, 40), "overdue": (240, 150, 140), "due": (235, 235, 235),
               "not arrived": (205, 205, 205)}
        y = self.top + self.Hp + int(self.row * 0.25)
        x = 10
        d.text((x, y), "orders:", font=self.fp, fill=(25, 25, 25))
        x += self.fp.getlength("orders:") + 16
        for k in ("delivered", "late", "due", "overdue"):
            r = self.fp.size * 0.45
            d.rounded_rectangle([x, y + self.fp.size * 0.15, x + 2 * r, y + self.fp.size * 0.15 + 2 * r], radius=3,
                                fill=col[k], outline=(60, 60, 60))
            d.text((x + 2 * r + 6, y), k, font=self.fp, fill=(25, 25, 25))
            x += 2 * r + 30 + self.fp.getlength(k)
        y0 = y + int(self.row * 1.05)
        for k, j in enumerate(self.jobs):
            r, c = divmod(k, self.per_row)
            x, y = 10 + c * self.chip_w, y0 + r * (self.chip_h + 6)
            st, _ = self._status(j, t)
            d.rounded_rectangle([x, y, x + self.chip_w - 6, y + self.chip_h], radius=6, fill=col[st],
                                outline=(200, 30, 30) if j.rush else (90, 90, 90), width=3 if j.rush else 1)
            self.tray(d, x + 4 + self.fp.size * 0.35, y + self.chip_h / 2, j.product, self.fp.size * 0.3)
            d.text((x + (self.chip_w - 6) / 2 + self.fp.size * 0.3, y + self.chip_h / 2), str(j.id + 1), font=self.fp,
                   fill=(255, 255, 255) if st in ("delivered", "late") else (25, 25, 25), anchor="mm")

    def panel_draw(self, d, t, done, late):
        if self.mode == "grid":
            return self.panel_grid(d, t)
        y0 = self.top + self.Hp + int(self.row * 0.4)
        cw = self.W / self.cols
        rows = math.ceil(len(self.jobs) / self.cols)
        for k, j in enumerate(self.jobs):
            c, r = divmod(k, rows)
            x, y = 10 + c * cw, y0 + r * self.row
            big = self.fp is not self.fs
            self.tray(d, x + 7 + (self.fp.size * 0.3 if big else 0), y + self.fp.size * 0.6, j.product, self.fp.size * 0.4)
            name = "order {} {}{}".format(j.id + 1, j.product, " RUSH" if j.rush else "")
            tx = x + (self.fp.size + 14 if big else 18)
            d.text((tx, y), name, font=self.fp, fill=(25, 25, 25))
            td = j.t_done if j.t_done is not None and j.t_done <= t + 1e-9 else None
            if td is not None:
                txt, col = ("LATE  delivered {}".format(clock(td)), (200, 30, 30)) if td > j.due + 1e-9 else (
                    "delivered {}".format(clock(td)), (30, 140, 50))
            elif j.arrival > t:
                txt, col = "arrives {}".format(clock(j.arrival)), (150, 150, 150)
            elif t > j.due:
                txt, col = "OVERDUE (due {})".format(clock(j.due)), (200, 30, 30)
            else:
                txt, col = "due {}".format(clock(j.due)), (90, 90, 90)
            if big:                              # short wording so two columns fit
                txt = txt.replace("LATE  delivered", "LATE").replace("OVERDUE (due ", "OVERDUE (").replace("arrives", "at")
            d.text((tx + self.fp.getlength(name) + 10, y), txt, font=self.fp, fill=col)

    def draw(self, snap, t):
        img = self.static.copy()
        d = ImageDraw.Draw(img)
        for n in self.F.stations:
            if n in snap["stations"]:
                self.station(d, n, snap["stations"][n])
        for i, name in enumerate(self.robots):
            rb = snap["robots"][name]
            pos, hd = self.robot_pose(rb, t)
            self.robot(d, i, name, pos, hd, rb[5], rb[4])
        done = sum(1 for j in self.jobs if j.t_done is not None and j.t_done <= t + 1e-9)
        late = sum(1 for j in self.jobs if j.t_done is not None and j.t_done <= t + 1e-9 and j.t_done > j.due + 1e-9)
        over = sum(1 for j in self.jobs if (j.t_done is None or j.t_done > t) and t > j.due)
        d.text((12, self.top * 0.32), self.title, font=self.fb, fill=(20, 20, 20), anchor="lm")
        head = "t = {}     delivered {}/{}     late {}     overdue {}".format(clock(t), done, len(self.jobs), late, over)
        d.text((self.W - 12, self.top * 0.32), head, font=self.fb, fill=(200, 30, 30) if late or over else (20, 20, 20), anchor="rm")
        lx, ly = 12, self.top * 0.76
        d.text((lx, ly), "stack lights:", font=self.fs, fill=(30, 30, 30), anchor="lm")
        lx += self.fs.getlength("stack lights:") + 10
        for k, c in STATE.items():               # legend
            r = self.fs.size * 0.4
            d.ellipse([lx, ly - r, lx + 2 * r, ly + r], fill=c, outline=(20, 20, 20))
            d.text((lx + 2 * r + 4, ly), k, font=self.fs, fill=(30, 30, 30), anchor="lm")
            lx += 2 * r + 16 + self.fs.getlength(k)
        note = "yellow boxes: junctions (one robot at a time)     ⚡ charger     small squares: trays (colour = product)"
        d.text((lx + 20, ly), note, font=self.fs, fill=(90, 90, 90), anchor="lm")
        self.panel_draw(d, t, done, late)
        return img


def _frames(sim, times, view):
    tr, k = sim.trace, 0
    for t in times:
        while k + 1 < len(tr) and tr[k + 1]["t"] <= t:
            k += 1
        yield view.draw(tr[k], t)


def video(sim, path, speed=None, fps=24, duration=90.0, hold=3.0, width=1500, title="", cols=None, panel_text=None,
          panel="auto"):
    """MP4 of a recorded run at `speed` simulated seconds per video second (default: the run takes `duration` s)."""
    import imageio_ffmpeg
    import numpy as np
    view = View(sim, width, title, cols, panel_text, panel)
    t_end = sim.trace[-1]["t"]
    speed = speed or max(1.0, t_end / duration)
    n = int(math.ceil(t_end / speed * fps))
    times = [min(t_end, i * speed / fps) for i in range(n + 1)] + [t_end] * int(hold * fps)
    w = imageio_ffmpeg.write_frames(path, (view.W, view.H), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                    quality=7, macro_block_size=2)
    w.send(None)
    for img in _frames(sim, times, view):
        w.send(np.ascontiguousarray(np.asarray(img)))
    w.close()
    return path, speed


def gif(sim, path, frames=240, fps=12, width=1100, title="", cols=None, panel_text=None, panel="auto"):
    """GIF of the same view with `frames` frames over the run (panel_text, panel: see View)."""
    view = View(sim, width, title, cols, panel_text, panel)
    t_end = sim.trace[-1]["t"]
    times = [t_end * i / frames for i in range(frames + 1)]
    imgs = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in _frames(sim, times, view)]
    dur = [int(1000 / fps)] * (len(imgs) - 1) + [2500]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0)
    return path
