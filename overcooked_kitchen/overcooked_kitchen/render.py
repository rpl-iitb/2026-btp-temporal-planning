"""Rendering with Overcooked-AI sprites (PIL only): frame(sim), gif(sim, path), gantt(sim, path).

The kitchen is composed at 1x (15 px tiles) from the upstream sheets plus assets/extra.png, then scaled with
nearest neighbour; HUD and timeline are drawn below the kitchen. On the sampled clock a running cook is drawn
between poses, interpolated along the BFS path.
"""
"""This file is UN-VERIFIED"""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

from overcooked_kitchen.layout import INGREDIENTS, NORTH, SOUTH, EAST, WEST

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
T = 15
HATS = ("blue", "green", "orange", "purple")
FACE = {NORTH: "NORTH", SOUTH: "SOUTH", EAST: "EAST", WEST: "WEST"}
FLOOR_STATION = {"counter": "counter", "board": "board", "pot": "pot", "onion_crate": "onions",
                 "tomato_crate": "tomatoes", "carrot_crate": "carrots", "rack": "dishes", "dropoff": "dropoff",
                 "sink": "sink", "serving": "serve"}
COOK_COLORS = [(40, 90, 170), (30, 130, 60), (210, 120, 20), (120, 50, 150)]
ACT_COLOR = {
    "move": (190, 190, 190), "carry": (150, 200, 120), "chop": (200, 150, 60), "add": (250, 225, 140),
    "start": (200, 60, 30), "plate": (120, 180, 230), "deliver": (230, 180, 20), "scrub": (100, 60, 30),
    "wash": (60, 120, 200), "cooking": (245, 140, 40), "ready": (60, 180, 175), "burnt": (30, 30, 30),
    "spoiled": (110, 140, 60), "failed": (220, 40, 40),
}
CATEGORY = {"take-raw": "carry", "put-raw": "carry", "take-raw-back": "carry", "take-chopped": "carry",
            "put-chopped": "carry", "chop": "chop", "add-onion": "add", "add-tomato": "add", "add-carrot": "add",
            "start-cooking": "start", "take-clean-plate": "plate", "dish-up": "plate", "deliver": "deliver",
            "scrub-pot": "scrub", "take-dirty-plate": "wash", "put-in-sink": "wash", "wash-plate": "wash",
            "take-from-sink": "wash", "rack-plate": "wash", "put-plate": "carry", "take-plate": "carry"}


_FONTS = {}


def _font(size):
    """PIL's bundled default font at a size (same layout on every system); loaded once."""
    if size not in _FONTS:
        _FONTS[size] = ImageFont.load_default(size=size)
    return _FONTS[size]


def _wrap(text, font, maxw):
    """Greedy word wrap into lines no wider than maxw (a single over-long word is split by characters)."""
    lines, cur = [], ""
    for word in text.split(" "):
        cand = word if not cur else cur + " " + word
        if font.getlength(cand) <= maxw:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        cur = ""
        while font.getlength(word) > maxw and len(word) > 1:
            k = len(word)
            while k > 1 and font.getlength(word[:k]) > maxw:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    lines.append(cur)
    return lines


class Sprites:
    """All sprite sheets (chefs, objects, soups, terrain, extra); tiles by frame name, with or without '.png'."""

    def __init__(self, scale=4):
        self.scale = scale
        self.raw = {}
        self._cache = {}
        for sheet in ("chefs", "objects", "soups", "terrain", "extra"):
            with open(os.path.join(ASSETS, sheet + ".json")) as f:
                data = json.load(f)
            img = Image.open(os.path.join(ASSETS, sheet + ".png")).convert("RGBA")
            if "textures" in data:
                frames = {fr["filename"]: fr for fr in data["textures"][0]["frames"]}
            else:
                frames = data["frames"]
            for name, fr in frames.items():
                f = fr["frame"]
                self.raw[name[:-4] if name.endswith(".png") else name] = img.crop(
                    (f["x"], f["y"], f["x"] + f["w"], f["y"] + f["h"]))

    def get(self, name):
        """Unscaled 15x15 RGBA tile."""
        name = name[:-4] if name.endswith(".png") else name
        return self.raw[name]

    def img(self, name):
        """Tile scaled by `scale` (nearest neighbour)."""
        if name not in self._cache:
            t = self.get(name)
            self._cache[name] = t.resize((t.width * self.scale, t.height * self.scale), Image.NEAREST)
        return self._cache[name]


_SPRITES = {}


def _sprites(scale):
    if scale not in _SPRITES:
        _SPRITES[scale] = Sprites(scale)
    return _SPRITES[scale]


# ---- naming helpers --------------------------------------------------------------------------
def _dominant(comp):
    o, t, c = comp
    return max((("carrot", c), ("tomato", t), ("onion", o)), key=lambda kv: kv[1])[0]


def _mix(comp):
    o, t, c = comp
    return c == 0 and 1 <= o + t <= 3


def dish_icon(comp):
    """Plated soup sprite of a composition (onions, tomatoes, carrots)."""
    if _mix(comp):
        return "soup_done_tomato_{}_onion_{}".format(comp[1], comp[0])
    return "soup-{}-dish".format(_dominant(comp))


def pot_sprite(comp, state):
    n = sum(comp)
    if state == "dirty":
        return "pot-dirty"
    if state == "burnt":
        return "pot-explosion"
    if n == 0:
        return "pot"
    if state == "ready":
        if _mix(comp):
            return "soup_cooked_tomato_{}_onion_{}".format(comp[1], comp[0])
        return "soup-{}-cooked".format(_dominant(comp))
    if _mix(comp):
        return "soup_idle_tomato_{}_onion_{}".format(comp[1], comp[0])
    return "soup-{}-{}-cooking".format(_dominant(comp), min(3, n))


def _tint(img, rgb):
    out = img.copy()
    px = out.load()
    for y in range(out.height):
        for x in range(out.width):
            r, g, b, a = px[x, y]
            if a:
                px[x, y] = (r * rgb[0] // 255, g * rgb[1] // 255, b * rgb[2] // 255, a)
    return out


def _effective(pot, t, burn):
    """Pot state at time t from its snapshot (timers that fire between frames are applied)."""
    st = pot["state"]
    if st == "cooking" and pot["ready_at"] is not None and t >= pot["ready_at"]:
        st = "ready" if pot.get("recipe", True) else "spoiled"
    if st == "ready" and burn is not None and pot["ready_at"] is not None and t >= pot["ready_at"] + burn:
        st = "burnt"
    return st


def _path(lay, src, dst, limit=400):
    """Poses (x, y, facing) along a shortest path src -> dst, both inclusive."""
    out, cur = [src], src
    for _ in range(limit):
        if cur == dst:
            break
        d = lay.next_step(cur, dst)
        if d is None:
            break
        n = (cur[0] + d[0], cur[1] + d[1])
        cur = (n[0], n[1], d) if lay.is_floor(n) else (cur[0], cur[1], d)
        out.append(cur)
    return out


class _Scene:
    """What a frame needs from a sim: static layout and orders, plus one snapshot with running tasks."""

    def __init__(self, sim):
        w = sim.world
        self.lay, self.kind, self.params = w.lay, w.kind, w.params
        self.orders = [dict(o) for o in w.orders]
        self.cooks = list(sim.cooks)
        self.stepped = bool(sim.clock.stepped)
        self.title = sim.scenario.get("scenario_id", "")
        self.paths = {}


def _running_view(sim):
    return {c: {"pose": r.task.pose, "start": r.start, "end": r.end, "action": r.task.action}
            for c, r in sim.running.items()}


def _cook_poses(scene, snap, running, t):
    """[(px, py, facing)] in 1x pixels for every cook at time t."""
    out = []
    for i, p in enumerate(snap["players"]):
        x, y, f = p["pos"][0], p["pos"][1], tuple(p["facing"])
        run = running.get(scene.cooks[i]) if not scene.stepped else None
        if run is not None and run.get("pose") is not None:
            key = ((x, y, f), tuple(run["pose"]))
            if key not in scene.paths:
                scene.paths[key] = _path(scene.lay, key[0], key[1])
            path = scene.paths[key]
            steps = len(path) - 1
            done = max(0.0, min(float(steps), t - run["start"]))
            k = int(math.floor(done))
            if k >= steps:
                x, y, f = path[-1][0], path[-1][1], path[-1][2]
                out.append((x * T, y * T, f))
                continue
            frac = done - k
            a, b = path[k], path[k + 1]
            out.append((int(round((a[0] + (b[0] - a[0]) * frac) * T)), int(round((a[1] + (b[1] - a[1]) * frac) * T)),
                        b[2] if frac > 0 else a[2]))
            continue
        out.append((x * T, y * T, f))
    return out


class Renderer:
    def __init__(self, scale=4, cols=None, hud_text=None):
        self.sp = _sprites(scale)
        self.scale = scale
        self.cols = cols            # order columns in the HUD (default: one per 300 px)
        self.hud_text = hud_text    # font size (px) of a compact HUD: one large line per order (for slides)
        self.k = max(1.0, scale / 3)    # HUD text and icons grow with the kitchen past scale 3
        self.f_s, self.f_b, self.f_t = _font(int(12 * self.k)), _font(int(14 * self.k)), _font(7)

    # ---- kitchen (1x) ----------------------------------------------------------------------------
    def _bar(self, d, x, y, frac, col, row=0):
        frac = max(0.0, min(1.0, frac))
        y0 = y + 11 - 3 * row
        d.rectangle([x + 1, y0, x + 13, y0 + 2], fill=(20, 20, 20, 255))
        n = int(round(11 * frac))
        if n > 0:
            d.rectangle([x + 2, y0 + 1, x + 1 + n, y0 + 1], fill=col + (255,))

    def _paste_item(self, canvas, name, x, y, size=T, dx=0, dy=0):
        spr = self.sp.get(name)
        if size != T:
            spr = spr.resize((size, size), Image.NEAREST)
        canvas.alpha_composite(spr, (x + (T - size) // 2 + dx, y + (T - size) // 2 + dy))

    def _item_name(self, item):
        if item is None:
            return None
        if item[0] == "raw":
            return item[1]
        if item[0] == "chopped":
            return item[1] + "-chopped"
        if item[1] == "dirty":
            return "dish-dirty"
        if item[1] == "soup":
            return dish_icon(tuple(item[2]))
        return "dish"

    def _kitchen(self, scene, snap, running, t):
        lay = scene.lay
        canvas = Image.new("RGBA", (lay.W * T, lay.H * T), (0, 0, 0, 255))
        d = ImageDraw.Draw(canvas)
        burn = scene.params.get("burn")
        for y in range(lay.H):
            for x in range(lay.W):
                canvas.alpha_composite(self.sp.get("floor"), (x * T, y * T))
                k = scene.kind.get((x, y))
                if k is None:
                    if not lay.is_floor((x, y)):
                        canvas.alpha_composite(self.sp.get("counter"), (x * T, y * T))
                    continue
                canvas.alpha_composite(self.sp.get(FLOOR_STATION[k]), (x * T, y * T))
        for c, item in snap["surface"].items():
            name = self._item_name(item)
            px, py = c[0] * T, c[1] * T
            if name:
                self._paste_item(canvas, name, px, py)
            if item is not None and item[0] == "raw" and snap["progress"].get(c):
                self._bar(d, px, py, snap["progress"][c] / scene.params["chop"][item[1]], (240, 200, 40))
        for c, pot in snap["pots"].items():
            px, py = c[0] * T, c[1] * T
            comp = tuple(pot["contents"].count(i) for i in INGREDIENTS)
            st = _effective(pot, t, burn)
            spr = self.sp.get(pot_sprite(comp, "dirty" if st == "spoiled" else st))
            if st == "spoiled":
                spr = _tint(spr, (130, 210, 110))
            canvas.alpha_composite(spr, (px, py))
            if st == "cooking" and pot["ready_at"] is not None and pot["cook_start"] is not None:
                span = max(1e-9, pot["ready_at"] - pot["cook_start"])
                self._bar(d, px, py, (t - pot["cook_start"]) / span, (60, 200, 80))
            elif st == "ready" and burn is not None and pot["ready_at"] is not None:
                self._bar(d, px, py, (pot["ready_at"] + burn - t) / burn, (230, 40, 40))
            elif st in ("dirty", "burnt", "spoiled") and pot["scrub"]:
                self._bar(d, px, py, pot["scrub"] / scene.params["scrub_pot"], (160, 110, 60))
        for c, plates in snap["stack"].items():
            if not plates:
                continue
            px, py = c[0] * T, c[1] * T
            name = "dish-dirty" if scene.kind[c] == "dropoff" else "dish"
            self._paste_item(canvas, name, px, py, 11, 0, -1)
            d.text((px + 10, py + 8), str(len(plates)), fill=(255, 255, 255, 255), font=self.f_t)
        for c, st in snap["sink"].items():
            px, py = c[0] * T, c[1] * T
            if st is not None:
                self._paste_item(canvas, "dish" if st == "clean" else "dish-dirty", px, py, 9, 0, 1)
                if st == "dirty" and snap["progress"].get(c):
                    self._bar(d, px, py, snap["progress"][c] / scene.params["wash_plate"], (90, 160, 240))
        eat = scene.params["eat"]
        for pid, ret in snap["diners"].items():
            c = snap.get("diner_cells", {}).get(pid)
            if c is not None:
                self._bar(d, c[0] * T, c[1] * T, 1 - (ret - t) / eat, (235, 150, 190), row=1)
        poses = _cook_poses(scene, snap, running, t)
        order = sorted(range(len(poses)), key=lambda i: poses[i][1])
        for i in order:
            self._chef(canvas, i, snap["players"][i]["held"], *poses[i])
        return canvas

    def _chef(self, canvas, i, held, px, py, facing):
        face = FACE[facing]
        over, name = None, face
        if held is not None:
            if held[0] == "raw" and held[1] in ("onion", "tomato"):
                name = "{}-{}".format(face, held[1])
            elif held[0] == "raw":
                over = held[1]
            elif held[0] == "chopped":
                over = held[1] + "-chopped"
            elif held[1] == "clean":
                name = face + "-dish"
            elif held[1] == "dirty":
                name, over = face + "-dish", "dish-dirty"
            else:
                comp = tuple(held[2])
                if comp[2] == 0 and comp[0] and not comp[1]:
                    name = face + "-soup-onion"
                elif comp[2] == 0 and comp[1] and not comp[0]:
                    name = face + "-soup-tomato"
                else:
                    name, over = face + "-dish", dish_icon(comp)
        canvas.alpha_composite(self.sp.get(name), (px, py))
        if over is not None:
            self._paste_item(canvas, over, px, py, 9, facing[0] * 4, 2 + facing[1] * 3)
        canvas.alpha_composite(self.sp.get("{}-{}hat".format(face, HATS[i % len(HATS)])), (px, py))

    # ---- HUD and timeline ---------------------------------------------------------------------------
    def hud_compact(self, scene, snap, t, width):
        """The HUD with large text: a status line, then one line per order (dish icon, id, status), same height
        in every frame."""
        items, icons, fs = [], [], self.hud_text
        f = _font(fs)
        pad, ic = int(fs * 0.4), int(fs * 1.45)
        head = "t = {:g}    served {}/{}    late {}    burnt {}".format(round(t), len(snap["served"]), len(scene.orders),
                                                                  len(snap["late"]), snap["burnt"])
        items.append((head, pad, int(fs * 0.3), f, (20, 20, 20)))
        y = int(fs * 1.7)
        cols = self.cols or 3
        cw, row = width // cols, int(fs * 1.45)
        rows = math.ceil(len(scene.orders) / cols)
        for k, o in enumerate(scene.orders):
            c, r = divmod(k, rows)
            x, yy = pad + c * cw, y + r * row
            done, dl = snap["served"].get(o["id"]), o.get("deadline")
            if done is not None:
                late_by = snap["late"].get(o["id"])
                mark, col = ("late +{:g}".format(round(late_by)), (200, 30, 30)) if late_by else (
                    "served {:g}".format(round(done)), (30, 140, 50))
            elif dl is not None and t > dl:
                mark, col = "overdue", (200, 30, 30)
            else:
                mark, col = ("due {:g}".format(dl) if dl is not None else "waiting"), (110, 110, 110)
            icons.append((x, yy + (row - ic) // 2 - int(fs * 0.1), dish_icon(tuple(o["comp"])), ic))
            items.append(("{} {}".format(o["id"], mark), x + ic + int(fs * 0.3), yy, f, col))
        return y + rows * row + pad, items, icons

    def hud_items(self, scene, snap, t, width):
        """(height, text items [(text, x, y, font, colour)], icons [(x, y, sprite)]) of the HUD at `width` px."""
        if self.hud_text:
            return self.hud_compact(scene, snap, t, width)
        items, icons, k = [], [], self.k
        pad = int(8 * k)
        late = sum(snap["late"].values())
        head = "{}   t = {:g}   served {}/{}   late {}   total lateness {:g}   burnt {}   spoiled {}".format(
            scene.title, round(t, 2), len(snap["served"]), len(scene.orders), len(snap["late"]), round(late, 2),
            snap["burnt"], snap["spoiled"])
        y = int(6 * k)
        for line in _wrap(head, self.f_b, width - 2 * pad):
            items.append((line, pad, y, self.f_b, (20, 20, 20)))
            y += int(18 * k)
        y += int(4 * k)
        cols = self.cols or max(1, width // 300)
        cw = width // cols
        blocks = []
        for o in scene.orders:
            done, dl = snap["served"].get(o["id"]), o.get("deadline")
            if done is not None:
                late_by = snap["late"].get(o["id"])
                mark, col = ("LATE +{:g} (served {:g})".format(late_by, done), (200, 30, 30)) if late_by else (
                    "served {:g}".format(done), (30, 140, 50))
            else:
                mark, col = ("due {}".format(dl) if dl is not None else "pending"), (80, 80, 80)
                if dl is not None and t > dl:
                    mark, col = "OVERDUE (due {})".format(dl), (200, 30, 30)
            tw = cw - int(34 * k) - pad
            blocks.append(([(ln, (20, 20, 20)) for ln in _wrap("{} {}".format(o["id"], o.get("dish", "")), self.f_s, tw)]
                           + [(ln, col) for ln in _wrap(mark, self.f_s, tw)], dish_icon(tuple(o["comp"]))))
        for r in range(0, len(blocks), cols):
            row = blocks[r:r + cols]
            for c, (lines, icon) in enumerate(row):
                x = pad + c * cw
                icons.append((x, y, icon))
                for j, (ln, col) in enumerate(lines):
                    items.append((ln, x + int(34 * k), y + int(14 * k) * j, self.f_s, col))
            y += max(int(36 * k), int(14 * k) * max(len(b[0]) for b in row) + int(4 * k))
        y += 2
        for i, c in enumerate(scene.cooks):
            for ln in _wrap("{}: {}".format(c, snap.get("doing", {}).get(c, "idle")), self.f_s, width - 2 * pad):
                items.append((ln, pad, y, self.f_s, COOK_COLORS[i % 4]))
                y += int(15 * k)
        return y + int(6 * k), items, icons

    def _hud(self, scene, snap, t, width):
        h, items, icons = self.hud_items(scene, snap, t, width)
        img = Image.new("RGB", (width, h), (245, 240, 230))
        d = ImageDraw.Draw(img)
        for x, y, name, *size in icons:
            icon = self.sp.get(name).resize(((size[0] if size else int(30 * self.k)),) * 2, Image.NEAREST)
            img.paste(icon, (x, y), icon)
        for text, x, y, font, col in items:
            d.text((x, y), text, font=font, fill=col)
        return img

    def _lanes(self, sim, tmax):
        """[(label, [(t0, t1, category)])] one lane per cook and per pot, from sim.log."""
        log = [e for e in sim.log if e["t"] <= tmax or e["type"] == "start"]
        lanes = []
        for c in sim.cooks:
            segs, open_ = [], {}
            for e in log:
                if e.get("cook") != c:
                    continue
                if e["type"] in ("start", "yield"):
                    open_ = {"t": e["t"], "action": e.get("action"), "end": e.get("end")}
                elif e["type"] in ("end", "task_failed", "abort") and open_:
                    cat = "failed" if e["type"] == "task_failed" else CATEGORY.get(open_["action"], "move")
                    segs.append((open_["t"], e["t"], cat))
                    open_ = {}
            if open_:
                segs.append((open_["t"], tmax if open_["end"] is None else min(open_["end"], tmax),
                             CATEGORY.get(open_["action"], "move")))
            lanes.append((c, [s for s in segs if s[0] < tmax]))
        for cell in sorted(sim.world.pots, key=lambda c: (c[1], c[0])):
            ev = sorted([(e["t"], e["type"], e) for e in sim.log if e.get("cell") == cell
                         and e["type"] in ("end", "soup_ready", "burnt", "spoiled") and e["t"] <= tmax],
                        key=lambda x: x[0])
            segs, cur = [], None
            for when, typ, e in ev:
                if typ == "end" and e["event"] == "start":
                    cur = (when, "cooking")
                elif typ == "soup_ready" and cur:
                    segs.append((cur[0], when, "cooking"))
                    cur = (when, "ready")
                elif typ in ("burnt", "spoiled") and cur:
                    segs.append((cur[0], when, cur[1]))
                    cur = (when, typ)
                elif typ == "end" and e["event"] == "dish_up" and cur:
                    segs.append((cur[0], when, cur[1]))
                    cur = None
                elif typ == "end" and e["event"] == "scrub" and cur:
                    segs.append((cur[0], when, cur[1]))
                    cur = None
            if cur:
                segs.append((cur[0], tmax, cur[1]))
            lanes.append(("pot ({},{})".format(*cell), [s for s in segs if s[0] < tmax]))
        return lanes

    def timeline_items(self, lanes, width):
        """(height, text items, legend swatches [(x, y, key)]) of the timeline panel; legend wraps at `width`."""
        lane_h, top = 18, 30
        items = [("Timeline: one lane per cook and pot", 8, 6, self.f_b, (20, 20, 20))]
        for k, (label, _) in enumerate(lanes):
            items.append((label, 8, top + k * lane_h + 2, self.f_s, (30, 30, 30)))
        x, y = 8, top + lane_h * len(lanes) + 8
        swatches = []
        for key in ("move", "carry", "chop", "add", "start", "plate", "deliver", "wash", "scrub", "cooking", "ready",
                    "burnt", "failed"):
            w = 12 + self.f_s.getlength(key)
            if x + w > width - 8 and x > 8:
                x, y = 8, y + 16
            swatches.append((x, y, key))
            items.append((key, x + 12, y - 2, self.f_s, (40, 40, 40)))
            x += int(w) + 12
        return y + 22, items, swatches

    def _timeline(self, sim, tmax, width, horizon=None):
        lanes = self._lanes(sim, tmax)
        lane_h, x0, top = 18, 90, 30
        h, items, swatches = self.timeline_items(lanes, width)
        img = Image.new("RGB", (width, h), (255, 255, 255))
        d = ImageDraw.Draw(img)
        span = max(1e-9, horizon if horizon else max(tmax, 1))
        sc = (width - x0 - 10) / span
        for k, (label, segs) in enumerate(lanes):
            y = top + k * lane_h
            d.rectangle([x0, y + 1, width - 10, y + lane_h - 2], fill=(248, 248, 248), outline=(220, 220, 220))
            for a, b, cat in segs:
                d.rectangle([x0 + int(a * sc), y + 2, max(x0 + int(a * sc) + 1, x0 + int(b * sc)), y + lane_h - 3],
                            fill=ACT_COLOR.get(cat, (180, 180, 180)))
        cx = x0 + int(min(tmax, span) * sc)
        d.line([cx, top - 2, cx, top + lane_h * len(lanes)], fill=(220, 30, 30), width=2)
        for x, y, key in swatches:
            d.rectangle([x, y, x + 9, y + 9], fill=ACT_COLOR[key], outline=(120, 120, 120))
        for text, x, y, font, col in items:
            d.text((x, y), text, font=font, fill=col)
        return img

    # ---- composition -----------------------------------------------------------------------------------
    def compose(self, scene, snap, running, t, hud=True, timeline_img=None):
        kit = self._kitchen(scene, snap, running, t)
        kit = kit.resize((kit.width * self.scale, kit.height * self.scale), Image.NEAREST).convert("RGB")
        parts = [kit]
        if hud:
            parts.append(self._hud(scene, snap, t, kit.width))
        if timeline_img is not None:
            parts.append(timeline_img)
        out = Image.new("RGB", (kit.width, sum(p.height for p in parts)), (245, 240, 230))
        y = 0
        for p in parts:
            out.paste(p, (0, y))
            y += p.height
        return out


def _doing(sim):
    return {c: "{} @ {}".format(r.task.action, r.task.station) for c, r in sim.running.items()}


def frame(sim, hud=True, timeline=False, scale=4):
    """One image of the sim at its current time (any clock; also a bare Sim at t=0)."""
    rd = Renderer(scale)
    scene = _Scene(sim)
    snap = sim.world.snapshot()
    snap["doing"] = _doing(sim)
    tl = rd._timeline(sim, sim.t, scene.lay.W * T * scale) if timeline else None
    return rd.compose(scene, snap, _running_view(sim), sim.t, hud, tl)


def gif(sim, path, fps=8, scale=3, cols=None, hud_text=None):
    """Animated GIF of a recorded run (Sim(..., record=True)); chefs glide between decision points; `cols`: order
    columns in the HUD (default: one per 300 px); `hud_text`: font size of a compact, large-text HUD."""
    if not sim.frames:
        raise ValueError("gif() needs recorded frames: construct Sim(..., record=True) and run it first")
    rd = Renderer(scale, cols, hud_text)
    scene = _Scene(sim)
    frames = []
    for k, fr in enumerate(sim.frames):
        end = sim.frames[k + 1]["t"] if k + 1 < len(sim.frames) else fr["t"]
        n = 1 if end <= fr["t"] else int(min(20, max(1, math.ceil(end - fr["t"]))))
        snap = dict(fr["world"])
        snap["doing"] = {c: "{} @ {}".format(r["action"], r["station"]) for c, r in fr["running"].items()}
        for j in range(n):
            t = fr["t"] + (end - fr["t"]) * j / n
            frames.append(rd.compose(scene, snap, fr["running"], t, True))
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    dur = [int(1000 / fps)] * (len(pal) - 1) + [2000]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=dur, loop=0)
    return path


def video(sim, path, fps=24, speed=3.0, scale=4, cols=None, hold=3.0, hud_text=None):
    """MP4 video of a recorded run (Sim(..., record=True)) at `speed` simulated seconds per second of video, chefs
    moving smoothly; the last frame is held `hold` seconds. Needs imageio-ffmpeg (pip install imageio-ffmpeg)."""
    import imageio_ffmpeg
    import numpy as np
    if not sim.frames:
        raise ValueError("video() needs recorded frames: construct Sim(..., record=True) and run it first")
    rd = Renderer(scale, cols, hud_text)
    scene = _Scene(sim)
    snaps = []
    for fr in sim.frames:
        snap = dict(fr["world"])
        snap["doing"] = {c: "{} @ {}".format(r["action"], r["station"]) for c, r in fr["running"].items()}
        snaps.append(snap)
    first = rd.compose(scene, snaps[0], sim.frames[0]["running"], sim.frames[0]["t"], True)
    W = first.width + first.width % 2
    kit_h = scene.lay.H * T * scale
    t_end = sim.frames[-1]["t"]
    hud_h = max(rd.hud_items(scene, s, t, first.width)[0] for s, fr in zip(snaps, sim.frames)
                for t in (fr["t"], t_end))
    H = kit_h + hud_h + 40
    H += H % 2
    w = imageio_ffmpeg.write_frames(path, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=7,
                                    macro_block_size=2)
    w.send(None)
    k, n = 0, int(math.ceil(t_end / speed * fps))
    for i in range(n + 1 + int(hold * fps)):
        t = min(t_end, i * speed / fps)
        while k + 1 < len(sim.frames) and sim.frames[k + 1]["t"] <= t:
            k += 1
        img = rd.compose(scene, snaps[k], sim.frames[k]["running"], t, True)
        canvas = Image.new("RGB", (W, H), (245, 240, 230))
        canvas.paste(img.crop((0, 0, min(img.width, W), min(img.height, H))), (0, 0))
        w.send(np.ascontiguousarray(np.asarray(canvas)))
    w.close()
    return path


def gantt(sim, path, scale=4):
    """PNG timeline of the run: one lane per cook and per pot, from sim.log."""
    rd = Renderer(scale)
    tmax = max([sim.t] + [e["t"] for e in sim.log] + [e["end"] for e in sim.log if e.get("end")])
    img = rd._timeline(sim, tmax, 1000, horizon=max(tmax, 1))
    img.save(path)
    return path


def text_boxes(sim, timeline=True, scale=4):
    """Laid-out HUD and timeline text as [(text, x0, y0, x1, y1)] plus the frame width (for layout checks)."""
    rd = Renderer(scale)
    scene = _Scene(sim)
    width = scene.lay.W * T * scale
    snap = sim.world.snapshot()
    snap["doing"] = _doing(sim)
    _, items, _ = rd.hud_items(scene, snap, sim.t, width)
    if timeline:
        _, more, _ = rd.timeline_items(rd._lanes(sim, sim.t), width)
        items = items + more
    return [(t, x, y, x + f.getlength(t), y + f.size) for t, x, y, f, _ in items], width
