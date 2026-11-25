"""Procedural media for the Gazebo world: bevelled OBJ meshes (smooth edges, UVs in metres), detail textures per surface
type (paint, powder coat, brushed metal, wood, epoxy floor, rubber, cardboard, plaster), OGRE materials that tint them,
and sign / screen textures. Written next to the world so the world needs no extra install (needs Pillow + numpy).
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

KIND = {"floor": "floor", "aisle": "paint", "outside": "plaster", "wall": "plaster", "dado": "plaster", "col": "plaster", "alu": "brushed", "steel": "brushed",
        "dark": "powder", "dark2": "powder", "black": "powder", "wood": "wood", "card": "card", "esd": "rubber", "jeans": "fabric", "vest": "fabric"}
TILE = {"floor": 2.0, "plaster": 1.0, "brushed": 0.5, "powder": 0.5, "paint": 0.6, "wood": 0.8, "card": 0.5, "rubber": 0.4, "fabric": 0.3}


def _noise(n, sigma, rng):
    a = rng.normal(0, 1, (n, n)).astype(np.float32)
    im = Image.fromarray(((a - a.min()) / (np.ptp(a) + 1e-9) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    x = np.asarray(im, np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def texture(kind, n=512, seed=0):
    """Near-white detail texture (mean ~0.93) that the material's diffuse colour tints."""
    rng = np.random.default_rng(seed)
    if kind == "brushed":
        a = rng.normal(0, 1, (n, n)).astype(np.float32)
        a = np.asarray(Image.fromarray(((a - a.min()) / np.ptp(a) * 255).astype(np.uint8)).resize((n // 32, n)).resize((n, n), Image.BILINEAR), np.float32)
        a = (a - a.mean()) / (a.std() + 1e-6)
        v = 0.92 + 0.045 * a + 0.015 * _noise(n, 1, rng)
    elif kind == "wood":
        y = np.linspace(0, 1, n)[:, None]
        warp = 0.06 * _noise(n, 18, rng)
        rings = np.sin((y * 34 + warp * 8 + 0.15 * np.sin(np.linspace(0, 6, n))[None, :]) * math.pi)
        v = 0.86 + 0.07 * rings + 0.03 * _noise(n, 2, rng)
    elif kind == "floor":
        v = 0.94 + 0.012 * _noise(n, 40, rng) + 0.007 * _noise(n, 1.5, rng)
        for k in (0, n // 2):
            v[k:k + 1, :] -= 0.07
            v[:, k:k + 1] -= 0.07
    elif kind == "rubber":
        yy, xx = np.mgrid[:n, :n]
        v = 0.9 + 0.05 * (np.sin(xx / 3.0) * np.sin(yy / 3.0)) + 0.02 * _noise(n, 1, rng)
    elif kind == "card":
        xx = np.arange(n)[None, :]
        v = 0.9 + 0.035 * np.sin(xx / 2.5) + 0.04 * _noise(n, 6, rng)
    elif kind == "fabric":
        yy, xx = np.mgrid[:n, :n]
        v = 0.9 + 0.04 * np.sin(xx * 1.3) * np.sin(yy * 1.3) + 0.03 * _noise(n, 1, rng)
    elif kind == "plaster":
        v = 0.94 + 0.01 * _noise(n, 12, rng) + 0.006 * _noise(n, 1, rng)
    elif kind == "powder":
        v = 0.92 + 0.025 * _noise(n, 0.8, rng)
    else:
        v = 0.95 + 0.006 * _noise(n, 10, rng) + 0.004 * _noise(n, 0.7, rng)
    return Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def _font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def sign(text, sub="", fg=(25, 25, 25), bg=(250, 200, 20), w=1024, h=256):
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([8, 8, w - 9, h - 9], outline=fg, width=10)
    d.text((40, 30 if sub else 60), text, fill=fg, font=_font(120 if len(text) < 12 else 90))
    if sub:
        d.text((44, 170), sub, fill=fg, font=_font(52))
    return im


def screen(title, rng, w=512, h=320, accent=(40, 200, 120)):
    """HMI-style screen: title bar, status, bars and a trend line."""
    im = Image.new("RGB", (w, h), (12, 22, 34))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 44], fill=(20, 70, 120))
    d.text((12, 8), title, fill=(235, 240, 245), font=_font(26))
    d.rounded_rectangle([w - 120, 8, w - 12, 36], 8, fill=accent)
    d.text((w - 104, 10), "RUN", fill=(10, 20, 10), font=_font(22))
    for i in range(5):
        v = rng.uniform(0.3, 0.95)
        d.rectangle([16, 64 + i * 34, 16 + int((w / 2 - 40) * v), 88 + i * 34], fill=(60, 150, 220))
        d.text((w / 2 - 10, 62 + i * 34), "{:3.0f}%".format(v * 100), fill=(200, 210, 220), font=_font(20))
    pts = [(w / 2 + 50 + k * 8, 250 - 60 * math.sin(k / 3.0) * rng.uniform(0.6, 1.0) - 40) for k in range(26)]
    d.line(pts, fill=accent, width=3)
    d.rectangle([w / 2 + 40, 70, w - 16, 300], outline=(60, 80, 100), width=2)
    return im


class Media:
    """Collects meshes, textures and materials while the world is built; write() puts them in <out>/media."""

    def __init__(self, out):
        self.root = os.path.join(os.path.abspath(out), "media")
        self.meshes, self.mats, self.images = {}, {}, {}
        self.rng = np.random.default_rng(7)

    @property
    def scripts(self):
        return "<uri>file://{0}/materials/scripts</uri><uri>file://{0}/materials/textures</uri>".format(self.root)

    def material(self, rgb, kind, glow=False, unlit_image=None):
        key = (kind, tuple(round(c, 3) for c in rgb), glow, unlit_image)
        if key not in self.mats:
            self.mats[key] = "FS/{}_{}".format(kind, len(self.mats))
        return "<material><script>{}<name>{}</name></script></material>".format(self.scripts, self.mats[key])

    def image(self, name, im):
        self.images[name] = im
        return name

    def mesh(self, sx, sy, sz, tile, bevel=None):
        """Bevelled box OBJ (chamfered edges with smooth normals; UVs in metres / tile) -> file path."""
        r = bevel if bevel is not None else min(0.012, 0.18 * min(sx, sy, sz))
        key = (round(sx, 4), round(sy, 4), round(sz, 4), round(r, 4), tile)
        if key not in self.meshes:
            self.meshes[key] = os.path.join(self.root, "meshes", "b{}.obj".format(len(self.meshes)))
        return self.meshes[key]

    def write(self):
        for d in ("meshes", "materials/scripts", "materials/textures"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        for (sx, sy, sz, r, tile), path in self.meshes.items():
            open(path, "w").write(obj_bevel_box(sx / 2, sy / 2, sz / 2, r, tile))
        kinds = {k[0] for k in self.mats}
        for k in kinds:
            if k not in self.images:
                texture(k, seed=hash(k) % 1000).save(os.path.join(self.root, "materials/textures", "fs_{}.png".format(k)))
        for name, im in self.images.items():
            im.save(os.path.join(self.root, "materials/textures", "fs_{}.png".format(name)))
        out = []
        for (kind, rgb, glow, unlit), name in self.mats.items():
            tex = "fs_{}.png".format(unlit or kind)
            if unlit:
                out.append("material %s\n{\n technique\n {\n  pass\n  {\n   lighting off\n   texture_unit\n   {\n    texture %s\n    filtering trilinear\n   }\n  }\n }\n}\n" % (name, tex))
                continue
            em = " emissive {0} {1} {2} 1\n".format(*rgb) if glow else ""
            spec = {"brushed": "0.45 0.45 0.45 1 40", "paint": "0.25 0.25 0.25 1 24", "floor": "0.18 0.18 0.18 1 30", "wood": "0.08 0.08 0.08 1 8"}.get(kind, "0.1 0.1 0.1 1 10")
            out.append("material %s\n{\n technique\n {\n  pass\n  {\n   ambient %.3f %.3f %.3f 1\n   diffuse %.3f %.3f %.3f 1\n   specular %s\n%s   texture_unit\n   {\n    texture %s\n    filtering trilinear\n   }\n  }\n }\n}\n" % (name, rgb[0] * 0.7, rgb[1] * 0.7, rgb[2] * 0.7, rgb[0], rgb[1], rgb[2], spec, em, tex))
        open(os.path.join(self.root, "materials/scripts", "factorysim.material"), "w").write("\n".join(out))


def obj_bevel_box(a, b, c, r, tile):
    """OBJ text of a box with half extents (a, b, c) and chamfer r; each vertex carries its face normal, so the chamfers
    shade as rounded edges; UVs are world metres / tile, projected per face."""
    r = max(1e-4, min(r, a * 0.9, b * 0.9, c * 0.9))
    V, N, T, F = [], [], [], []

    def vert(p, axis, sign):
        n = [0.0, 0.0, 0.0]
        n[axis] = float(sign)
        u, v = [(p[1], p[2]), (p[0], p[2]), (p[0], p[1])][axis]
        V.append(p)
        N.append(n)
        T.append((u / tile, v / tile))
        return len(V)

    corner = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corner[(sx, sy, sz)] = (vert((sx * a, sy * (b - r), sz * (c - r)), 0, sx), vert((sx * (a - r), sy * b, sz * (c - r)), 1, sy), vert((sx * (a - r), sy * (b - r), sz * c), 2, sz))

    def quad(p, q, s, t, out):
        F.append((p, q, s) if not out else (p, s, q))
        F.append((p, s, t) if not out else (p, t, s))

    for axis in range(3):
        for sg in (-1, 1):
            cs = [k for k in corner if k[axis] == sg]
            o = [i for i in range(3) if i != axis]
            cs.sort(key=lambda k: math.atan2(k[o[1]], k[o[0]]))
            ids = [corner[k][axis] for k in cs]
            if (axis == 1) ^ (sg < 0):
                ids = ids[::-1]
            quad(ids[0], ids[1], ids[2], ids[3], False)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for si in (-1, 1):
            for sj in (-1, 1):
                def key(sk):
                    s = [0, 0, 0]
                    s[i], s[j], s[k] = si, sj, sk
                    return tuple(s)
                lo, hi = key(-1), key(1)
                p, q, s_, t = corner[lo][i], corner[hi][i], corner[hi][j], corner[lo][j]
                out = (si * sj > 0) ^ (k == 1)
                quad(p, q, s_, t, not out)
    for k, (ix, iy, iz) in corner.items():
        F.append((ix, iy, iz) if k[0] * k[1] * k[2] > 0 else (ix, iz, iy))
    G = []
    for f in F:
        p0, p1, p2 = (np.array(V[i - 1]) for i in f)
        g = np.cross(p1 - p0, p2 - p0)
        n = sum(np.array(N[i - 1]) for i in f)
        G.append(f if float(np.dot(g, n)) >= 0 else (f[0], f[2], f[1]))
    F = G
    lines = ["v {:.5f} {:.5f} {:.5f}".format(*p) for p in V] + ["vt {:.5f} {:.5f}".format(*t) for t in T] + ["vn {:.3f} {:.3f} {:.3f}".format(*n) for n in N]
    lines += ["f " + " ".join("{0}/{0}/{0}".format(i) for i in f) for f in F]
    return "\n".join(lines) + "\n"
