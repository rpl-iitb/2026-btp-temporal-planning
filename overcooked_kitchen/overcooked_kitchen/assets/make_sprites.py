"""Draws the sprites Overcooked-AI lacks (carrot, board, sink, dirty things...) into assets/extra.png + extra.json.

Same 15x15 pixel style and palette as the upstream sheets (colours are sampled from them). Run once: python make_sprites.py
"""
"""This file is UN-VERIFIED"""
import json
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
T = 15
ORANGE, ORANGE_DK, LEAF = (237, 145, 33, 255), (200, 100, 10, 255), (0, 102, 0, 255)
BEIGE = (204, 204, 153)


def _sheet(name):
    with open(os.path.join(HERE, name + ".json")) as f:
        data = json.load(f)
    return Image.open(os.path.join(HERE, name + ".png")).convert("RGBA"), data["frames"]


def _crop(sheet, frames, name):
    f = frames[name]["frame"]
    return sheet.crop((f["x"], f["y"], f["x"] + T, f["y"] + T))


def _blank():
    return Image.new("RGBA", (T, T), (0, 0, 0, 0))


def _px(img, pts, col):
    for x, y in pts:
        img.putpixel((x, y), col)


def _carrot(img, ox=0, oy=0, size=1.0):
    d = ImageDraw.Draw(img)
    if size >= 1:
        _px(img, [(6 + ox, 2 + oy), (7 + ox, 3 + oy), (8 + ox, 2 + oy), (7 + ox, 2 + oy)], LEAF)
        for k, w in enumerate((5, 5, 4, 4, 3, 3, 2, 1)):
            x0 = 7 - w // 2 + ox
            d.line([x0, 4 + k + oy, x0 + w - 1, 4 + k + oy], fill=ORANGE)
        _px(img, [(6 + ox, 6 + oy), (8 + ox, 8 + oy)], ORANGE_DK)
    else:
        _px(img, [(1 + ox, oy), (ox, oy)], LEAF)
        _px(img, [(ox, 1 + oy), (1 + ox, 1 + oy), (2 + ox, 1 + oy), (ox, 2 + oy), (1 + ox, 2 + oy), (1 + ox, 3 + oy)],
            ORANGE)
        _px(img, [(2 + ox, 1 + oy)], ORANGE_DK)
    return img


def _chopped(col, dark):
    img = _blank()
    d = ImageDraw.Draw(img)
    for x, y in ((3, 4), (8, 3), (5, 8), (10, 8)):
        d.rectangle([x, y, x + 2, y + 2], fill=col)
        d.line([x, y + 2, x + 2, y + 2], fill=dark)
    return img


def _recolour(img):
    out = img.copy()
    for y in range(T):
        for x in range(T):
            r, g, b, a = out.getpixel((x, y))
            if a and abs(r - g) < 8 and r - b > 25:
                k = r / 204.0
                out.putpixel((x, y), (min(255, int(237 * k)), min(255, int(145 * k)), int(33 * k), a))
    return out


def build():
    terrain, tf = _sheet("terrain")
    objects, of = _sheet("objects")
    counter = _crop(terrain, tf, "counter.png")
    pot = _crop(terrain, tf, "pot.png")
    grey = pot.getpixel((2, 12))
    dark = pot.getpixel((4, 4))
    dish = _crop(objects, of, "dish.png")
    out = {}

    out["carrot"] = _carrot(_blank())

    crate = counter.copy()
    for ox, oy in ((1, 1), (5, 1), (9, 1), (3, 7), (8, 7)):
        _carrot(crate, ox + 1, oy, 0.5)
    out["carrots"] = crate

    board = counter.copy()
    d = ImageDraw.Draw(board)
    d.rectangle([2, 3, 12, 11], fill=(222, 184, 120, 255), outline=(180, 130, 60, 255))
    d.line([4, 5, 9, 5], fill=(196, 150, 84, 255))
    d.line([4, 9, 10, 9], fill=(196, 150, 84, 255))
    d.line([10, 4, 10, 7], fill=grey)
    d.line([10, 8, 10, 10], fill=(110, 70, 20, 255))
    out["board"] = board

    out["onion-chopped"] = _chopped(BEIGE + (255,), (150, 150, 112, 255))
    out["tomato-chopped"] = _chopped((255, 0, 0, 255), (160, 0, 0, 255))
    out["carrot-chopped"] = _chopped(ORANGE, ORANGE_DK)

    sink = Image.new("RGBA", (T, T), grey)
    d = ImageDraw.Draw(sink)
    d.rectangle([2, 4, 12, 12], fill=(60, 120, 200, 255), outline=dark)
    d.line([3, 6, 11, 6], fill=(150, 200, 250, 255))
    d.rectangle([6, 1, 8, 3], fill=(200, 200, 200, 255))
    d.point((7, 4), fill=(150, 200, 250, 255))
    out["sink"] = sink

    drop = counter.copy()
    d = ImageDraw.Draw(drop)
    d.rectangle([1, 8, 13, 13], fill=grey, outline=dark)
    for k, y in enumerate((9, 7, 5)):
        d.ellipse([3 + k, y, 11 - k, y + 3], fill=(215, 205, 190, 255), outline=(150, 120, 80, 255))
    out["dropoff"] = drop

    dd = dish.copy()
    for y in range(T):
        for x in range(T):
            r, g, b, a = dd.getpixel((x, y))
            if a:
                dd.putpixel((x, y), (int(r * 0.85), int(g * 0.8), int(b * 0.65), a))
    _px(dd, [(5, 6), (9, 5), (8, 9), (6, 8), (10, 8)], (110, 70, 30, 255))
    out["dish-dirty"] = dd

    pd = pot.copy()
    d = ImageDraw.Draw(pd)
    d.rectangle([3, 6, 11, 9], fill=(90, 60, 20, 255))
    _px(pd, [(4, 7), (7, 7), (9, 8), (5, 9)], (60, 90, 30, 255))
    out["pot-dirty"] = pd

    for n in (1, 2, 3):
        out["soup-carrot-{}-cooking".format(n)] = _recolour(_crop(objects, of, "soup-onion-{}-cooking.png".format(n)))
    out["soup-carrot-cooked"] = _recolour(_crop(objects, of, "soup-onion-cooked.png"))
    out["soup-carrot-dish"] = _recolour(_crop(objects, of, "soup-onion-dish.png"))
    return out


def main():
    frames = build()
    names = list(frames)
    sheet = Image.new("RGBA", (1 + 17 * len(names), 17), (0, 0, 0, 0))
    meta = {}
    for k, n in enumerate(names):
        x = 1 + 17 * k
        sheet.paste(frames[n], (x, 1))
        meta[n + ".png"] = {"frame": {"x": x, "y": 1, "w": T, "h": T}, "rotated": False, "trimmed": False,
                            "spriteSourceSize": {"x": 0, "y": 0, "w": T, "h": T}, "sourceSize": {"w": T, "h": T}}
    sheet.save(os.path.join(HERE, "extra.png"))
    with open(os.path.join(HERE, "extra.json"), "w") as f:
        json.dump({"frames": meta, "meta": {"image": "extra.png", "size": {"w": sheet.width, "h": sheet.height}}},
                  f, indent=1)


if __name__ == "__main__":
    main()
