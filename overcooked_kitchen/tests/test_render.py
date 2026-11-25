"""Tests for the sprite renderer: frames, extra sprites, GIF and Gantt output."""
"""This file is UN-VERIFIED"""
import json
import os

import pytest
from PIL import Image

from overcooked_kitchen import render
from overcooked_kitchen.baselines import make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.layout import Layout

EXTRA = ["carrot", "carrots", "board", "onion-chopped", "tomato-chopped", "carrot-chopped", "sink", "dropoff",
         "dish-dirty", "pot-dirty", "soup-carrot-1-cooking", "soup-carrot-2-cooking", "soup-carrot-3-cooking",
         "soup-carrot-cooked", "soup-carrot-dish"]


def test_frame_size_k02():
    sim = Sim("k02_two_dishes")
    lay = Layout(sim.scenario["grid"])
    img = render.frame(sim)
    assert img.width == lay.W * 15 * 4
    assert img.height >= lay.H * 15 * 4


def test_all_extra_sprites_exist():
    with open(os.path.join(render.ASSETS, "extra.json")) as f:
        frames = json.load(f)["frames"]
    names = {n[:-4] for n in frames}
    assert set(EXTRA) <= names
    sp = render.Sprites()
    for n in EXTRA:
        assert sp.get(n).size == (15, 15)


def test_gif_writes(tmp_path):
    sim = Sim("m01_three_cooks", record=True)
    sim.run(make("dispatcher"))
    p = render.gif(sim, str(tmp_path / "run.gif"))
    with Image.open(p) as im:
        assert im.n_frames > 10


def test_gif_needs_record():
    with pytest.raises(ValueError, match="record=True"):
        render.gif(Sim("k01_one_soup"), "x.gif")


def test_burnt_pot_uses_explosion_sprite():
    sim = Sim("k01_one_soup")
    cell = next(iter(sim.world.pots))
    before = render.frame(sim, hud=False)
    sim.world.pots[cell].state = "burnt"
    after = render.frame(sim, hud=False)
    box = (cell[0] * 60, cell[1] * 60, cell[0] * 60 + 60, cell[1] * 60 + 60)
    boom = render.Sprites().img("pot-explosion")
    ref = Image.new("RGBA", (60, 60))
    ref.alpha_composite(render.Sprites().img("floor"))
    ref.alpha_composite(render.Sprites().img("pot"))
    ref.alpha_composite(boom)
    assert after.crop(box).tobytes() == ref.convert("RGB").tobytes()
    assert after.crop(box).tobytes() != before.crop(box).tobytes()


@pytest.mark.parametrize("clock", ["sampled", "grid"])
def test_frame_mid_run_and_timeline(clock, tmp_path):
    sim = Sim("m01_three_cooks", clock=clock)
    pol = make("dispatcher")
    for _ in range(12):
        c = sim.next_decision()
        if c is None:
            break
        sim.act(c, pol(sim, c, sim.legal(c)))
    img = render.frame(sim, timeline=True)
    assert img.height > Layout(sim.scenario["grid"]).H * 60
    sim.run(pol)
    out = render.gantt(sim, str(tmp_path / "g.png"))
    assert os.path.getsize(out) > 0


@pytest.mark.parametrize("name", ["k01_one_soup", "k02_two_dishes", "k03_service_cleanup", "k04_burn_risk",
                                  "m02_four_cooks"])
def test_hud_text_not_clipped(name):
    sim = Sim(name)
    boxes, width = render.text_boxes(sim)
    assert boxes
    for text, x0, y0, x1, y1 in boxes:
        assert x0 >= 0 and x1 <= width, (text, x1, width)
    sim.world.snapshot()
    img = render.frame(sim, timeline=True)
    assert img.width == width


def test_wrap_splits_long_text():
    f = render._font(14)
    text = "word " * 30 + "x" * 80
    lines = render._wrap(text, f, 200)
    assert len(lines) > 3
    assert all(f.getlength(ln) <= 200 for ln in lines)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_narrow_hud_wraps_status():
    rd = render.Renderer()
    sim = Sim("k02_two_dishes")
    snap = sim.world.snapshot()
    snap["doing"] = {}
    h, items, _ = rd.hud_items(render._Scene(sim), snap, 0, 200)
    assert len([i for i in items if i[2] < 60 and i[1] == 8]) >= 2
    assert all(x + f.getlength(t) <= 200 for t, x, y, f, _ in items)


def _opaque(img):
    return {(x, y): img.getpixel((x, y)) for x in range(img.width) for y in range(img.height)
            if img.getpixel((x, y))[3] == 255}


def test_hat_colours_per_cook():
    sim = Sim("m02_four_cooks")
    img = render.frame(sim, hud=False).convert("RGBA")
    sp = render.Sprites()
    seen = set()
    for i, p in enumerate(sim.world.players):
        hat = sp.get("NORTH-{}hat".format(render.HATS[i]))
        px = _opaque(hat)
        assert px
        for (x, y), col in px.items():
            assert img.getpixel((p.pos[0] * 60 + x * 4 + 2, p.pos[1] * 60 + y * 4 + 2)) == col
        seen.add(next(iter(px.values())))
    assert len(seen) == 4


@pytest.mark.parametrize("held,over", [(("raw", "carrot"), "carrot"), (("chopped", "onion"), "onion-chopped"),
                                        (("plate", "dirty", None), "dish-dirty"),
                                        (("plate", "soup", (0, 0, 1)), "soup-carrot-dish")])
def test_held_item_overlay(held, over):
    rd = render.Renderer()
    bare = Image.new("RGBA", (15, 15), (0, 0, 0, 0))
    with_item = bare.copy()
    rd._chef(bare, 0, None, 0, 0, render.EAST)
    rd._chef(with_item, 0, held, 0, 0, render.EAST)
    spr = rd.sp.get(over)
    colours = {c for c in _opaque(spr).values()}
    changed = [with_item.getpixel(k) for k in _opaque(with_item) if bare.getpixel(k) != with_item.getpixel(k)]
    assert changed
    assert any(c in colours for c in changed)


def test_held_raw_onion_uses_chef_frame():
    rd = render.Renderer()
    img = Image.new("RGBA", (15, 15), (0, 0, 0, 0))
    rd._chef(img, 0, ("raw", "onion"), 0, 0, render.EAST)
    exp = rd.sp.get("EAST-onion").copy()
    exp.alpha_composite(rd.sp.get("EAST-bluehat"))
    assert img.tobytes() == exp.tobytes()


def test_spoiled_pot_is_tinted_dirty_pot():
    sim = Sim("k01_one_soup")
    cell = next(iter(sim.world.pots))
    box = (cell[0] * 60, cell[1] * 60, cell[0] * 60 + 60, cell[1] * 60 + 60)
    sim.world.pots[cell].state = "dirty"
    dirty = render.frame(sim, hud=False).crop(box)
    sim.world.pots[cell].state = "spoiled"
    spoiled = render.frame(sim, hud=False).crop(box)
    assert spoiled.tobytes() != dirty.tobytes()
    sp = render.Sprites()
    ref = Image.new("RGBA", (60, 60))
    ref.alpha_composite(sp.img("floor"))
    ref.alpha_composite(sp.img("pot"))
    tinted = render._tint(sp.get("pot-dirty"), (130, 210, 110)).resize((60, 60), Image.NEAREST)
    ref.alpha_composite(tinted)
    assert spoiled.tobytes() == ref.convert("RGB").tobytes()
