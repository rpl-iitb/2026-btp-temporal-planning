"""Human play: take any cook with the keyboard (AUTO / TASK / DRIVE per cook) while the others follow a policy.

Play is window-free and testable (handle_key with key names, update(dt)); main() shows it in a pygame window.
Grid clock only: update(dt) advances Sim.tick() at `rate` steps / s.
"""
"""This file is UN-VERIFIED"""
import argparse
import json
import sys
from typing import List, Optional

from overcooked_kitchen import grid
from overcooked_kitchen.baselines import make
from overcooked_kitchen.core import Sim
from overcooked_kitchen.layout import NORTH, SOUTH, EAST, WEST
from overcooked_kitchen.tasks import WAIT

ARROWS = {"up": NORTH, "down": SOUTH, "left": WEST, "right": EAST, "space": grid.INTERACT}
WASD = {"w": NORTH, "s": SOUTH, "a": WEST, "d": EAST, "e": grid.INTERACT}
MODE_KEYS = {"z": "auto", "t": "task", "v": "drive"}
MIN_RATE, MAX_RATE = 0.5, 64.0
HELP = ("1-4 cook  z AUTO  t TASK  v DRIVE  arrows+space / wasd+e drive  [ ] menu, Enter run  "
        "p pause  +/- speed  r restart  q quit")


class Play:
    def __init__(self, scenario, cooks=None, policy="dispatcher", rate=4.0, seed=0, record_path=None):
        """cooks: names (or 1-based numbers) of the cooks that start in DRIVE mode; the others are AUTO."""
        self.scenario, self.seed, self.policy_name = scenario, seed, policy
        self.rate, self.record_path = float(rate), record_path
        self.initial = [c if isinstance(c, str) else "c{}".format(c) for c in (cooks or [])]
        self._reset()

    def _reset(self):
        self.sim = Sim(self.scenario, clock="grid", seed=self.seed)
        self.policy = make(self.policy_name, seed=self.seed)
        for c in self.initial:
            self.sim.set_mode(c, "drive")
        self.selected = self.initial[0] if self.initial else self.sim.cooks[0]
        self.cursor = {c: None for c in self.sim.cooks}
        self._at = {c: 0 for c in self.sim.cooks}
        self.paused = False
        self.quit = False
        self.steps: List[dict] = []
        self._acc = 0.0
        self._saved = False

    @property
    def over(self) -> bool:
        return self.sim.success or self.sim.failure is not None

    # ---- keys -------------------------------------------------------------------------
    def menu(self, cook) -> List[str]:
        return ["{} @ {}".format(t.action, t.station) for t in self._tasks(cook)]

    def _tasks(self, cook):
        return [t for t in self.sim.legal(cook) if t != WAIT]

    def handle_key(self, key: str) -> None:
        sim, cook = self.sim, self.selected
        if key in ("1", "2", "3", "4"):
            if int(key) <= len(sim.cooks):
                self.selected = sim.cooks[int(key) - 1]
        elif key in MODE_KEYS:
            sim.set_mode(cook, MODE_KEYS[key])
        elif key in ARROWS:
            if sim.mode[cook] == "drive":
                sim.drive(cook, ARROWS[key])
        elif key in WASD:
            other = self._next_driven()
            if other is not None:
                sim.drive(other, WASD[key])
        elif key in ("[", "]"):
            n = len(self._tasks(cook))
            if n and sim.mode[cook] == "task":
                self._select(cook, (self.highlight(cook) + (1 if key == "]" else -1)) % n)
        elif key == "return":
            self._confirm(cook)
        elif key == "p":
            self.paused = not self.paused
        elif key == "+":
            self.rate = min(MAX_RATE, self.rate * 2)
        elif key == "-":
            self.rate = max(MIN_RATE, self.rate / 2)
        elif key == "r":
            self._reset()
        elif key == "q":
            self.quit = True

    def _next_driven(self) -> Optional[str]:
        cooks = self.sim.cooks
        k = cooks.index(self.selected)
        for c in cooks[k + 1:] + cooks[:k]:
            if self.sim.mode[c] == "drive":
                return c
        return None

    def highlight(self, cook) -> int:
        """Index of the highlighted menu entry: follows its label; if that is gone, the nearest remaining entry."""
        labels = self.menu(cook)
        if not labels:
            return 0
        if self.cursor[cook] in labels:
            self._at[cook] = labels.index(self.cursor[cook])
        else:
            self._at[cook] = min(self._at[cook], len(labels) - 1)
            self.cursor[cook] = labels[self._at[cook]]
        return self._at[cook]

    def _select(self, cook, k):
        self._at[cook] = k
        self.cursor[cook] = self.menu(cook)[k]

    def _confirm(self, cook):
        sim = self.sim
        if sim.mode[cook] != "task" or cook in sim.running:
            return
        label = self.cursor[cook]
        for t in self._tasks(cook):
            if "{} @ {}".format(t.action, t.station) == label:
                sim.act(cook, t)
                self.cursor[cook] = None
                return

    # ---- time -------------------------------------------------------------------------
    def update(self, dt: float) -> None:
        if self.paused or self.over:
            return
        self._acc += dt * self.rate
        while self._acc >= 1 and not self.over:
            self._acc -= 1
            self._step()
        if self.over:
            self._acc = 0.0
            self._finish()

    def _step(self):
        sim = self.sim
        for c in sim.pending():
            if sim.mode[c] == "auto":
                sim.act(c, self.policy(sim, c, sim.legal(c)))
        t, modes = sim.t, dict(sim.mode)
        sim.tick()
        names = {c: grid.ACTION_NAMES[grid.ACTIONS.index(a)] for c, a in zip(sim.cooks, sim.clock.last_actions)}
        self.steps.append({"t": t, "actions": names, "modes": modes})

    def _finish(self):
        if self.record_path and not self._saved:
            self.save(self.record_path)

    def save(self, path) -> None:
        sim = self.sim
        data = {"scenario": sim.scenario.get("scenario_id"), "seed": self.seed, "steps": self.steps,
                "metrics": sim.metrics(), "log": sim.log}
        with open(path, "w") as f:
            json.dump(data, f, default=str)
        self._saved = True

    # ---- display ----------------------------------------------------------------------
    def status(self) -> List[str]:
        sim = self.sim
        parts = ["{}{}:{}".format("*" if c == self.selected else "", c, sim.mode[c].upper()) for c in sim.cooks]
        head = "t={} rate={:g}/s {}  {}".format(sim.t, self.rate, "PAUSED" if self.paused else "", "  ".join(parts))
        if sim.success:
            head += "  SERVED ALL"
        elif sim.failure:
            head += "  ENDED: " + sim.failure
        lines = [head.strip()]
        if sim.mode[self.selected] == "task":
            m = self.menu(self.selected)
            if m:
                k = self.highlight(self.selected)
                lines.append("menu: " + "  ".join(("> " if i == k else "") + s for i, s in enumerate(m)))
        lines.append(HELP)
        return lines


def _key_name(event, pygame) -> str:
    name = pygame.key.name(event.key)
    aliases = {"enter": "return", "keypad enter": "return", "keypad +": "+", "=": "+", "keypad -": "-",
               "[+]": "+", "[-]": "-", "[enter]": "return"}
    if event.unicode in ("+", "-", "[", "]"):
        return event.unicode
    return aliases.get(name, name)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="overcooked_kitchen.play", description="Play a kitchen scenario.")
    ap.add_argument("--scenario", default="k01_one_soup")
    ap.add_argument("--cooks", type=int, default=1, help="number of cooks you drive (the others are AUTO)")
    ap.add_argument("--policy", default="dispatcher")
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", default=None, help="save the trajectory JSON here when the game ends")
    args = ap.parse_args(argv)
    try:
        import pygame
    except ImportError:
        sys.exit("human play needs pygame: pip install -e '.[play]'  (or pip install pygame)")
    from overcooked_kitchen.render import frame
    n = len(Sim(args.scenario).cooks)
    cooks = max(0, args.cooks)
    if cooks > n:
        print("--cooks {}: {} has {} cook(s); you drive {}".format(args.cooks, args.scenario, n, n))
        cooks = n
    play = Play(args.scenario, list(range(1, cooks + 1)), args.policy, args.rate, args.seed, args.record)
    pygame.init()
    font = pygame.font.SysFont("monospace", 14)
    img = frame(play.sim, hud=True, timeline=True)
    bar = 18 * 3 + 8
    screen = pygame.display.set_mode((img.width, img.height + bar))
    pygame.display.set_caption("overcooked_kitchen: " + str(args.scenario))
    clock = pygame.time.Clock()
    while not play.quit:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                play.quit = True
            elif e.type == pygame.KEYDOWN:
                play.handle_key(_key_name(e, pygame))
        play.update(clock.tick(30) / 1000.0)
        img = frame(play.sim, hud=True, timeline=True)
        screen.fill((20, 20, 20))
        screen.blit(pygame.image.frombytes(img.convert("RGB").tobytes(), img.size, "RGB"), (0, 0))
        for k, line in enumerate(play.status()):
            screen.blit(font.render(line[:img.width // 8], True, (230, 230, 230)), (6, img.height + 4 + 18 * k))
        pygame.display.flip()
    if play.record_path:
        play.save(play.record_path)
    pygame.quit()


if __name__ == "__main__":
    main()
