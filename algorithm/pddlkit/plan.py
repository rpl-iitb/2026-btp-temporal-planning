"""Temporal plans: parsing and the quantities every stage reports.

Definitions (used consistently in all results):
  makespan            = max_i (start_i + duration_i)  -- wall-clock length of the schedule
  total_action_time   = sum_i duration_i              -- work done; >= makespan / #parallel lanes
  num_actions         = number of timed actions in the plan
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

_LINE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*:\s*\(([^)]*)\)\s*(?:\[\s*([0-9]*\.?[0-9]+)\s*\])?")


@dataclass(frozen=True)
class TimedAction:
    start: float
    name: str
    args: Tuple[str, ...]
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    def to_line(self) -> str:
        return "{:.3f}: ({}) [{:.3f}]".format(self.start, " ".join((self.name,) + self.args), self.duration)


@dataclass
class TemporalPlan:
    actions: List[TimedAction]

    @classmethod
    def parse(cls, text: str) -> "TemporalPlan":
        acts = []
        for line in text.splitlines():
            if line.strip().startswith(";"):
                continue
            m = _LINE.match(line)
            if m:
                toks = m.group(2).lower().split()
                acts.append(TimedAction(float(m.group(1)), toks[0], tuple(toks[1:]),
                                        float(m.group(3)) if m.group(3) else 0.0))
        return cls(sorted(acts, key=lambda a: (a.start, a.name, a.args)))

    def to_text(self) -> str:
        return "\n".join(a.to_line() for a in self.actions) + "\n"

    @property
    def makespan(self) -> float:
        return max((a.end for a in self.actions), default=0.0)

    @property
    def total_action_time(self) -> float:
        return sum(a.duration for a in self.actions)

    def __len__(self):
        return len(self.actions)

    def objects_used(self) -> set:
        return {x for a in self.actions for x in a.args}
