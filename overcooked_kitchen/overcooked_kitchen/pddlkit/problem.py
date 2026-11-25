"""High-level temporal planning problem (O, I, G, T of Pi = <D, O, I, G, T>).

This is the source of truth for every generated PDDL problem. Compared with
raw PDDL it keeps:
  * typed objects with benchmark annotations (`attrs`: e.g. a pot's grid
    cell), so graphs and executors never have to re-derive them;
  * deadlines as first-class goal attributes -- the PDDL generator decides how
    to encode them (timed initial literals, see DomainSpec.deadline_encoding);
  * explicit provenance (scenario, simulator commit, hashes).

Serialised as JSON (to_dict / from_dict). Ported from earlier research code.
"""
"""This file is UN-VERIFIED"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"

Fact = Tuple[str, ...]


@dataclass
class Obj:
    name: str
    type: str
    attrs: dict = field(default_factory=dict)


@dataclass
class Goal:
    fact: Fact
    deadline: Optional[float] = None   # latest simulator timestep at which the goal must hold


@dataclass
class TimedLiteral:
    time: float
    fact: Fact
    value: bool = True


@dataclass
class HLProblem:
    instance_id: str
    benchmark: str
    domain_name: str
    objects: List[Obj]
    facts: List[Fact]
    fluents: Dict[Tuple[str, ...], float]      # (function, arg1, ...) -> value
    goals: List[Goal]
    timed_literals: List[TimedLiteral] = field(default_factory=list)
    metric: str = "makespan"
    annotations: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # ---- access -------------------------------------------------------------------
    @property
    def object_types(self) -> Dict[str, str]:
        return {o.name: o.type for o in self.objects}

    def obj(self, name) -> Obj:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def of_type(self, ty, spec=None) -> List[str]:
        if spec is None:
            return sorted(o.name for o in self.objects if o.type == ty)
        return sorted(o.name for o in self.objects if spec.is_subtype(o.type, ty))

    def restrict(self, keep, instance_id=None) -> "HLProblem":
        """Pi|S: keep only objects in `keep` and every fact, fluent and timed
        literal whose object arguments are all kept (constants always stay).
        Goal objects must be kept."""
        names = {o.name for o in self.objects}
        keep = set(keep) & names
        if not self.goal_objects <= keep:
            raise ValueError("restriction drops goal objects {}".format(sorted(self.goal_objects - keep)))
        ok = lambda f: all(a in keep or a not in names for a in f[1:])  # noqa: E731
        return HLProblem(
            instance_id=instance_id or self.instance_id, benchmark=self.benchmark, domain_name=self.domain_name,
            objects=[o for o in self.objects if o.name in keep], facts=[f for f in self.facts if ok(f)],
            fluents={k: v for k, v in self.fluents.items() if ok(k)}, goals=list(self.goals),
            timed_literals=[t for t in self.timed_literals if ok(t.fact)], metric=self.metric,
            annotations=dict(self.annotations), provenance=dict(self.provenance, restricted_from=self.instance_id))

    def static_predicates(self, spec) -> set:
        return spec.static_predicates() - {t.fact[0] for t in self.timed_literals}

    @property
    def goal_objects(self) -> set:
        return {a for g in self.goals for a in g.fact[1:]}

    # ---- validation against the domain --------------------------------------------
    def validate(self, spec) -> "HLProblem":
        """Typing / arity / reference checks. Raises ValueError listing every
        problem; returns self so calls can be chained."""
        errors = []
        types = dict(spec.constants)
        for o in self.objects:
            if o.name in types:
                errors.append("duplicate object or constant name {}".format(o.name))
            if o.type not in spec.types and o.type != "object":
                errors.append("object {} has unknown type {}".format(o.name, o.type))
            types[o.name] = o.type

        def check(fact, table, where):
            if not fact or fact[0] not in table:
                errors.append("{}: undeclared {}".format(where, fact[0] if fact else "<empty>"))
                return
            sig = table[fact[0]]
            if len(sig) != len(fact) - 1:
                errors.append("{}: {} expects {} args, got {}".format(where, fact[0], len(sig), len(fact) - 1))
                return
            for arg, p in zip(fact[1:], sig):
                if arg not in types:
                    errors.append("{}: unknown object {} in {}".format(where, arg, fact))
                elif not spec.is_subtype(types[arg], p.type):
                    errors.append("{}: {} is {}, {} needs {}".format(where, arg, types[arg], fact[0], p.type))

        for f in self.facts:
            check(f, spec.predicates, "init")
        if len(set(self.facts)) != len(self.facts):
            errors.append("init: duplicate facts")
        for k, v in self.fluents.items():
            check(k, spec.functions, "fluent")
            if not isinstance(v, (int, float)):
                errors.append("fluent {} has non-numeric value {!r}".format(k, v))
        for g in self.goals:
            check(g.fact, spec.predicates, "goal")
            if g.deadline is not None and g.deadline <= 0:
                errors.append("goal {} has non-positive deadline".format(g.fact))
            if g.deadline is not None and not spec.deadline_encoding:
                errors.append("goal {} has a deadline but the domain declares no deadline_encoding".format(g.fact))
            if g.deadline is not None and spec.deadline_encoding and \
                    g.fact[0] != spec.deadline_encoding["goal_predicate"]:
                errors.append("deadline on {} but only {} goals can carry deadlines".format(
                    g.fact, spec.deadline_encoding["goal_predicate"]))
        for tl in self.timed_literals:
            check(tl.fact, spec.predicates, "timed literal")
            if tl.time <= 0:
                errors.append("timed literal {} at non-positive time".format(tl.fact))
        # durative-action durations must be defined for every fluent the domain reads
        if self.metric not in ("makespan",):
            errors.append("unsupported metric {}".format(self.metric))
        if errors:
            raise ValueError("problem {} invalid:\n  ".format(self.instance_id) + "\n  ".join(errors))
        return self

    # ---- (de)serialisation --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "benchmark": self.benchmark,
            "domain": self.domain_name,
            "objects": [{"name": o.name, "type": o.type, "attrs": o.attrs} for o in self.objects],
            "init": {
                "facts": [list(f) for f in sorted(self.facts)],
                "fluents": [{"function": k[0], "args": list(k[1:]), "value": v}
                            for k, v in sorted(self.fluents.items())],
            },
            "timed_literals": [{"time": t.time, "fact": list(t.fact), "value": t.value}
                               for t in sorted(self.timed_literals, key=lambda t: (t.time, t.fact))],
            "goals": [{"fact": list(g.fact), "deadline": g.deadline} for g in self.goals],
            "metric": self.metric,
            "annotations": self.annotations,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HLProblem":
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version {}".format(d.get("schema_version")))
        return cls(
            instance_id=d["instance_id"], benchmark=d["benchmark"], domain_name=d["domain"],
            objects=[Obj(o["name"], o["type"], o.get("attrs", {})) for o in d["objects"]],
            facts=[tuple(f) for f in d["init"]["facts"]],
            fluents={(f["function"],) + tuple(f["args"]): f["value"] for f in d["init"]["fluents"]},
            goals=[Goal(tuple(g["fact"]), g.get("deadline")) for g in d["goals"]],
            timed_literals=[TimedLiteral(t["time"], tuple(t["fact"]), t.get("value", True))
                            for t in d.get("timed_literals", [])],
            metric=d.get("metric", "makespan"),
            annotations=d.get("annotations", {}), provenance=d.get("provenance", {}))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, sort_keys=False)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path) -> "HLProblem":
        with open(path) as f:
            return cls.from_dict(json.load(f))
