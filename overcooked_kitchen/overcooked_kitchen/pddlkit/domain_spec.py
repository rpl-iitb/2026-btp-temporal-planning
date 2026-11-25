"""Lifted temporal domain specification (the "D" of Pi = <D, O, I, G, T>).

A DomainSpec is loaded from a declarative YAML file (the package's domain.yaml). It is
benchmark-agnostic: types, constants, predicates, numeric functions and
durative actions with at-start / over-all / at-end conditions and
at-start / at-end effects -- the PDDL 2.1 durative-action model.

Literals are written as strings, e.g. "at ?c ?from" or "not idle ?c".
A duration is a number or a function term string, e.g. "travel ?from ?to".

`extends: <other.domain.yaml>` merges another spec first.
Ported from earlier research code.
"""
"""This file is UN-VERIFIED"""
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import yaml

TIME_POINTS_COND = ("at_start", "over_all", "at_end")
TIME_POINTS_EFF = ("at_start", "at_end")


@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[str, ...]
    negated: bool = False

    @classmethod
    def parse(cls, text: str) -> "Atom":
        toks = text.replace("(", " ").replace(")", " ").split()
        neg = False
        if toks and toks[0] == "not":
            neg, toks = True, toks[1:]
        if not toks:
            raise ValueError("empty literal: {!r}".format(text))
        return cls(toks[0], tuple(toks[1:]), neg)

    def __str__(self):
        body = " ".join((self.pred,) + self.args)
        return "not " + body if self.negated else body


@dataclass(frozen=True)
class Param:
    name: str   # with leading '?'
    type: str


def parse_params(text: str) -> List[Param]:
    """'?c - cook ?a ?b - pose' -> [Param(?c, cook), Param(?a, pose), Param(?b, pose)]"""
    toks, out, pending = text.split(), [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "-":
            ty = toks[i + 1]
            out += [Param(p, ty) for p in pending]
            pending = []
            i += 2
            continue
        if not t.startswith("?"):
            raise ValueError("bad parameter list: {!r}".format(text))
        pending.append(t)
        i += 1
    out += [Param(p, "object") for p in pending]
    return out


@dataclass
class DurativeAction:
    name: str
    params: List[Param]
    duration: Union[int, float, Atom]           # constant or function term
    conditions: Dict[str, List[Atom]]           # keys: TIME_POINTS_COND
    effects: Dict[str, List[Atom]]              # keys: TIME_POINTS_EFF
    sim: dict = field(default_factory=dict)     # executor binding (benchmark-specific, documented)
    doc: str = ""
    duration_min: Optional[float] = None        # set: flexible duration in [duration_min, duration]

    @property
    def param_names(self):
        return [p.name for p in self.params]


@dataclass
class DomainSpec:
    name: str
    requirements: List[str]
    types: Dict[str, str]                         # type -> parent
    constants: Dict[str, str]                     # constant -> type
    predicates: Dict[str, List[Param]]
    functions: Dict[str, List[Param]]
    actions: Dict[str, DurativeAction]
    deadline_encoding: Optional[dict] = None
    doc: str = ""
    source_files: List[str] = field(default_factory=list)

    # ---- type helpers -------------------------------------------------------------
    def ancestors(self, ty: str) -> List[str]:
        out = [ty]
        while ty in self.types and self.types[ty] != ty:
            ty = self.types[ty]
            out.append(ty)
            if ty == "object":
                break
        if "object" not in out:
            out.append("object")
        return out

    def is_subtype(self, ty: str, parent: str) -> bool:
        return parent in self.ancestors(ty)

    # ---- static analysis ----------------------------------------------------------
    def fluent_predicates(self) -> set:
        """Predicates changed by some action effect."""
        return {a.pred for act in self.actions.values() for pt in TIME_POINTS_EFF for a in act.effects[pt]}

    def static_predicates(self) -> set:
        """Predicates no action and no deadline timed literal can change. A
        problem's own timed literals can make more predicates dynamic; use
        HLProblem.static_predicates(spec) for the per-instance answer."""
        dyn = self.fluent_predicates()
        if self.deadline_encoding:
            dyn.add(self.deadline_encoding["window_predicate"])
        return set(self.predicates) - dyn

    def sha256(self) -> str:
        h = hashlib.sha256()
        for f in self.source_files:
            with open(f, "rb") as fh:
                h.update(fh.read())
        return h.hexdigest()

    def check(self):
        """Internal consistency: every literal uses a declared predicate with the
        right arity, every argument is a parameter or constant, types exist."""
        errors = []
        known_types = set(self.types) | {"object"}
        for t, parent in self.types.items():
            if parent not in known_types:
                errors.append("type {} has unknown parent {}".format(t, parent))
        for c, t in self.constants.items():
            if t not in known_types:
                errors.append("constant {} has unknown type {}".format(c, t))
        for name, params in list(self.predicates.items()) + list(self.functions.items()):
            for p in params:
                if p.type not in known_types:
                    errors.append("{}: unknown type {}".format(name, p.type))
        for act in self.actions.values():
            pnames = {p.name: p.type for p in act.params}
            if len(pnames) != len(act.params):
                errors.append("{}: duplicate parameter".format(act.name))

            def check_atom(a, where, table):
                if a.pred not in table:
                    errors.append("{} {}: undeclared {}".format(act.name, where, a.pred))
                    return
                sig = table[a.pred]
                if len(sig) != len(a.args):
                    errors.append("{} {}: {} expects {} args".format(act.name, where, a.pred, len(sig)))
                    return
                for arg, p in zip(a.args, sig):
                    ty = pnames.get(arg) if arg.startswith("?") else self.constants.get(arg)
                    if ty is None:
                        errors.append("{} {}: unbound {}".format(act.name, where, arg))
                    elif not self.is_subtype(ty, p.type):
                        errors.append("{} {}: {} has type {}, {} needs {}".format(
                            act.name, where, arg, ty, a.pred, p.type))

            for pt in TIME_POINTS_COND:
                for a in act.conditions[pt]:
                    check_atom(a, pt, self.predicates)
            for pt in TIME_POINTS_EFF:
                for a in act.effects[pt]:
                    check_atom(a, pt, self.predicates)
            if isinstance(act.duration, Atom):
                check_atom(act.duration, "duration", self.functions)
            elif not act.duration > 0:
                errors.append("{}: non-positive duration".format(act.name))
        if self.deadline_encoding:
            for k in ("goal_predicate", "window_predicate"):
                if self.deadline_encoding.get(k) not in self.predicates:
                    errors.append("deadline_encoding.{} not a predicate".format(k))
        if errors:
            raise ValueError("domain spec {} invalid:\n  ".format(self.name) + "\n  ".join(errors))
        return self


# ---- loading ---------------------------------------------------------------------
def _parse_sig(spec) -> List[Param]:
    if spec is None or spec == "":
        return []
    return parse_params(spec)


def _merge(base: dict, ext: dict) -> dict:
    out = dict(base)
    for k, v in ext.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        elif k == "actions" and isinstance(v, list):
            names = {a["name"] for a in v}
            out[k] = [a for a in out.get(k, []) if a["name"] not in names] + v
        elif k == "requirements" and isinstance(v, list):
            out[k] = list(dict.fromkeys(out.get(k, []) + v))
        else:
            out[k] = v
    return out


def _load_raw(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)
    files = [os.path.abspath(path)]
    if raw.get("extends"):
        base_path = os.path.join(os.path.dirname(path), raw.pop("extends"))
        base, base_files = _load_raw(base_path)
        raw = _merge(base, raw)
        files = base_files + files
    return raw, files


def load_domain_spec(path: str) -> DomainSpec:
    raw, files = _load_raw(path)
    actions = {}
    for a in raw.get("actions", []):
        dur = a["duration"]
        duration = dur if isinstance(dur, (int, float)) else Atom.parse(str(dur))
        cond = {pt: [Atom.parse(s) for s in (a.get("condition", {}) or {}).get(pt, []) or []]
                for pt in TIME_POINTS_COND}
        eff = {pt: [Atom.parse(s) for s in (a.get("effect", {}) or {}).get(pt, []) or []]
               for pt in TIME_POINTS_EFF}
        actions[a["name"]] = DurativeAction(a["name"], _parse_sig(a.get("params")), duration, cond, eff,
                                            a.get("sim", {}) or {}, a.get("doc", ""), a.get("duration_min"))
    spec = DomainSpec(
        name=raw["name"],
        requirements=list(raw.get("requirements", [])),
        types=dict(raw.get("types", {})),
        constants=dict(raw.get("constants", {}) or {}),
        predicates={k: _parse_sig(v) for k, v in (raw.get("predicates") or {}).items()},
        functions={k: _parse_sig(v) for k, v in (raw.get("functions") or {}).items()},
        actions=actions,
        deadline_encoding=raw.get("deadline_encoding"),
        doc=raw.get("doc", ""),
        source_files=files,
    )
    return spec.check()
