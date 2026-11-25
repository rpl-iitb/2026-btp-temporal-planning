"""PDDL 2.1 text (as factory_sim/pddl.py writes it) -> the algorithm's instance format: a domain spec YAML dict and an
HL problem. Covers the subset FactorySim uses: typed objects, durative actions with at-start / over-all / at-end
literals (positive or `not`), durations that are numbers or function terms, numeric fluents in :init, timed
initial literals and a conjunctive goal.
"""
import re

from algorithm.pddlkit import Goal, HLProblem, Obj, TimedLiteral

TP = {"at start": "at_start", "over all": "over_all", "at end": "at_end"}


def sexpr(text):
    toks = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text).lower())
    stack, cur = [], []
    for t in toks:
        if t == "(":
            stack.append(cur)
            cur = []
        elif t == ")":
            done, cur = cur, stack.pop()
            cur.append(done)
        else:
            cur.append(t)
    return cur[0]


def typed(tokens):
    """['?a', '?b', '-', 'node', 'x', '-', 't'] -> [(name, type)]"""
    out, pending, i = [], [], 0
    while i < len(tokens):
        if tokens[i] == "-":
            out += [(p, tokens[i + 1]) for p in pending]
            pending, i = [], i + 2
        else:
            pending.append(tokens[i])
            i += 1
    return out + [(p, "object") for p in pending]


def sig(tokens):
    return " ".join("{} - {}".format(n, t) for n, t in typed(tokens))


def _lit(x):
    return "not " + " ".join(x[1]) if x[0] == "not" else " ".join(x)


def _conj(x):
    return x[1:] if x and x[0] == "and" else ([x] if x else [])


def read_domain(text):
    d = sexpr(text)
    out = {"name": d[1][1], "requirements": [], "types": {}, "constants": {}, "predicates": {}, "functions": {}, "actions": []}
    for sec in d[2:]:
        key = sec[0]
        if key == ":requirements":
            out["requirements"] = sec[1:]
        elif key == ":types":
            out["types"] = {n: t for n, t in typed(sec[1:])}
        elif key == ":constants":
            out["constants"] = {n: t for n, t in typed(sec[1:])}
        elif key == ":predicates":
            out["predicates"] = {p[0]: sig(p[1:]) for p in sec[1:]}
        elif key == ":functions":
            out["functions"] = {p[0]: sig(p[1:]) for p in sec[1:] if isinstance(p, list)}
        elif key == ":durative-action":
            a = {"name": sec[1]}
            kv = dict(zip(sec[2::2], sec[3::2]))
            a["params"] = sig(kv[":parameters"])
            dur = kv[":duration"][2]
            a["duration"] = float(dur) if not isinstance(dur, list) else " ".join(dur)
            for field, name in ((":condition", "condition"), (":effect", "effect")):
                parts = {}
                for c in _conj(kv[field]):
                    parts.setdefault(TP[c[0] + " " + c[1]], []).append(_lit(c[2]))
                a[name] = parts
            out["actions"].append(a)
    return out


def read_problem(text, benchmark, instance_id=None):
    p = sexpr(text)
    name, dom = p[1][1], None
    objs, facts, fluents, tils, goals = [], [], {}, [], []
    for sec in p[2:]:
        if sec[0] == ":domain":
            dom = sec[1]
        elif sec[0] == ":objects":
            objs = [Obj(n, t) for n, t in typed(sec[1:])]
        elif sec[0] == ":init":
            for x in sec[1:]:
                if x[0] == "=":
                    fluents[tuple(x[1])] = float(x[2])
                elif x[0] == "at" and len(x) == 3 and not isinstance(x[1], list):
                    lit = x[2]
                    neg = lit[0] == "not"
                    tils.append(TimedLiteral(float(x[1]), tuple(lit[1] if neg else lit), not neg))
                else:
                    facts.append(tuple(x))
        elif sec[0] == ":goal":
            goals = [Goal(tuple(g)) for g in _conj(sec[1])]
    return HLProblem(instance_id=instance_id or name, benchmark=benchmark, domain_name=dom, objects=objs, facts=facts,
                     fluents=fluents, goals=goals, timed_literals=tils)
