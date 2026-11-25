"""Read generated PDDL back and compare it with its source of truth.

This guards the writer itself: a round trip must recover exactly the HL
objects, facts, fluents, timed literals and goals (problem) and exactly the
spec's parameters, durations, conditions and effects (domain). It is a
deliberately small reader for the subset the writer emits, not a general
PDDL parser -- VAL's Parser is the independent syntax/type check.
"""
import re


def tokenize(text):
    text = re.sub(r";[^\n]*", "", text)
    return re.findall(r"\(|\)|[^\s()]+", text.lower())


def parse_sexpr(text):
    toks = tokenize(text)
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
    if stack:
        raise ValueError("unbalanced parentheses")
    if len(cur) != 1:
        raise ValueError("expected one top-level expression, got {}".format(len(cur)))
    return cur[0]


def _sections(expr):
    return {e[0]: e for e in expr[2:] if isinstance(e, list) and e and isinstance(e[0], str)}


def _num(x):
    v = float(x)
    return int(v) if v.is_integer() else v


def read_problem(text):
    e = parse_sexpr(text)
    sec = _sections(e)
    objects, pending = {}, []
    toks = sec[":objects"][1:]
    i = 0
    while i < len(toks):
        if toks[i] == "-":
            for n in pending:
                objects[n] = toks[i + 1]
            pending, i = [], i + 2
        else:
            pending.append(toks[i])
            i += 1
    facts, fluents, tils = set(), {}, set()
    for item in sec[":init"][1:]:
        if item[0] == "=":
            fluents[tuple(item[1])] = _num(item[2])
        elif item[0] == "at" and len(item) == 3 and not isinstance(item[1], list) \
                and re.fullmatch(r"[0-9.]+", item[1]):
            lit = item[2]
            if lit[0] == "not":
                tils.add((float(item[1]), tuple(lit[1]), False))
            else:
                tils.add((float(item[1]), tuple(lit), True))
        else:
            facts.add(tuple(item))
    g = sec[":goal"][1]
    goals = [tuple(x) for x in g[1:]] if g[0] == "and" else [tuple(g)]
    return {"name": e[1][1], "domain": sec[":domain"][1], "objects": objects, "facts": facts,
            "fluents": fluents, "tils": tils, "goals": goals, "metric": sec.get(":metric")}


def read_domain(text):
    e = parse_sexpr(text)
    actions = {}
    for item in e[2:]:
        if item[0] != ":durative-action":
            continue
        name, kv = item[1], {}
        for k in range(2, len(item), 2):
            kv[item[k]] = item[k + 1]
        params = [t for t in kv[":parameters"] if t.startswith("?")]
        dmin, dur = None, kv[":duration"]
        if dur[0] == "and":            # (and (>= ?duration lo) (<= ?duration hi))
            dmin, dur = _num(dur[1][2]), dur[2]
        dur = dur[2]
        conds = {"at start": set(), "over all": set(), "at end": set()}
        for c in kv[":condition"][1:]:
            conds[c[0] + " " + c[1]].add(_lit(c[2]))
        effs = {"at start": set(), "at end": set()}
        for c in kv[":effect"][1:]:
            effs[c[0] + " " + c[1]].add(_lit(c[2]))
        actions[name] = {"params": params, "duration": tuple(dur) if isinstance(dur, list) else _num(dur),
                         "duration_min": dmin,
                         "conditions": conds, "effects": effs}
    return {"name": e[1][1], "actions": actions}


def _lit(x):
    return ("not",) + tuple(x[1]) if x[0] == "not" else tuple(x)


# ---- comparisons -----------------------------------------------------------------------
def check_problem_roundtrip(spec, problem, pddl_text):
    """Return a list of discrepancies (empty = exact round trip)."""
    from .writer import deadline_literals
    r = read_problem(pddl_text)
    errs = []
    want_objs = {o.name: o.type for o in problem.objects}
    if r["objects"] != want_objs:
        errs.append("objects differ: {}".format(set(r["objects"].items()) ^ set(want_objs.items())))
    if r["facts"] != set(problem.facts):
        errs.append("init facts differ: {}".format(r["facts"] ^ set(problem.facts)))
    want_fl = {k: v for k, v in problem.fluents.items()}
    if r["fluents"] != want_fl:
        errs.append("fluents differ: {}".format(set(r["fluents"].items()) ^ set(want_fl.items())))
    want_tils = {(float(t.time), tuple(t.fact), t.value) for t in problem.timed_literals}
    want_tils |= {(t, w, False) for t, w in deadline_literals(spec, problem)}
    if r["tils"] != want_tils:
        errs.append("timed literals differ: {}".format(r["tils"] ^ want_tils))
    if r["goals"] != [tuple(g.fact) for g in problem.goals]:
        errs.append("goals differ")
    if r["domain"] != spec.name or r["name"] != problem.instance_id.lower():
        errs.append("header differs")
    return errs


def check_domain_roundtrip(spec, pddl_text):
    r = read_domain(pddl_text)
    errs = []
    if set(r["actions"]) != set(spec.actions):
        errs.append("action names differ: {}".format(set(r["actions"]) ^ set(spec.actions)))
    for name, act in spec.actions.items():
        got = r["actions"].get(name)
        if got is None:
            continue
        if got["params"] != act.param_names:
            errs.append("{}: parameters differ".format(name))
        want_dur = (act.duration.pred,) + act.duration.args if hasattr(act.duration, "pred") else act.duration
        if got["duration"] != want_dur or got.get("duration_min") != act.duration_min:
            errs.append("{}: duration differs".format(name))
        for pt, key in (("at_start", "at start"), ("over_all", "over all"), ("at_end", "at end")):
            want = {(("not",) if a.negated else ()) + (a.pred,) + a.args for a in act.conditions[pt]}
            if got["conditions"][key] != want:
                errs.append("{}: {} conditions differ".format(name, key))
        for pt, key in (("at_start", "at start"), ("at_end", "at end")):
            want = {(("not",) if a.negated else ()) + (a.pred,) + a.args for a in act.effects[pt]}
            if got["effects"][key] != want:
                errs.append("{}: {} effects differ".format(name, key))
    return errs
