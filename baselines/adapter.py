"""Runs the original FLAX / iFLAX code (vendor/) on temporal tasks by replacing only its I/O: PDDLGym
states, Fast Downward -> POPF on the reduced temporal task, STRIPS VAL -> temporal VAL on the full task.
Also the code shared by B1-B3. See baselines/README.md for every adaptation.
"""

"""This file is UN-VERIFIED"""


import dataclasses
import os
import sys
import threading

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from pddlgym.structs import LiteralConjunction, Predicate, State, Type  # noqa: E402
from planning.planner import Planner, PlanningFailure, PlanningTimeout  # noqa: E402

from pddl_generator import write_domain  # noqa: E402
from algorithm import split_dirs  # noqa: E402
from algorithm.planning import Budget, Workspace, attempt  # noqa: E402


# ---- PDDLGym view of a temporal task -------------------------------------------------------
class TaskGoal(LiteralConjunction):
    def __init__(self, literals, task, relaxed=False):
        super().__init__(literals)
        self.task = task
        self.relaxed = relaxed


class TaskDomain:
    """Enough of a pddlgym PDDLDomain for FLAX / iFLAX."""
    operators_as_actions = True

    def __init__(self, spec):
        self.spec = spec
        self.domain_name = spec.name
        self.predicates = {p: Predicate(p, len(sig), var_types=[Type(x.type) for x in sig])
                           for p, sig in spec.predicates.items()}
        for a in spec.actions.values():
            self.predicates[a.name] = Predicate(a.name, len(a.params), var_types=[Type(x.type) for x in a.params])
        self.actions = list(spec.actions)
        self.type_to_parent_types = {Type(t): {Type(x) for x in spec.ancestors(t)} for t in list(spec.types) + ["object"]}

    def write(self, path):
        with open(path, "w") as f:
            f.write(write_domain(self.spec))


_DOMAINS, _DLOCK = {}, threading.Lock()


def domain_for(spec):
    with _DLOCK:
        if spec.name not in _DOMAINS:
            _DOMAINS[spec.name] = TaskDomain(spec)
        return _DOMAINS[spec.name]


def entity(task, name):
    return Type(task.type_of(name))(name)


def task_to_state(task):
    dom = domain_for(task.spec)
    lit = lambda f: dom.predicates[f[0]](*[entity(task, a) for a in f[1:]])  # noqa: E731
    objs = list(task.objects) + list(task.constants)      # pddlgym states include domain constants
    return State(frozenset(lit(f) for f in task.init), frozenset(entity(task, o) for o in objs),
                 TaskGoal([lit(g) for g in task.goals], task))


def _name(v):
    return getattr(v, "name", v)


def state_to_task(state):
    """Reduced (or rule-relaxed) temporal task described by a pddlgym State."""
    from algorithm.grounding import Task
    task, relaxed = state.goal.task, state.goal.relaxed
    objs = {_name(o) for o in state.objects} & set(task.objects)
    objs |= set(task.goal_objects)
    names = set(task.objects)
    ok = lambda f: all(a in objs or a not in names for a in f[1:])  # noqa: E731
    facts = sorted({(l.predicate.name,) + tuple(_name(v) for v in l.variables) for l in state.literals})
    facts = [f for f in facts if f[0] in task.spec.predicates and ok(f)]
    p = task.problem
    q = dataclasses.replace(
        p, objects=[o for o in p.objects if o.name in objs], facts=facts,
        fluents={k: v for k, v in p.fluents.items() if ok(k)},
        goals=[dataclasses.replace(g, deadline=None) for g in p.goals] if relaxed else list(p.goals),
        timed_literals=[] if relaxed else [t for t in p.timed_literals if ok(t.fact)])
    return Task(task.spec, q, _domain_file=task._domain_file)


class PlanList(list):
    """A plan in the form FLAX expects (list of action Literals with .variables),
    carrying the temporal plan and its VAL verdict."""
    temporal = None
    valid = False
    attempt = None


def plan_to_literals(task, plan):
    dom = domain_for(task.spec)
    out = PlanList(dom.predicates[a.name](*[entity(task, x) for x in a.args]) for a in plan.actions)
    out.temporal = plan
    return out


# ---- planner + validation -------------------------------------------------------------------------
class RunContext:
    """Budget, workspace and attempt log of one baseline run."""

    def __init__(self, budget_s=1e9, deploy=False):
        self.budget = Budget(budget_s)
        self.ws = Workspace()
        self.attempts = []
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.deploy = deploy
        self.log = True

    def close(self):
        self.cancel.set()
        self.ws.close()


_DEFAULT, _DEFAULT_LOCK = [], threading.Lock()


def default_context():
    """Process-wide context for planners the original code creates itself (iFLAX's
    training loop builds TrackedFD() per task, in fresh threads): no overall budget
    (callers pass per-call timeouts), no attempt log, workspace removed at exit."""
    with _DEFAULT_LOCK:
        if not _DEFAULT:
            import atexit
            ctx = RunContext()
            ctx.log = False
            atexit.register(ctx.ws.close)
            _DEFAULT.append(ctx)
        return _DEFAULT[0]


class TemporalPOPF(Planner):
    """flax Planner interface on POPF + temporal VAL."""

    def __init__(self, ctx=None, alias_flag=None):
        super().__init__()
        self.ctx = ctx
        self.last_num_evaluated = None

    def __call__(self, domain, state, timeout):
        ctx = self.ctx or default_context()
        task, relaxed = state.goal.task, state.goal.relaxed
        if timeout is None or timeout <= 0 or ctx.cancel.is_set():
            raise PlanningTimeout("no time left")
        sub = state_to_task(state)
        S = set(sub.objects)
        if relaxed:   # rough plan of the rule-relaxed task: checked against that task, as in FLAX
            a = attempt(sub, None, S, ctx.budget, "relaxed", ctx.ws, cap=timeout, cancel=ctx.cancel,
                        check_proof=False)
        else:
            a = attempt(task, None, S, ctx.budget, "plan", ctx.ws, cap=timeout, cancel=ctx.cancel,
                        check_proof=False, plan_task=sub)
        if ctx.log:
            with ctx.lock:
                ctx.attempts.append(a)
        self.last_num_evaluated = a.states
        if a.outcome == "valid":
            if ctx.deploy and not relaxed:
                ctx.cancel.set()        # first valid plan wins; stop sibling recovery branches
            out = plan_to_literals(task, a.plan)
            out.valid, out.attempt = True, a
            return out
        if a.outcome in ("timeout", "cancelled", "budget_exhausted"):
            raise PlanningTimeout(a.outcome)
        raise PlanningFailure(a.outcome + (": " + a.detail if a.detail else ""))


class TrackedPOPF(TemporalPOPF):
    """Stand-in for iflax/fd_tracked.py::TrackedFD (Fast Downward)."""


# ---- rule wrappers (installed into the vendored modules) ---------------------------------------------
def install_rule_wrappers():
    import iflax_planner
    import planning.my_planner as mp
    if getattr(mp, "_adapter_wrapped", False):
        return
    orig_comp, orig_relax = mp.apply_complementary_rules, mp.apply_relaxation_rules

    def comp_fixpoint(state, cur_objects, complementary_rules):
        cur = set(cur_objects)
        while True:
            new, dummy = orig_comp(state, cur, complementary_rules)
            if new == cur:
                return new, dummy
            cur = new

    def relax_no_time(state, relaxation_rules, domain, force_include_goal_objects=True):
        objs, st = orig_relax(state, relaxation_rules, domain, force_include_goal_objects)
        g = state.goal
        return objs, st.with_goal(TaskGoal(g.literals, g.task, relaxed=True))

    for mod in (mp, iflax_planner):
        mod.apply_complementary_rules = comp_fixpoint
        mod.apply_relaxation_rules = relax_no_time
    from guidance.gnn_guidance import GNNSearchGuidance
    GNNSearchGuidance._state_to_graph = tolerant_state_to_graph
    mp._adapter_wrapped = True


def tolerant_state_to_graph(self, state):
    """flax guidance/gnn_guidance.py::GNNSearchGuidance._state_to_graph, unchanged except
    that types / predicates missing from the training vocabulary are skipped."""
    import numpy as np
    assert self._node_feature_to_index is not None, "Must initialize first"
    all_objects = sorted(state.objects)
    node_to_objects = dict(enumerate(all_objects))
    objects_to_node = {v: k for k, v in node_to_objects.items()}
    num_objects = len(all_objects)
    G = self.wrap_goal_literal
    R = self.reverse_binary_literal
    nidx, eidx = self._node_feature_to_index, self._edge_feature_to_index
    graph_input = {"n_node": np.array(num_objects)}
    input_node_features = np.zeros((num_objects, self._num_node_features))
    for obj_index, obj in enumerate(all_objects):
        if obj.var_type in nidx:
            input_node_features[obj_index, nidx[obj.var_type]] = 1
    for lit in state.literals:
        if lit.predicate.arity == 1 and lit.predicate in nidx:
            input_node_features[objects_to_node[lit.variables[0]], nidx[lit.predicate]] = 1
    for lit in state.goal.literals:
        if lit.predicate.arity == 1 and G(lit.predicate) in nidx:
            input_node_features[objects_to_node[lit.variables[0]], nidx[G(lit.predicate)]] = 1
    graph_input["nodes"] = input_node_features
    all_edge_features = np.zeros((num_objects, num_objects, self._num_edge_features))
    for bin_lit in state.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [bin_lit, R(bin_lit)]:
            if lit.predicate in eidx:
                all_edge_features[objects_to_node[lit.variables[0]], objects_to_node[lit.variables[1]],
                                  eidx[lit.predicate]] = 1
    for bin_lit in state.goal.literals:
        if bin_lit.predicate.arity != 2:
            continue
        for lit in [G(bin_lit), G(R(bin_lit))]:
            if lit.predicate in eidx:
                all_edge_features[objects_to_node[lit.variables[0]], objects_to_node[lit.variables[1]],
                                  eidx[lit.predicate]] = 1
    adjacency_mat = np.any(all_edge_features, axis=2)
    receivers, senders, edges = [], [], []
    for sender, receiver in np.argwhere(adjacency_mat):
        senders.append(sender)
        receivers.append(receiver)
        edges.append(all_edge_features[sender, receiver])
    n_edge = len(edges)
    graph_input["edges"] = np.reshape(edges, [n_edge, self._num_edge_features])
    graph_input["receivers"] = np.reshape(receivers, [n_edge]).astype(np.int64)
    graph_input["senders"] = np.reshape(senders, [n_edge]).astype(np.int64)
    graph_input["n_edge"] = np.reshape(n_edge, [1]).astype(np.int64)
    graph_input["globals"] = None
    return graph_input, node_to_objects


# ---- pddlgym.make stand-in (used by the original training code) ------------------------------------
class TaskEnv:
    """Enough of a pddlgym env for GNNSearchGuidance / IFlaxGNNGuidance: one problem per task."""
    operators_as_actions = True

    def __init__(self, tasks):
        self.tasks = tasks
        self.problems = [type("P", (), {"problem_fname": t.name})() for t in tasks]
        self.domain = domain_for(tasks[0].spec)
        self._idx = 0

    def fix_problem_index(self, idx):
        self._idx = idx

    def reset(self):
        return task_to_state(self.tasks[self._idx]), {}


_ENVS = {}


def register_env(name, tasks):
    _ENVS[name] = TaskEnv(tasks)


def make_env(env_id):
    name = env_id[len("PDDLEnv"):-len("-v0")]
    if name not in _ENVS:
        raise KeyError("no task set registered as {!r} (adapter.register_env)".format(name))
    return _ENVS[name]


def graph_literals(state):
    """GNN input view: FLAX's graph code supports arity <= 2 only."""
    return state.with_literals(l for l in state.literals if l.predicate.arity <= 2)


# ---- shared by B1-B3: rules, result rows, frozen scores, training data ---------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RULES = {"complementary": os.path.join(HERE, "rules", "overcooked_complementary_rules.json"),
         "relaxation": os.path.join(HERE, "rules", "overcooked_relaxation_rules.json")}


def run_result(name, task, ctx, plan, branch, notes):
    from algorithm.v1 import RunResult
    att = [a for a in ctx.attempts]
    valid = [a for a in att if a.outcome == "valid" and a.stage != "relaxed"]
    first = min(valid, key=lambda a: a.t_end) if valid else None
    best = plan.attempt if plan is not None and getattr(plan, "valid", False) else None
    return RunResult(
        method=name, instance=task.name, budget_s=ctx.budget.total, solved=best is not None,
        makespan=best.makespan if best else None, first_makespan=first.makespan if first else None,
        first_valid_s=first.t_end if first else None, total_s=ctx.budget.elapsed(),
        n_objects=len(best.objects) if best else None, n_total=len(task.objects),
        branch=branch if best else None, attempts=[a.row() for a in att], ledger=ctx.budget.report()["ledger_s"],
        planner_calls=sum(1 for a in att if a.planner_s > 0), states=sum(a.states or 0 for a in att),
        proofs=0, invalid_on_full=sum(1 for a in att if a.outcome == "invalid_on_full"),
        timeouts=sum(1 for a in att if a.outcome == "timeout"), notes=notes,
        plan_text=best.plan.to_text() if best else None, objects=sorted(best.objects) if best else None)


class FrozenScores:
    """Scores computed once per task (thread-safe view for the parallel 3R branches)."""

    def __init__(self, scores):
        self.scores = scores

    def score_object(self, obj, state):
        return self.scores[_name(obj)]


_SCORE_LOCK = threading.Lock()


def frozen_scores(guidance, task, state):
    with _SCORE_LOCK:
        return {_name(o): float(guidance.score_object(o, state)) for o in state.objects}




def load_split(name):
    """Tasks of a split, each with its source folder (for the certificate plan)."""
    from algorithm.grounding import Task
    out = []
    for d in split_dirs(name):
        t = Task.load(d)
        t.source_dir = d
        out.append(t)
    return out


def model_prefix(method, seed):
    """results/b2/b2_seed0 (+ _train.pt, _meta.json, ...)."""
    from algorithm import results_dir
    return os.path.join(results_dir(method.lower()), "{}_seed{}".format(method.lower(), seed))


_CACHE, _CACHE_LOCK = {}, threading.Lock()


def load_guidance(method, seed, make_guidance, train_split="train"):
    """Rebuild the guidance and load its checkpoint (the original code's mode="test"); cached."""
    with _CACHE_LOCK:
        key = (method, seed)
        if key not in _CACHE:
            install_rule_wrappers()
            register_env(train_split, load_split(train_split))
            g = make_guidance(seed)
            g.train(train_split, timeout=None, mode="test")
            _CACHE[key] = g
        return _CACHE[key]


def train_main(method, make_guidance, train_timeout, argv=None):
    """Shared CLI of b2.py / b3.py: train with the original training code, write <prefix>_meta.json."""
    import argparse
    import json
    import random
    import time
    import numpy as np
    import torch
    ap = argparse.ArgumentParser(description="train " + method)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", default="train")
    ap.add_argument("--flax_epochs", type=int, default=301, help="B2")
    ap.add_argument("--epochs", type=int, default=6, help="B3: bilevel epochs")
    ap.add_argument("--budget", type=float, default=15.0, help="B3: planning time per task in the bilevel loop")
    ap.add_argument("--label_cap", type=float, default=30.0, help="B2 teacher: POPF cap on the full task")
    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--max_tasks", type=int, default=0, help="smoke tests only")
    args = ap.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_rule_wrappers()
    t0 = time.time()
    tasks = load_split(args.train)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    register_env(args.train, tasks)
    g = make_guidance(args.seed, args)
    g.train(args.train, timeout=train_timeout(args), mode="train")
    meta = {"method": method, "args": vars(args), "train_tasks": len(tasks), "seconds": round(time.time() - t0, 1),
            "code": "baselines/vendor (FLAX sair-lab/flax 42a1030 + BTP iFLAX), adapter baselines/adapter.py"}
    with open(model_prefix(method, args.seed) + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("done in {:.0f}s".format(time.time() - t0), flush=True)
