"""iFlax's Algorithm 1: score-guided iterative expansion, falling back to
a parallel Repair / Restart / Rollback (3R) recovery strategy when the
initial expansion gets stuck 
"""
import time
import json
import tempfile
import concurrent.futures

from pddlgym.structs import State
from pddlgym.spaces import LiteralSpace
from pddlgym.parser import PDDLProblemParser

from planning import Planner, PlanningFailure, PlanningTimeout, validate_strips_plan
from planning.my_planner import apply_relaxation_rules, apply_complementary_rules


def _restrict_state(state, objects):
    lits = {lit for lit in state.literals if all(v in objects for v in lit.variables)}
    return State(lits, objects, state.goal)


def _write_pddl_files(domain, state):
    """Writes out the *original, unpruned* task once, for plan validation --
    same pattern flax's planners use (validate against the real task, even
    though search happens on pruned/relaxed sub-tasks)."""
    act_preds = [domain.predicates[a] for a in list(domain.actions)]
    act_space = LiteralSpace(act_preds, type_to_parent_types=domain.type_to_parent_types)
    dom_file = tempfile.NamedTemporaryFile(delete=False).name
    prob_file = tempfile.NamedTemporaryFile(delete=False).name
    domain.write(dom_file)
    lits = set(state.literals)
    if not domain.operators_as_actions:
        lits |= set(act_space.all_ground_literals(state, valid_only=False))
    PDDLProblemParser.create_pddl_file(
        prob_file, state.objects, lits, "myproblem",
        domain.domain_name, state.goal, fast_downward_order=True)
    return dom_file, prob_file


def _plan_and_validate(base_planner, domain, state, timeout, dom_file, prob_file):
    """PLAN(tau, dt) from Algorithm 1."""
    plan = base_planner(domain, state, timeout)
    if not validate_strips_plan(domain_file=dom_file, problem_file=prob_file, plan=plan):
        raise PlanningFailure("Invalid plan")
    return plan


class IFlaxPlanner(Planner):
    """Implements Algorithm 1 end to end: score-guided iterative expansion,
    then parallel Repair/Restart/Rollback recovery if that gets stuck.
    """

    def __init__(self, is_strips_domain, base_planner_factory, search_guider, seed,
                 gamma=0.9, max_iterations=1000, force_include_goal_objects=True,
                 complementary_rules=None, relaxation_rules=None,
                 expansion_frac=0.6, mode="deploy"):
        """
        base_planner_factory: zero-arg callable returning a fresh TrackedFD.
            Each of the (possibly concurrent) recovery branches needs its
            own planner instance so their `last_num_evaluated` bookkeeping
            and temp files don't clash.
        expansion_frac: fraction of the total per-task timeout given to the
            initial score-guided expansion (T1 in the paper); the rest is
            the shared 3R recovery budget (T2).
        mode: "deploy" (first successful branch wins, others abandoned) or
            "train" (wait for all branches, keep fewest evaluated states).
        """
        super().__init__()
        self._is_strips_domain = is_strips_domain
        self._gamma = gamma
        self._max_iterations = max_iterations
        self._base_planner_factory = base_planner_factory
        self._guidance = search_guider
        self._seed = seed
        self._force_include_goal_objects = force_include_goal_objects
        with open(complementary_rules) as f:
            self._complementary_rules = json.load(f)
        with open(relaxation_rules) as f:
            self._relaxation_rules = json.load(f)
        self._expansion_frac = expansion_frac
        assert mode in ("deploy", "train")
        self._mode = mode
        self.last_result = None  # diagnostics from the most recent __call__
        self.last_trace = []  # list of {"stage", "num_objects", ...} -- active-set
                               # size over time, for visualize_mazenamo.py's evolution panel

    # ---- Score-guided iterative expansion (Algorithm 1, lines 1-11) ----
    # Also used by Restart (as ITERATIVEEXP) with a non-default base_objects.
    def _iterative_expansion(self, domain, state, base_planner, timeout,
                              base_objects=None, threshold_init=None, stage="expansion"):
        """Returns (plan_or_None, active_objects, prev_active_objects, timed_out)."""
        dom_file, prob_file = _write_pddl_files(domain, state)
        start_time = time.time()

        cur_objects = set(base_objects) if base_objects is not None else set()
        if self._force_include_goal_objects:
            for lit in state.goal.literals:
                cur_objects |= set(lit.variables)

        object_to_score = {o: self._guidance.score_object(o, state)
                            for o in state.objects if o not in cur_objects}
        threshold = threshold_init if threshold_init is not None else self._gamma
        prev_objects = set(cur_objects)

        for _ in range(self._max_iterations):
            if time.time() - start_time >= timeout:
                return None, cur_objects, prev_objects, True

            unused_objs = sorted(state.objects - cur_objects)
            new_objs = set()
            while unused_objs:
                threshold *= self._gamma
                new_objs = {o for o in unused_objs if object_to_score[o] > threshold}
                if new_objs:
                    break
            prev_objects = set(cur_objects)
            cur_objects |= new_objs

            dummy_state = _restrict_state(state, cur_objects)
            print("[Trying to plan with {} objects of {} total, threshold is {}...]"
                  .format(len(cur_objects), len(state.objects), threshold), flush=True)
            self.last_trace.append({
                "stage": stage, "num_objects": len(cur_objects),
                "total_objects": len(state.objects), "threshold": threshold,
            })
            try:
                remaining = timeout - (time.time() - start_time)
                plan = _plan_and_validate(base_planner, domain, dummy_state, remaining,
                                           dom_file, prob_file)
                return plan, cur_objects, prev_objects, False
            except PlanningFailure:
                if len(cur_objects) == len(state.objects):
                    return None, cur_objects, prev_objects, False
                continue
            except PlanningTimeout:
                return None, cur_objects, prev_objects, True

        return None, cur_objects, prev_objects, False

    # ---- Thread 1: Repair (Algorithm 1, lines 14-17) ----
    def _repair(self, domain, state, active_objects, timeout):
        base_planner = self._base_planner_factory()
        dom_file, prob_file = _write_pddl_files(domain, state)
        start = time.time()

        _, relaxed_state = apply_relaxation_rules(
            state, self._relaxation_rules, domain, self._force_include_goal_objects)
        relaxed_plan = base_planner(domain, relaxed_state, timeout - (time.time() - start))
        objects_in_relaxed_plan = {o for act in relaxed_plan for o in act.variables}

        _, repair_state = apply_complementary_rules(
            state, active_objects | (objects_in_relaxed_plan & state.objects),
            self._complementary_rules)
        self.last_trace.append({
            "stage": "repair", "num_objects": len(repair_state.objects),
            "total_objects": len(state.objects), "threshold": None,
        })
        plan = _plan_and_validate(base_planner, domain, repair_state,
                                   timeout - (time.time() - start), dom_file, prob_file)
        return plan, base_planner.last_num_evaluated

    # ---- Thread 2: Restart (Algorithm 1, lines 18-19) ----
    def _restart(self, domain, state, goal_objects, timeout):
        base_planner = self._base_planner_factory()
        start = time.time()

        _, relaxed_state = apply_relaxation_rules(
            state, self._relaxation_rules, domain, self._force_include_goal_objects)
        relaxed_plan = base_planner(domain, relaxed_state, timeout - (time.time() - start))
        objects_in_relaxed_plan = {o for act in relaxed_plan for o in act.variables}

        restart_objects, _ = apply_complementary_rules(
            state, goal_objects | (objects_in_relaxed_plan & state.objects),
            self._complementary_rules)

        plan, _, _, _ = self._iterative_expansion(
            domain, state, base_planner, timeout - (time.time() - start),
            base_objects=restart_objects, stage="restart")
        if plan is None:
            raise PlanningFailure("Restart failed to find a plan")
        return plan, base_planner.last_num_evaluated

    # ---- Thread 3: Rollback (Algorithm 1, line 20) ----
    def _rollback(self, domain, state, prev_active_objects, timeout):
        base_planner = self._base_planner_factory()
        dom_file, prob_file = _write_pddl_files(domain, state)
        start_time = time.time()

        cur_objects = set(prev_active_objects)
        remaining_by_score = sorted(
            (o for o in state.objects if o not in cur_objects),
            key=lambda o: self._guidance.score_object(o, state), reverse=True)

        for obj in remaining_by_score:
            if time.time() - start_time >= timeout:
                break
            cur_objects.add(obj)
            dummy_state = _restrict_state(state, cur_objects)
            self.last_trace.append({
                "stage": "rollback", "num_objects": len(cur_objects),
                "total_objects": len(state.objects), "threshold": None,
            })
            try:
                remaining = timeout - (time.time() - start_time)
                plan = _plan_and_validate(base_planner, domain, dummy_state, remaining,
                                           dom_file, prob_file)
                return plan, base_planner.last_num_evaluated
            except PlanningFailure:
                continue
            except PlanningTimeout:
                break
        raise PlanningFailure("Rollback failed to find a plan")

    # ---- Full Algorithm 1 ----
    def __call__(self, domain, state, timeout):
        self.last_trace = []
        base_planner = self._base_planner_factory()
        t1 = timeout * self._expansion_frac
        t2 = timeout - t1

        plan, active_objects, prev_objects, _ = self._iterative_expansion(
            domain, state, base_planner, t1)

        if plan is not None:
            self.last_result = {"branch": "iterative_expansion",
                                 "num_evaluated": base_planner.last_num_evaluated}
            return plan

        goal_objects = set()
        for lit in state.goal.literals:
            goal_objects |= set(lit.variables)

        branches = {
            "repair": lambda: self._repair(domain, state, active_objects, t2),
            "restart": lambda: self._restart(domain, state, goal_objects, t2),
            "rollback": lambda: self._rollback(domain, state, prev_objects, t2),
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): name for name, fn in branches.items()}

            if self._mode == "deploy":
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    try:
                        plan, num_eval = fut.result()
                    except (PlanningFailure, PlanningTimeout):
                        continue
                    self.last_result = {"branch": name, "num_evaluated": num_eval}
                    return plan
                raise PlanningFailure("All 3R recovery branches failed")

            # mode == "train": wait for all branches, keep fewest evaluated states.
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    plan, num_eval = fut.result()
                    results[name] = (plan, num_eval if num_eval is not None else float("inf"))
                except (PlanningFailure, PlanningTimeout):
                    continue
            if not results:
                raise PlanningFailure("All 3R recovery branches failed")
            best_branch = min(results, key=lambda n: results[n][1])
            self.last_result = {"branch": best_branch, "num_evaluated": results[best_branch][1]}
            return results[best_branch][0]
