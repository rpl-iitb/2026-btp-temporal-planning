"""The robot dispatcher as V8's constructive scheduler for FactorySim (the scheduler protocol of algorithm/v8.py): a
nominal rollout of factory_dispatcher in algorithm/menv.py is a plan (releases, processing and conveyors start as soon
as they can; robots follow the dispatch rules). First plans: the policies nearest / progress (and score, with a learned
score); search: the same with random tie-breaking, until the deadline.
"""
from algorithm import v10
from algorithm.menv import MAEnv, plan_text
from algorithm.pddlkit import TemporalPlan, write_domain, write_problem

from factory_test.factory_dispatcher import Dispatcher


class Objects:
    def __init__(self, plan, policy):
        self.plan, self.policy = plan, policy

    def objects(self):
        return self.plan.objects_used()


class FactoryScheduler:
    def __init__(self, spec):
        self.spec, self._dom = spec, None

    def _task(self, problem):
        if self._dom is None:
            self._dom = v10.Domain(write_domain(self.spec))
        return v10.Task(self._dom, v10.Problem(write_problem(self.spec, problem)))

    def _run(self, T, policy, score, seed=None):
        env = MAEnv(T, "robot", noise=None, seed=0)
        res = env.rollout(Dispatcher(env, policy, score, seed))
        if not res["success"]:
            return None
        plan = TemporalPlan.parse(plan_text(res["steps"]))
        return plan, Objects(plan, policy)

    def first_plans(self, problem, score, k):
        T = self._task(problem)
        runs = [self._run(T, p, score) for p in (("nearest", "progress", "score") if score else ("nearest", "progress"))]
        return sorted((r for r in runs if r), key=lambda r: r[0].makespan)

    def search(self, problem, score, k, max_runs=4000, deadline_fn=None):
        T, out = self._task(problem), []
        for n in range(max_runs):
            if deadline_fn is not None and deadline_fn():
                break
            pol = ("nearest", "progress", "score" if score else "nearest")[n % 3]
            r = self._run(T, pol, score, seed=n)
            if r:
                out.append((r[0].makespan, r[0], r[1], pol))
        return sorted(out, key=lambda x: x[0])
