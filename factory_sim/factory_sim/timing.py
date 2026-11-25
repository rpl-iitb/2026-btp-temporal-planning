"""Duration models: a number, or {fixed: x} | {uniform: [a, b]} | {normal: [mean, sd]} | {lognormal: [mean, cv]} | {exponential: mean} | {empirical: [x1, x2, ...]}."""
import math


def _kind(m):
    if isinstance(m, (int, float)):
        return "fixed", float(m)
    if not isinstance(m, dict) or len(m) != 1:
        raise ValueError("bad duration model: {!r}".format(m))
    return next(iter(m.items()))


def mean(m):
    k, v = _kind(m)
    if k == "fixed" or k == "exponential":
        return float(v)
    if k == "uniform":
        return (v[0] + v[1]) / 2.0
    if k in ("normal", "lognormal"):
        return float(v[0])
    if k == "empirical":
        return float(sum(v)) / len(v)
    raise ValueError("unknown duration model: " + k)


def sample(m, rng, cv_scale=1.0):
    k, v = _kind(m)
    if k == "fixed" or cv_scale == 0:
        return mean(m)
    if k == "uniform":
        c = mean(m)
        return c + (rng.uniform(v[0], v[1]) - c) * cv_scale
    if k == "normal":
        return max(0.0, rng.normal(v[0], v[1] * cv_scale))
    if k == "lognormal":
        cv = v[1] * cv_scale
        if cv <= 0:
            return float(v[0])
        s2 = math.log(1 + cv * cv)
        return float(v[0] * rng.lognormal(-s2 / 2, math.sqrt(s2)))
    if k == "exponential":
        return float(rng.exponential(v))
    if k == "empirical":
        return float(v[rng.integers(len(v))])
    raise ValueError("unknown duration model: " + k)
