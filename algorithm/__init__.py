"""T-iFLAX: temporal object-selection planning (v1 ... v14), domain-independent.

The algorithm reads PDDL tasks given as instance folders (hl_problem.json + a domain spec YAML) and writes results;
it imports nothing from the simulators. A test harness (factory_test/, overcooked_test/) tells it where things are,
by environment variables or configure() *before* importing a version:

    TIFLAX_DATA     root of the instances: <root>/<split>/<instance>/hl_problem.json
    TIFLAX_RESULTS  where results and trained models go (default ./results)
    TIFLAX_BENCHES  benchmark name -> its instance root, "name=path:name=path" (versions that take --bench)
    TIFLAX_DOMAINS  benchmark name -> domain spec YAML, "name=path:name=path"
                    (or a domain.yaml in the instance folder or its parents)
    TIFLAX_TOOLS    folder with popf / Validate / Parser / optic-clp (also TIFLAX_POPF ... or PATH)
"""
import glob
import os

PDDL = os.environ.get("TIFLAX_DATA", os.path.join(os.getcwd(), "data"))
RESULTS = os.environ.get("TIFLAX_RESULTS", os.path.join(os.getcwd(), "results"))
DOMAINS = dict(x.split("=", 1) for x in os.environ.get("TIFLAX_DOMAINS", "").split(":") if "=" in x)
BENCHES = dict(x.split("=", 1) for x in os.environ.get("TIFLAX_BENCHES", "").split(":") if "=" in x)


def configure(data=None, results=None, domains=None, benches=None):
    """Set the data / results folders and the registries (call before importing a version module)."""
    global PDDL, RESULTS
    PDDL = data or PDDL
    RESULTS = results or RESULTS
    DOMAINS.update(domains or {})
    BENCHES.update(benches or {})


def domain_yaml_for(benchmark, instance_dir=None):
    """Domain spec YAML of a benchmark: the registry, else a domain.yaml in the instance folder or a parent."""
    if benchmark in DOMAINS:
        return DOMAINS[benchmark]
    d = os.path.abspath(instance_dir) if instance_dir else None
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "domain.yaml")):
            return os.path.join(d, "domain.yaml")
        d = os.path.dirname(d)
    raise KeyError("no domain spec for benchmark {}: set TIFLAX_DOMAINS or configure(domains=...)".format(benchmark))


def results_dir(name):
    """<RESULTS>/<name>/ (created)."""
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return d


def split_dirs(split, bench=None):
    """Instance folders of a dataset split (<TIFLAX_DATA or the benchmark's root>/<split>/*), sorted."""
    root = BENCHES[bench] if bench else PDDL
    return sorted(d for d in glob.glob(os.path.join(root, split, "*")) if os.path.isdir(d))
