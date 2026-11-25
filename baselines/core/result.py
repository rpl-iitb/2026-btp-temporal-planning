"""Result record of one run (same fields as the algorithm's, so evaluations can compare them)."""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class RunResult:
    method: str
    instance: str
    budget_s: float
    solved: bool
    makespan: Optional[float] = None
    first_makespan: Optional[float] = None
    first_valid_s: Optional[float] = None
    total_s: float = 0.0
    n_objects: Optional[int] = None
    n_total: int = 0
    branch: Optional[str] = None
    attempts: List[dict] = field(default_factory=list)
    ledger: Dict[str, float] = field(default_factory=dict)
    planner_calls: int = 0
    states: int = 0
    proofs: int = 0
    invalid_on_full: int = 0
    timeouts: int = 0
    notes: List[str] = field(default_factory=list)
    plan_text: Optional[str] = None
    objects: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)
