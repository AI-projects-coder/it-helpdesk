"""Baseline storage and regression detection.

A regression is flagged when a quality metric drops by more than the
tolerance (absolute), when latency/tokens grow by more than the relative
tolerance, or when a ticket that passed in the baseline now fails.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from helpdesk.config import get_settings

QUALITY_METRICS = ["correctness", "groundedness", "tool_use", "classification", "resolution_match"]
COST_METRICS = ["latency_p50_ms", "tokens_avg"]
PASS_THRESHOLD = 0.5  # a ticket "passes" when its correctness is at least this


@dataclass
class Regression:
    metric: str
    baseline: float
    current: float
    detail: str


@dataclass
class ComparisonReport:
    regressions: list[Regression] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    newly_failing: list[str] = field(default_factory=list)
    config_changes: list[str] = field(default_factory=list)  # e.g. judge prompt bumped: deltas aren't like-for-like

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions or self.newly_failing)


def baseline_path() -> Path:
    return get_settings().eval_dir / "baseline.json"


def save_baseline(run: dict, path: Path | None = None) -> Path:
    path = path or baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2))
    return path


def load_baseline(path: Path | None = None) -> dict | None:
    path = path or baseline_path()
    return json.loads(path.read_text()) if path.exists() else None


def compare(current: dict, baseline: dict, tolerance: float = 0.05, cost_tolerance: float = 0.5) -> ComparisonReport:
    report = ComparisonReport()
    cur, base = current["summary"], baseline["summary"]

    cur_cfg, base_cfg = current.get("config", {}), baseline.get("config", {})
    for k in sorted(cur_cfg.keys() | base_cfg.keys()):
        if cur_cfg.get(k) != base_cfg.get(k):
            report.config_changes.append(f"{k}: {base_cfg.get(k)} -> {cur_cfg.get(k)}")

    for m in QUALITY_METRICS:
        if m not in cur or m not in base:
            continue
        delta = round(cur[m] - base[m], 6)  # avoid 0.93 - 0.88 = 0.05000000000000004
        if delta < -tolerance:
            report.regressions.append(Regression(m, base[m], cur[m], f"dropped {-delta:.3f} (> {tolerance})"))
        elif delta > tolerance:
            report.improvements.append(f"{m} +{delta:.3f}")

    for m in COST_METRICS:
        if base.get(m) and cur.get(m) is not None:
            growth = (cur[m] - base[m]) / base[m]
            if growth > cost_tolerance:
                report.regressions.append(Regression(m, base[m], cur[m], f"grew {growth:.0%} (> {cost_tolerance:.0%})"))

    base_pass = {t["ticket_id"] for t in baseline["tickets"] if t["scores"]["correctness"] >= PASS_THRESHOLD}
    for t in current["tickets"]:
        if t["ticket_id"] in base_pass and t["scores"]["correctness"] < PASS_THRESHOLD:
            report.newly_failing.append(t["ticket_id"])
    return report
