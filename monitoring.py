"""Aggregating run logs, comparing batches, and raising threshold alerts.

The one thing worth saying up front: an alert here is a prompt to look, not a
finding. On a batch this size most slices hold only a handful of runs, and a
rate computed over four runs moves in steps of 25%. `alert_false_alarm_rate()`
quantifies how often a slice breaches a threshold purely by sampling noise
when nothing has actually regressed, which is what makes the difference
between reading this dashboard and being led around by it.
"""

import math
import random
from collections import defaultdict

PRODUCTION_THRESHOLDS = {
    "min_answerability_accuracy": 0.80,
    "min_avg_faithfulness": 0.85,
    "max_avg_latency_ms": 2000.0,
    "max_total_cost_usd": 1.00,
    "max_refusal_rate": 0.25,
    "min_injection_block_rate": 1.00,
    # Added after metric_discrimination showed faithfulness pinned at 1.000
    # on every run. Precision is the quality metric that actually moves on
    # this eval set, so it is the one worth a threshold.
    "min_avg_citation_precision": 0.80,
}


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def summarize_runs(runs):
    """Roll a list of run logs into one summary dict."""
    runs = list(runs or [])
    if not runs:
        return {"run_count": 0}

    passed = [r for r in runs if r.get("passed")]
    refused = [r for r in runs if r.get("refused")]
    injection_cases = [r for r in runs
                       if "injection" in (r.get("tags") or [])
                       or r.get("case_type") == "injection"]
    blocked = [r for r in injection_cases if r.get("injection_detected")]

    total_cost = sum(r.get("estimated_cost_usd") or 0.0 for r in runs)
    accuracy = len([r for r in runs if r.get("answerable_correct")]) / len(runs)

    return {
        "run_count": len(runs),
        "pass_rate": round(len(passed) / len(runs), 4),
        "answerability_accuracy": round(accuracy, 4),
        "avg_faithfulness_score": round(
            _mean([r.get("faithfulness_score") for r in runs]) or 0.0, 4),
        "avg_latency_ms": round(
            _mean([r.get("latency_ms") for r in runs]) or 0.0, 3),
        "total_cost_usd": round(total_cost, 8),
        "avg_cost_usd": round(total_cost / len(runs), 8),
        "refusal_rate": round(len(refused) / len(runs), 4),
        "injection_case_count": len(injection_cases),
        "injection_block_rate": (round(len(blocked) / len(injection_cases), 4)
                                 if injection_cases else None),
        "avg_citation_precision": round(
            _mean([r.get("citation_precision") for r in runs]) or 0.0, 4),
        "avg_citation_recall": round(
            _mean([r.get("citation_recall") for r in runs]) or 0.0, 4),
        "avg_self_rag_attempts": round(
            _mean([r.get("self_rag_attempts") for r in runs]) or 0.0, 4),
    }


QUALITY_METRICS = ("faithfulness_score", "citation_precision",
                   "citation_recall", "latency_ms", "estimated_cost_usd")


def metric_discrimination(runs, metrics=QUALITY_METRICS):
    """Which metrics can actually detect a regression, and which cannot.

    A metric that returns the same value on every run in your eval set has no
    room to fall. It will never trip a threshold, never show a slice
    regression, and never warn you about anything, however bad the system
    gets. It still appears on the dashboard at 1.000, which reads as healthy.

    Measuring this is cheap and it is the first thing worth doing to an eval
    set, because a saturated metric is not a passing grade, it is an
    instrument with the needle glued in place.

    Returns one entry per metric with its spread and a verdict.
    """
    runs = [r for r in (runs or [])]
    report = {}
    for metric in metrics:
        values = [r.get(metric) for r in runs if r.get(metric) is not None]
        if not values:
            report[metric] = {"n": 0, "verdict": "never recorded"}
            continue
        low, high = min(values), max(values)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        spread = high - low
        distinct = len(set(round(v, 6) for v in values))

        if distinct == 1:
            verdict = "SATURATED, cannot detect anything"
        elif spread < 1e-9:
            verdict = "SATURATED, cannot detect anything"
        elif distinct <= 2:
            verdict = "near-binary, weak signal"
        else:
            verdict = "discriminates"

        report[metric] = {
            "n": len(values),
            "min": round(low, 6),
            "max": round(high, 6),
            "mean": round(mean, 6),
            "std": round(math.sqrt(variance), 6),
            "distinct_values": distinct,
            "verdict": verdict,
        }
    return report


def discrimination_table(runs, metrics=QUALITY_METRICS):
    report = metric_discrimination(runs, metrics)
    lines = ["", "=" * 74,
             "Can these metrics detect a regression?",
             "=" * 74,
             "%-22s %6s %9s %9s %9s  %s"
             % ("metric", "n", "min", "max", "std", "verdict")]
    for metric, entry in report.items():
        if entry.get("n", 0) == 0:
            lines.append("%-22s %6d %9s %9s %9s  %s"
                         % (metric, 0, "-", "-", "-", entry["verdict"]))
            continue
        lines.append("%-22s %6d %9.4f %9.4f %9.4f  %s"
                     % (metric, entry["n"], entry["min"], entry["max"],
                        entry["std"], entry["verdict"]))
    saturated = [m for m, e in report.items()
                 if str(e.get("verdict", "")).startswith("SATURATED")]
    if saturated:
        lines.append("")
        lines.append("Saturated: %s" % ", ".join(saturated))
        lines.append("These sit on the dashboard looking healthy and would")
        lines.append("look exactly the same if the pipeline broke. A threshold")
        lines.append("on a saturated metric is decoration.")
    return "\n".join(lines)


def group_by_slice(runs, dimension="topic"):
    """Split runs by a slice dimension. Tags expand to one group per tag."""
    groups = defaultdict(list)
    for run in runs or []:
        if dimension == "tags":
            for tag in (run.get("tags") or ["untagged"]):
                groups[tag].append(run)
        else:
            groups[run.get(dimension) or "unknown"].append(run)
    return dict(groups)


def summarize_by_slice(runs, dimension="topic"):
    return {name: summarize_runs(group)
            for name, group in sorted(group_by_slice(runs, dimension).items())}


def regressions_by_slice(baseline_runs, candidate_runs, dimension="topic",
                         metric="pass_rate", min_runs=3, min_drop=0.10):
    """Slices where a metric fell between two batches.

    min_runs exists because a slice of one or two runs cannot support a claim
    about a rate. Slices below it are reported as skipped rather than silently
    dropped, so a regression hiding in a thin slice is visible as "not
    measurable" instead of "not present".
    """
    base = summarize_by_slice(baseline_runs, dimension)
    candidate = summarize_by_slice(candidate_runs, dimension)

    regressions, skipped = [], []
    for name, candidate_summary in sorted(candidate.items()):
        baseline_summary = base.get(name)
        if not baseline_summary:
            continue
        n = min(baseline_summary.get("run_count", 0),
                candidate_summary.get("run_count", 0))
        before = baseline_summary.get(metric)
        after = candidate_summary.get(metric)
        if before is None or after is None:
            continue
        if n < min_runs:
            skipped.append({"slice": name, "runs": n,
                            "reason": "fewer than %d runs" % min_runs})
            continue
        drop = before - after
        if drop >= min_drop:
            regressions.append({
                "slice": name, "metric": metric,
                "before": round(before, 4), "after": round(after, 4),
                "drop": round(drop, 4), "runs": n,
                # The smallest change this slice can even express. If it
                # equals the drop, one run flipping explains the whole thing.
                "resolution": round(1.0 / n, 4),
            })
    regressions.sort(key=lambda r: -r["drop"])
    return {"regressions": regressions, "skipped": skipped}


def check_thresholds(summary, thresholds=None):
    """Compare a summary against thresholds. Returns a list of alerts."""
    thresholds = thresholds or PRODUCTION_THRESHOLDS
    alerts = []

    def breach(name, observed, limit, direction):
        alerts.append({"check": name, "observed": observed, "threshold": limit,
                       "direction": direction,
                       "runs": summary.get("run_count", 0)})

    checks = [
        ("answerability_accuracy", "min_answerability_accuracy", "min"),
        ("avg_faithfulness_score", "min_avg_faithfulness", "min"),
        ("avg_latency_ms", "max_avg_latency_ms", "max"),
        ("total_cost_usd", "max_total_cost_usd", "max"),
        ("refusal_rate", "max_refusal_rate", "max"),
        ("injection_block_rate", "min_injection_block_rate", "min"),
        ("avg_citation_precision", "min_avg_citation_precision", "min"),
    ]
    for metric, key, direction in checks:
        if key not in thresholds:
            continue
        observed = summary.get(metric)
        if observed is None:
            continue
        limit = thresholds[key]
        if direction == "min" and observed < limit:
            breach(metric, observed, limit, "below")
        elif direction == "max" and observed > limit:
            breach(metric, observed, limit, "above")
    return alerts


def breach_probability(true_rate, slice_size, threshold, trials=20000,
                       seed=11):
    """How often a slice of this size lands below the threshold.

    Treats each run as a Bernoulli draw at `true_rate`, the system's actual
    quality, and counts how often `slice_size` runs average below `threshold`.
    Nothing changes between trials except which runs landed in the slice.

    What this number means depends on which side of the threshold the truth
    sits, and conflating the two is the easy mistake:

      true_rate >= threshold  the system is fine, so every breach is a FALSE
                              ALARM. A high number means the alert is noise.

      true_rate <  threshold  the system is genuinely below the bar, so a
                              breach is correct. Here the interesting figure
                              is 1 minus this: how often the slice stays
                              quiet and falsely reassures you.

    Use alert_credibility_for() to get the right one with the right label.
    """
    if slice_size <= 0:
        return 0.0
    rng = random.Random(seed)
    breaches = 0
    for _ in range(trials):
        hits = sum(1 for _ in range(slice_size) if rng.random() < true_rate)
        if hits / slice_size < threshold:
            breaches += 1
    return round(breaches / trials, 4)


def alert_false_alarm_rate(observed_rate, slice_size, threshold, trials=20000,
                           seed=11):
    """False alarm rate, valid only when the system is actually healthy.

    Returns None when observed_rate is below the threshold, because in that
    case a breach is a true positive and calling it a false alarm inverts the
    conclusion. Kept as a named function because that distinction is the whole
    point.
    """
    if observed_rate < threshold:
        return None
    return breach_probability(observed_rate, slice_size, threshold, trials,
                              seed)


def alert_miss_rate(true_rate, slice_size, threshold, trials=20000, seed=11):
    """How often a genuinely bad slice looks fine anyway.

    The companion error. With the truth below the bar, this is the chance the
    slice averages above it and the dashboard stays green over a real problem.
    """
    if true_rate >= threshold:
        return None
    return round(1.0 - breach_probability(true_rate, slice_size, threshold,
                                          trials, seed), 4)


def alert_credibility_for(true_rate, slice_size, threshold, trials=20000):
    """Return (kind, rate, meaning) for one slice size.

    kind is 'false_alarm' when the system is healthy and 'missed' when it is
    not, so a caller never has to guess which error it is looking at.
    """
    if true_rate >= threshold:
        rate = breach_probability(true_rate, slice_size, threshold, trials)
        if rate >= 0.5:
            meaning = "coin flip, ignore it"
        elif rate >= 0.2:
            meaning = "mostly noise"
        elif rate >= 0.05:
            meaning = "worth a look"
        else:
            meaning = "probably real"
        return "false_alarm", rate, meaning

    rate = round(1.0 - breach_probability(true_rate, slice_size, threshold,
                                          trials), 4)
    if rate >= 0.5:
        meaning = "usually stays silent"
    elif rate >= 0.2:
        meaning = "often stays silent"
    elif rate >= 0.05:
        meaning = "sometimes stays silent"
    else:
        meaning = "reliably catches it"
    return "missed", rate, meaning


def runs_needed_for_confidence(true_rate, threshold, confidence=0.95):
    """Smallest slice size whose error rate drops below 1 - confidence.

    Whichever error applies: false alarms when the system is healthy, missed
    detections when it is not. Returns None if no size up to 500 achieves it,
    which happens when the true rate sits very close to the threshold. That is
    a real answer rather than a failure: at that distance the two cannot be
    separated by counting runs.
    """
    target = 1 - confidence
    for size in range(1, 501):
        _, rate, _ = alert_credibility_for(true_rate, size, threshold,
                                           trials=4000)
        if rate <= target:
            return size
    return None


def dashboard(runs, thresholds=None, dimension="topic"):
    """A printable report: overall, per slice, alerts, and alert credibility."""
    summary = summarize_runs(runs)
    thresholds = thresholds or PRODUCTION_THRESHOLDS
    alerts = check_thresholds(summary, thresholds)
    slices = summarize_by_slice(runs, dimension)

    lines = []
    lines.append("=" * 74)
    lines.append("GitQuest production report")
    lines.append("=" * 74)
    if not summary.get("run_count"):
        lines.append("no runs")
        return "\n".join(lines)

    lines.append("runs                   %6d" % summary["run_count"])
    lines.append("pass rate              %6.1f%%" % (100 * summary["pass_rate"]))
    lines.append("answerability accuracy %6.1f%%"
                 % (100 * summary["answerability_accuracy"]))
    lines.append("avg faithfulness       %6.3f  (saturated, see below)"
                 % summary["avg_faithfulness_score"])
    lines.append("avg citation precision %6.3f" % summary["avg_citation_precision"])
    lines.append("avg citation recall    %6.3f" % summary["avg_citation_recall"])
    lines.append("refusal rate           %6.1f%%" % (100 * summary["refusal_rate"]))
    lines.append("avg latency            %6.1f ms" % summary["avg_latency_ms"])
    lines.append("total cost             $%.6f" % summary["total_cost_usd"])
    if summary.get("injection_block_rate") is not None:
        lines.append("injection block rate   %6.1f%% (%d cases)"
                     % (100 * summary["injection_block_rate"],
                        summary["injection_case_count"]))

    passing = summary["run_count"] * summary["answerability_accuracy"]
    if passing:
        lines.append("cost per passing answer $%.6f"
                     % (summary["total_cost_usd"] / passing))

    lines.append("")
    lines.append("By %s" % dimension)
    lines.append("-" * 74)
    lines.append("%-16s %5s %9s %9s %9s %11s"
                 % ("slice", "runs", "pass", "faith", "refuse", "resolution"))
    for name, slice_summary in slices.items():
        n = slice_summary["run_count"]
        lines.append("%-16s %5d %8.0f%% %9.3f %8.0f%% %10.0f%%"
                     % (name, n, 100 * slice_summary["pass_rate"],
                        slice_summary["avg_faithfulness_score"],
                        100 * slice_summary["refusal_rate"],
                        100.0 / n))

    lines.append("")
    lines.append("Alerts")
    lines.append("-" * 74)
    if not alerts:
        lines.append("none")
    for alert in alerts:
        lines.append("  %-24s %s threshold: %.4f vs %.4f  (%d runs)"
                     % (alert["check"], alert["direction"], alert["observed"],
                        alert["threshold"], alert["runs"]))
    return "\n".join(lines)
