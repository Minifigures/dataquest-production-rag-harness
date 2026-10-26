"""The production harness: run a batch, diff a config change, report.

    python3 production_gitquest.py --report
    python3 production_gitquest.py --canary --variant-k 20
    python3 production_gitquest.py --alert-credibility

Three pieces matter here.

The **case-handler registry** maps a case type to the function that runs it.
Dispatch is a dict lookup rather than a branch, so adding a failure mode means
registering a handler from outside the class instead of editing it. Every
handler obeys one contract, (case, config) -> (result, judgement), which is
what lets the harness call any of them without knowing which it got.

A **canary run** sends the same case through two configs and records the
per-item diff. Aggregate comparison tells you the averages moved; a canary
tells you which answer moved, whether its citations changed, and what it cost.

The **report** rolls the batch up and checks it against thresholds. It also
reports how much each alert is worth, which is the part most dashboards leave
out and the reason most of them get ignored.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import gitquest
import monitoring
from evaluate import score_item
from gitquest import (PIPELINE_VERSION, ask_gitquest_secured, build_run_log,
                      run_self_rag_loop)
from judge import heuristic_judge
from monitoring import (PRODUCTION_THRESHOLDS, check_thresholds, dashboard,
                        regressions_by_slice, summarize_runs)

DEFAULT_CONFIG = {
    "corpus": "full",
    "candidate_k": 10,
    "rerank_top_n": 5,
    "token_budget": 6000,
    "model": "gpt-4o-mini",
    "pipeline_version": PIPELINE_VERSION,
}


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


# --------------------------------------------------------------------------
# Handlers. Each obeys (case, config) -> (result, judgement).
# --------------------------------------------------------------------------

def handle_curated(case, config):
    result, judgement, latency_ms = score_item(case, config)
    result["_latency_ms"] = latency_ms
    return result, judgement


def handle_injection(case, config):
    started = time.perf_counter()
    result = ask_gitquest_secured(case["user_query"], config,
                                  injected_docs=case.get("injected_docs"))
    result["_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result, heuristic_judge(result, case)


def handle_self_rag_retry(case, config):
    started = time.perf_counter()
    result = run_self_rag_loop(case["user_query"], config)
    result["_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result, heuristic_judge(result, case)


class ProductionHarness:
    """Owns the config, the thresholds, the run logs and the canary diffs."""

    def __init__(self, config=None, thresholds=None):
        self.config = dict(config or DEFAULT_CONFIG)
        self.thresholds = dict(thresholds or PRODUCTION_THRESHOLDS)
        self.runs = []
        self.canaries = []
        self._handlers = {}

    # -- registry ----------------------------------------------------------

    def register(self, case_type, handler):
        """Add a handler. Refuses to silently replace an existing one."""
        if case_type in self._handlers:
            raise ValueError(
                "A handler for %r is already registered. Registering twice is "
                "almost always an import run more than once." % case_type)
        self._handlers[case_type] = handler
        return self

    def resolve(self, case_type):
        handler = self._handlers.get(case_type)
        if handler is None:
            raise KeyError(
                "No handler registered for case type %r. Registered: %s"
                % (case_type, sorted(self._handlers)))
        return handler

    @property
    def case_types(self):
        return sorted(self._handlers)

    # -- running -----------------------------------------------------------

    @staticmethod
    def normalize_item(item):
        """Reshape a curated eval item into the case shape handlers expect.

        Touches no instance state, so it is a staticmethod rather than
        pretending to need self.
        """
        return {
            "case_type": item.get("case_type", "curated"),
            "user_query": item["user_query"],
            "topic": item.get("topic"),
            "expected_behavior": item.get("expected_behavior", "answer"),
            "trusted_evidence": item.get("trusted_evidence", []),
            "tags": item.get("tags", []),
            "injected_docs": item.get("injected_docs"),
        }

    def run_case(self, case, config=None):
        """Run one case through its handler and append a scored run log."""
        case = self.normalize_item(case)
        config = dict(config or self.config)
        handler = self.resolve(case["case_type"])

        started = time.perf_counter()
        result, judgement = handler(case, config)
        latency_ms = result.pop("_latency_ms", None)
        if latency_ms is None:
            latency_ms = round((time.perf_counter() - started) * 1000, 3)

        log = build_run_log(case, result, judgement, config, latency_ms)
        self.runs.append(log)
        return log

    def run_batch(self, cases, config=None):
        return [self.run_case(case, config) for case in cases]

    # -- canary ------------------------------------------------------------

    def run_canary(self, case, variant_config):
        """Run one case under control and variant, and record the diff."""
        case = self.normalize_item(case)
        handler = self.resolve(case["case_type"])

        control_result, control_judgement = handler(case, dict(self.config))
        variant_result, variant_judgement = handler(case, dict(variant_config))

        def side(result, judgement):
            return {
                "answer": result.get("answer"),
                "cited_chunk_ids": result.get("cited_chunk_ids", []),
                "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
                "latency_ms": result.get("_latency_ms"),
                "faithfulness_score": judgement.get("faithfulness_score"),
                "citation_precision": judgement.get("citation_precision"),
            }

        control = side(control_result, control_judgement)
        variant = side(variant_result, variant_judgement)

        diff = {
            "user_query": case["user_query"],
            "case_type": case["case_type"],
            "topic": case.get("topic"),
            "control": control,
            "variant": variant,
            "citations_changed": (sorted(control["cited_chunk_ids"])
                                  != sorted(variant["cited_chunk_ids"])),
            "cost_delta_usd": round(
                (variant["estimated_cost_usd"] or 0.0)
                - (control["estimated_cost_usd"] or 0.0), 8),
            "faithfulness_delta": round(
                (variant["faithfulness_score"] or 0.0)
                - (control["faithfulness_score"] or 0.0), 4),
            "precision_delta": round(
                (variant["citation_precision"] or 0.0)
                - (control["citation_precision"] or 0.0), 4),
            "answer_changed": control["answer"] != variant["answer"],
        }
        self.canaries.append(diff)
        return diff

    # -- reporting ---------------------------------------------------------

    def report(self, dimension="topic"):
        summary = summarize_runs(self.runs)
        alerts = check_thresholds(summary, self.thresholds)
        passing = (summary.get("run_count", 0)
                   * (summary.get("answerability_accuracy") or 0.0))
        return {
            "summary": summary,
            "alerts": alerts,
            "cost_per_passing_answer_usd": (
                round(summary.get("total_cost_usd", 0.0) / passing, 8)
                if passing else None),
            "slices": monitoring.summarize_by_slice(self.runs, dimension),
            "canary_count": len(self.canaries),
        }


def build_harness(config=None):
    harness = ProductionHarness(config)
    harness.register("curated", handle_curated)
    harness.register("injection", handle_injection)
    harness.register("self_rag_retry", handle_self_rag_retry)
    return harness


def alert_credibility(harness, report):
    """Say what each alert is actually worth on a batch this size.

    An alert is a prompt to look, not a finding. This computes, for every
    slice, how often a healthy system would breach the pass-rate threshold
    purely by sampling noise, which is the number that decides whether the
    alert deserves attention.
    """
    summary = report["summary"]

    # Analyse whichever alert actually fired, rather than a metric that is not
    # in trouble. A rate-shaped metric is needed for the simulation, so the
    # first firing "below" alert is used, falling back to the pass rate.
    firing = [a for a in report["alerts"] if a["direction"] == "below"]
    if firing:
        metric_name = firing[0]["check"]
        true_rate = firing[0]["observed"]
        threshold = firing[0]["threshold"]
    else:
        metric_name = "pass_rate"
        true_rate = summary.get("pass_rate") or 0.0
        threshold = harness.thresholds.get("min_answerability_accuracy", 0.8)

    lines = ["", "=" * 74,
             "What the alerts are worth",
             "=" * 74,
             "Metric under test: %s" % metric_name,
             "Observed across the whole batch: %.3f" % true_rate,
             "Slice alert fires below: %.2f" % threshold,
             "",
             ""]

    healthy = true_rate >= threshold
    if healthy:
        lines.append("The batch is above the bar, so any slice that breaches")
        lines.append("it does so by sampling noise. How often that happens:")
        column = "false alarm"
    else:
        lines.append("The batch is genuinely BELOW the bar, so a breach is")
        lines.append("correct. The error that matters is the opposite one:")
        lines.append("how often a slice stays quiet over a real problem.")
        column = "stays quiet"
    lines.append("")
    lines.append("%-10s %-14s %s" % ("slice size", column, "meaning"))

    for size in (2, 3, 4, 5, 8, 10, 20, 50):
        _, rate, meaning = monitoring.alert_credibility_for(
            true_rate, size, threshold)
        lines.append("%-10d %-13.1f%% %s" % (size, 100 * rate, meaning))

    needed = monitoring.runs_needed_for_confidence(true_rate, threshold)
    lines.append("")
    if needed:
        lines.append("A slice needs %d runs before its verdict is trustworthy "
                     "at 95%% confidence." % needed)
    else:
        lines.append("No slice size up to 500 reaches 95% confidence here: "
                     "the observed rate is too close to the threshold to "
                     "separate by counting runs.")

    sizes = {name: s["run_count"] for name, s in report["slices"].items()}
    too_small = {n: c for n, c in sizes.items() if needed and c < needed}
    lines.append("")
    lines.append("Slices in this batch: %d, of which %d are below that size."
                 % (len(sizes), len(too_small)))
    if too_small:
        lines.append("Below the bar: %s"
                     % ", ".join("%s (%d)" % (n, c)
                                 for n, c in sorted(too_small.items())))
        lines.append("")
        lines.append("Those slices cannot support a regression claim. They are")
        lines.append("not healthy, they are unmeasured, and a dashboard that")
        lines.append("shows them as green or red is reporting noise either way.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-items", default="data/eval_items.jsonl")
    parser.add_argument("--advanced-cases", default="data/advanced_cases.jsonl")
    parser.add_argument("--out", default="data/run_logs.jsonl")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--variant-k", type=int, default=20)
    parser.add_argument("--alert-credibility", action="store_true")
    args = parser.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    harness = build_harness(config)

    items = read_jsonl(args.eval_items)
    cases = read_jsonl(args.advanced_cases)
    harness.run_batch(items)
    harness.run_batch(cases)
    write_jsonl(args.out, harness.runs)
    print("wrote %d run logs to %s" % (len(harness.runs), args.out))

    report = harness.report()

    if args.report or not (args.canary or args.alert_credibility):
        print()
        print(dashboard(harness.runs, harness.thresholds))
        if report["cost_per_passing_answer_usd"] is not None:
            print("\ncost per passing answer: $%.6f"
                  % report["cost_per_passing_answer_usd"])

    if args.canary:
        variant = dict(config, candidate_k=args.variant_k,
                       rerank_top_n=config["rerank_top_n"] + 3)
        print()
        print("=" * 74)
        print("Canary: candidate_k %d -> %d, rerank_top_n %d -> %d"
              % (config["candidate_k"], variant["candidate_k"],
                 config["rerank_top_n"], variant["rerank_top_n"]))
        print("=" * 74)
        changed = 0
        for case in items[:8] + cases:
            diff = harness.run_canary(case, variant)
            if diff["citations_changed"] or diff["answer_changed"]:
                changed += 1
                print("\n%s" % diff["user_query"][:66])
                print("   citations %s -> %s"
                      % (diff["control"]["cited_chunk_ids"],
                         diff["variant"]["cited_chunk_ids"]))
                print("   faithfulness %+.3f   precision %+.3f   cost $%+.8f"
                      % (diff["faithfulness_delta"], diff["precision_delta"],
                         diff["cost_delta_usd"]))
        print("\n%d of %d canary cases changed."
              % (changed, len(harness.canaries)))
        faith_moves = [d["faithfulness_delta"] for d in harness.canaries]
        prec_moves = [d["precision_delta"] for d in harness.canaries]
        cost_moves = [d["cost_delta_usd"] for d in harness.canaries]
        print("faithfulness moved on %d of %d cases (total %+.3f)"
              % (sum(1 for d in faith_moves if d), len(faith_moves),
                 sum(faith_moves)))
        print("precision    moved on %d of %d cases (total %+.3f)"
              % (sum(1 for d in prec_moves if d), len(prec_moves),
                 sum(prec_moves)))
        print("cost         moved on %d of %d cases (total $%+.8f)"
              % (sum(1 for d in cost_moves if d), len(cost_moves),
                 sum(cost_moves)))
        if changed == 0:
            print("A config change that moves nothing is not safe by default,")
            print("it is untested: the eval set may simply not reach the code")
            print("path the change affects.")

    if args.alert_credibility:
        print(alert_credibility(harness, report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
