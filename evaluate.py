"""Scoring a curated eval item end to end."""

import time

import gitquest
import judge


def score_item(item, config=None, runner=None):
    """Run one curated item through the pipeline and judge the result.

    runner is injectable so the harness can score a variant config, or a test
    can substitute a stub, without this module knowing about either.
    """
    config = config or {}
    runner = runner or gitquest.ask_gitquest_secured

    started = time.perf_counter()
    result = runner(item["user_query"], config)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)

    judgement = judge.heuristic_judge(result, item)
    return result, judgement, latency_ms
