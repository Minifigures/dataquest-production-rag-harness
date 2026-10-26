# Production RAG Harness

Dataquest guided project, *Production RAG Application*. A harness that runs a
RAG pipeline against an eval set, diffs config changes case by case, and
reports the batch against thresholds.

```
python3 production_gitquest.py --report
python3 production_gitquest.py --canary --variant-k 20
python3 production_gitquest.py --alert-credibility
python3 -m unittest -v test_harness      # 50 tests
```

Everything runs offline with no API key. Retrieval is BM25 over a committed
16-chunk git corpus and answers are composed extractively. That is a
requirement, not a shortcut: **a pipeline whose output varies between runs
cannot be used to detect a regression**, because every comparison is
confounded by the pipeline itself. The judges are deterministic for the same
reason, since an LLM judge adds its own noise to the number you are trying to
read.

## What the dashboard said, and what was actually true

The brief hints that "most of the alerts it raises on this data are not the
failures they look like". That turned out to be the smaller half of the
problem. Here is the report on 26 runs:

```
runs                       26
pass rate               100.0%
answerability accuracy  100.0%
avg faithfulness        1.000
refusal rate             11.5%
injection block rate    100.0% (3 cases)
```

A clean sweep. Every case passed, grounding is perfect, every injection was
blocked. Nothing to do.

Except the pipeline cites the wrong chunk about half the time, and none of
those numbers can see it.

### Two of the three quality metrics are incapable of failing

`metric_discrimination()` measures the spread of every metric across the batch
and reports whether it could detect anything:

```
metric                      n       min       max       std  verdict
faithfulness_score         26    1.0000    1.0000    0.0000  SATURATED, cannot detect anything
citation_precision         26    0.0000    1.0000    0.3067  discriminates
citation_recall            26    0.0000    1.0000    0.3331  discriminates
latency_ms                 26    0.0310    0.6880    0.1270  discriminates
estimated_cost_usd         26    0.0000    0.0001    0.0000  discriminates
```

Faithfulness is exactly 1.0000 on all 26 runs, standard deviation zero. The
reason is structural: the answer is composed from sentences copied out of the
retrieved chunks, so every content word in the answer is by construction
present in the context. The metric is measuring whether copied text resembles
the text it was copied from.

The `min_avg_faithfulness: 0.85` threshold in the config therefore **cannot
fire, ever, for any input.** It is not a safety net, it is a decoration that
looks like one. The same would be true of any extractive or heavily templated
pipeline, which is a large share of real RAG systems.

So the first thing worth doing to an eval set is not running it. It is
checking which of your metrics have any room to move. A saturated metric is
not a passing grade, it is an instrument with the needle glued in place.

Adding the metric that does move changes the verdict completely:

```
avg citation precision  0.532
avg citation recall     0.846

Alerts
  avg_citation_precision   below threshold: 0.5320 vs 0.8000  (26 runs)
```

Pass rate 100%, and 47% of citations point at chunks that are not the evidence
for the question. Asked why branching is fast, the pipeline cites the chunk
about `git bisect`.

### The canary proves the blind spot

A canary run sends the same cases through two configs and diffs them per item.
Widening retrieval from `candidate_k=10, rerank_top_n=5` to `20, 8`:

```
Why is branching in git so fast?
   citations ['git-branch-01'] -> ['git-bisect-01', 'git-branch-01']
   faithfulness +0.000   precision -0.500   cost $+0.00003615

Why should I avoid rebasing shared commits?
   citations ['git-log-01', 'git-rebase-01'] -> ['git-bisect-01', 'git-log-01', 'git-rebase-01']
   faithfulness +0.000   precision -0.167   cost $+0.00002340

3 of 14 canary cases changed.
faithfulness moved on 0 of 14 cases (total +0.000)
precision    moved on 2 of 14 cases (total -0.667)
cost         moved on 9 of 14 cases (total $+0.00021840)
```

The change pulled irrelevant chunks into the citations and raised cost on 9 of
14 cases. **Faithfulness moved on zero of them.** A team gating deploys on
faithfulness would have shipped a measurable quality regression with a green
dashboard and a cost increase, and had no instrument that could have said
otherwise.

### Every slice on the dashboard is too small to mean anything

The per-topic table looks like a monitoring surface. It holds 10 slices of 2
to 4 runs each, so each slice's pass rate can only move in steps of 25% to 50%.

`--alert-credibility` simulates how much a slice verdict is worth, treating
each run as a draw at the batch's observed rate:

```
Metric under test: avg_citation_precision
Observed across the whole batch: 0.532
Slice alert fires below: 0.80

The batch is genuinely BELOW the bar, so a breach is
correct. The error that matters is the opposite one:
how often a slice stays quiet over a real problem.

slice size stays quiet    meaning
2          29.0         % often stays silent
3          15.7         % sometimes stays silent
4          8.5          % sometimes stays silent
10         8.4          % sometimes stays silent
20         1.2          % reliably catches it
50         0.0          % reliably catches it

A slice needs 9 runs before its verdict is trustworthy at 95% confidence.
Slices in this batch: 10, of which 10 are below that size.
```

All ten. Not one slice in this dashboard can support a claim about its topic.
They are not healthy and they are not failing, they are **unmeasured**, and a
dashboard that colours them green or red is reporting noise in both
directions.

Writing that analysis caught a mistake in my own first version of it. I had
labelled every breach a "false alarm", which is only correct when the true
rate sits **above** the threshold. Here precision is genuinely below the bar,
so a breach is a true positive and the interesting error is the opposite one:
how often a slice stays quiet over a real problem. `alert_false_alarm_rate()`
now returns `None` when the system is unhealthy and `alert_miss_rate()`
returns `None` when it is healthy, so the two cannot be confused again, and
`alert_credibility_for()` returns which error it is along with the number.

## The design the brief asks for

**A case-handler registry.** Dispatch is a dict lookup rather than a branch
per case type, so a new failure mode is registered from outside the class with
no edit to it. `register()` raises rather than replacing an existing handler,
because a silent replacement is almost always a module imported twice and the
resulting behaviour is extremely hard to trace.

**One handler contract**, `(case, config) -> (result, judgement)`. A test
walks every registered handler and calls it through the same contract, so a
handler that drifts fails immediately rather than at dispatch time.

**Canary diffs** recording both sides' answer, citations, cost, latency,
faithfulness and precision, plus whether citations changed and the cost delta.

**A batch report** with cost per passing answer and threshold checks.

## Other decisions

**`regressions_by_slice` skips thin slices loudly.** A slice under `min_runs`
is reported in a `skipped` list rather than dropped, so a regression hiding in
a two-run slice shows up as "not measurable" instead of "not present". Each
regression also carries its `resolution`, the smallest change the slice can
express. When the resolution equals the drop, one run flipping explains the
entire regression.

**Refusals score 1.0 on faithfulness.** Declining to answer invents nothing.
Scoring a refusal as unfaithful would push the pipeline toward answering
questions it cannot support.

**An injection that silences the system is still a successful attack.** Cases
tagged `resist` require the pipeline to answer the real question *and* flag
the injection. A refusal fails them, which is tested.

**Self-RAG cost is summed across attempts, not taken from the last one.** A
retry is not free, and reporting only the final attempt hides exactly the
spike you would want to explain later.

## Files

| File | Contents |
|---|---|
| `corpus.py` | 16 committed git documentation chunks |
| `gitquest.py` | retrieval, rerank, token budget, injection defence, self-RAG |
| `judge.py` | deterministic faithfulness, citation precision and recall |
| `evaluate.py` | scoring one curated item |
| `monitoring.py` | summaries, slices, thresholds, discrimination, alert statistics |
| `production_gitquest.py` | the harness: registry, batch, canary, report |
| `data/eval_items.jsonl` | 20 curated items, 3 of which should be refused |
| `data/advanced_cases.jsonl` | 3 injection cases, 3 self-RAG retry cases |
| `test_harness.py` | 50 tests |

## What I would do next

Replace the extractive answerer with a real model behind the same interface.
That is the point at which faithfulness stops being saturated and starts
earning its threshold, and `metric_discrimination()` is how you would confirm
it rather than assume it. The eval set also needs to grow to roughly 9 runs
per slice before any per-topic claim is worth making, which on 10 topics means
about 90 curated items rather than 20.
