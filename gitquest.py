"""The GitQuest RAG pipeline.

Retrieval, reranking, a token budget, prompt assembly, answer generation, a
prompt-injection defence, and a self-RAG retry loop.

Everything is deterministic and runs with no network and no API key. Retrieval
is lexical (BM25-style), and the answer is composed extractively from the
retrieved chunks. That is a real constraint, not a shortcut: a pipeline whose
output changes between runs cannot be used to detect a regression, which is
the entire purpose of the harness built on top of it.
"""

import math
import re
from collections import Counter

import corpus

PIPELINE_VERSION = "gitquest-1.2.0"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "does", "for", "from", "how", "i", "if", "in", "into", "is", "it", "its",
    "my", "of", "on", "or", "that", "the", "their", "them", "then", "there",
    "they", "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
}

# Cost model, in dollars per 1000 tokens. Fixed numbers so cost is a
# reproducible function of the prompt rather than a billing surprise.
COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.0006

REFUSAL = ("I cannot answer that from the documentation I have been given.")

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) instructions",
    r"disregard (all |any |the )?(previous|prior|above)",
    r"you are now",
    r"system prompt",
    r"reveal your (instructions|prompt|system)",
    r"print your (instructions|prompt)",
    r"new instructions:",
    r"</?(system|instructions)>",
]


def tokenize(text):
    words = re.findall(r"[a-z0-9\-]+", str(text).lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def estimate_tokens(text):
    """Roughly four characters per token, the usual working approximation."""
    return max(1, len(str(text)) // 4)


def _build_index(chunks):
    documents = [tokenize(c["text"]) for c in chunks]
    frequency = Counter()
    for document in documents:
        for term in set(document):
            frequency[term] += 1
    return documents, frequency


_DOCUMENTS, _DOC_FREQ = _build_index(corpus.CHUNKS)
_AVG_LEN = sum(len(d) for d in _DOCUMENTS) / len(_DOCUMENTS)


def retrieve(query, candidate_k=10, chunks=None):
    """BM25 lexical retrieval. Returns [{chunk_id, topic, text, score}]."""
    chunks = chunks if chunks is not None else corpus.CHUNKS
    documents, frequency = (_DOCUMENTS, _DOC_FREQ) if chunks is corpus.CHUNKS \
        else _build_index(chunks)
    average = (sum(len(d) for d in documents) / len(documents)) if documents else 1

    query_terms = tokenize(query)
    total = len(documents)
    k1, b = 1.5, 0.75

    scored = []
    for chunk, document in zip(chunks, documents):
        counts = Counter(document)
        length = len(document) or 1
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            df = frequency.get(term, 0) or 1
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            tf = counts[term]
            score += idf * (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * length / average))
        if score > 0:
            scored.append(dict(chunk, score=round(score, 6)))

    # Sort by score, then chunk_id, so ties never reorder between runs.
    scored.sort(key=lambda c: (-c["score"], c["chunk_id"]))
    return scored[:candidate_k]


def rerank(query, candidates, top_n=5):
    """Promote chunks that cover more distinct query terms.

    BM25 rewards repeating one rare term. A chunk touching four of the query's
    five terms is usually the better context even when it scores lower, so
    coverage breaks the tie.
    """
    query_terms = set(tokenize(query))
    reranked = []
    for candidate in candidates:
        terms = set(tokenize(candidate["text"]))
        covered = len(query_terms & terms)
        coverage = covered / len(query_terms) if query_terms else 0.0
        reranked.append(dict(candidate,
                             coverage=round(coverage, 4),
                             rerank_score=round(
                                 candidate["score"] * (1 + coverage), 6)))
    reranked.sort(key=lambda c: (-c["rerank_score"], c["chunk_id"]))
    return reranked[:top_n]


def apply_token_budget(chunks, token_budget=6000, reserve=512):
    """Drop the lowest ranked chunks until the context fits.

    Returns (kept, dropped_count). Enforcing this in the pipeline rather than
    hoping the context fits is what stops a long retrieval from silently
    truncating the most relevant chunk at the provider's end.
    """
    budget = max(0, token_budget - reserve)
    kept, used = [], 0
    dropped = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk["text"])
        if used + cost > budget:
            dropped += 1
            continue
        kept.append(chunk)
        used += cost
    return kept, dropped


def detect_injection(text):
    """Return the injection patterns a piece of text matches."""
    lowered = str(text).lower()
    return [pattern for pattern in INJECTION_PATTERNS
            if re.search(pattern, lowered)]


def sanitize_documents(documents):
    """Quarantine retrieved documents that try to issue instructions.

    Retrieved text is data. When a document contains something shaped like an
    instruction, it is almost certainly an injection, because documentation
    does not address the model. Quarantined documents are excluded from the
    context and reported, rather than dropped quietly.
    """
    clean, quarantined = [], []
    for document in documents:
        hits = detect_injection(document.get("text", ""))
        if hits:
            quarantined.append(dict(document, injection_patterns=hits))
        else:
            clean.append(document)
    return clean, quarantined


def build_prompt(query, chunks):
    parts = ["Answer the question using only the numbered sources below.",
             "Cite the chunk_id of every source you use.",
             "If the sources do not contain the answer, say so.", ""]
    for index, chunk in enumerate(chunks, start=1):
        parts.append("[%d] (%s) %s" % (index, chunk["chunk_id"], chunk["text"]))
    parts.append("")
    parts.append("Question: %s" % query)
    return "\n".join(parts)


def _compose_answer(query, chunks):
    """Build an extractive answer from the retrieved chunks.

    Picks the sentences that overlap the query most. Not a language model, and
    does not pretend to be: the point is a grounded, reproducible answer that
    the faithfulness check can actually score.
    """
    if not chunks:
        return REFUSAL, []

    query_terms = set(tokenize(query))
    scored_sentences = []
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.])\s+", chunk["text"]):
            sentence = sentence.strip()
            if not sentence:
                continue
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                scored_sentences.append((overlap, chunk["chunk_id"], sentence))

    if not scored_sentences:
        return REFUSAL, []

    scored_sentences.sort(key=lambda s: (-s[0], s[1], s[2]))
    chosen = scored_sentences[:3]
    cited = sorted({chunk_id for _, chunk_id, _ in chosen})
    answer = " ".join(sentence for _, _, sentence in chosen)
    return answer, cited


def ask_gitquest(query, config=None):
    """The plain pipeline: retrieve, rerank, budget, answer."""
    config = config or {}
    candidates = retrieve(query, config.get("candidate_k", 10))
    reranked = rerank(query, candidates, config.get("rerank_top_n", 5))
    kept, dropped = apply_token_budget(
        reranked, config.get("token_budget", 6000))
    answer, cited = _compose_answer(query, kept)

    prompt = build_prompt(query, kept)
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(answer)
    cost = (input_tokens / 1000.0 * COST_PER_1K_INPUT
            + output_tokens / 1000.0 * COST_PER_1K_OUTPUT)

    return {
        "query": query,
        "answer": answer,
        "cited_chunk_ids": cited,
        "retrieved_chunk_ids": [c["chunk_id"] for c in kept],
        "dropped_for_budget": dropped,
        "quarantined_chunk_ids": [],
        "injection_detected": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "pipeline_version": PIPELINE_VERSION,
        "refused": answer == REFUSAL,
    }


def ask_gitquest_secured(query, config=None, injected_docs=None):
    """The pipeline with prompt-injection defence.

    injected_docs simulates poisoned documents reaching retrieval, which is how
    the adversarial cases in the eval set exercise this path.
    """
    config = config or {}

    query_injection = detect_injection(query)

    candidates = retrieve(query, config.get("candidate_k", 10))
    if injected_docs:
        candidates = list(candidates) + [
            dict(document, score=99.0) for document in injected_docs]
        candidates.sort(key=lambda c: (-c.get("score", 0), c["chunk_id"]))

    clean, quarantined = sanitize_documents(candidates)

    reranked = rerank(query, clean, config.get("rerank_top_n", 5))
    kept, dropped = apply_token_budget(
        reranked, config.get("token_budget", 6000))
    answer, cited = _compose_answer(query, kept)

    prompt = build_prompt(query, kept)
    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(answer)
    cost = (input_tokens / 1000.0 * COST_PER_1K_INPUT
            + output_tokens / 1000.0 * COST_PER_1K_OUTPUT)

    return {
        "query": query,
        "answer": answer,
        "cited_chunk_ids": cited,
        "retrieved_chunk_ids": [c["chunk_id"] for c in kept],
        "dropped_for_budget": dropped,
        "quarantined_chunk_ids": [q["chunk_id"] for q in quarantined],
        "injection_detected": bool(quarantined or query_injection),
        "query_injection_patterns": query_injection,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(cost, 8),
        "pipeline_version": PIPELINE_VERSION,
        "refused": answer == REFUSAL,
    }


def run_self_rag_loop(query, config=None, max_attempts=2):
    """Retry with wider retrieval when the first answer is poorly grounded.

    Self-RAG in its simplest useful form: score the answer, and if it is below
    the bar, widen retrieval and try again. Returns the best attempt and the
    full trace, because knowing a retry happened is what makes a latency or
    cost spike explainable later.
    """
    import judge

    config = dict(config or {})
    attempts = []
    best = None

    for attempt in range(1, max_attempts + 1):
        result = ask_gitquest_secured(query, config)
        score = judge.faithfulness_score(result["answer"],
                                         result["retrieved_chunk_ids"])
        attempts.append({"attempt": attempt,
                         "faithfulness": score,
                         "cited": result["cited_chunk_ids"],
                         "candidate_k": config.get("candidate_k", 10)})
        if best is None or score > best[1]:
            best = (result, score)
        if score >= judge.FAITHFULNESS_BAR:
            break
        # Widen retrieval for the next attempt.
        config["candidate_k"] = config.get("candidate_k", 10) + 5
        config["rerank_top_n"] = config.get("rerank_top_n", 5) + 2

    result, score = best
    result = dict(result)
    result["self_rag_attempts"] = len(attempts)
    result["self_rag_trace"] = attempts
    result["faithfulness"] = score
    # Every attempt cost money, so the reported cost is the sum, not the last.
    result["estimated_cost_usd"] = round(
        result["estimated_cost_usd"] * len(attempts), 8)
    return result


def build_run_log(case, result, judgement, config, latency_ms):
    """One row per run, the unit everything downstream aggregates."""
    return {
        "user_query": case.get("user_query"),
        "case_type": case.get("case_type", "curated"),
        "tags": case.get("tags", []),
        "topic": case.get("topic"),
        "answer": result.get("answer"),
        "cited_chunk_ids": result.get("cited_chunk_ids", []),
        "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
        "refused": result.get("refused", False),
        "injection_detected": result.get("injection_detected", False),
        "quarantined_chunk_ids": result.get("quarantined_chunk_ids", []),
        "self_rag_attempts": result.get("self_rag_attempts", 1),
        "faithfulness_score": judgement.get("faithfulness_score"),
        "citation_precision": judgement.get("citation_precision"),
        "citation_recall": judgement.get("citation_recall"),
        "answerable_correct": judgement.get("answerable_correct"),
        "passed": judgement.get("passed"),
        "estimated_cost_usd": result.get("estimated_cost_usd", 0.0),
        "latency_ms": latency_ms,
        "pipeline_version": config.get("pipeline_version", PIPELINE_VERSION),
        "model": config.get("model"),
        "candidate_k": config.get("candidate_k"),
        "rerank_top_n": config.get("rerank_top_n"),
    }
