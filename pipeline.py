#!/usr/bin/env python3
"""
pulse-pipeline  v2.3
Fetch → Filter arXiv → Filter → Cluster → Deduplicate → Generate → Write → Exit

Changes from v2.2:
  - NIM key fallback: if NIM_API_KEY hits a rate-limit or auth error, the
    pipeline transparently retries the same call with NIM_API_KEY_2 (if set).
    Key selection is tracked run-wide so all clusters in a failing run use the
    same fallback key rather than re-probing on every call.
    Reverts to NIM_API_KEY automatically on the next run.

Changes from v2.1:
  - step1b_filter_arxiv: pure-code relevance scorer for arXiv articles.
    Scores each paper on three signals (topic keywords, known-lab authorship,
    novelty language) then keeps only the top ARXIV_TOP_K per run.
    No extra LLM call — zero added latency or cost.

Changes from v2.0:
  - MIN_CLUSTER_SIZE removed: solo articles from editorial sources are now kept
  - step3_cluster returns ClusterResult dataclass tagging each cluster as
    "multi" or "solo" so the LLM prompt and DB write can adapt accordingly
  - Tier-1 solo articles always pass through; Tier-2 solos are gated by a
    configurable SOLO_TIER2_MIN_SNIPPET guard to suppress low-quality stubs
  - _build_prompt adapts its language for single-article clusters
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

UTC = timezone.utc

# ── Config ────────────────────────────────────────────────────────────────────
FETCH_TIMEOUT           = 8        # seconds per feed request
MAX_WORKERS             = 10       # parallel feed fetchers
CLUSTER_THRESHOLD       = 0.75     # cosine similarity floor for grouping
SNIPPET_LEN             = 400      # chars of content passed to LLM
AGE_CUTOFF_HOURS        = 48
PRUNE_AFTER_DAYS        = 7
LLM_TOP_K               = 5        # max articles sent to LLM per cluster

# v2.1 — solo article controls
# MIN_CLUSTER_SIZE is gone. Solo articles are always kept, subject to:
SOLO_TIER2_MIN_SNIPPET  = 80       # Tier-2 solo must have ≥ this many snippet
                                   # chars or it's treated as a stub and dropped.
                                   # Tier-1 solos always pass regardless.

# v2.2 — arXiv relevance filter
ARXIV_TOP_K             = 5        # keep only the N highest-scoring arXiv papers per run
ARXIV_MIN_SCORE         = 1        # papers scoring below this are dropped even if
                                   # they would land in the top-K (safety floor)
ARXIV_SOURCE_NAMES      = {        # source_name values that identify arXiv feeds
    "arXiv cs.AI",
    "arXiv cs.LG (Machine Learning)",
}

# ── Taxonomy ──────────────────────────────────────────────────────────────────
ALLOWED_CATEGORIES  = {"Tech", "Dev", "Business", "Market", "Governance", "Science", "Society"}
ALLOWED_ENTERPRISES = {
    "OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Microsoft", "Apple",
    "Amazon", "Nvidia", "xAI", "Mistral", "Samsung", "Baidu", "IBM", "Salesforce",
    "Adobe", "Stability AI", "Hugging Face", "Cohere", "Other",
}
ALLOWED_REGIONS     = {
    "USA", "Europe", "China", "Asia-Pacific", "Middle East", "Latin America", "Africa", "Global",
}
ALLOWED_STORY_TYPES = {
    "Research", "Product Launch", "Funding", "Acquisition",
    "Policy", "Incident", "Partnership", "Personnel", "Opinion",
}
ALLOWED_IMPACT      = {"Breakthrough", "Significant", "Routine"}
ALLOWED_AUDIENCES   = {"General Public", "Developers", "Investors", "Policymakers", "Researchers"}

# ── Clients ───────────────────────────────────────────────────────────────────
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

# NIM dual-key client manager
# Primary key is always NIM_API_KEY.
# NIM_API_KEY_2 is optional — used as a one-run fallback if the primary key
# fails with a rate-limit (429) or auth (401/403) error.
# The active key is stored in _nim_active_key and can be switched at most once
# per run, so all clusters after a failure use the same fallback key.
_NIM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
_NIM_PRIMARY_KEY = os.environ["NIM_API_KEY"]
_NIM_FALLBACK_KEY: str | None = os.environ.get("NIM_API_KEY_2")  # optional

# Errors that warrant a key switch rather than a hard failure
_NIM_RETRYABLE_CODES = {429, 401, 403}

class _NimClientManager:
    """
    Thin wrapper that holds two OpenAI clients (primary + optional fallback)
    and exposes a single .complete() method that handles key switching
    transparently.

    Key-switch semantics:
    - Switches at most once per pipeline run (run-level, not call-level).
    - Once switched, all subsequent calls in the same run use the fallback key.
    - The switch is logged clearly so it shows up in CI logs.
    - If no fallback key is configured, or if the fallback also fails,
      the exception propagates to _call_llm as before.
    """

    def __init__(self, primary_key: str, fallback_key: str | None) -> None:
        self._primary  = OpenAI(base_url=_NIM_BASE_URL, api_key=primary_key)
        self._fallback = (
            OpenAI(base_url=_NIM_BASE_URL, api_key=fallback_key)
            if fallback_key else None
        )
        self._active        = self._primary
        self._switched      = False
        self._primary_label = "NIM_API_KEY"

    def _should_fallback(self, exc: Exception) -> bool:
        """Return True if this exception is a retryable key error."""
        if self._switched or self._fallback is None:
            return False
        # openai SDK wraps HTTP errors as openai.APIStatusError with .status_code
        code = getattr(exc, "status_code", None)
        if code in _NIM_RETRYABLE_CODES:
            return True
        # Fallback: check the string representation for common patterns
        msg = str(exc).lower()
        return any(t in msg for t in ("rate limit", "rate_limit", "429",
                                      "unauthorized", "forbidden", "invalid api key"))

    def complete(self, **kwargs) -> object:
        """
        Call chat.completions.create(**kwargs) on the active client.
        If it fails with a retryable error and a fallback key is available,
        switch once and retry the same call.
        """
        try:
            return self._active.chat.completions.create(**kwargs)
        except Exception as exc:
            if self._should_fallback(exc):
                log.warning(
                    "NIM primary key error (%s) — switching to NIM_API_KEY_2 for this run.",
                    exc,
                )
                self._active   = self._fallback
                self._switched = True
                # Retry with fallback key — let any new exception propagate
                return self._active.chat.completions.create(**kwargs)
            raise


nim = _NimClientManager(_NIM_PRIMARY_KEY, _NIM_FALLBACK_KEY)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def strip_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(separator=" ").strip()


def parse_entry_date(entry) -> datetime | None:
    import calendar
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime.fromtimestamp(calendar.timegm(t), tz=UTC)
            except Exception:
                pass
    for attr in ("published", "updated"):
        s = getattr(entry, attr, "")
        if s:
            try:
                dt = dateutil_parser.parse(s)
                return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
            except Exception:
                pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH
# ═══════════════════════════════════════════════════════════════════════════════

def load_sources(path: str = "sources.yaml") -> list[dict]:
    with open(path) as fh:
        return yaml.safe_load(fh)["sources"]


def _fetch_one(source: dict) -> list[dict]:
    articles: list[dict] = []
    try:
        resp = requests.get(
            source["feed_url"],
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "pulse-pipeline/2.1"},
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        for entry in feed.entries:
            pub   = parse_entry_date(entry)
            title = strip_html(entry.get("title", "")).strip()
            url   = entry.get("link", "").strip()
            if not pub or not title or not url:
                continue
            raw = ""
            if hasattr(entry, "content") and entry.content:
                raw = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                raw = entry.summary or ""
            articles.append({
                "title":           title,
                "url":             url,
                "source_name":     source["name"],
                "tier":            source.get("tier", 2),
                "published_at":    pub,
                "content_snippet": strip_html(raw)[:SNIPPET_LEN],
            })
    except Exception as e:
        log.warning("Feed failed [%s]: %s", source["name"], e)
    return articles


def step1_fetch(sources: list[dict]) -> list[dict]:
    log.info("STEP 1: Fetching %d feeds…", len(sources))
    all_articles: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in sources}
        for fut in as_completed(futures):
            all_articles.extend(fut.result())
    log.info("  Fetched %d articles total.", len(all_articles))
    return all_articles


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1b — arXiv RELEVANCE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Scoring tables ────────────────────────────────────────────────────────────

# Topic keywords — matched against title + snippet (lowercased).
# Weight reflects how central the topic is to Pulse's audience.
_ARXIV_TOPIC_KEYWORDS: list[tuple[int, list[str]]] = [
    # weight 3 — core applied / product-adjacent topics
    (3, [
        "large language model", "llm", "language model",
        "foundation model", "generative ai", "generative model",
        "multimodal", "vision language", "text-to-image", "text-to-video",
        "agent", "agentic", "tool use", "tool-use", "function calling",
        "rlhf", "reinforcement learning from human feedback",
        "instruction tuning", "fine-tuning", "finetuning",
        "retrieval augmented", "rag",
        "inference", "inference efficiency", "speculative decoding",
        "quantization", "model compression", "distillation",
        "prompt", "in-context learning", "chain-of-thought", "reasoning",
        "code generation", "code llm", "coding model",
    ]),
    # weight 2 — high-impact theoretical / safety / scaling
    (2, [
        "scaling law", "scaling behavior", "emergent",
        "alignment", "ai safety", "constitutional ai",
        "hallucination", "factuality", "faithfulness",
        "reward model", "preference learning", "dpo", "ppo",
        "transformer", "attention mechanism", "mixture of experts", "moe",
        "diffusion model", "flow matching",
        "benchmark", "evaluation", "leaderboard",
        "robotics", "embodied", "autonomous",
        "medical ai", "clinical", "healthcare ai",
        "speech", "audio language model", "tts",
    ]),
    # weight 1 — broader ML (relevant but lower priority)
    (1, [
        "neural network", "deep learning",
        "graph neural", "gnn",
        "federated learning",
        "continual learning", "lifelong learning",
        "explainability", "interpretability",
        "bias", "fairness", "toxicity",
        "privacy", "differential privacy",
        "zero-shot", "few-shot", "meta-learning",
    ]),
]

# Lab / institution signals — presence in title or snippet boosts score.
# These labs consistently produce high-impact work.
_ARXIV_LAB_SIGNALS: list[str] = [
    "openai", "anthropic", "google deepmind", "deepmind", "google brain",
    "meta ai", "fair,", "meta fair",
    "microsoft research",
    "nvidia research",
    "stanford", "mit csail", "berkeley", "carnegie mellon", "cmu",
    "oxford", "cambridge", "eth zurich",
    "hugging face", "mistral",
    "allen institute", "ai2",
]

# Novelty language — phrases that suggest an original contribution
# vs. a survey, commentary, or incremental tweak.
_ARXIV_NOVELTY_PHRASES: list[str] = [
    "we propose", "we introduce", "we present",
    "we develop", "we design", "we release",
    "novel", "new approach", "new method", "new framework",
    "outperforms", "state-of-the-art", "sota",
    "surpasses", "significantly improves", "substantially better",
    "first to", "first model", "first framework",
]


def _score_arxiv(article: dict) -> float:
    """
    Score an arXiv article on three independent signals.
    Returns a float; higher = more relevant.

    Signal 1 — topic keywords  (0–∞, weighted sum, capped at 6)
    Signal 2 — known-lab boost (+1.5 if any lab signal found)
    Signal 3 — novelty language (+1.0 if any novelty phrase found)

    Maximum possible score ≈ 8.5, but anything ≥ 3 is solid.
    """
    haystack = (
        (article.get("title") or "") + " " +
        (article.get("content_snippet") or "")
    ).lower()

    # Signal 1 — topic keywords
    topic_score = 0.0
    for weight, keywords in _ARXIV_TOPIC_KEYWORDS:
        for kw in keywords:
            if kw in haystack:
                topic_score += weight
    topic_score = min(topic_score, 6.0)  # cap so one very chatty abstract can't dominate

    # Signal 2 — known lab authorship
    lab_boost = 1.5 if any(lab in haystack for lab in _ARXIV_LAB_SIGNALS) else 0.0

    # Signal 3 — novelty language
    novelty_boost = 1.0 if any(phrase in haystack for phrase in _ARXIV_NOVELTY_PHRASES) else 0.0

    return topic_score + lab_boost + novelty_boost


def step1b_filter_arxiv(articles: list[dict]) -> list[dict]:
    """
    Split the article list into arXiv and non-arXiv.
    Score and rank arXiv articles; keep top ARXIV_TOP_K above ARXIV_MIN_SCORE.
    Recombine and return.
    """
    arxiv, others = [], []
    for a in articles:
        if a["source_name"] in ARXIV_SOURCE_NAMES:
            arxiv.append(a)
        else:
            others.append(a)

    if not arxiv:
        return others

    # Score every arXiv article
    scored = [(a, _score_arxiv(a)) for a in arxiv]

    # Debug log so you can tune thresholds later
    for a, score in sorted(scored, key=lambda x: -x[1])[:10]:
        log.debug(
            "  arXiv score %.2f  [%s]  %s",
            score, a["source_name"], a["title"][:80],
        )

    # Keep top-K above the minimum floor
    kept = [
        a for a, score in sorted(scored, key=lambda x: -x[1])
        if score >= ARXIV_MIN_SCORE
    ][:ARXIV_TOP_K]

    log.info(
        "  arXiv filter: %d total → %d kept (top-%d, min_score=%.1f)",
        len(arxiv), len(kept), ARXIV_TOP_K, ARXIV_MIN_SCORE,
    )

    return others + kept


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def step2_filter(articles: list[dict]) -> list[dict]:
    log.info("STEP 2: Filtering…")
    cutoff = datetime.now(UTC) - timedelta(hours=AGE_CUTOFF_HOURS)

    recent = [a for a in articles if a["published_at"] >= cutoff]
    log.info("  Age filter: %d → %d", len(articles), len(recent))

    try:
        rows = (
            supabase.table("sources")
            .select("url")
            .gte("published_at", cutoff.isoformat())
            .execute()
            .data
        )
        known_urls = {r["url"] for r in rows}
    except Exception as e:
        log.warning("  DB URL fetch failed (%s) — skipping URL dedup.", e)
        known_urls = set()

    fresh = [a for a in recent if a["url"] not in known_urls]
    log.info("  URL dedup: %d → %d new articles.", len(recent), len(fresh))
    return fresh


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterResult:
    articles:  list[dict]
    is_multi:  bool           # True = 2+ sources; False = solo article
    min_tier:  int = field(init=False)

    def __post_init__(self):
        self.min_tier = min(a.get("tier", 2) for a in self.articles)


def step3_cluster(articles: list[dict]) -> list[ClusterResult]:
    """
    Groups semantically similar articles. Solo articles are now kept
    (previously dropped by MIN_CLUSTER_SIZE = 2).

    Tier-2 solos with very short snippets are filtered as stubs — they're
    typically feed entries with no real content (e.g. title-only arXiv entries
    from minor sources, or paywalled summaries with no excerpt).
    """
    log.info("STEP 3: Clustering %d articles…", len(articles))

    if not articles:
        return []

    model      = SentenceTransformer("all-MiniLM-L6-v2")
    headlines  = [a["title"] for a in articles]
    embeddings = model.encode(headlines, normalize_embeddings=True, show_progress_bar=False)
    sim        = cosine_similarity(embeddings)

    assigned: set[int]           = set()
    results:  list[ClusterResult] = []

    for i in range(len(articles)):
        if i in assigned:
            continue
        group = [i]
        for j in range(i + 1, len(articles)):
            if j not in assigned and sim[i][j] >= CLUSTER_THRESHOLD:
                group.append(j)
                assigned.add(j)
        assigned.add(i)

        cluster_articles = [articles[k] for k in group]
        is_multi         = len(group) >= 2

        if not is_multi:
            # Solo article — apply stub guard for Tier-2 sources
            article = cluster_articles[0]
            tier    = article.get("tier", 2)
            snippet = article.get("content_snippet", "")
            if tier >= 2 and len(snippet) < SOLO_TIER2_MIN_SNIPPET:
                log.debug(
                    "  Dropping stub solo [%s] '%s' (snippet=%d chars)",
                    article["source_name"], article["title"][:60], len(snippet),
                )
                continue

        results.append(ClusterResult(articles=cluster_articles, is_multi=is_multi))

    multi_count = sum(1 for r in results if r.is_multi)
    solo_count  = len(results) - multi_count
    log.info(
        "  Clusters: %d multi-source | %d solo-source",
        multi_count, solo_count,
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DEDUPLICATE
# ═══════════════════════════════════════════════════════════════════════════════

def _cluster_id(cluster: list[dict]) -> str:
    titles = [
        a["title"]
        for a in sorted(cluster, key=lambda x: x["published_at"])[:3]
    ]
    return hashlib.sha256(
        json.dumps(titles, sort_keys=True).encode()
    ).hexdigest()[:32]


def step4_deduplicate(
    clusters: list[ClusterResult],
) -> tuple[list[dict], list[dict]]:
    log.info("STEP 4: Deduplicating %d clusters…", len(clusters))
    cids = [_cluster_id(c.articles) for c in clusters]

    try:
        rows = (
            supabase.table("stories")
            .select("cluster_id, id, source_count")
            .in_("cluster_id", cids)
            .execute()
            .data
        )
        existing = {r["cluster_id"]: r for r in rows}
    except Exception as e:
        log.warning("  DB lookup failed (%s) — treating all as new.", e)
        existing = {}

    new_clusters:     list[dict] = []
    updated_clusters: list[dict] = []

    for cid, cr in zip(cids, clusters):
        if cid in existing:
            updated_clusters.append({
                "cluster_id":            cid,
                "story_id":              existing[cid]["id"],
                "existing_source_count": existing[cid]["source_count"],
                "cluster":               cr.articles,
                "is_multi":              cr.is_multi,
            })
        else:
            new_clusters.append({
                "cluster_id": cid,
                "cluster":    cr.articles,
                "is_multi":   cr.is_multi,
            })

    log.info("  New: %d | To update: %d", len(new_clusters), len(updated_clusters))
    return new_clusters, updated_clusters


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GENERATE
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are a news analyst and summariser. "
    "You write for a general audience — curious, intelligent, not technical.\n"
    "Always respond with valid JSON only. No preamble. No markdown. "
    "No explanation outside the JSON."
)


def _build_prompt(cluster: list[dict], is_multi: bool) -> str:
    top  = sorted(cluster, key=lambda a: (a.get("tier", 2), -a["published_at"].timestamp()))[:LLM_TOP_K]
    n    = len(top)

    if is_multi:
        intro = f"Here are {n} articles covering the same AI news story:"
    else:
        intro = "Here is 1 article about an AI news story:"

    body = "".join(
        f"\nSource: {a['source_name']}\nHeadline: {a['title']}\nExcerpt: {a['content_snippet']}\n"
        for a in top
    )

    return f"""{intro}
{body}
Return a JSON object with exactly these fields:

{{
  "headline": "The single clearest headline for this story. Max 15 words. No clickbait.",
  "summary_short": "2–3 sentences. What happened, why it matters, what comes next. Plain English. No jargon.",
  "summary_long": "4–5 sentences. Same tone, more detail. Still plain English.",
  "categories": ["one or more of: Tech, Dev, Business, Market, Governance, Science, Society"],
  "enterprises": ["zero or more of: OpenAI, Anthropic, Google DeepMind, Meta AI, Microsoft, Apple, Amazon, Nvidia, xAI, Mistral, Samsung, Baidu, IBM, Salesforce, Adobe, Stability AI, Hugging Face, Cohere, Other — include only organisations that are a PRIMARY SUBJECT of the story, not merely mentioned in passing. Return [] if none qualify."],
  "regions": ["one or more of: USA, Europe, China, Asia-Pacific, Middle East, Latin America, Africa, Global"],
  "story_type": "exactly one of: Research, Product Launch, Funding, Acquisition, Policy, Incident, Partnership, Personnel, Opinion",
  "impact_level": "exactly one of: Breakthrough, Significant, Routine — Breakthrough = genuinely novel or historically notable; Significant = meaningful but incremental; Routine = minor update, patch, personnel news, or administrative. When in doubt assign Significant over Breakthrough.",
  "audience": ["one or more of: General Public, Developers, Investors, Policymakers, Researchers"]
}}"""


_ENTERPRISE_ALIASES: dict[str, str] = {
    "Meta":               "Meta AI",
    "Meta AI Research":   "Meta AI",
    "Facebook":           "Meta AI",
    "Facebook AI":        "Meta AI",
    "Google":             "Google DeepMind",
    "Google AI":          "Google DeepMind",
    "DeepMind":           "Google DeepMind",
    "Google Research":    "Google DeepMind",
    "xAI / Grok":         "xAI",
    "Grok":               "xAI",
    "HuggingFace":        "Hugging Face",
    "Huggingface":        "Hugging Face",
    "NVIDIA":             "Nvidia",
    "Reliance":           "Other",
}

def _normalize_enterprises(values: list) -> list[str]:
    result = []
    seen: set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        canonical = _ENTERPRISE_ALIASES.get(v, v if v in ALLOWED_ENTERPRISES else "Other")
        if canonical not in seen:
            result.append(canonical)
            seen.add(canonical)
    return result


def _validate(data: dict) -> list[str]:
    errs: list[str] = []

    def chk_str(key: str, max_words: int) -> None:
        v = data.get(key)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"{key}: empty or missing")
        elif len(v.split()) > max_words:
            errs.append(f"{key}: too long — {len(v.split())} words (max {max_words})")

    def chk_list(key: str, allowed: set, required_nonempty: bool = True) -> None:
        v = data.get(key)
        if not isinstance(v, list):
            errs.append(f"{key}: must be a list"); return
        if required_nonempty and not v:
            errs.append(f"{key}: must not be empty"); return
        bad = [x for x in v if x not in allowed]
        if bad:
            errs.append(f"{key}: invalid values {bad}")

    chk_str("headline",      20)
    chk_str("summary_short", 60)
    chk_str("summary_long",  120)
    chk_list("categories",   ALLOWED_CATEGORIES)
    chk_list("enterprises",  ALLOWED_ENTERPRISES, required_nonempty=False)
    chk_list("regions",      ALLOWED_REGIONS)
    chk_list("audience",     ALLOWED_AUDIENCES)

    if data.get("story_type") not in ALLOWED_STORY_TYPES:
        errs.append(f"story_type: invalid '{data.get('story_type')}'")
    if data.get("impact_level") not in ALLOWED_IMPACT:
        errs.append(f"impact_level: invalid '{data.get('impact_level')}'")

    return errs


def _call_llm(cluster: list[dict], is_multi: bool) -> dict | None:
    try:
        rsp = nim.complete(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(cluster, is_multi)},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        raw     = rsp.choices[0].message.content or ""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data    = json.loads(cleaned)
        if isinstance(data.get("enterprises"), list):
            data["enterprises"] = _normalize_enterprises(data["enterprises"])
        errs    = _validate(data)
        if errs:
            log.warning("  Validation errors: %s | raw: %.300s", errs, raw)
            return None
        return data
    except Exception as e:
        log.warning("  LLM call failed: %s", e)
        return None


def step5_generate(new_clusters: list[dict]) -> list[dict]:
    log.info("STEP 5: Generating summaries for %d new clusters…", len(new_clusters))
    enriched: list[dict] = []
    for item in new_clusters:
        gen = _call_llm(item["cluster"], item["is_multi"])
        if gen is None:
            log.warning("  Skipping cluster %s — LLM failed or invalid.", item["cluster_id"])
            continue
        enriched.append({**item, "generated": gen})
    log.info("  Generated: %d/%d", len(enriched), len(new_clusters))
    return enriched


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def step6_write(
    enriched: list[dict],
    updated:  list[dict],
) -> tuple[int, int, int]:
    new_count = updated_count = pruned_count = 0

    for item in enriched:
        cluster = item["cluster"]
        gen     = item["generated"]
        latest  = max(a["published_at"] for a in cluster)

        try:
            result = supabase.table("stories").insert({
                "cluster_id":       item["cluster_id"],
                "headline":         gen["headline"],
                "summary_short":    gen["summary_short"],
                "summary_long":     gen["summary_long"],
                "categories":       gen["categories"],
                "enterprises":      gen["enterprises"],
                "regions":          gen["regions"],
                "story_type":       gen["story_type"],
                "impact_level":     gen["impact_level"],
                "audience":         gen["audience"],
                "source_count":     len(cluster),
                "latest_source_at": latest.isoformat(),
            }).execute()
            story_id = result.data[0]["id"]
        except Exception as e:
            log.error("  Insert story failed [%s]: %s", item["cluster_id"], e)
            continue

        try:
            supabase.table("sources").insert([
                {
                    "story_id":     story_id,
                    "source_name":  a["source_name"],
                    "url":          a["url"],
                    "published_at": a["published_at"].isoformat(),
                }
                for a in cluster
            ]).execute()
        except Exception as e:
            log.error("  Insert sources failed for story %s: %s", story_id, e)

        new_count += 1

    for item in updated:
        cluster = item["cluster"]
        latest  = max(a["published_at"] for a in cluster)
        try:
            supabase.table("stories").update({
                "source_count":     item["existing_source_count"] + len(cluster),
                "latest_source_at": latest.isoformat(),
            }).eq("cluster_id", item["cluster_id"]).execute()

            supabase.table("sources").insert([
                {
                    "story_id":     item["story_id"],
                    "source_name":  a["source_name"],
                    "url":          a["url"],
                    "published_at": a["published_at"].isoformat(),
                }
                for a in cluster
            ]).execute()
            updated_count += 1
        except Exception as e:
            log.error("  Update failed [%s]: %s", item["cluster_id"], e)

    cutoff = (datetime.now(UTC) - timedelta(days=PRUNE_AFTER_DAYS)).isoformat()
    try:
        result = supabase.table("stories").delete().lt("first_seen_at", cutoff).execute()
        pruned_count = len(result.data) if result.data else 0
    except Exception as e:
        log.error("  Pruning failed: %s", e)

    return new_count, updated_count, pruned_count


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    log.info("═══ pulse-pipeline v2.3 starting ═══")

    sources      = load_sources()
    articles     = step1_fetch(sources)
    articles     = step1b_filter_arxiv(articles)   # ← arXiv scorer (v2.2)
    new_articles = step2_filter(articles)

    if not new_articles:
        log.info("No new articles this run — exiting cleanly.")
        sys.exit(0)

    clusters = step3_cluster(new_articles)
    if not clusters:
        log.info("No clusters formed this run — exiting cleanly.")
        sys.exit(0)

    new_clusters, updated_clusters = step4_deduplicate(clusters)
    enriched_clusters              = step5_generate(new_clusters)
    new_count, updated_count, pruned_count = step6_write(enriched_clusters, updated_clusters)

    log.info(
        "═══ Run complete. New: %d | Updated: %d | Pruned: %d | Duration: %.1fs ═══",
        new_count, updated_count, pruned_count, time.time() - t0,
    )


if __name__ == "__main__":
    main()