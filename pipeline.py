#!/usr/bin/env python3
"""
pulse-pipeline  v2.0
Fetch → Filter → Cluster → Deduplicate → Generate → Write → Exit
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
FETCH_TIMEOUT       = 8          # seconds per feed request
MAX_WORKERS         = 10         # parallel feed fetchers
CLUSTER_THRESHOLD   = 0.82       # cosine similarity floor
MIN_CLUSTER_SIZE    = 2          # drop single-article "clusters"
SNIPPET_LEN         = 400        # chars of content passed to LLM
AGE_CUTOFF_HOURS    = 48
PRUNE_AFTER_DAYS    = 7
LLM_TOP_K           = 5          # max articles sent to LLM per cluster

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
    os.environ["SUPABASE_SERVICE_KEY"],  # service key — never anon
)
nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NIM_API_KEY"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def strip_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(separator=" ").strip()


def parse_entry_date(entry) -> datetime | None:
    """Try multiple feedparser date attributes; always return UTC-aware datetime."""
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
            headers={"User-Agent": "pulse-pipeline/2.0"},
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
# STEP 2 — FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def step2_filter(articles: list[dict]) -> list[dict]:
    log.info("STEP 2: Filtering…")
    cutoff = datetime.now(UTC) - timedelta(hours=AGE_CUTOFF_HOURS)

    # 2a — age
    recent = [a for a in articles if a["published_at"] >= cutoff]
    log.info("  Age filter: %d → %d", len(articles), len(recent))

    # 2b — URL dedup against sources table
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

def step3_cluster(articles: list[dict]) -> list[list[dict]]:
    log.info("STEP 3: Clustering %d articles…", len(articles))
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    headlines  = [a["title"] for a in articles]
    embeddings = model.encode(headlines, normalize_embeddings=True, show_progress_bar=False)
    sim        = cosine_similarity(embeddings)

    assigned: set[int]         = set()
    clusters: list[list[dict]] = []

    for i in range(len(articles)):
        if i in assigned:
            continue
        group = [i]
        for j in range(i + 1, len(articles)):
            if j not in assigned and sim[i][j] >= CLUSTER_THRESHOLD:
                group.append(j)
                assigned.add(j)
        assigned.add(i)
        clusters.append([articles[k] for k in group])

    clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]
    log.info("  Formed %d clusters (≥%d articles each).", len(clusters), MIN_CLUSTER_SIZE)
    return clusters


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
    clusters: list[list[dict]],
) -> tuple[list[dict], list[dict]]:
    """Returns (new_clusters, updated_clusters)."""
    log.info("STEP 4: Deduplicating %d clusters…", len(clusters))
    cids = [_cluster_id(c) for c in clusters]

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

    for cid, cluster in zip(cids, clusters):
        if cid in existing:
            updated_clusters.append({
                "cluster_id":            cid,
                "story_id":              existing[cid]["id"],
                "existing_source_count": existing[cid]["source_count"],
                "cluster":               cluster,
            })
        else:
            new_clusters.append({"cluster_id": cid, "cluster": cluster})

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


def _build_prompt(cluster: list[dict]) -> str:
    # Tier 1 first; within same tier, most recent first
    top  = sorted(cluster, key=lambda a: (a.get("tier", 2), -a["published_at"].timestamp()))[:LLM_TOP_K]
    n    = len(top)
    body = "".join(
        f"\nSource: {a['source_name']}\nHeadline: {a['title']}\nExcerpt: {a['content_snippet']}\n"
        for a in top
    )
    return f"""Here are {n} articles covering the same AI news story:
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

    chk_str("headline",     20)
    chk_str("summary_short", 60)
    chk_str("summary_long", 120)
    chk_list("categories",  ALLOWED_CATEGORIES)
    chk_list("enterprises", ALLOWED_ENTERPRISES, required_nonempty=False)
    chk_list("regions",     ALLOWED_REGIONS)
    chk_list("audience",    ALLOWED_AUDIENCES)

    if data.get("story_type") not in ALLOWED_STORY_TYPES:
        errs.append(f"story_type: invalid '{data.get('story_type')}'")
    if data.get("impact_level") not in ALLOWED_IMPACT:
        errs.append(f"impact_level: invalid '{data.get('impact_level')}'")

    return errs


def _call_llm(cluster: list[dict]) -> dict | None:
    try:
        rsp = nim_client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_prompt(cluster)},
            ],
            max_tokens=1000,
            temperature=0.2,
        )
        raw     = rsp.choices[0].message.content or ""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        data    = json.loads(cleaned)
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
        gen = _call_llm(item["cluster"])
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

    # 6a + 6b — insert new stories + their sources
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
                    "story_id":    story_id,
                    "source_name": a["source_name"],
                    "url":         a["url"],
                    "published_at": a["published_at"].isoformat(),
                }
                for a in cluster
            ]).execute()
        except Exception as e:
            log.error("  Insert sources failed for story %s: %s", story_id, e)

        new_count += 1

    # 6c — update existing stories with new source counts
    for item in updated:
        cluster = item["cluster"]
        latest  = max(a["published_at"] for a in cluster)
        try:
            supabase.table("stories").update({
                "source_count":    item["existing_source_count"] + len(cluster),
                "latest_source_at": latest.isoformat(),
            }).eq("cluster_id", item["cluster_id"]).execute()

            supabase.table("sources").insert([
                {
                    "story_id":    item["story_id"],
                    "source_name": a["source_name"],
                    "url":         a["url"],
                    "published_at": a["published_at"].isoformat(),
                }
                for a in cluster
            ]).execute()
            updated_count += 1
        except Exception as e:
            log.error("  Update failed [%s]: %s", item["cluster_id"], e)

    # 6d — prune stories older than 7 days (cascades to sources)
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
    log.info("═══ pulse-pipeline v2.0 starting ═══")

    sources      = load_sources()
    articles     = step1_fetch(sources)
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
