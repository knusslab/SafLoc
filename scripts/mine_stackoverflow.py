#!/usr/bin/env python3
"""
mine_stackoverflow.py — Dataset SO miner for the SafLoc / TC-RAG benchmark.

Mirror of scripts/mine_knative.py for Stack Overflow. Extracts (Log, Code,
Config) triplets from accepted Q/A pairs on Knative-related tags. Output
schema matches dataset_c.jsonl so the existing categorisation script
(log_only / log_config / log_code / log_code_config) drops in unchanged.

Usage:
    export STACKEXCHANGE_KEY=...   # optional — raises quota 300 -> 10 000/day
    python scripts/mine_stackoverflow.py \
        --tags knative knative-serving knative-eventing kubernetes-serverless \
        --since 2024-12-01 \
        --out data/dataset_so.jsonl
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from dateutil.parser import isoparse


API = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
PAGE_SIZE = 100  # API max

DEFAULT_TAGS = [
    "knative",
    "knative-serving",
    "knative-eventing",
    "kubernetes-serverless",
]
DEFAULT_SINCE = "2024-12-01"
DEFAULT_OUT = "data/dataset_so.jsonl"

# Same fault-signal keywords as the GitHub miner (§5.3 Stage 2).
ERROR_KEYWORDS = [
    "error", "failed", "exception", "denied", "timeout",
    "crashloopbackoff", "oomkilled", "evicted", "403", "500",
]

# Markdown fenced blocks (SO renders these as <pre><code>...) and the HTML form.
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
PRE_CODE_RE = re.compile(
    r"<pre(?:\s[^>]*)?><code(?:\s[^>]*)?>([\s\S]*?)</code></pre>",
    re.IGNORECASE,
)
TYPED_LOG_FENCE_RE = re.compile(
    r"```(?:log|text|bash|console|shell|sh)?\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)
YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)\s*\n([\s\S]*?)```", re.IGNORECASE)
CODE_FENCE_RE = re.compile(
    r"```(?:python|py|go|golang|js|javascript|ts|typescript|java)\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

# Heuristics for HTML-wrapped <pre><code> blocks.
YAML_HINT_RE = re.compile(
    r"^\s*(apiVersion:|kind:\s+(Service|Deployment|ConfigMap|Role|"
    r"ClusterRole|RoleBinding|ClusterRoleBinding|InMemoryChannel|Broker|"
    r"Trigger|Subscription)|metadata:\s*$)",
    re.MULTILINE,
)
CODE_HINT_RE = re.compile(
    r'\b(def\s+\w+\s*\(|func\s+\w+\s*\(|package\s+\w+|import\s+[\'"]?\w|'
    r'class\s+\w+|function\s+\w+\s*\()', re.MULTILINE,
)


# ---------------------------------------------------------------------------
# HTTP with quota / backoff handling
# ---------------------------------------------------------------------------

def api_get(path: str, *, key: Optional[str], **params) -> dict:
    """GET against the Stack Exchange API, honoring `backoff` and quota."""
    params.setdefault("site", SITE)
    if key:
        params.setdefault("key", key)
    qs = urlencode(params, doseq=True)
    url = f"{API}/{path}?{qs}"

    backoff = 5
    while True:
        try:
            req = Request(url, headers={"Accept-Encoding": "gzip"})
            with urlopen(req, timeout=60) as resp:
                raw = resp.read()
                # Stack Exchange always gzips.
                if resp.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                data = json.loads(raw.decode("utf-8"))
        except HTTPError as e:
            if e.code in (429, 502, 503, 504):
                print(f"[http {e.code}] sleep {backoff}s", file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue
            raise
        except URLError as e:
            print(f"[url-error {e}] sleep {backoff}s", file=sys.stderr)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue

        if "error_id" in data:
            # 502 = throttled. Sleep and retry.
            if data.get("error_id") == 502:
                wait = max(backoff, 30)
                print(f"[throttled] {data.get('error_message')} — sleep {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                backoff = min(backoff * 2, 300)
                continue
            raise RuntimeError(f"SE API error: {data}")

        # Server-requested cool-down on the next call.
        if "backoff" in data:
            time.sleep(int(data["backoff"]) + 1)
        return data


def paginate(path: str, *, key: Optional[str], **params):
    """Yield items across all pages, respecting `has_more` + `quota_remaining`."""
    page = 1
    while True:
        data = api_get(path, key=key, page=page, pagesize=PAGE_SIZE, **params)
        for item in data.get("items", []):
            yield item
        qr = data.get("quota_remaining")
        if qr is not None and qr < 5:
            print(f"[quota] only {qr} requests left — stopping", file=sys.stderr)
            return
        if not data.get("has_more"):
            return
        page += 1


# ---------------------------------------------------------------------------
# Body parsing
# ---------------------------------------------------------------------------

def _extract_blocks_from_body(body: str) -> list[str]:
    """Return raw text of every <pre><code> block in an HTML body."""
    return [html.unescape(m.group(1)) for m in PRE_CODE_RE.finditer(body)]


def _is_yaml_block(text: str) -> bool:
    return bool(YAML_HINT_RE.search(text))


def _is_code_block(text: str) -> bool:
    return bool(CODE_HINT_RE.search(text))


def passes_fault_signal(body_html: str) -> tuple[bool, list[str]]:
    """Stage 2: body has a <pre><code> block AND ≥1 error keyword."""
    if not body_html:
        return False, []
    if not PRE_CODE_RE.search(body_html):
        return False, []
    text = html.unescape(re.sub(r"<[^>]+>", " ", body_html)).lower()
    hits = [kw for kw in ERROR_KEYWORDS if kw in text]
    return (len(hits) > 0), hits


def extract_log(body_html: str) -> Optional[str]:
    """Longest <pre><code> block whose text contains an error keyword."""
    if not body_html:
        return None
    blocks = _extract_blocks_from_body(body_html)
    err_blocks = [
        b.strip() for b in blocks
        if any(kw in b.lower() for kw in ERROR_KEYWORDS)
        and not _is_yaml_block(b)
    ]
    if err_blocks:
        return max(err_blocks, key=len)
    return None


def extract_yaml(body_html: str) -> Optional[str]:
    if not body_html:
        return None
    blocks = _extract_blocks_from_body(body_html)
    yaml_blocks = [b.strip() for b in blocks if _is_yaml_block(b)]
    return max(yaml_blocks, key=len) if yaml_blocks else None


def extract_code(body_html: str) -> Optional[str]:
    if not body_html:
        return None
    blocks = _extract_blocks_from_body(body_html)
    # Code = block matching code hints AND not YAML AND not predominantly logs.
    code_blocks = [
        b.strip() for b in blocks
        if _is_code_block(b)
        and not _is_yaml_block(b)
        and not any(kw in b.lower() for kw in ("error", "failed", "exception"))
    ]
    return max(code_blocks, key=len) if code_blocks else None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Counters:
    scanned: int = 0
    no_accepted: int = 0
    skipped_date: int = 0
    failed_signal: int = 0
    no_answer_body: int = 0
    empty_triplet: int = 0
    written: int = 0
    keywords: dict[str, int] = field(default_factory=dict)


def mine_tag(
    tag: str,
    *,
    key: Optional[str],
    since: datetime,
    out_fh,
    seen: set[int],
    max_questions: Optional[int],
    query: Optional[str] = None,
) -> Counters:
    """Mine either a tag (`tag` non-empty) or a free-text query (`query` set)."""
    label = f"query:'{query}'" if query else f"tag:{tag}"
    print(f"\n=== {label} (since {since.date().isoformat()}) ===",
          file=sys.stderr)
    c = Counters()

    if query:
        # Free-text search via /search/advanced. Accept any question with
        # ≥1 answer; if no accepted answer, fall back to the top-scored one.
        q_iter = paginate(
            "search/advanced",
            key=key,
            q=query,
            answers=1,
            order="desc",
            sort="creation",
            fromdate=int(since.timestamp()),
            filter="withbody",
        )
    else:
        q_iter = paginate(
            "questions",
            key=key,
            tagged=tag,
            order="desc",
            sort="creation",
            fromdate=int(since.timestamp()),
            filter="withbody",  # include question body
        )

    for q in q_iter:
        if max_questions is not None and c.written >= max_questions:
            break
        c.scanned += 1
        if c.scanned % 25 == 0:
            print(f"  scanned={c.scanned} written={c.written}", file=sys.stderr)

        qid = q["question_id"]
        if qid in seen:
            continue
        accepted_id = q.get("accepted_answer_id")
        if not accepted_id and not query:
            # Tag mode keeps the strict accepted-answer requirement.
            c.no_accepted += 1
            continue
        if q["creation_date"] < since.timestamp():
            c.skipped_date += 1
            continue

        ok, hits = passes_fault_signal(q.get("body", ""))
        if not ok:
            c.failed_signal += 1
            continue
        for h in hits:
            c.keywords[h] = c.keywords.get(h, 0) + 1

        # Fetch the chosen answer body.
        if accepted_id:
            answer_id = accepted_id
            answer_kind = "accepted"
        else:
            # Query mode fallback: top-scored answer for this question.
            ans_data = api_get(
                f"questions/{qid}/answers",
                key=key, filter="withbody",
                order="desc", sort="votes", pagesize=1,
            )
            if not ans_data.get("items"):
                c.no_answer_body += 1
                continue
            answer_id = ans_data["items"][0]["answer_id"]
            answer_kind = "top_voted"

        a_data = api_get(
            f"answers/{answer_id}",
            key=key, filter="withbody",
        )
        if not a_data.get("items"):
            c.no_answer_body += 1
            continue
        answer = a_data["items"][0]
        a_body = answer.get("body", "")
        q_body = q.get("body", "")

        log = extract_log(q_body) or extract_log(a_body)
        # Fix-side modalities prefer the answer; fall back to question for K.
        code_diff = extract_code(a_body) or extract_code(q_body)
        config_diff = extract_yaml(a_body) or extract_yaml(q_body)

        if log is None and code_diff is None and config_diff is None:
            c.empty_triplet += 1
            continue

        record = {
            "issue_id": f"stackoverflow#{qid}",
            "repository": (
                f"stackoverflow:query:{query}" if query
                else f"stackoverflow:{tag}"
            ),
            "issue_number": qid,
            "issue_url": q["link"],
            "closing_pr_url": (
                f"https://stackoverflow.com/a/{answer_id}"
            ),
            "closed_at": datetime.fromtimestamp(
                q.get("last_activity_date", q["creation_date"]),
                tz=timezone.utc,
            ).isoformat(),
            "keywords_hit": hits,
            "log": log,
            "code_diff": code_diff,
            "config_diff": config_diff,
            "artifact_availability": {
                "log_available": log is not None,
                "log_source": "question_body_pre_code" if log else None,
                "code_available": code_diff is not None,
                "code_source": "answer_body_pre_code" if code_diff else None,
                "config_available": config_diff is not None,
                "config_source": "answer_body_yaml" if config_diff else None,
            },
            "stackoverflow": {
                "tag": tag,
                "query": query,
                "question_score": q.get("score"),
                "answer_score": answer.get("score"),
                "answer_kind": answer_kind,
                "view_count": q.get("view_count"),
                "matched_tags": q.get("tags"),
            },
        }
        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_fh.flush()
        seen.add(qid)
        c.written += 1

    return c


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_seen(out_path: Path) -> set[int]:
    seen: set[int] = set()
    if not out_path.exists():
        return seen
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                if "issue_number" in rec:
                    seen.add(int(rec["issue_number"]))
            except (json.JSONDecodeError, ValueError):
                continue
    return seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    p.add_argument("--query", default=None,
                   help="Free-text search term. If set, runs /search/advanced "
                        "with accepted=True instead of tag-based mining; "
                        "--tags is ignored.")
    p.add_argument("--since", default=DEFAULT_SINCE)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--max-questions", type=int, default=None,
                   help="Per-tag/query cap (smoke testing).")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    key = os.getenv("STACKEXCHANGE_KEY") or None
    if key:
        print(f"[auth] using STACKEXCHANGE_KEY (len={len(key)})", file=sys.stderr)
    else:
        print("[auth] anonymous (300 req/day quota)", file=sys.stderr)

    since = isoparse(args.since)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(out_path) if args.resume else set()
    if args.resume:
        print(f"[resume] {len(seen)} ids already in {out_path}", file=sys.stderr)

    totals = Counters()
    with out_path.open("a", encoding="utf-8") as out_fh:
        if args.query:
            c = mine_tag(
                tag="", query=args.query, key=key, since=since,
                out_fh=out_fh, seen=seen, max_questions=args.max_questions,
            )
            print(
                f"[query:'{args.query}'] scanned={c.scanned} no_accepted={c.no_accepted} "
                f"skipped_date={c.skipped_date} failed_signal={c.failed_signal} "
                f"no_answer_body={c.no_answer_body} empty_triplet={c.empty_triplet} "
                f"written={c.written}",
                file=sys.stderr,
            )
            for k, v in vars(c).items():
                if k == "keywords":
                    for kk, vv in v.items():
                        totals.keywords[kk] = totals.keywords.get(kk, 0) + vv
                elif isinstance(v, int):
                    setattr(totals, k, getattr(totals, k) + v)
        else:
            for tag in args.tags:
                c = mine_tag(
                    tag, key=key, since=since, out_fh=out_fh,
                    seen=seen, max_questions=args.max_questions,
                )
                print(
                    f"[tag:{tag}] scanned={c.scanned} no_accepted={c.no_accepted} "
                    f"skipped_date={c.skipped_date} failed_signal={c.failed_signal} "
                    f"no_answer_body={c.no_answer_body} empty_triplet={c.empty_triplet} "
                    f"written={c.written}",
                    file=sys.stderr,
                )
                for k, v in vars(c).items():
                    if k == "keywords":
                        for kk, vv in v.items():
                            totals.keywords[kk] = totals.keywords.get(kk, 0) + vv
                    elif isinstance(v, int):
                        setattr(totals, k, getattr(totals, k) + v)

    print("\n=== TOTAL ===", file=sys.stderr)
    print(json.dumps(totals.__dict__, indent=2), file=sys.stderr)
    print(f"output: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
