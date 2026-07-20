#!/usr/bin/env python3
"""
mine_knative.py — Dataset B miner for the SafLoc / TC-RAG benchmark.

Extracts (Log, Code, Config) triplets from closed Knative issues that have a
linked (merged) Pull Request. Aligned with §5.3–§5.5 of
Dataset_collection_methodology_v2.md.

Usage:
    export GITHUB_TOKEN=ghp_xxx
    python scripts/mine_knative.py \
        --repos knative/serving knative/eventing \
        --since 2024-12-01 \
        --out data/dataset_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from dateutil.parser import isoparse
from github import Github, GithubException, RateLimitExceededException
from github.Issue import Issue
from github.PullRequest import PullRequest


# ---------------------------------------------------------------------------
# Configuration (per §5.3 / §5.4)
# ---------------------------------------------------------------------------

DEFAULT_REPOS = ["knative/serving", "knative/eventing"]
DEFAULT_SINCE = "2024-12-01"  # closed strictly after November 2024
DEFAULT_OUT = "data/dataset_b.jsonl"

# §5.3 Stage 2 fault-signal keywords (case-insensitive match on body).
ERROR_KEYWORDS = [
    "error", "failed", "exception", "denied", "timeout",
    "crashloopbackoff", "oomkilled", "evicted", "403", "500",
]

CONFIG_EXTS = (".yaml", ".yml")
CODE_EXTS = (".go", ".py", ".js", ".ts")

# Path prefixes / suffixes that add noise rather than diagnostic signal.
EXCLUDED_PATH_PREFIXES = (".github/", "vendor/", "hack/", "third_party/")
EXCLUDED_CODE_SUFFIXES = ("_test.go",)
EXCLUDED_CODE_DIRS = ("test/", "tests/")

# Regex: any fenced code block (used for Stage-2 presence test).
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
# Regex: typed fenced log/text/bash block (§5.4 priority 1).
TYPED_LOG_BLOCK_RE = re.compile(
    r"```(?:log|text|bash|console|shell|sh)?\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)
# Regex: YAML fenced block (config fallback from issue body).
YAML_BLOCK_RE = re.compile(r"```(?:yaml|yml)\s*\n([\s\S]*?)```", re.IGNORECASE)
# Regex: inline JSON error object (§5.4 priority 2).
JSON_ERROR_RE = re.compile(r"\{[^{}]*\"error\"[^{}]*\}")


# ---------------------------------------------------------------------------
# Rate-limit-safe wrapper
# ---------------------------------------------------------------------------

def safe_call(gh: Github, fn, *args, **kwargs):
    """Invoke a PyGithub call, sleeping through rate-limit / secondary 403s."""
    backoff = 30
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededException:
            reset = gh.get_rate_limit().core.reset
            now = datetime.now(timezone.utc)
            wait = max(5, int((reset - now).total_seconds()) + 5)
            print(f"[rate-limit] sleeping {wait}s (until {reset.isoformat()})",
                  file=sys.stderr)
            time.sleep(wait)
        except GithubException as exc:
            # Secondary rate limit / abuse detection → exponential backoff.
            if exc.status in (403, 429):
                print(f"[secondary-limit] {exc.status} — sleeping {backoff}s",
                      file=sys.stderr)
                time.sleep(backoff)
                backoff = min(backoff * 2, 600)
                continue
            raise


def safe_iter(gh: Github, paginated) -> Iterable:
    """Iterate a PaginatedList while honoring rate limits."""
    it = iter(paginated)
    while True:
        try:
            yield safe_call(gh, next, it)
        except StopIteration:
            return


# ---------------------------------------------------------------------------
# Filters (Stages 1–3)
# ---------------------------------------------------------------------------

def passes_fault_signal(body: Optional[str]) -> tuple[bool, list[str]]:
    """Stage 2 — body must have a fenced block AND ≥1 error keyword."""
    if not body:
        return False, []
    if not FENCED_BLOCK_RE.search(body):
        return False, []
    lower = body.lower()
    hits = [kw for kw in ERROR_KEYWORDS if kw in lower]
    return (len(hits) > 0), hits


def find_closing_pr(gh: Github, issue: Issue) -> Optional[PullRequest]:
    """
    Stage 3 — resolve the PR that closed this issue via timeline events.

    Preference order:
      1. `closed` event whose `source` is a merged PR in the same repo.
      2. Most recent merged `cross-referenced` PR in the same repo.
    """
    repo_full = issue.repository.full_name
    cross_ref_candidate: Optional[PullRequest] = None

    for event in safe_iter(gh, issue.get_timeline()):
        ev_type = getattr(event, "event", None)

        # 1. Direct "closed" via PR.
        if ev_type == "closed":
            source = getattr(event, "source", None)
            issue_ref = getattr(source, "issue", None) if source else None
            if issue_ref is not None and getattr(issue_ref, "pull_request", None):
                pr = _issue_to_pr(gh, issue_ref, repo_full)
                if pr and pr.merged:
                    return pr

        # 2. Cross-reference fallback.
        if ev_type == "cross-referenced":
            source = getattr(event, "source", None)
            issue_ref = getattr(source, "issue", None) if source else None
            if issue_ref is not None and getattr(issue_ref, "pull_request", None):
                pr = _issue_to_pr(gh, issue_ref, repo_full)
                if pr and pr.merged:
                    cross_ref_candidate = pr  # keep the latest

    return cross_ref_candidate


def _issue_to_pr(gh: Github, issue_ref, repo_full: str) -> Optional[PullRequest]:
    """Convert a timeline source-issue reference to a PullRequest in same repo."""
    try:
        ref_repo = issue_ref.repository.full_name
    except Exception:
        ref_repo = None
    if ref_repo and ref_repo != repo_full:
        return None
    try:
        return safe_call(gh, issue_ref.as_pull_request)
    except GithubException:
        return None


# ---------------------------------------------------------------------------
# Triplet extraction (§5.4)
# ---------------------------------------------------------------------------

def extract_log(body: str) -> Optional[str]:
    """Longest fenced block that contains an error keyword; JSON fallback."""
    if not body:
        return None
    blocks = TYPED_LOG_BLOCK_RE.findall(body)
    err_blocks = [
        b.strip() for b in blocks
        if any(kw in b.lower() for kw in ERROR_KEYWORDS)
    ]
    if err_blocks:
        return max(err_blocks, key=len)
    json_match = JSON_ERROR_RE.search(body)
    return json_match.group(0) if json_match else None


def _file_excluded(path: str, *, is_code: bool) -> bool:
    if any(path.startswith(p) for p in EXCLUDED_PATH_PREFIXES):
        return True
    if is_code:
        if path.endswith(EXCLUDED_CODE_SUFFIXES):
            return True
        if any(path.startswith(d) for d in EXCLUDED_CODE_DIRS):
            return True
    return False


def extract_diffs(gh: Github, pr: PullRequest) -> tuple[Optional[str], Optional[str]]:
    """Return (code_diff, config_diff) concatenated from PR file patches."""
    code_parts: list[str] = []
    config_parts: list[str] = []
    for f in safe_iter(gh, pr.get_files()):
        patch = getattr(f, "patch", None)
        if not patch:
            continue
        name = f.filename
        if name.endswith(CODE_EXTS) and not _file_excluded(name, is_code=True):
            code_parts.append(f"--- {name}\n{patch}")
        elif name.endswith(CONFIG_EXTS) and not _file_excluded(name, is_code=False):
            config_parts.append(f"--- {name}\n{patch}")
    code = "\n".join(code_parts) if code_parts else None
    config = "\n".join(config_parts) if config_parts else None
    return code, config


def extract_yaml_from_body(body: str) -> Optional[str]:
    if not body:
        return None
    blocks = YAML_BLOCK_RE.findall(body)
    return blocks[0].strip() if blocks else None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Counters:
    scanned: int = 0
    skipped_pr: int = 0
    skipped_date: int = 0
    failed_signal: int = 0
    no_pr: int = 0
    empty_triplet: int = 0
    written: int = 0
    keywords: dict[str, int] = field(default_factory=dict)


def mine_repo(
    gh: Github,
    repo_full: str,
    since: datetime,
    out_fh,
    seen: set[str],
    max_issues: Optional[int],
) -> Counters:
    print(f"\n=== {repo_full} (since {since.date().isoformat()}) ===",
          file=sys.stderr)
    c = Counters()
    repo = safe_call(gh, gh.get_repo, repo_full)

    issues = safe_call(
        gh, repo.get_issues,
        state="closed", sort="updated", direction="desc", since=since,
    )

    for issue in safe_iter(gh, issues):
        if max_issues is not None and c.written >= max_issues:
            break
        c.scanned += 1
        if c.scanned % 25 == 0:
            print(f"  scanned={c.scanned} written={c.written}", file=sys.stderr)

        # Skip PRs (the issues endpoint returns them too).
        if issue.pull_request is not None:
            c.skipped_pr += 1
            continue

        closed_at = issue.closed_at
        if closed_at is None or closed_at.replace(tzinfo=timezone.utc) < since:
            c.skipped_date += 1
            continue

        issue_id = f"{repo_full}#{issue.number}"
        if issue_id in seen:
            continue

        ok, hits = passes_fault_signal(issue.body)
        if not ok:
            c.failed_signal += 1
            continue
        for h in hits:
            c.keywords[h] = c.keywords.get(h, 0) + 1

        pr = find_closing_pr(gh, issue)
        if pr is None:
            c.no_pr += 1
            continue

        log = extract_log(issue.body)
        code_diff, config_diff = extract_diffs(gh, pr)
        if config_diff is None:
            config_diff = extract_yaml_from_body(issue.body)
            config_source = "issue_body_yaml" if config_diff else None
        else:
            config_source = "closing_pr_diff"

        if log is None and code_diff is None and config_diff is None:
            c.empty_triplet += 1
            continue

        record = {
            "issue_id": issue_id,
            "repository": repo_full,
            "issue_number": issue.number,
            "issue_url": issue.html_url,
            "closing_pr_url": pr.html_url,
            "closed_at": closed_at.astimezone(timezone.utc).isoformat(),
            "keywords_hit": hits,
            "log": log,
            "code_diff": code_diff,
            "config_diff": config_diff,
            "artifact_availability": {
                "log_available": log is not None,
                "log_source": "issue_body_code_block" if log else None,
                "code_available": code_diff is not None,
                "code_source": "closing_pr_diff" if code_diff else None,
                "config_available": config_diff is not None,
                "config_source": config_source,
            },
        }
        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_fh.flush()
        seen.add(issue_id)
        c.written += 1

    return c


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_seen(out_path: Path) -> set[str]:
    seen: set[str] = set()
    if not out_path.exists():
        return seen
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "issue_id" in rec:
                    seen.add(rec["issue_id"])
            except json.JSONDecodeError:
                continue
    return seen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repos", nargs="+", default=DEFAULT_REPOS,
                   help="GitHub repos to mine (default: %(default)s)")
    p.add_argument("--since", default=DEFAULT_SINCE,
                   help="Only issues closed on/after this date (YYYY-MM-DD).")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Output JSONL path (default: %(default)s)")
    p.add_argument("--max-issues", type=int, default=None,
                   help="Per-repo cap (smoke testing).")
    p.add_argument("--resume", action="store_true",
                   help="Skip issue_ids already present in the output file.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN env var is not set.", file=sys.stderr)
        return 2

    since = isoparse(args.since)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(out_path) if args.resume else set()
    if args.resume:
        print(f"[resume] {len(seen)} issue_ids already in {out_path}",
              file=sys.stderr)

    gh = Github(token, per_page=100)
    totals = Counters()

    with out_path.open("a", encoding="utf-8") as out_fh:
        for repo in args.repos:
            c = mine_repo(gh, repo, since, out_fh, seen, args.max_issues)
            print(
                f"[{repo}] scanned={c.scanned} skipped_pr={c.skipped_pr} "
                f"skipped_date={c.skipped_date} failed_signal={c.failed_signal} "
                f"no_pr={c.no_pr} empty_triplet={c.empty_triplet} "
                f"written={c.written}",
                file=sys.stderr,
            )
            totals.scanned += c.scanned
            totals.skipped_pr += c.skipped_pr
            totals.skipped_date += c.skipped_date
            totals.failed_signal += c.failed_signal
            totals.no_pr += c.no_pr
            totals.empty_triplet += c.empty_triplet
            totals.written += c.written
            for k, v in c.keywords.items():
                totals.keywords[k] = totals.keywords.get(k, 0) + v

    print("\n=== TOTAL ===", file=sys.stderr)
    print(json.dumps(totals.__dict__, indent=2), file=sys.stderr)
    print(f"output: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
