#!/usr/bin/env python3
"""Dry-run-first wrapper around the official Agent Retrieval Bench CLI.

This code does not implement BM25, RepoMap, metric formulas or dataset download.
It preserves the upstream evaluation path and records provenance of real runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_ROOT.parent
RANKERS = ("lexical", "bm25", "repomap")
RELEASE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def build_command(
    executable: str,
    ranker: str,
    derived: Path,
    corpus: Path,
    summary: Path,
    details: Path,
) -> list[str]:
    """Build the documented upstream `arb` CLI, without custom scoring."""
    if ranker not in RANKERS:
        raise ValueError(f"Unsupported ranker: {ranker}")
    if ranker == "repomap":
        args = [executable, "eval-repomap"]
    else:
        args = [executable, "eval-baseline"]
    args += ["--derived", str(derived), "--corpus", str(corpus)]
    if ranker != "repomap":
        args += ["--ranker", ranker]
    args += [
        "--candidate-filter", "all_files", "--no-keep-list",
        "--out", str(summary), "--details", str(details),
    ]
    return args


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path,
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker", choices=RANKERS, default="lexical")
    parser.add_argument("--release", default="v2_edit2ripple")
    parser.add_argument("--arb-root", type=Path, default=REPO_ROOT / "third_party" / "agent-retrieval-bench")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None, help="Safe output subdirectory label")
    parser.add_argument("--execute", action="store_true", help="Actually run official evaluator; default only previews")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not RELEASE_PATTERN.fullmatch(args.release):
        print("Invalid release identifier", file=sys.stderr)
        return 2
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not RELEASE_PATTERN.fullmatch(run_id):
        print("Invalid run identifier", file=sys.stderr)
        return 2

    arb_root = args.arb_root.resolve()
    data_root = (args.data_root or arb_root / "data").resolve()
    derived = data_root / "benchmark" / args.release
    corpus = data_root / "corpus" / args.release
    out_dir = PAPER_ROOT / "results" / f"{run_id}_{args.ranker}"
    summary = out_dir / "summary.json"
    details = out_dir / "details.jsonl"
    arb_executable = shutil.which("arb")
    command = build_command("arb", args.ranker, derived, corpus, summary, details)
    sample_file = derived / "samples.jsonl"
    missing = []
    if sys.version_info < (3, 10):
        missing.append("Python 3.10 or newer")
    if not arb_executable:
        missing.append("official arb CLI on PATH")
    if not sample_file.is_file():
        missing.append(str(sample_file))
    if not corpus.is_dir():
        missing.append(str(corpus))
    if not arb_root.is_dir():
        missing.append(str(arb_root))

    preview = {
        "mode": "execute" if args.execute else "dry_run",
        "release": args.release,
        "ranker": args.ranker,
        "candidate_filter": "all_files",
        "command": command,
        "missing_prerequisites": missing,
        "will_download_data": False,
        "will_run_model_inference": False,
    }
    if not args.execute:
        print(json.dumps(preview, indent=2))
        return 0

    if missing:
        print("Cannot run benchmark; prerequisites missing:\n- " + "\n- ".join(missing), file=sys.stderr)
        return 2
    if out_dir.exists():
        print(f"Refusing to overwrite existing run directory: {out_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(command, cwd=arb_root, capture_output=True, text=True, check=False)
        rc = completed.returncode
        captured = completed.stdout + "\n--- STDERR ---\n" + completed.stderr
    except OSError as exc:
        rc = 127
        captured = f"Execution failed: {exc!r}"
    (out_dir / "command.log").write_text(captured, encoding="utf-8")

    record = {
        "status": "completed" if rc == 0 and summary.is_file() and details.is_file() else "failed",
        "exit_code": rc,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "release": args.release,
        "ranker": args.ranker,
        "command": command,
        "arb_root": str(arb_root),
        "arb_git_sha": git_head(arb_root),
        "repository_git_sha": git_head(REPO_ROOT),
        "samples_sha256": file_sha256(sample_file),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "summary_path": str(summary),
        "details_path": str(details),
        "no_dataset_download": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "exit_code": rc, "manifest": str(out_dir / "manifest.json")}, indent=2))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
