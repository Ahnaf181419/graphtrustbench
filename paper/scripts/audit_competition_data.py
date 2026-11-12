#!/usr/bin/env python3
"""Read-only schema and graph-integrity preflight for the public Kaggle development set.

This does not download Kaggle data, inspect hidden data, construct gold labels,
or compute benchmark results. Reference patches are never emitted.
"""

from __future__ import annotations

import argparse
import array
import ast
from datetime import datetime, timezone
import math
import os
import platform
import struct
import subprocess
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import sys


SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def locate_release(data_root: Path) -> Path:
    """Find documented `tasks.jsonl` in a release directory or its `published/` child."""
    candidates = (data_root, data_root / "published")
    for candidate in candidates:
        if (candidate / "tasks.jsonl").is_file():
            return candidate
    raise FileNotFoundError(
        f"No tasks.jsonl in {data_root} or {data_root / 'published'}; "
        "download the *public development* dataset using your own Kaggle session first"
    )


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_tasks(task_file: Path) -> list[dict]:
    tasks: list[dict] = []
    ids: set[str] = set()
    with task_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_number}: task must be a JSON object")
            task_id = obj.get("instance_id")
            if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
                raise ValueError(f"Line {line_number}: missing instance_id")
            if task_id in ids:
                raise ValueError(f"Line {line_number}: duplicate instance_id {task_id}")
            ids.add(task_id)
            if not isinstance(obj.get("repo"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", obj["repo"]) or obj["repo"].split("/")[-1] in (".", ".."):
                raise ValueError(f"Line {line_number}: invalid repo")
            if not isinstance(obj.get("base_commit"), str) or not SHA_PATTERN.fullmatch(obj["base_commit"]):
                raise ValueError(f"Line {line_number}: invalid base_commit")
            if not isinstance(obj.get("problem_statement"), str) or not obj["problem_statement"].strip():
                raise ValueError(f"Line {line_number}: missing problem_statement")
            tasks.append(obj)
    if not tasks:
        raise ValueError("No tasks found in tasks.jsonl")
    return tasks


def balanced_sample(tasks: list[dict], limit: int, seed: int) -> list[dict]:
    """Select at least one from each repository when the limit allows it."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in tasks:
        groups[item["repo"]].append(item)
    rng = random.Random(seed)
    selected: list[dict] = []
    for repo in sorted(groups):
        if len(selected) == limit:
            break
        selected.append(rng.choice(sorted(groups[repo], key=lambda x: x["instance_id"])))
    used = {t["instance_id"] for t in selected}
    remaining = sorted((t for t in tasks if t["instance_id"] not in used), key=lambda x: x["instance_id"])
    selected += rng.sample(remaining, k=min(max(0, limit - len(selected)), len(remaining)))
    return sorted(selected, key=lambda x: x["instance_id"])


def audit_graph(graph_file: Path) -> dict:
    graph = json.loads(graph_file.read_text(encoding="utf-8"))
    if not isinstance(graph, dict):
        raise ValueError("Graph root must be a JSON object")
    nodes = graph.get("nodes")
    edges = graph.get("edges", graph.get("links"))
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Expected nodes and edges/links lists")
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    string_ids = [node_id for node_id in node_ids if isinstance(node_id, str) and node_id]
    valid_ids = set(string_ids)
    missing_node_ids = len(nodes) - len(string_ids)
    duplicate_node_ids = len(string_ids) - len(valid_ids)
    dangling = 0
    edge_types: Counter[str] = Counter()
    malformed_edges = 0
    for edge in edges:
        if not isinstance(edge, dict):
            malformed_edges += 1
            continue
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            malformed_edges += 1
            continue
        if source not in valid_ids or target not in valid_ids:
            dangling += 1
        edge_types[str(edge.get("type", "unknown"))] += 1
    return {
        "repo_name": graph.get("graph", {}).get("repo_name") if isinstance(graph.get("graph", {}), dict) else None,
        "nodes_without_text": sum(not isinstance(n, dict) or not isinstance(n.get("text"), str) or not n["text"].strip() for n in nodes),
        "nodes_with_explicit_path": sum(isinstance(n, dict) and any(k in n for k in ("path", "file", "file_path")) for n in nodes),
        "nodes": len(nodes),
        "edges": len(edges),
        "directed": graph.get("directed"),
        "multigraph": graph.get("multigraph"),
        "missing_node_ids": missing_node_ids,
        "duplicate_node_ids": duplicate_node_ids,
        "malformed_edges": malformed_edges,
        "dangling_edges": dangling,
        "edge_types": dict(sorted(edge_types.items())),
    }


def resolve_artifact(root: Path, task: dict, kind: str) -> Path:
    """Support exported commit filenames and official task aliases; reject conflicts."""
    directory, extension = {"graph": ("graphs", "json"), "embedding": ("embeddings", "npz"), "snapshot": ("snapshots", "tgz")}[kind]
    stem = task["repo"].split("/")[-1] + "_" + task["base_commit"]
    paths = [root / directory / f"{task['instance_id']}.{extension}"]
    if kind != "snapshot":
        paths.append(root / directory / f"{stem}.{extension}")
    existing = [p for p in paths if p.is_file()]
    for p in existing:
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Artifact escapes release directory: {p}")
    if len(existing) == 2 and not existing[0].samefile(existing[1]) and digest(existing[0]) != digest(existing[1]):
        raise ValueError(f"Conflicting {kind} aliases for {task['instance_id']}")
    # Prefer the commit filename so task hard-link aliases share one cache entry.
    return existing[-1] if existing else paths[0]


def audit_embeddings(path: Path, node_ids: set[str]) -> dict:
    """Validate NPY payloads without pickle or loading the whole corpus into RAM."""
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    nonfinite = zero = malformed = 0
    keys = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                raise ValueError(f"Unexpected embedding member: {info.filename}")
            keys.append(info.filename[:-4])
            with archive.open(info) as handle:
                prefix = handle.read(8)
                if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY" or prefix[6:8] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                    raise ValueError("Invalid NPY header")
                width = 2 if prefix[6] == 1 else 4
                length = int.from_bytes(handle.read(width), "little")
                if not 0 < length <= 65536:
                    raise ValueError("Invalid NPY header length")
                header = ast.literal_eval(handle.read(length).decode("utf-8"))
                if not isinstance(header, dict):
                    raise ValueError("NPY header must be a dictionary")
                shape = header.get("shape")
                dtype = header.get("descr")
                shapes[str(shape)] += 1
                dtypes[str(dtype)] += 1
                if not isinstance(shape, tuple) or len(shape) != 1 or not isinstance(shape[0], int) or not 0 < shape[0] <= 65536 or dtype not in ("<f4", ">f4", "=f4") or header.get("fortran_order") is not False:
                    malformed += 1
                    continue
                payload = handle.read(shape[0] * 4 + 1)
                if len(payload) != shape[0] * 4:
                    malformed += 1
                    continue
                values = array.array("f", payload)
                if (dtype == ">f4" and sys.byteorder == "little") or (dtype == "<f4" and sys.byteorder == "big"):
                    values.byteswap()
                nonfinite += sum(not math.isfinite(v) for v in values)
                zero += not any(values)
    keyset = set(keys)
    return {"vectors": len(keys), "duplicate_keys": len(keys) - len(keyset),
            "shapes": dict(shapes), "dtypes": dict(dtypes), "malformed_vectors": malformed,
            "nonfinite_values": nonfinite, "zero_vectors": zero,
            "missing_node_keys": len(node_ids - keyset), "extra_node_keys": len(keyset - node_ids)}


def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return str(path)


def audit_snapshot(path: Path) -> dict:
    """Read through an archive without extraction, including gzip CRC validation."""
    import gzip
    members = size = python_files = duplicates = unsafe_links = 0
    names = set()
    refs = {}
    head = None
    # gzip is explicit so the unread trailer is consumed and CRC checked too.
    with gzip.open(path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                name = safe_member_name(member.name)
                members += 1
                size += member.size
                duplicates += name in names
                names.add(name)
                python_files += member.isfile() and name.endswith(".py") and not name.startswith(".git/")
                if member.issym() or member.islnk():
                    unsafe_links += PurePosixPath(member.linkname).is_absolute() or ".." in PurePosixPath(member.linkname).parts
                if member.isfile() and (name == ".git/HEAD" or name.startswith(".git/refs/")):
                    if member.size > 4096:
                        raise ValueError("Oversized Git ref")
                    value = archive.extractfile(member).read().decode("utf-8").strip()
                    if name == ".git/HEAD":
                        head = value
                    else:
                        refs[name] = value
        while stream.read(1024 * 1024):
            pass
    return {"members": members, "uncompressed_bytes": size, "python_files": python_files,
            "duplicate_member_names": duplicates, "potentially_external_links": unsafe_links,
            "git_head": head, "git_refs": refs,
            "provenance": "archive readable; export/import can rewrite commit IDs; pre-fix content not yet verified"}


def graph_failed(stats: dict) -> bool:
    return any(stats.get(k, 0) for k in ("dangling_edges", "malformed_edges", "duplicate_node_ids", "missing_node_ids")) or not stats.get("nodes") or stats.get("directed") is not True


def analyze(data_root: Path, limit: int, seed: int, scan_snapshots: bool = False) -> dict:
    release_root = locate_release(data_root)
    task_file = release_root / "tasks.jsonl"
    tasks = read_tasks(task_file)
    selected_ids = {t["instance_id"] for t in balanced_sample(tasks, limit, seed)}
    audits = []
    assets = {}
    graph_cache = {}
    embedding_cache = {}
    missing: Counter[str] = Counter({"graph": 0, "embedding": 0, "snapshot": 0})
    for item in tasks:
        row = {k: item[k] for k in ("instance_id", "repo", "base_commit")}
        row["has_reference_patch"] = isinstance(item.get("patch"), str) and bool(item["patch"].strip())
        row["has_test_patch"] = isinstance(item.get("test_patch"), str) and bool(item["test_patch"].strip())
        row["problem_statement_chars"] = len(item["problem_statement"])
        row["errors"] = []
        paths = {}
        for kind in ("graph", "embedding", "snapshot"):
            try:
                path = resolve_artifact(release_root, item, kind)
                paths[kind] = path
                present = path.is_file()
            except ValueError as exc:
                row["errors"].append(str(exc))
                present = False
            row[kind + "_present"] = present
            missing[kind] += not present
            if not present:
                row["errors"].append(f"Missing or conflicting {kind}")
                continue
            relative = path.relative_to(release_root).as_posix()
            row[kind + "_path"] = relative
            if relative not in assets:
                assets[relative] = {"bytes": path.stat().st_size}
                if kind != "snapshot" or scan_snapshots:
                    assets[relative]["sha256"] = digest(path)
        if row["graph_present"]:
            path = paths["graph"]
            if path not in graph_cache:
                try:
                    stats = audit_graph(path)
                    graph = json.loads(path.read_text(encoding="utf-8"))
                    ids = {n["id"] for n in graph["nodes"] if isinstance(n, dict) and isinstance(n.get("id"), str)}
                    graph_cache[path] = (stats, ids, None)
                except (OSError, ValueError, TypeError) as exc:
                    graph_cache[path] = ({}, set(), str(exc))
            stats, node_ids, error = graph_cache[path]
            row["graph_audit"] = stats
            expected = item["repo"].split("/")[-1] + "_" + item["base_commit"]
            if error or graph_failed(stats) or stats.get("repo_name") not in (None, expected):
                row["graph_error"] = error or "Invalid graph structure or repo_name/base_commit mismatch"
                row["errors"].append(row["graph_error"])
            if row["embedding_present"]:
                ep = paths["embedding"]
                cache_key = (ep, path)
                if cache_key not in embedding_cache:
                    try:
                        embedding_cache[cache_key] = (audit_embeddings(ep, node_ids), None)
                    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile, EOFError, struct.error) as exc:
                        embedding_cache[cache_key] = ({}, str(exc))
                emb, err = embedding_cache[cache_key]
                row["embedding_audit"] = emb
                if err or any(emb.get(k, 0) for k in ("duplicate_keys", "malformed_vectors", "nonfinite_values", "missing_node_keys", "extra_node_keys", "zero_vectors")) or len(emb.get("shapes", {})) != 1:
                    row["errors"].append("Invalid embeddings: " + (err or "shape/key/value integrity failure"))
        if scan_snapshots and row["snapshot_present"]:
            try:
                row["snapshot_audit"] = audit_snapshot(paths["snapshot"])
                if row["snapshot_audit"]["duplicate_member_names"] or not row["snapshot_audit"]["python_files"] or not row["snapshot_audit"]["git_head"]:
                    row["errors"].append("Snapshot lacks Python/Git data or contains duplicate entries")
            except (OSError, ValueError, tarfile.TarError, EOFError) as exc:
                row["errors"].append("Invalid snapshot: " + str(exc))
        if not row["has_reference_patch"]:
            row["errors"].append("Missing reference patch for evaluation")
        row["schema_usable"] = not row["errors"]
        audits.append(row)
        if scan_snapshots:
            print(f"Audited {len(audits)}/{len(tasks)}: {item['instance_id']}", file=sys.stderr, flush=True)
    selected = sorted((r for r in audits if r["instance_id"] in selected_ids), key=lambda r: r["instance_id"])
    unique_stats = [value[0] for value in graph_cache.values() if value[0]]
    edge_types: Counter[str] = Counter()
    for stats in unique_stats:
        edge_types.update(stats["edge_types"])
    pairs: dict[str, list[str]] = defaultdict(list)
    for item in tasks:
        pairs[item["repo"] + "@" + item["base_commit"]].append(item["instance_id"])
    fingerprint = hashlib.sha256(json.dumps({"tasks_sha256": digest(task_file), "assets": assets}, sort_keys=True).encode()).hexdigest()
    return {
        "status": "passed_schema_preflight" if all(r["schema_usable"] for r in audits) else "failed_preflight",
        "dataset_root": str(release_root), "tasks_sha256": digest(task_file),
        "dataset_fingerprint": fingerprint,
        "fingerprint_scope": "all consumed files" if scan_snapshots else "tasks/graphs/embeddings hashes; snapshots sizes only",
        "total_tasks": len(tasks), "total_repositories": len({t["repo"] for t in tasks}),
        "repository_distribution": dict(sorted(Counter(t["repo"] for t in tasks).items())),
        "usable_tasks": sum(r["schema_usable"] for r in audits),
        "usable_definition": "valid task, graph and embeddings, present snapshot and patch; archive readability checked only when scan_snapshots=true; NOT verified gold mapping or pre-fix provenance",
        "missing_files": dict(missing), "scan_snapshots": scan_snapshots,
        "unique_base_commits": len(pairs), "shared_base_commits": {k: v for k, v in pairs.items() if len(v) > 1},
        "unique_graphs": len(graph_cache), "unique_graph_nodes": sum(g["nodes"] for g in unique_stats),
        "unique_graph_edges": sum(g["edges"] for g in unique_stats), "edge_types": dict(sorted(edge_types.items())),
        "tasks_with_graph_errors": sum(not r["graph_present"] or "graph_error" in r for r in audits),
        "seed": seed, "selected_count": len(selected),
        "selected_graph_errors": sum(not r["graph_present"] or "graph_error" in r for r in selected),
        "selected": selected, "tasks": audits, "assets": dict(sorted(assets.items())),
        "retrieval_scores_computed": False, "gold_labels_constructed": False,
    }


def write_new_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12, help="Pilot size; ALL tasks are audited")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--scan-snapshots", action="store_true", help="Hash/read all archives; no extraction")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pilot-output", type=Path, default=None)
    args = parser.parse_args(argv)
    for target in (args.output, args.pilot_output):
        if target and target.exists():
            print(f"Refusing to overwrite: {target}", file=sys.stderr)
            return 2
    started = time.perf_counter()
    try:
        report = analyze(args.data_root.expanduser().resolve(), args.limit, args.seed, args.scan_snapshots)
        repo = Path(__file__).resolve().parents[2]
        report["provenance"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.perf_counter() - started,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "script_sha256": digest(Path(__file__)), "python": platform.python_version(),
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
        }
        if args.output:
            write_new_json(args.output, report)
            print(f"Report: {args.output}")
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        if args.pilot_output:
            fields = ("instance_id", "repo", "base_commit", "graph_path", "embedding_path", "snapshot_path")
            pilot = [{k: row[k] for k in fields if k in row} for row in report["selected"]]
            write_new_json(args.pilot_output, {"status": report["status"], "seed": args.seed,
                "selection": "one random task per repo, remaining slots uniformly sampled; seed 2026; no gold used",
                "dataset_fingerprint": report["dataset_fingerprint"], "tasks_sha256": report["tasks_sha256"], "tasks": pilot})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Preflight blocked: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "passed_schema_preflight" else 1


if __name__ == "__main__":
    raise SystemExit(main())
