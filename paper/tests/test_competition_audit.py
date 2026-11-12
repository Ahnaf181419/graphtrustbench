"""Synthetic-only tests; no Kaggle content or credentials are required."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import struct
import tarfile
import unittest
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_competition_data.py"
spec = importlib.util.spec_from_file_location("competition_audit", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(root: Path) -> None:
    published = root / "published"
    (published / "graphs").mkdir(parents=True)
    tasks = [
        {"instance_id": "alpha_1", "repo": "org/alpha", "base_commit": "a" * 40, "problem_statement": "fix A", "patch": "secret gold A"},
        {"instance_id": "beta_2", "repo": "org/beta", "base_commit": "b" * 40, "problem_statement": "fix B", "patch": "secret gold B"},
        {"instance_id": "alpha_3", "repo": "org/alpha", "base_commit": "c" * 40, "problem_statement": "fix C", "patch": "secret gold C"},
    ]
    (published / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    graph = {"directed": True, "multigraph": True, "nodes": [{"id": "module.fn"}, {"id": "module.helper"}], "edges": [{"source": "module.fn", "target": "module.helper", "type": "calls"}]}
    for task in tasks:
        (published / "graphs" / f"{task['instance_id']}.json").write_text(json.dumps(graph), encoding="utf-8")


class TestKaggleDataAudit(unittest.TestCase):
    def test_embedding_keys_shape_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vectors.npz"
            header = repr({"descr": "<f4", "fortran_order": False, "shape": (2,)}).encode() + b"\n"
            payload = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pkg.fn.npy", payload + struct.pack("<ff", 1, float("nan")))
            stats = audit.audit_embeddings(path, {"pkg.fn", "pkg.other"})
            self.assertEqual(stats["vectors"], 1)
            self.assertEqual(stats["nonfinite_values"], 1)
            self.assertEqual(stats["missing_node_keys"], 1)
            self.assertEqual(stats["shapes"], {"(2,)": 1})

    def test_embedding_pickle_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "objects.npz"
            header = repr({"descr": "|O", "fortran_order": False, "shape": (1,)}).encode() + b"\n"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pkg.fn.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"not a pickle")
            self.assertEqual(audit.audit_embeddings(path, {"pkg.fn"})["malformed_vectors"], 1)

    def test_snapshot_stream_counts_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshot.tgz"
            with tarfile.open(path, "w:gz") as archive:
                for name, content in [("./pkg/a.py", b"pass\n"), ("./.git/HEAD", b"ref: refs/heads/main\n")]:
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            stats = audit.audit_snapshot(path)
            self.assertEqual(stats["python_files"], 1)
            self.assertEqual(stats["members"], 2)
            with tarfile.open(path, "w:gz") as archive:
                archive.addfile(tarfile.TarInfo("../../outside"))
            with self.assertRaisesRegex(ValueError, "Unsafe archive path"):
                audit.audit_snapshot(path)

    def test_commit_named_files_are_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            release = root / "published"
            task = audit.read_tasks(release / "tasks.jsonl")[0]
            original = release / "graphs/alpha_1.json"
            renamed = release / "graphs" / ("alpha_" + "a" * 40 + ".json")
            original.rename(renamed)
            self.assertEqual(audit.resolve_artifact(release, task, "graph"), renamed)
            self.assertEqual(audit.analyze(root, 3, 2026)["selected_graph_errors"], 0)

    def test_conflicting_aliases_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            release = root / "published"
            task = audit.read_tasks(release / "tasks.jsonl")[0]
            (release / "graphs" / ("alpha_" + "a" * 40 + ".json")).write_text("{}")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                audit.resolve_artifact(release, task, "graph")

    def test_identical_aliases_prefer_shared_commit_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            release = root / 'published'
            task = audit.read_tasks(release / 'tasks.jsonl')[0]
            canonical = release / 'graphs' / ('alpha_' + 'a' * 40 + '.json')
            canonical.write_bytes((release / 'graphs/alpha_1.json').read_bytes())
            self.assertEqual(audit.resolve_artifact(release, task, 'graph'), canonical)

    def test_shared_snapshots_are_reported_without_dropping_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / 'published/tasks.jsonl'
            tasks = audit.read_tasks(path)
            tasks[2]['base_commit'] = tasks[0]['base_commit']
            path.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
            report = audit.analyze(root, 3, 2026)
            self.assertEqual(report['total_tasks'], 3)
            self.assertEqual(report['unique_base_commits'], 2)
            self.assertEqual(report['shared_base_commits']['org/alpha@' + 'a' * 40], ['alpha_1', 'alpha_3'])

    def test_missing_assets_fail_cli_even_when_graphs_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                code = audit.main(["--data-root", str(root), "--limit", "1"])
            self.assertEqual(code, 1)
            report = audit.analyze(root, 1, 2026)
            self.assertEqual(report["missing_files"], {"graph": 0, "embedding": 3, "snapshot": 3})
            self.assertEqual(len(report["tasks"]), 3)
            self.assertEqual(report["usable_tasks"], 0)

    def test_empty_dataset_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tasks.jsonl"
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "No tasks"):
                audit.read_tasks(path)

    def test_unsafe_task_paths_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "published/tasks.jsonl"
            task = json.loads(path.read_text().splitlines()[0])
            for field, value in [("instance_id", "../outside"), ("repo", "org/../../outside"), ("problem_statement", " ")]:
                with self.subTest(field=field):
                    changed = dict(task, **{field: value})
                    path.write_text(json.dumps(changed) + "\n")
                    with self.assertRaises(ValueError):
                        audit.read_tasks(path)

    def test_graph_metadata_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            path = root / "published/graphs/alpha_1.json"
            obj = json.loads(path.read_text())
            obj["graph"] = {"repo_name": "beta_wrong"}
            path.write_text(json.dumps(obj))
            self.assertEqual(audit.analyze(root, 3, 2026)["selected_graph_errors"], 1)

    def test_sampling_is_input_order_invariant(self) -> None:
        tasks = [{"instance_id": f"{repo}_{i}", "repo": repo} for repo in ["o/a", "o/b"] for i in range(5)]
        self.assertEqual(audit.balanced_sample(tasks, 6, 2026), audit.balanced_sample(tasks[::-1], 6, 2026))

    def test_deterministic_sampling_and_cross_repo_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            one = audit.analyze(root, limit=2, seed=9)
            two = audit.analyze(root, limit=2, seed=9)
            self.assertEqual(one["selected"], two["selected"])
            self.assertEqual({r["repo"] for r in one["selected"]}, {"org/alpha", "org/beta"})
            self.assertEqual(one["selected_graph_errors"], 0)
            self.assertFalse(one["retrieval_scores_computed"])

    def test_no_patch_content_leaked_to_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            rendered = json.dumps(audit.analyze(root, limit=3, seed=3))
            self.assertNotIn("secret gold", rendered)
            self.assertIn("has_reference_patch", rendered)

    def test_detect_dangling_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            graph_file = root / "published" / "graphs" / "alpha_1.json"
            graph = json.loads(graph_file.read_text(encoding="utf-8"))
            graph["edges"].append({"source": "module.fn", "target": "missing", "type": "calls"})
            graph_file.write_text(json.dumps(graph), encoding="utf-8")
            result = audit.analyze(root, limit=3, seed=3)
            self.assertEqual(result["selected_graph_errors"], 1)

    def test_duplicate_instance_id_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root)
            task_file = root / "published" / "tasks.jsonl"
            first = task_file.read_text(encoding="utf-8").splitlines()[0]
            with task_file.open("a", encoding="utf-8") as handle:
                handle.write(first + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate instance_id"):
                audit.read_tasks(task_file)

    def test_missing_dataset_is_explicit_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                audit.locate_release(Path(temp))


if __name__ == "__main__":
    unittest.main()
