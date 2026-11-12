"""No-download unit tests for the official benchmark command wrapper."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline.py"
spec = importlib.util.spec_from_file_location("paper_run_baseline", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestBaselineRunner(unittest.TestCase):
    def test_lexical_uses_official_evaluator_and_unfiltered_corpus(self) -> None:
        cmd = runner.build_command(
            "arb", "lexical", Path("derived"), Path("corpus"),
            Path("summary.json"), Path("details.jsonl"),
        )
        self.assertEqual(cmd[:2], ["arb", "eval-baseline"])
        self.assertEqual(cmd[cmd.index("--ranker") + 1], "lexical")
        self.assertEqual(cmd[cmd.index("--candidate-filter") + 1], "all_files")
        self.assertIn("--no-keep-list", cmd)
        self.assertIn("--details", cmd)

    def test_repomap_uses_upstream_repomap_command(self) -> None:
        cmd = runner.build_command(
            "arb", "repomap", Path("derived"), Path("corpus"),
            Path("summary.json"), Path("details.jsonl"),
        )
        self.assertEqual(cmd[:2], ["arb", "eval-repomap"])
        self.assertNotIn("--ranker", cmd)

    def test_invalid_ranker_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.build_command("arb", "pretend", Path("a"), Path("b"), Path("c"), Path("d"))

    def test_dry_run_never_creates_results_or_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = runner.main(["--arb-root", str(root), "--run-id", "unit_dryrun"])
            self.assertEqual(code, 0)
            data = json.loads(stdout.getvalue())
            self.assertEqual(data["mode"], "dry_run")
            self.assertTrue(data["missing_prerequisites"])
            self.assertFalse(data["will_download_data"])
            self.assertFalse((runner.PAPER_ROOT / "results" / "unit_dryrun_lexical").exists())

    def test_execute_refuses_missing_data_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with contextlib.redirect_stderr(io.StringIO()):
                code = runner.main(["--execute", "--arb-root", str(root), "--run-id", "unit_missing"])
            self.assertEqual(code, 2)
            self.assertFalse((runner.PAPER_ROOT / "results" / "unit_missing_lexical").exists())

    def test_invalid_run_id_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--run-id", "../../unsafe"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
