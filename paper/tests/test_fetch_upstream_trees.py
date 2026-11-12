"""Upstream tree verification manifest: pinned GitHub tree SHAs for held-out tasks."""
import importlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
fetcher = importlib.import_module('fetch_upstream_trees')


def task(i='fastapi_1', repo='fastapi/fastapi', sha='a' * 40):
    return {'instance_id': i, 'repo': repo, 'base_commit': sha}


class TestFetchUpstreamTrees(unittest.TestCase):
    def test_build_url_matches_the_documented_git_database_api(self):
        self.assertEqual(fetcher.build_url(task()),
                         f"https://api.github.com/repos/fastapi/fastapi/git/commits/{'a'*40}")

    def test_tree_record_pins_the_reported_tree_sha(self):
        payload = {'sha': 'a' * 40, 'tree': {'sha': 'b' * 40, 'url': 'ignore'}}
        record = fetcher.tree_record(task(), lambda url: payload)
        self.assertEqual(record, {'instance_id': 'fastapi_1', 'repo': 'fastapi/fastapi',
                                  'base_commit': 'a' * 40, 'tree_sha': 'b' * 40,
                                  'url': fetcher.build_url(task())})

    def test_missing_or_malformed_tree_fails_closed(self):
        for payload in ({}, {'tree': {}}, {'tree': {'sha': ''}}, {'tree': None}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                fetcher.tree_record(task(), lambda url: payload)

    def test_live_fetch_sends_the_documented_github_token_when_provided(self):
        import os
        from fetch_upstream_trees import live_fetch
        captured = {}
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"tree": {"sha": "' + b'c' * 40 + b'"}}'
        def fake_urlopen(request, timeout):
            captured['headers'] = dict(request.header_items())
            return FakeResponse()
        import fetch_upstream_trees, urllib.request
        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            os.environ['GITHUB_TOKEN'] = 'unit-test-token'
            payload = live_fetch('https://api.github.com/x')
            self.assertEqual(payload['tree']['sha'], 'c' * 40)
            self.assertIn('unit-test-token', captured['headers'].get('Authorization', ''))
        finally:
            urllib.request.urlopen = original
            os.environ.pop('GITHUB_TOKEN', None)

    def test_live_fetch_retries_transient_network_errors_instead_of_dying(self):
        import fetch_upstream_trees, urllib.request
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b'{"tree": {"sha": "' + b'c' * 40 + b'"}}'
        calls = []
        def flaky_urlopen(request, timeout):
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError('The read operation timed out')
            return FakeResponse()
        original_open, original_sleep = urllib.request.urlopen, fetcher.time.sleep
        fetcher.time.sleep = lambda seconds: None
        urllib.request.urlopen = flaky_urlopen
        try:
            payload = fetcher.live_fetch('https://api.github.com/x')
            self.assertEqual(payload['tree']['sha'], 'c' * 40)
            self.assertEqual(len(calls), 3, 'two transient failures then success')
        finally:
            urllib.request.urlopen = original_open
            fetcher.time.sleep = original_sleep

    def test_existing_manifest_entries_are_reused_without_an_api_call(self):
        tasks = [task('a_1', 'o/a', 'a' * 40), task('b_2', 'o/b', 'b' * 40)]
        existing = {'tasks': [{'instance_id': 'a_1', 'repo': 'o/a', 'base_commit': 'a' * 40,
                               'tree_sha': 'f' * 40, 'url': 'cached'}]}
        calls = []
        def fake(url):
            calls.append(url)
            return {'tree': {'sha': 'c' * 40}}
        manifest = fetcher.build_manifest(tasks, fake, 'now', existing=existing)
        self.assertEqual(len(calls), 1, 'only the missing task should be fetched')
        self.assertEqual(manifest['tasks'][0]['tree_sha'], 'f' * 40)
        self.assertEqual(manifest['tasks'][1]['tree_sha'], 'c' * 40)

    def test_assembly_covers_every_task_exactly_once(self):
        tasks = [task('a_1', 'o/a', 'a' * 40), task('b_2', 'o/b', 'b' * 40)]
        seen = []
        def fake(url):
            seen.append(url)
            return {'tree': {'sha': url[-5] * 40}}
        manifest = fetcher.build_manifest(tasks, fake, '2026-11-08T00:00:00Z')
        self.assertEqual([r['instance_id'] for r in manifest['tasks']], ['a_1', 'b_2'])
        self.assertEqual(len(seen), 2)
        self.assertEqual(manifest['tasks'][0]['tree_sha'], 'a' * 40)


if __name__ == '__main__':
    unittest.main()
