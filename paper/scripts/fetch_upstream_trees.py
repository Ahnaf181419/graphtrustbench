#!/usr/bin/env python3
"""Fetch pinned upstream Git tree SHAs from GitHub's public Git database API. Read-only, fail-closed."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from audit_competition_data import write_new_json

API = 'https://api.github.com/repos/{repo}/git/commits/{sha}'
USER_AGENT = 'graphtrustbench-provenance-check'


def build_url(task: dict) -> str:
    return API.format(repo=task['repo'], sha=task['base_commit'])


def tree_record(task: dict, fetch) -> dict:
    """Turn one API payload into a pinned record; reject anything without a usable tree SHA."""
    payload = fetch(build_url(task))
    tree = payload.get('tree') if isinstance(payload, dict) else None
    tree_sha = tree.get('sha') if isinstance(tree, dict) else None
    if not isinstance(tree_sha, str) or not tree_sha.strip():
        raise ValueError(f"No upstream tree SHA for {task['instance_id']}")
    return {'instance_id': task['instance_id'], 'repo': task['repo'],
            'base_commit': task['base_commit'], 'tree_sha': tree_sha, 'url': build_url(task)}


def build_manifest(tasks: list[dict], fetch, retrieved_on: str, existing: dict | None = None) -> dict:
    """Pin upstream trees for every task, reusing already-verified entries to spare API budget."""
    cached = {row['instance_id']: row for row in (existing or {}).get('tasks', [])}
    records = []
    for task in tasks:
        prior = cached.get(task['instance_id'])
        if prior and prior.get('tree_sha') and prior.get('base_commit') == task['base_commit']:
            records.append(prior)
            continue
        records.append(tree_record(task, fetch))
    return {'retrieved_on': retrieved_on, 'source': 'GitHub Git database API (public, read-only)',
            'tasks': records}


def live_fetch(url: str) -> dict:
    """Fetch one GitHub API payload, waiting out the unauthenticated rate-limit window."""
    headers = {'User-Agent': USER_AGENT, 'Accept': 'application/vnd.github+json'}
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    attempt = 0
    while True:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                raise
            reset = exc.headers.get('x-ratelimit-reset')
            if reset and token:
                raise SystemExit('Rate limited even with GITHUB_TOKEN; try again later')
            wait = 60
            if reset:
                try:
                    wait = max(30, int(reset) - int(time.time()) + 5)
                except ValueError:
                    wait = 300
            print(f'Rate limited; waiting {wait}s for the next window', flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # HTTPError is a URLError subclass and is handled above; this branch is transient
            # network trouble (SSL read timeouts, resets), which must not kill a 91-request run.
            if attempt >= 3:
                raise
            attempt += 1
            print(f'Transient network error ({exc}); retry {attempt}/3 in 20s', flush=True)
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heldout', type=Path, required=True)
    parser.add_argument('--existing', type=Path, default=None,
                        help='Prior tree manifest whose verified entries are reused')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing manifest')
    tasks = json.loads(args.heldout.read_text())['tasks']
    existing = json.loads(args.existing.read_text()) if args.existing else None
    try:
        manifest = build_manifest(tasks, live_fetch, datetime.now(timezone.utc).isoformat(), existing=existing)
    except urllib.error.HTTPError as exc:
        # 403 is the unauthenticated rate limit; fail closed rather than pin partial provenance.
        raise SystemExit(f'Upstream fetch failed ({exc.code}); wait for the rate-limit window or export GITHUB_TOKEN')
    write_new_json(args.output, manifest)
    print(f'Pinned {len(manifest["tasks"])} upstream trees -> {args.output}')


if __name__ == '__main__':
    main()
