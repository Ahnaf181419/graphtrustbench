"""Run pytest inside a container with no network: the only way repo_ops executes upstream tests
and model-written patches."""
import os
import subprocess
from pathlib import Path

from .repo_ops import pytest_args, read_outcomes

SCRIPTS = Path(__file__).resolve().parents[1]          # mounted read-only so gtb6.outcomes imports


MEMORY = '6g'       # per container, swap included; a runaway test is killed instead of the host's processes


def make(image, python='python', timeout=900, memory=MEMORY):
    def run(repo, targets):
        if not targets:
            return {}
        repo = Path(repo).resolve()
        cmd = ['docker', 'run', '--rm', '--network', 'none', '--memory', memory, '--memory-swap', memory, '--user', f'{os.getuid()}:{os.getgid()}',
               '-v', f'{repo}:/repo', '-v', f'{SCRIPTS}:/gtb:ro', '-w', '/repo',
               '-e', 'PYTHONPATH=/gtb', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'HOME=/tmp',
               '-e', 'GTB_OUTCOMES=/repo/.gtb_outcomes.json',
               image, 'timeout', str(timeout), python, *pytest_args(repo, targets)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout + 120)
        except subprocess.TimeoutExpired:
            pass
        return read_outcomes(repo / '.gtb_outcomes.json')
    return run
