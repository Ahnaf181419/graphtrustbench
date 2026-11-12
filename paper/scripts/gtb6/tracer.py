"""Q2: a pytest plugin that records which call edges the tests actually execute.

Usage (inside the task's environment, from the repository root):
    GTB_SRC=<abs path of the package source dir> GTB_OUT=<out.json> \
    python -m pytest -p gtb6.tracer <test ids> -q

Records (caller_module, caller_qualname, callee_module, callee_qualname) for every Python-level
call whose caller and callee code both live under GTB_SRC. Test code is excluded unless it lives
under GTB_SRC. Needs Python 3.11+ (code.co_qualname).
"""
import json
import os
import sys
import threading

_SRC = os.path.realpath(os.environ.get('GTB_SRC', os.getcwd()))
_ROOT = os.path.dirname(_SRC)          # module names are relative to the source dir's parent
_OUT = os.environ.get('GTB_OUT', 'gtb_trace.json')
_edges = set()
_modcache = {}


def _module(filename):
    m = _modcache.get(filename)
    if m is None:
        path = os.path.realpath(filename)
        if not path.startswith(_SRC + os.sep) or not path.endswith('.py'):
            m = ''
        else:
            rel = os.path.relpath(path, _ROOT)[:-3].replace(os.sep, '.')
            m = rel[:-9] if rel.endswith('.__init__') else rel
        _modcache[filename] = m
    return m


def _profile(frame, event, arg):
    if event != 'call':
        return
    callee = frame.f_code
    cm = _module(callee.co_filename)
    if not cm:
        return
    back = frame.f_back
    if back is None:
        return
    rm = _module(back.f_code.co_filename)
    if not rm:
        return
    _edges.add((rm, back.f_code.co_qualname, cm, callee.co_qualname))


def pytest_sessionstart(session):
    threading.setprofile(_profile)
    sys.setprofile(_profile)


def pytest_sessionfinish(session, exitstatus):
    sys.setprofile(None)
    threading.setprofile(None)
    with open(_OUT, 'w') as fh:
        json.dump(sorted(_edges), fh)


def to_node_ids(edges, resolve):
    """Map traced edges to graph IDs. resolve(module, qualname) -> graph node id or None.
    '<locals>' frames (closures, decorator wrappers) and '<module>' frames usually do not map;
    their share is reported as 'unmapped'."""
    mapped, unmapped = set(), 0
    for rm, rq, cm, cq in edges:
        u, v = resolve(rm, rq), resolve(cm, cq)
        if u and v and u != v:
            mapped.add((u, v))
        else:
            unmapped += 1
    return mapped, unmapped
