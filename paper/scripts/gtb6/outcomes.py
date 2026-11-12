"""A pytest plugin that writes {nodeid: outcome} to $GTB_OUTCOMES.
An outcome is 'failed' if setup, call or teardown failed, 'skipped' if skipped, else 'passed'."""
import json
import os

_res = {}


def _rel(config, nodeid):
    """Node id relative to the invocation dir (pytest's own ids are relative to its rootdir)."""
    prefix = os.path.relpath(config.rootpath, config.invocation_params.dir).replace(os.sep, '/')
    return nodeid if prefix == '.' else f'{prefix}/{nodeid}'


def pytest_addoption(parser):
    parser.addoption('--gtb-select', default=None, help='JSON list of node ids (or whole files) to run')


def pytest_collection_modifyitems(session, config, items):
    path = config.getoption('--gtb-select')
    if not path:
        return
    with open(path) as fh:
        wanted = set(json.load(fh))
    keep, drop = [], []
    for it in items:
        nid = _rel(config, it.nodeid)
        (keep if nid in wanted or nid.split('::', 1)[0] in wanted else drop).append(it)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_runtest_logreport(report):
    prev = _res.get(report.nodeid)
    if report.failed:
        _res[report.nodeid] = 'failed'
    elif report.skipped and prev != 'failed':
        _res[report.nodeid] = 'skipped'
    elif report.when == 'call' and report.passed and prev is None:
        _res[report.nodeid] = 'passed'


def pytest_sessionfinish(session, exitstatus):
    # node ids are relative to pytest's rootdir; rewrite them relative to the invocation dir so they can
    # be passed back as command-line targets (rich keeps a pytest.ini in tests/)
    res = {_rel(session.config, k): v for k, v in _res.items()}
    with open(os.environ.get('GTB_OUTCOMES', 'gtb_outcomes.json'), 'w') as fh:
        json.dump(res, fh, indent=0, sort_keys=True)
