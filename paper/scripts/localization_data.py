"""Read-only retrieval inputs. No reference patch or evaluation labels in this API."""
from __future__ import annotations

import ast
import hashlib
import copy
import textwrap
from collections import defaultdict
import io
from pathlib import Path, PurePosixPath
import tarfile
import tokenize

from audit_competition_data import safe_member_name


def snapshot_tree(path: Path) -> dict:
    """Recompute the worktree's Git tree ID without trusting its embedded history."""
    tree = {}
    files = byte_count = 0
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = safe_member_name(member.name)
            if name == '.git' or name.startswith('.git/') or member.isdir():
                continue
            if not (member.isfile() or member.issym()):
                raise ValueError(f'Unsupported snapshot member type: {name}')
            parent = tree
            parts = PurePosixPath(name).parts
            for part in parts[:-1]:
                parent = parent.setdefault(part, {})
                if not isinstance(parent, dict):
                    raise ValueError(f'File/directory collision: {name}')
            if parts[-1] in parent:
                raise ValueError(f'Duplicate tree entry: {name}')
            if member.issym():
                payload = member.linkname.encode('utf-8', 'surrogateescape')
                blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).digest()
                mode = b'120000'
            else:
                sha = hashlib.sha1(b'blob ' + str(member.size).encode() + b'\0')
                handle = archive.extractfile(member)
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    sha.update(chunk)
                blob = sha.digest()
                mode = b'100755' if member.mode & 0o111 else b'100644'
            parent[parts[-1]] = (mode, blob)
            files += 1
            byte_count += member.size

    def tree_id(entries: dict) -> bytes:
        body = b''
        for name, value in sorted(entries.items(), key=lambda item: item[0].encode('utf-8', 'surrogateescape') + (b'/' if isinstance(item[1], dict) else b'')):
            mode, sha = (b'40000', tree_id(value)) if isinstance(value, dict) else value
            body += mode + b' ' + name.encode('utf-8', 'surrogateescape') + b'\0' + sha
        return hashlib.sha1(b'tree ' + str(len(body)).encode() + b'\0' + body).digest()

    return {'tree_sha': tree_id(tree).hex(), 'files': files, 'file_bytes': byte_count}


def retrieval_task(task: dict) -> dict:
    return {key: task[key] for key in ('instance_id', 'repo', 'base_commit', 'problem_statement')}


def read_snapshot(path: Path, wanted: set[str] | None = None) -> dict[str, str]:
    """Read Python source, or evaluation-only requested files, without extraction."""
    sources = {}
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = safe_member_name(member.name)
            if name.startswith('.git/') or not member.isfile():
                continue
            if (wanted is None and not name.endswith('.py')) or (wanted is not None and name not in wanted):
                continue
            if name in sources:
                raise ValueError(f'Duplicate source member: {name}')
            if member.size > 16 * 1024 * 1024:
                raise ValueError(f'Source exceeds 16 MiB safety bound: {name}')
            raw = archive.extractfile(member).read()
            encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0] if name.endswith('.py') else 'utf-8'
            sources[name] = raw.decode(encoding)
    return sources


def definition_fingerprint(node: ast.AST) -> str:
    """Functions compare complete ASTs; supplied class stubs expose only headers."""
    normalized = copy.copy(node)
    normalized.decorator_list = []
    if isinstance(normalized, ast.ClassDef):
        normalized.body = []
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def extract_candidates(sources: dict[str, str], nodes: list[dict]) -> tuple[dict, dict]:
    """Bind source-compatible exact IDs, then unique naming aliases, without labels."""
    definitions = defaultdict(list)
    alias_definitions = defaultdict(dict)
    parse_errors = {}
    for path, text in sorted(sources.items()):
        if not path.endswith('.py'):
            continue
        parts = list(PurePosixPath(path).with_suffix('').parts)
        if parts[0] == 'src':
            parts = parts[1:]
        modules = ['.'.join(parts)]
        if parts[-1] == '__init__':
            modules.append('.'.join(parts[:-1]))
        module_aliases = {'.'.join(m.split('.')[i:]) for m in modules for i in range(len(m.split('.')))}
        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError as exc:
            parse_errors[path] = f'{exc.msg} (line {exc.lineno})'
            continue
        lines = text.splitlines(keepends=True)

        def visit(node: ast.AST, scope: list[str], classes: list[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    nested = scope + [child.name]
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    definition = {'path': path, 'start': start, 'end': child.end_lineno,
                                  'text': ''.join(lines[start - 1:child.end_lineno]),
                                  'fingerprint': definition_fingerprint(child)}
                    for module in modules:
                        definitions['.'.join([module, *nested])].append(definition)
                    nearest_class = [] if isinstance(child, ast.ClassDef) else classes[-1:]
                    for module in module_aliases:
                        for symbol_scope in (nested, classes + [child.name], nearest_class + [child.name]):
                            alias_definitions['.'.join([module, *symbol_scope])][(path, start, child.end_lineno)] = definition
                    visit(child, nested, classes + [child.name] if isinstance(child, ast.ClassDef) else classes)
                else:
                    visit(child, scope, classes)
        visit(tree, [], [])
    candidates = {}
    missing = []
    ambiguous = []
    alias_mapped = []
    ambiguous_aliases = []
    conflicting_exact = []
    seen = set()
    for node in sorted(nodes, key=lambda n: n['id']):
        node_id = node['id']
        if node_id in seen:
            raise ValueError(f'Duplicate graph ID: {node_id}')
        seen.add(node_id)
        matches = definitions.get(node_id, [])
        binding = 'exact'
        try:
            parsed = ast.parse(textwrap.dedent(node.get('text', ''))).body
        except (SyntaxError, TypeError):
            parsed = []
        fingerprint = (definition_fingerprint(parsed[0]) if len(parsed) == 1 and
                       isinstance(parsed[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None)
        if matches and fingerprint:
            compatible = [d for d in matches if d['fingerprint'] == fingerprint]
            if len(compatible) != len(matches):
                conflicting_exact.append(node_id)
            matches = compatible
        if not matches:
            if fingerprint:
                matches = [d for d in alias_definitions.get(node_id, {}).values() if d['fingerprint'] == fingerprint]
            if len(matches) > 1:
                ambiguous_aliases.append(node_id)
                matches = []
            if matches:
                alias_mapped.append(node_id)
                binding = 'source_verified_alias'
        if not matches or len({m['path'] for m in matches}) != 1:
            missing.append(node_id)
            continue
        if len(matches) > 1:
            ambiguous.append(node_id)
        candidates[node_id] = {'id': node_id, 'path': matches[0]['path'], 'binding': binding,
            'spans': [[m['start'], m['end']] for m in matches],
            'text': '\n'.join(m['text'] for m in matches)}
    return candidates, {'graph_nodes': len(nodes), 'mapped_nodes': len(candidates),
                        'missing_node_ids': missing, 'ambiguous_node_ids': ambiguous,
                        'alias_mapped_node_ids': alias_mapped, 'ambiguous_alias_node_ids': ambiguous_aliases,
                        'conflicting_exact_node_ids': conflicting_exact,
                        'source_parse_errors': parse_errors}
