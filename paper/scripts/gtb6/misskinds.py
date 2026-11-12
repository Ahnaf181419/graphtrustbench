"""Registered classification of a traced edge the supplied graph misses, by its callee.
Priority order: dunder > decorated > method > function; 'unresolved' if the definition is not found."""
import ast
import os

KINDS = ('dunder', 'decorated', 'method', 'function', 'unresolved')


class DefIndex:
    """qualname -> (is_decorated, in_class) for one module file."""

    def __init__(self, path):
        self.defs = {}
        try:
            tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
        except (SyntaxError, OSError):
            return
        self._walk(tree.body, '', False)

    def _walk(self, body, prefix, in_class):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + node.name
                self.defs[q] = (bool(node.decorator_list), in_class)
                self._walk(node.body, q + '.<locals>.', False)
            elif isinstance(node, ast.ClassDef):
                q = prefix + node.name
                self.defs[q] = (bool(node.decorator_list), False)
                self._walk(node.body, q + '.', True)


def module_path(src_parent, module):
    base = os.path.join(src_parent, *module.split('.'))
    return base + '.py' if os.path.exists(base + '.py') else os.path.join(base, '__init__.py')


_CACHE = {}


def clear_cache():
    """Call after every checkout: the index is per file path, not per commit."""
    _CACHE.clear()


def classify(src_parent, module, qualname):
    key = (src_parent, module)
    if key not in _CACHE:
        _CACHE[key] = DefIndex(module_path(src_parent, module))
    info = _CACHE[key].defs.get(qualname)
    if info is None:
        return 'unresolved'
    last = qualname.rsplit('.', 1)[-1]
    decorated, in_class = info
    if last.startswith('__') and last.endswith('__'):
        return 'dunder'
    if decorated:
        return 'decorated'
    if in_class:
        return 'method'
    return 'function'
