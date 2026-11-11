"""The import structure the backend relies on, checked on the source itself.

Lower layers know nothing about higher ones, and no two modules import each other, directly or through
a chain, even with function-local imports.
"""
import ast
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SKIPPED_DIRECTORIES = {"tests", "tools", "cache", "saved", ".venv", "__pycache__"}

# What each lower layer may import (any import, including function-local ones). Higher layers
# (video, tracking, routers, app) may import anything below them.
ALLOWED = {
    "schemas": {"schemas"},
    "core": {"core", "schemas"},
    "sessions": {"sessions", "core", "schemas", "utils"},
}


def backend_modules() -> dict[str, Path]:
    modules = {}
    for path in sorted(BACKEND.rglob("*.py")):
        relative = path.relative_to(BACKEND)
        if SKIPPED_DIRECTORIES.intersection(relative.parts):
            continue
        name = ".".join(relative.with_suffix("").parts)
        modules[name[: -len(".__init__")] if name.endswith(".__init__") else name] = path
    return modules


def imports_of(path: Path, modules: dict[str, Path]) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in found if name in modules}


def package(module: str) -> str:
    return module.split(".")[0]


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = backend_modules()
        cls.edges = {name: imports_of(path, cls.modules) for name, path in cls.modules.items()}

    def test_lower_layers_import_nothing_above_them(self):
        violations = [
            f"{source} imports {target}"
            for source, targets in sorted(self.edges.items())
            if package(source) in ALLOWED
            for target in sorted(targets)
            if package(target) not in ALLOWED[package(source)]
        ]
        self.assertEqual(violations, [], "A lower layer reaches into a higher one:\n" + "\n".join(violations))

    def test_no_two_modules_import_each_other_even_through_a_chain(self):
        index, low, stack, on_stack, cycles, counter = {}, {}, [], set(), [], [0]

        def visit(node):
            index[node] = low[node] = counter[0]
            counter[0] += 1
            stack.append(node)
            on_stack.add(node)
            for target in self.edges[node]:
                if target not in index:
                    visit(target)
                    low[node] = min(low[node], low[target])
                elif target in on_stack:
                    low[node] = min(low[node], index[target])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    cycles.append(sorted(component))

        for node in self.edges:
            if node not in index:
                visit(node)
        self.assertEqual(cycles, [], f"Import cycles between modules: {cycles}")

    def test_the_check_sees_the_real_modules(self):
        # A guard against the test passing because it found nothing to check.
        for expected in ("core.context", "sessions.publication", "video.propagation", "tracking.runtime", "app.lifecycle"):
            self.assertIn(expected, self.modules)
        self.assertIn("core.annotations", self.edges["tracking.results"])
        self.assertGreater(sum(len(targets) for targets in self.edges.values()), 100)


if __name__ == "__main__":
    unittest.main()
