"""The import structure the backend relies on, checked on the source itself (function-local imports count).

Every backend module belongs to a package in IMPORTS below, which lists, lowest layer first, what that package
may import from other packages. Three rules keep it true:

- a package imports only what its entry allows, so a layer cannot reach upward;
- every module is covered, so a new top-level file cannot sit outside the check;
- no two modules import each other, directly or through a chain.

The table is a ceiling, not an inventory: an allowance nothing uses yet is harmless.

`video` and `tracking` use each other's services (loading a video reads a tracking result, tracking reuses the
video's frame and SAM2 handling), and that is accepted. What makes it safe is the last rule: at the module level
nothing loops back. They are the only pair allowed to do this.
"""
import ast
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

SKIPPED_DIRECTORIES = {"tests", "tools", "cache", "saved", ".venv", "__pycache__"}

# package -> the other packages it may import. Lowest layer first.
IMPORTS = {
    "schemas": set(),                                                  # request, response and stored shapes
    "utils": set(),                                                    # overlay drawing, atomic JSON files
    "mask_manifest": {"utils"},                                        # propagation manifests: frame index, streamed reads
    "core": {"schemas"},                                               # session state, annotations, jobs, undo
    "sessions": {"core", "mask_manifest", "schemas", "utils"},         # session files on disk: save, publish, cleanup
    "sam2_frame_store": set(),                                         # model wrappers; they know nothing of the app
    "co_tracker": {"core"},
    "sam2_video_masker": {"core", "sam2_frame_store", "schemas"},
    "video": {"core", "mask_manifest", "sam2_video_masker", "schemas", "sessions", "tracking", "utils"},     # feature services
    "tracking": {"co_tracker", "core", "schemas", "sessions", "utils", "video"},
    "routers": {"core", "schemas", "tracking", "video"},               # HTTP
    "app": {"core", "routers", "sessions"},                            # assembly and startup/shutdown
    "api": {"app"},                                                    # entry point
}

# Pairs of packages that may import each other. Their modules still may not form a cycle.
MUTUAL = {frozenset({"video", "tracking"})}


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
        # (importing module, imported module) pairs that cross from one package into another
        cls.crossings = sorted((source, target) for source, targets in cls.edges.items()
                               for target in targets if package(source) != package(target))

    def test_no_package_imports_beyond_what_it_is_allowed(self):
        violations = [f"{source} imports {target}" for source, target in self.crossings
                      if package(target) not in IMPORTS[package(source)]]
        self.assertEqual(violations, [], "Imports the layering does not allow:\n" + "\n".join(violations))

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


if __name__ == "__main__":
    unittest.main()
