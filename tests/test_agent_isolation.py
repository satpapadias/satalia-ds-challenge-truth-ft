"""The agents are pure MCP clients.

Every capability an agent has is reached by calling an MCP tool. None of them
imports truthclf, and the agent container image is built without the truthclf
package and without the libraries it depends on, so `import truthclf` fails
inside it. The Dockerfile's `agent` stage asserts that at build time.

This file asserts the same property from the source, so it is checked on every
test run rather than only when an image is built. A build takes a minute and
needs a daemon; this takes milliseconds and catches the mistake at the moment it
is made.

The failure it guards against is easy to make and hard to see: reaching into
truthclf from an agent still works locally, still passes every functional test,
and quietly converts a tool call into an in-process function call -- which is
precisely the distinction between an agent and a library wrapped in HTTP.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENTS = ROOT / "truthclf_agents"

# Libraries the agents must not need. They belong to the MCP servers, which are
# the only processes that do modelling or data work.
FORBIDDEN_RUNTIME = ("truthclf", "truthclf_mcp", "sklearn", "scipy", "pandas",
                     "statsmodels", "numpy", "together", "tiktoken", "diskcache")

AGENT_MODULES = ["truthclf_agents.orchestrator", "truthclf_agents.zero_shot",
                 "truthclf_agents.fine_tuned", "truthclf_agents.explainer"]


def _agent_sources():
    return sorted(AGENTS.glob("*.py"))


def _imported_roots(path: pathlib.Path) -> set[str]:
    """Top-level package names imported by one module, from its syntax tree.

    Read statically rather than by importing, so a module is checked even when
    its dependencies are not installed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which stays inside the package.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_there_are_agent_sources_to_check():
    """Guard the guard: an empty glob would make every test below vacuous."""
    assert len(_agent_sources()) >= 6


@pytest.mark.parametrize("path", _agent_sources(), ids=lambda p: p.name)
def test_no_agent_module_imports_truthclf_or_the_data_stack(path):
    offending = _imported_roots(path) & set(FORBIDDEN_RUNTIME)
    assert not offending, (
        f"{path.name} imports {sorted(offending)}. Agents reach capabilities "
        "only through MCP tool calls; this would make the agent image "
        "unbuildable and turn a tool call into an in-process function call.")


def test_the_dockerfile_enforces_it_at_build_time():
    """The static check above is not the only line of defence.

    A test can be deleted; a build that fails cannot be ignored. This asserts the
    Dockerfile still carries the check, so removing one does not silently leave
    the property unguarded.
    """
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS agent" in dockerfile
    assert "--only-group agents" in dockerfile
    assert "--no-install-project" in dockerfile
    for mod in ("truthclf", "sklearn", "pandas"):
        assert mod in dockerfile, f"the build-time check no longer covers {mod}"


def test_the_agent_dependency_group_excludes_the_data_stack():
    """The group is what the agent image installs, so its contents are the
    image's contents."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    group = text.split("[dependency-groups]", 1)[1].split("[tool.", 1)[0]
    lowered = group.lower()
    for banned in ("scikit-learn", "scipy", "pandas", "statsmodels",
                   "numpy", "together", "tiktoken", "diskcache"):
        assert banned not in lowered, (
            f"{banned!r} is in the agents dependency group; the agent image "
            "would then carry the data-science stack it exists to exclude")


def test_agents_import_without_truthclf_on_the_path(tmp_path):
    """The end-to-end version: import every agent in a interpreter that cannot
    see truthclf at all, which is the situation inside the agent image."""
    # A directory containing only the agent package, so truthclf is unreachable
    # even though it is installed in this environment.
    link = tmp_path / "truthclf_agents"
    link.symlink_to(AGENTS, target_is_directory=True)

    env = dict(os.environ)
    # Drop the project root from the path, and disable the site-packages entry
    # that provides the editable truthclf install.
    env["PYTHONPATH"] = str(tmp_path)
    env["PYTHONNOUSERSITE"] = "1"

    code = (
        "import sys\n"
        f"sys.path = [p for p in sys.path if p not in {{{str(ROOT)!r}, ''}}]\n"
        "import importlib\n"
        "try:\n"
        "    importlib.import_module('truthclf')\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('truthclf was importable; test setup is wrong')\n"
        + "".join(f"importlib.import_module({m!r})\n" for m in AGENT_MODULES)
        + "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=str(tmp_path))
    if "truthclf was importable" in proc.stdout + proc.stderr:
        pytest.skip("truthclf is installed non-editably; the build-time check "
                    "in the Dockerfile covers this case")
    assert proc.returncode == 0, (
        f"an agent failed to import without truthclf present:\n{proc.stderr[-2000:]}")
    assert "ok" in proc.stdout
