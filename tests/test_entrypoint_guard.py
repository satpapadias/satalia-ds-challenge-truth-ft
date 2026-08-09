"""The evaluation entrypoint must FAIL on an incomplete cache, not fabricate.

Regression test for a defect where a fresh clone (the schema-2 cache is
gitignored) ran the default invocation to completion with exit code 0 and
printed an explainer table built from 300 silent p=0.5 defaults.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _run(args, cwd, env=None):
    e = dict(os.environ)
    e.pop("TOGETHER_API_KEY", None)
    e.update(env or {})
    return subprocess.run([sys.executable, "scripts/regenerate_results.py", *args],
                          cwd=cwd, env=e, capture_output=True, text=True, timeout=900)


def _clone(tmp_path):
    """A checkout without the gitignored schema-2 cache."""
    dst = tmp_path / "clone"
    subprocess.run(["git", "archive", "HEAD", "-o", str(tmp_path / "t.tar")],
                   cwd=ROOT, check=True)
    dst.mkdir()
    subprocess.run(["tar", "-xf", str(tmp_path / "t.tar"), "-C", str(dst)], check=True)
    # Overlay the WORKING COPY of the package and scripts, so the test exercises
    # the code as it is now rather than as it was at HEAD. Copying only the
    # entrypoint left it calling an older truthclf/ and failing spuriously.
    import shutil
    for d in ("truthclf", "scripts"):
        shutil.rmtree(dst / d, ignore_errors=True)
        shutil.copytree(pathlib_Path(ROOT) / d, dst / d,
                        ignore=shutil.ignore_patterns("__pycache__"))
    assert not (dst / ".llm_cache").exists(), "clone must not carry the schema-2 cache"
    return dst


from pathlib import Path as pathlib_Path  # noqa: E402


def test_default_invocation_fails_without_the_schema2_cache(tmp_path):
    clone = _clone(tmp_path)
    r = _run([], clone)
    assert r.returncode != 0, (
        "entrypoint completed without the cache — it must not fabricate numbers\n"
        + r.stdout[-2000:])
    assert "refetch_quarantined" in (r.stdout + r.stderr), \
        "the failure must name the script that rebuilds the cache"


def test_failure_message_offers_the_offline_route(tmp_path):
    clone = _clone(tmp_path)
    r = _run([], clone)
    assert "--source archive" in (r.stdout + r.stderr)


def test_offline_invocation_succeeds_on_a_clean_clone(tmp_path):
    """The fully-offline path must still work from the tracked v1 archive."""
    clone = _clone(tmp_path)
    r = _run(["--source", "archive", "--explainer-source", "archive"], clone)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "0.668006" in r.stdout, "must reproduce the adopted zero-shot accuracy"
