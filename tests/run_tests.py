"""Standalone test runner (pytest not installed in this env).
Discovers `test_*` functions in the test modules, runs them, reports pass/fail.
Equivalent to `python -m pytest tests/` once pytest is available."""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_data
import test_metrics
import test_prompts
import test_calibration
import test_explain

MODULES = [test_data, test_metrics, test_prompts, test_calibration, test_explain]


def main() -> int:
    passed, failed = 0, 0
    for mod in MODULES:
        names = sorted(n for n in dir(mod) if n.startswith("test_"))
        print(f"\n{mod.__name__}  ({len(names)} tests)")
        for name in names:
            try:
                getattr(mod, name)()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{'='*50}\n  {passed} passed, {failed} failed\n{'='*50}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
