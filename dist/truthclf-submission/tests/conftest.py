from __future__ import annotations

import os
import sys

# Make `import truthclf` work whether run via pytest or the standalone runner.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
