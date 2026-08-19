import json
import sys
from dataclasses import asdict
from truthclf import experiments

# Take an optional argument for the number of rows, default to 10
n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10

# Sample 10 rows from the test split
rows = experiments.sample_rows("test", n=n_rows, seed=0)

# The /verify endpoint expects a list of dictionaries, not Row objects.
# It also expects the ground-truth labels in a separate 'labels' key.
payload = {
    "points": [asdict(row) for row in rows],
    "labels": [row.label for row in rows]
}

print(json.dumps(payload, indent=2))