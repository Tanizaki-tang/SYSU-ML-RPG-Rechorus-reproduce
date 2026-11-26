"""Summarize frozen-split history lengths.

The multi-scale decoder only has redundant evidence to ensemble when the
history is long.  This script reports the effective history length
``min(history_length, 50)`` per phase, plus the share of evaluation targets
that are repeated interactions.  It explains why fixed crop lengths transfer
to MovieLens-1M but collapse to a no-op on Grocery.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "frozen_v1"


def main():
    for dataset in sorted(path.name for path in DATA.iterdir()
                          if path.is_dir()):
        frames = {}
        for phase in ("train", "dev", "test"):
            path = DATA / dataset / f"{phase}.csv"
            if not path.exists():
                continue
            frames[phase] = pd.read_csv(
                path, sep="\t",
                usecols=["user_id", "item_id", "time", "row_id",
                         "history_length"],
            )

        print(f"\n## {dataset}")
        for phase, frame in frames.items():
            history = np.minimum(frame["history_length"].to_numpy(), 50)
            quantiles = np.percentile(history, [10, 25, 50, 75, 90])
            print(
                f"- {phase}: n={len(frame)}, "
                f"mean={history.mean():.1f}, "
                f"q10/25/50/75/90="
                f"{'/'.join(f'{x:.0f}' for x in quantiles)}, "
                f"share<50={(history < 50).mean():.3f}"
            )

        # 重复目标：目标是否在用户之前的历史里出现。
        complete = pd.concat([
            frame.assign(phase=phase) for phase, frame in frames.items()
        ]).sort_values(["user_id", "time", "row_id"], kind="mergesort")
        history = {}
        repeated = []
        for row in complete.itertuples(index=False):
            user = int(row.user_id)
            seen = history.setdefault(user, set())
            if row.phase in ("dev", "test"):
                repeated.append(int(row.item_id) in seen)
            seen.add(int(row.item_id))
        print(f"- repeat-target rate (dev+test): "
              f"{np.mean(repeated):.4f}")

        ids = json.loads(
            (DATA / dataset / "semantic_ids_m32.json").read_text(
                encoding="utf-8"
            )
        )
        print(f"- items={complete['item_id'].nunique()}, "
              f"m32 collisions={ids['extra_items_sharing_ids']}")


if __name__ == "__main__":
    main()
