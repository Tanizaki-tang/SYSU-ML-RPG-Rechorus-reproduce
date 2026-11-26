"""Summarize decoder experiments collected in results/exp/decoder/summary.json.

Prints, per dataset and phase:
  * mean and sample standard deviation across seeds for each configuration;
  * paired per-seed deltas of an experiment configuration against a baseline
    (same checkpoint, so the delta is a within-checkpoint comparison).

Example::

    python scripts/analyze_decoder_results.py --phase test \
        --base baseline_full --exp tta_rel_a
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "exp" / "decoder" / "summary.json"
METRICS = ("NDCG@10", "HR@10", "NDCG@5", "HR@5")


def mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, variance ** 0.5


def fmt(mean, std):
    return f"{mean:.5f} ± {std:.5f}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="test")
    parser.add_argument(
        "--datasets", default="MovieLens_1M,Grocery_and_Gourmet_Food"
    )
    parser.add_argument("--base", default="baseline_full")
    parser.add_argument("--exp", default="")
    args = parser.parse_args()

    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    for dataset in [x for x in args.datasets.split(",") if x]:
        subset = [
            row for row in rows
            if row["dataset"] == dataset and row["phase"] == args.phase
            and not row.get("eval_limit")
        ]
        if not subset:
            continue

        by_config = {}
        for row in subset:
            by_config.setdefault(row["config"], {})[row["seed"]] = row

        seeds = sorted(set().union(
            *[set(runs) for runs in by_config.values()]
        ))
        print(f"\n## {dataset} / {args.phase} / seeds={seeds}")
        print("| config | " + " | ".join(METRICS) + " | visited | eval s |")
        print("|---" * (len(METRICS) + 3) + "|")
        for config, runs in sorted(by_config.items()):
            cells = []
            for metric in METRICS:
                values = [
                    runs[seed]["metrics"][metric]
                    for seed in seeds if seed in runs
                ]
                mean, std = mean_std(values)
                cells.append(fmt(mean, std))
            visited = [
                runs[seed]["mean_visited_items"]
                for seed in seeds
                if seed in runs
                and runs[seed].get("mean_visited_items") is not None
            ]
            visited_text = "-"
            if visited:
                mean, std = mean_std(visited)
                visited_text = f"{mean:.0f} ± {std:.0f}"
            seconds = [
                runs[seed]["evaluation_seconds"]
                for seed in seeds
                if seed in runs and runs[seed].get("evaluation_seconds")
            ]
            seconds_text = "-"
            if seconds:
                mean, std = mean_std(seconds)
                seconds_text = f"{mean:.1f}"
            print(f"| {config} | " + " | ".join(cells)
                  + f" | {visited_text} | {seconds_text} |")

        if args.exp and args.exp in by_config and args.base in by_config:
            base, exp = by_config[args.base], by_config[args.exp]
            shared = sorted(set(base) & set(exp))
            print(f"\n配对差值 {args.exp} - {args.base}（{len(shared)} 个种子）")
            print("| metric | " + " | ".join(str(s) for s in shared)
                  + " | mean ± std |")
            print("|---" * (len(shared) + 2) + "|")
            for metric in METRICS:
                deltas = [
                    exp[seed]["metrics"][metric]
                    - base[seed]["metrics"][metric]
                    for seed in shared
                ]
                mean, std = mean_std(deltas)
                cells = " | ".join(f"{d:+.5f}" for d in deltas)
                print(f"| {metric} | {cells} | {mean:+.5f} ± {std:.5f} |")


if __name__ == "__main__":
    main()
