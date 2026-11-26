"""Run decoder-only comparison experiments for the RPG baseline.

All configurations reuse the frozen datasets, semantic IDs and checkpoints
recorded in ``handoff/baseline_assets.json``; only the decoding procedure
changes.  Each run writes the standard ReChorus JSON record, and this script
collects them into a single summary.

Examples::

    python scripts/run_decoder_experiments.py --datasets MovieLens_1M \
        --seeds 2026 --phases dev
    python scripts/run_decoder_experiments.py --configs baseline_full,tta_full \
        --datasets MovieLens_1M,Grocery_and_Gourmet_Food --seeds 2026,2027,2028 \
        --phases test
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "handoff" / "baseline_assets.json"
OUTDIR = ROOT / "results" / "exp" / "decoder"

# Decoder configurations.  ``graph`` selects the semantic-ID graph search and
# inherits the dataset's recorded graph parameters unless overridden here.
CONFIGS = {
    "baseline_full": {
        "decode": "full",
    },
    "baseline_graph": {
        "decode": "graph",
    },
    "tta_full": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
    },
    "tta_full_dense": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "50,45,40,35,30,25,20,15,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
    },
    "tta_graph": {
        "decode": "graph",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
    },
    "tta_full_znorm": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "digit_norm": "zscore",
    },
    "tta_full_confw": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "digit_weight": "confidence",
    },
    "tta_full_both": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "digit_norm": "zscore",
        "digit_weight": "confidence",
    },
    "tta_rel_a": {
        "decode": "full",
        "history_tta": "relative",
        "crop_ratios": "1.0,0.6,0.3",
        "tta_aggregate": "prob",
    },
    "tta_rel_b": {
        "decode": "full",
        "history_tta": "relative",
        "crop_ratios": "0.8,0.5,0.25",
        "tta_aggregate": "prob",
    },
    "tta_rel_c": {
        "decode": "full",
        "history_tta": "relative",
        "crop_ratios": "1.0,0.75,0.5",
        "tta_aggregate": "prob",
    },
    "tta_rel_d": {
        "decode": "full",
        "history_tta": "relative",
        "crop_ratios": "0.9,0.6,0.3",
        "tta_aggregate": "prob",
    },
    "mcd8": {
        "decode": "full",
        "mcd_samples": 8,
        "tta_aggregate": "prob",
    },
    "mcd16": {
        "decode": "full",
        "mcd_samples": 16,
        "tta_aggregate": "prob",
    },
    "mcd24": {
        "decode": "full",
        "mcd_samples": 24,
        "tta_aggregate": "prob",
    },
    "tta_mcd8": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "mcd_samples": 8,
    },
    "scale08": {"decode": "full", "logprob_scale": 0.8},
    "scale15": {"decode": "full", "logprob_scale": 1.5},
    "pop01": {"decode": "full", "pop_beta": 0.1},
    "pop01_tau20": {
        "decode": "full", "pop_beta": 0.1, "pop_length_tau": 20,
    },
    "pop02": {"decode": "full", "pop_beta": 0.2},
    "pop05": {"decode": "full", "pop_beta": 0.5},
    "pop02_tau20": {
        "decode": "full", "pop_beta": 0.2, "pop_length_tau": 20,
    },
    "tta_mcd8_graph": {
        "decode": "graph",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "mcd_samples": 8,
    },
    "tta_mcd8_pop": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "mcd_samples": 8,
        "pop_beta": 0.2,
        "pop_length_tau": 20,
    },
    "tta_pop": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "pop_beta": 0.2,
        "pop_length_tau": 20,
    },
    "adaptive": {
        "decode": "full",
        "history_tta": "crops",
        "crop_lengths": "40,20,10",
        "tta_aggregate": "prob",
        "tta_include_full": 0,
        "mcd_samples": 8,
        "pop_beta": 0.2,
        "pop_length_tau": 20,
        "tta_min_history": 20,
    },
}


def build_command(
    dataset, seed, phase, config, asset, log_file, eval_limit=0
):
    training = asset["training_config"]
    checkpoint = asset["checkpoints"][str(seed)]["file"]
    graph = asset["graph_config"]

    command = [
        sys.executable, "-B", "src/run_unified.py",
        "--model_name", "RPG", "--gpu", "0",
        "--path", "data/frozen_v1", "--dataset", dataset,
        "--sid_file", "semantic_ids_m32.npy", "--history_max", "50",
        "--emb_size", "128", "--ffn_size", "512",
        "--n_layers", "2", "--n_heads", "4",
        "--temperature", str(training["temperature"]),
        "--random_seed", str(seed),
        "--train", "0", "--load", "1", "--model_path", checkpoint,
        "--eval_phase", phase, "--eval_limit", str(eval_limit),
        "--decode", config["decode"],
        "--eval_batch_size", "32", "--item_chunk", "2048",
        "--topk", "5,10", "--metric", "NDCG,HR",
        "--main_metric", "NDCG@10",
        "--log_file", str(log_file),
    ]
    if config["decode"] == "graph":
        command += [
            "--graph_edges", str(graph["edges"]),
            "--graph_beams", str(graph["beams"]),
            "--graph_steps", str(graph["steps"]),
            "--decode_seed", str(graph["decode_seed"]),
        ]
    if "history_tta" in config:
        command += [
            "--history_tta", config["history_tta"],
            "--crop_lengths", config.get("crop_lengths", ""),
            "--crop_ratios", config.get("crop_ratios", ""),
            "--tta_include_full", str(config.get("tta_include_full", 1)),
        ]
    if "tta_aggregate" in config:
        command += ["--tta_aggregate", config["tta_aggregate"]]
    if "mcd_samples" in config:
        command += ["--mcd_samples", str(config["mcd_samples"])]
    if "tta_min_history" in config:
        command += ["--tta_min_history", str(config["tta_min_history"])]
    if "digit_norm" in config:
        command += ["--digit_norm", config["digit_norm"]]
    if "digit_weight" in config:
        command += ["--digit_weight", config["digit_weight"]]
    if "logprob_scale" in config:
        command += ["--logprob_scale", str(config["logprob_scale"])]
    if "pop_beta" in config:
        command += ["--pop_beta", str(config["pop_beta"])]
    if "pop_length_tau" in config:
        command += ["--pop_length_tau", str(config["pop_length_tau"])]
    return command


def summarize(dataset, phase, seeds):
    """跨种子汇总：均值与样本标准差。"""
    summary_file = OUTDIR / "summary.json"
    if not summary_file.exists():
        raise SystemExit(f"缺少 {summary_file}")
    rows = json.loads(summary_file.read_text(encoding="utf-8"))
    groups = {}
    for row in rows:
        if row["dataset"] != dataset or row["phase"] != phase:
            continue
        if row.get("eval_limit"):
            continue
        if seeds and row["seed"] not in seeds:
            continue
        groups.setdefault(row["config"], []).append(row)

    print(f"\n# {dataset} / {phase} / seeds={sorted({r['seed'] for g in groups.values() for r in g})}")
    print("| config | NDCG@10 | HR@10 | NDCG@5 | HR@5 | visited |")
    print("|---|---:|---:|---:|---:|---:|")
    for config, runs in sorted(groups.items()):
        cells = []
        for metric in ("NDCG@10", "HR@10", "NDCG@5", "HR@5"):
            values = [r["metrics"][metric] for r in runs]
            mean = sum(values) / len(values)
            cells.append(f"{mean:.5f}")
        visited = [r["mean_visited_items"] for r in runs
                   if r.get("mean_visited_items") is not None]
        visited_text = "-" if not visited else f"{sum(visited)/len(visited):.0f}"
        print(f"| {config} | " + " | ".join(cells) + f" | {visited_text} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", default=",".join(CONFIGS),
        help="逗号分隔的配置名，见 CONFIGS",
    )
    parser.add_argument(
        "--datasets",
        default="MovieLens_1M,Grocery_and_Gourmet_Food",
    )
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--phases", default="dev")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--eval_limit", type=int, default=0,
        help="只评估前 N 条样本，用于冒烟测试",
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="不运行实验，只对已有 summary.json 跨种子汇总",
    )
    args = parser.parse_args()

    if args.summarize:
        for dataset in args.datasets.split(","):
            for phase in args.phases.split(","):
                summarize(
                    dataset, phase,
                    {int(x) for x in args.seeds.split(",") if x},
                )
        return

    configs = [name for name in args.configs.split(",") if name]
    datasets = [name for name in args.datasets.split(",") if name]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    phases = [name for name in args.phases.split(",") if name]

    for name in configs:
        if name not in CONFIGS:
            raise SystemExit(f"未知配置 {name}，可选：{sorted(CONFIGS)}")

    assets = json.loads(ASSETS.read_text(encoding="utf-8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for dataset in datasets:
        asset = assets[dataset]
        for seed in seeds:
            for phase in phases:
                for name in configs:
                    config = CONFIGS[name]
                    stem = f"{dataset}_{seed}_{name}"
                    log_file = OUTDIR / f"{stem}.log"
                    record_file = log_file.with_suffix(
                        f".{phase}.{config['decode']}.json"
                    )
                    command = build_command(
                        dataset, seed, phase, config, asset, log_file,
                        eval_limit=args.eval_limit,
                    )
                    if args.dry_run:
                        print(" ".join(command))
                        continue

                    print(f"[run] {stem} phase={phase}", flush=True)
                    result = subprocess.run(
                        command, cwd=ROOT,
                        capture_output=True, text=True,
                    )
                    if result.stdout:
                        print(result.stdout.rstrip(), flush=True)
                    if result.returncode != 0:
                        print(result.stderr, file=sys.stderr)
                        raise SystemExit(
                            f"{stem} 失败，返回码 {result.returncode}"
                        )

                    record = json.loads(
                        record_file.read_text(encoding="utf-8")
                    )
                    entry = {
                        "dataset": dataset,
                        "seed": seed,
                        "phase": phase,
                        "config": name,
                        "decode": config["decode"],
                        "history_tta": record.get("history_tta"),
                        "crop_lengths": record.get("crop_lengths"),
                        "metrics": record["metrics"],
                        "evaluation_seconds": record.get(
                            "evaluation_seconds"
                        ),
                        "graph_build_seconds": record.get(
                            "graph_build_seconds"
                        ),
                        "mean_visited_items": record.get(
                            "mean_visited_items"
                        ),
                        "eval_limit": args.eval_limit,
                        "record_file": str(record_file.relative_to(ROOT)),
                    }
                    summary.append(entry)

    if args.dry_run:
        return

    summary_file = OUTDIR / "summary.json"
    existing = []
    if summary_file.exists():
        existing = json.loads(
            summary_file.read_text(encoding="utf-8")
        )
    key = lambda row: (
        row["dataset"], row["seed"], row["phase"], row["config"]
    )
    merged = {key(row): row for row in existing}
    merged.update({key(row): row for row in summary})
    merged = sorted(merged.values(), key=key)
    summary_file.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    header = (
        "| dataset | seed | phase | config | NDCG@10 | HR@10 "
        "| NDCG@5 | HR@5 | visited | eval s |"
    )
    print("\n" + header)
    print("|---" * 10 + "|")
    for row in merged:
        metrics = row["metrics"]
        visited = row.get("mean_visited_items")
        print(
            f"| {row['dataset']} | {row['seed']} | {row['phase']} "
            f"| {row['config']} | {metrics['NDCG@10']:.5f} "
            f"| {metrics['HR@10']:.5f} | {metrics['NDCG@5']:.5f} "
            f"| {metrics['HR@5']:.5f} "
            f"| {'-' if visited is None else f'{visited:.0f}'} "
            f"| {row['evaluation_seconds']:.1f} |"
        )
    print(f"\n汇总已写入 {summary_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
