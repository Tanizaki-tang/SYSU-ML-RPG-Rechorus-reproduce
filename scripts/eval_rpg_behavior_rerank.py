"""Evaluate train-only behavior reranking on the frozen RPG cohorts."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch.nn.utils.rnn import pad_sequence


BASELINE = Path("results/rpg_main_v1")
FROZEN_CONFIG = Path("results/rpg_behavior_rerank_config.json")
DATASETS = ("MovieLens_1M", "Grocery_and_Gourmet_Food")
TOP_K = 100
RECENT = 5
WEIGHTS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
SPLIT_SEED = 2026
PRIOR_MASS = 50.0
RELIABILITY_MASS = 5.0


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def train_signals(frame, n_items):
    ordered = frame.sort_values(["user_id", "time", "row_id"])
    items = ordered["item_id"].to_numpy()
    timestamps = ordered["time"].to_numpy()
    users = ordered["user_id"].to_numpy()
    same_user = users[:-1] == users[1:]
    sources, targets = items[:-1][same_user], items[1:][same_user]
    tied = (timestamps[:-1] == timestamps[1:])[same_user]
    prior = np.bincount(items, minlength=n_items).astype(np.float64) + 1
    prior[0] = 0
    prior /= prior.sum()

    def counts(left, right, weights):
        return coo_matrix((weights, (left, right)), shape=(n_items, n_items)).tocsr()

    reversed_left = np.concatenate((sources, targets))
    reversed_right = np.concatenate((targets, sources))
    forward = np.where(tied, 0.5, 1.0)
    time_weights = np.concatenate((forward, 0.5 * tied))
    graphs = {
        "cooccurrence": counts(
            reversed_left, reversed_right, np.full(2 * len(sources), 0.5)
        ),
        "directed": counts(sources, targets, np.ones(len(sources))),
        "time_aware": counts(
            reversed_left[time_weights > 0],
            reversed_right[time_weights > 0],
            time_weights[time_weights > 0],
        ),
    }
    return prior, graphs, {
        "train_adjacent_pairs": len(sources),
        "same_timestamp_pairs": int(tied.sum()),
    }


def rank_metrics(rank):
    safe_rank = np.maximum(rank, 1)
    result = {}
    for k in (5, 10):
        hit = (rank > 0) & (rank <= k)
        result[f"HR@{k}"] = hit.astype(float)
        result[f"NDCG@{k}"] = hit / np.log2(safe_rank + 1)
    result["HR@100"] = (rank > 0).astype(float)
    return result


def target_rank(predictions, targets):
    matches = predictions == targets[:, None]
    return np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)


def summarize(metrics, selection):
    return {name: float(value[selection].mean()) for name, value in metrics.items()}


def row_standardize(values, valid):
    count = valid.sum(axis=1, keepdims=True).clip(min=1)
    mean = np.where(valid, values, 0).sum(axis=1, keepdims=True) / count
    centered = np.where(valid, values - mean, 0)
    std = np.sqrt(np.square(centered).sum(axis=1, keepdims=True) / count)
    return np.divide(centered, std, out=np.zeros_like(centered), where=std > 1e-12)


def behavior_scores(graph, prior, histories, candidates):
    safe_prior = np.where(candidates > 0, prior[candidates], 1)
    total = np.zeros(candidates.shape, dtype=np.float64)
    mass = np.zeros((len(histories), 1), dtype=np.float64)
    # Index 0 is the most recent history item. Padding has zero weight.
    for index, items in enumerate(histories.T):
        counts = np.asarray(graph[np.broadcast_to(items[:, None], candidates.shape).ravel(),
                                  candidates.ravel()]).reshape(candidates.shape)
        lift = np.log1p(counts / (PRIOR_MASS * safe_prior))
        reliable_lift = lift * counts / (counts + RELIABILITY_MASS)
        weight = (items[:, None] > 0) * 0.8**index
        total += weight * reliable_lift
        mass += weight
    return total / mass.clip(min=1)


def bootstrap_delta(differences, users, seed):
    _, user_indices = np.unique(users, return_inverse=True)
    counts = np.bincount(user_indices)
    sums = np.bincount(user_indices, weights=differences)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(20):
        sampled = rng.integers(len(counts), size=(50, len(counts)))
        estimates.extend(sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


@torch.inference_mode()
def retrieve(model, corpus, frame, batch_size, chunk_size, blocked_items, mask_history, rank_items):
    device = next(model.parameters()).device
    recent = np.zeros((len(frame), RECENT), dtype=np.int64)
    ids = []
    scores = []
    for start in range(0, len(frame), batch_size):
        rows = frame.iloc[start:start + batch_size]
        histories, blocked = [], []
        for offset, row in enumerate(rows.itertuples(index=False)):
            complete = [item for item, _ in corpus.user_his[int(row.user_id)][:int(row.position)]]
            blocked.append(blocked_items(complete, row.item_id))
            tail = complete[-RECENT:][::-1]
            recent[start + offset, :len(tail)] = tail
            histories.append(torch.tensor(complete[-model.history_max:], dtype=torch.long))
        feed = {
            "history_items": pad_sequence(histories, batch_first=True).to(device),
            "lengths": torch.tensor([len(history) for history in histories], device=device),
        }
        probs = model.token_logits(feed).log_softmax(dim=-1)
        best_ids = torch.empty((len(rows), 0), dtype=torch.long, device=device)
        best_scores = probs.new_empty((len(rows), 0))
        for first in range(1, corpus.n_items, chunk_size):
            item_ids = torch.arange(first, min(first + chunk_size, corpus.n_items), device=device)
            item_scores = model.score_items(probs, item_ids)
            if not torch.isfinite(item_scores).all():
                raise ValueError("Non-finite item scores")
            item_scores = mask_history(item_ids, item_scores, blocked)
            best_ids, best_scores = rank_items(
                torch.cat((best_ids, item_ids.expand(len(rows), -1)), dim=1),
                torch.cat((best_scores, item_scores), dim=1),
                TOP_K,
            )
        ids.append(best_ids.cpu().numpy())
        scores.append(best_scores.cpu().numpy())
    return np.concatenate(ids), np.concatenate(scores), recent


def baseline_inputs(root, dataset, seed, phase, frozen):
    if frozen is not None:
        locked = frozen["datasets"][dataset]
        reference = locked["test_baseline_by_seed"][str(seed)]
        settings = {**frozen["model"], **locked["model"]}
        checkpoint = root / reference["checkpoint"]
        if digest(checkpoint) != reference["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch for {dataset}, seed {seed}")
        return checkpoint, reference, settings
    source = root / BASELINE
    config = read_json(source / "train.json")[dataset]["RPG"]["config"]
    tag = f"m{config['digits']}_lr{config['lr']}_t{config['temperature']}_s{seed}_full"
    checkpoint = source / "train" / dataset / "RPG" / tag / "best.pt"
    reference_folder = "train" if phase == "dev" else "test"
    reference = read_json(source / reference_folder / dataset / "RPG" / tag / "metrics.json")
    return checkpoint, reference, reference["arguments"]


def run_dataset(dataset, root, seed, reader_class, model_class, ranking_api, locked=None, phase="dev", frozen=None):
    checkpoint, reference, settings = baseline_inputs(root, dataset, seed, phase, frozen)
    model_args = model_class.parse_model_args(argparse.ArgumentParser()).parse_args([])
    for key in ("sid_file", "emb_size", "ffn_size", "n_layers", "n_heads", "temperature", "history_max"):
        setattr(model_args, key, settings[key])
    model_args.path = str(root / "data/frozen_v1")
    model_args.dataset = dataset
    model_args.sep = "\t"
    model_args.device = torch.device("cuda")
    corpus = reader_class(model_args)
    sid = Path(model_args.path) / dataset / model_args.sid_file
    sidecar = read_json(sid.with_suffix(".json"))
    sid_hash = digest(sid)
    if sidecar["sid_sha256"] != sid_hash or sidecar["binding"]["dataset_hash"] != corpus.dataset_hash:
        raise ValueError(f"Semantic ID checksum or dataset mismatch for {dataset}")
    model = model_class(model_args, corpus)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not torch.equal(state["codes"], model.codes):
        raise ValueError(f"Checkpoint and semantic IDs disagree for {dataset}")
    model.load_state_dict(state)
    model.to(model_args.device).eval()
    frame = corpus.data_df[phase].query("position > 0")
    row_hash = hashlib.sha256(json.dumps(frame["row_id"].tolist()).encode()).hexdigest()
    expected_dataset_hash = locked["dataset_hash"] if frozen is not None else reference["dataset_hash"]
    if corpus.dataset_hash != expected_dataset_hash or row_hash != reference["instance_hash"]:
        raise ValueError(f"Baseline {phase} cohort mismatch for {dataset}")
    if locked is not None:
        if locked["dataset_hash"] != corpus.dataset_hash or locked["sid_sha256"] != sid_hash:
            raise ValueError(f"Frozen pilot and dataset inputs differ for {dataset}")
        if phase == "dev" and locked["instance_hash"] != row_hash:
            raise ValueError(f"Frozen pilot validation cohort differs for {dataset}")
    print(f"Seed {seed} {dataset}: retrieving Top-{TOP_K} for {len(frame)} {phase} queries", flush=True)
    candidates, scores, histories = retrieve(
        model, corpus, frame, settings["eval_batch_size"], settings["item_chunk"], *ranking_api
    )
    targets = frame["item_id"].to_numpy()
    users = frame["user_id"].to_numpy()
    valid = candidates > 0
    base = rank_metrics(target_rank(candidates, targets))
    baseline = summarize(base, slice(None))
    for metric, expected in reference["metrics"].items():
        if not np.isclose(baseline[metric], expected, atol=1e-10, rtol=0):
            raise ValueError(f"Baseline {dataset} {metric} differs: {baseline[metric]} vs {expected}")

    unique_users = np.unique(users)
    calibration = None
    evaluation_rows = np.ones(len(frame), dtype=bool)
    evaluation_key = "test" if phase == "test" else "holdout"
    if phase == "dev":
        shuffled_users = np.random.default_rng(SPLIT_SEED).permutation(unique_users)
        calibration = np.isin(users, shuffled_users[:len(shuffled_users) // 2])
        evaluation_rows = ~calibration
    prior, graphs, graph_stats = train_signals(corpus.data_df["train"], corpus.n_items)
    directed = graphs["directed"]
    evidence = np.zeros(len(histories), dtype=bool)
    for items in histories.T:
        evidence |= np.asarray(directed[items, targets]).ravel() > 0
    evidence_groups = {
        "target_has_training_transition": evaluation_rows & evidence,
        "target_has_no_training_transition": evaluation_rows & ~evidence,
    }

    def by_evidence(metrics):
        return {
            label: {"queries": int(group.sum()), **summarize(metrics, group)}
            for label, group in evidence_groups.items() if group.any()
        }

    methods = {
        "rpg": {
            "weight": 0.0,
            evaluation_key: summarize(base, evaluation_rows),
            f"{evaluation_key}_by_target_evidence": by_evidence(base),
        }
    }
    if calibration is not None:
        methods["rpg"]["calibration"] = summarize(base, calibration)
    score_z = row_standardize(scores.astype(np.float64), valid)
    popularity_ndcg = None
    rerank_methods = ("popularity", "directed") if phase == "test" else ("popularity", *graphs)
    for name in rerank_methods:
        if name == "popularity":
            secondary = np.log(np.where(valid, prior[candidates], 1))
        else:
            secondary = behavior_scores(graphs[name], prior, histories, candidates)
        secondary_z = row_standardize(secondary, valid)
        trials = {}
        weights = WEIGHTS if locked is None else (locked["methods"][name]["weight"],)
        for weight in weights:
            fused = np.where(valid, score_z + weight * secondary_z, -np.inf)
            indices = np.argsort(-fused, axis=1, kind="stable")
            if weight == 0 and not np.array_equal(indices, np.broadcast_to(np.arange(TOP_K), indices.shape)):
                raise RuntimeError("Zero fusion weight changed the RPG ordering")
            ranked = np.take_along_axis(candidates, indices, axis=1)
            trials[weight] = rank_metrics(target_rank(ranked, targets))
        sweep = {}
        weight = weights[0]
        if calibration is not None:
            sweep = {candidate: summarize(metrics, calibration) for candidate, metrics in trials.items()}
            weight = max(weights, key=lambda value: sweep[value]["NDCG@10"])
        selected = trials[weight]
        result = summarize(selected, evaluation_rows)
        ndcg = selected["NDCG@10"][evaluation_rows]
        diff = ndcg - base["NDCG@10"][evaluation_rows]
        methods[name] = {
            "weight": weight,
            evaluation_key: result,
            f"{evaluation_key}_delta_ndcg_at_10": float(diff.mean()),
            f"{evaluation_key}_delta_ndcg_at_10_user_bootstrap_95ci": bootstrap_delta(
                diff, users[evaluation_rows], SPLIT_SEED
            ),
            f"{evaluation_key}_by_target_evidence": by_evidence(selected),
        }
        if sweep:
            methods[name].update(calibration=sweep[weight], calibration_sweep=sweep)
        if name == "popularity":
            popularity_ndcg = ndcg
        elif locked is not None:
            delta_pop = ndcg - popularity_ndcg
            methods[name][f"{evaluation_key}_delta_vs_popularity"] = float(delta_pop.mean())
            methods[name][f"{evaluation_key}_delta_vs_popularity_user_bootstrap_95ci"] = bootstrap_delta(
                delta_pop, users[evaluation_rows], SPLIT_SEED
            )
        print(f"{dataset}: {name} weight={weight:g}, {evaluation_key} NDCG@10={result['NDCG@10']:.6f}", flush=True)

    result = {
        "checkpoint": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": digest(checkpoint),
        "sid_sha256": sid_hash,
        "dataset_hash": corpus.dataset_hash,
        "instance_hash": row_hash,
        "examples": len(frame),
        "target_training_transition_coverage": float(evidence.mean()),
        f"rpg_full_{phase}": baseline,
        "training_graph": graph_stats,
        "methods": methods,
    }
    if calibration is not None:
        result.update(
            calibration_examples=int(calibration.sum()),
            holdout_examples=int(evaluation_rows.sum()),
            calibration_users=len(unique_users) // 2,
            holdout_users=len(unique_users) - len(unique_users) // 2,
        )
    else:
        result["users"] = len(unique_users)
    return result


def summarize_test(per_seed):
    runs = list(per_seed.values())
    first = runs[0]
    for run in runs[1:]:
        for key in ("dataset_hash", "instance_hash", "sid_sha256", "examples", "users"):
            if run[key] != first[key]:
                raise ValueError(f"Test seeds disagree on {key}")
    summary = {}
    for method, first_method in first["methods"].items():
        rows = [run["methods"][method] for run in runs]
        if any(row["weight"] != first_method["weight"] for row in rows):
            raise ValueError(f"Test seeds disagree on {method} weight")
        metrics = {}
        for key in first_method["test"]:
            values = [row["test"][key] for row in rows]
            metrics[key] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
        summary[method] = {"weight": first_method["weight"], "metrics": metrics}
    return {"examples": first["examples"], "users": first["users"], "summary": summary, "per_seed": per_seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    parser.add_argument("--seed", type=int, nargs="+", help="Checkpoint seeds, default 2026 on dev and 2026–2028 on test")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--locked-pilot", type=Path, help="Reuse weights from a previous validation pilot")
    inputs.add_argument("--config", type=Path, help=f"Frozen test settings, default {FROZEN_CONFIG}")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.config is not None and args.phase != "test":
        parser.error("--config is available for test evaluation")
    config_path = args.config
    if args.phase == "test" and args.locked_pilot is None and config_path is None:
        config_path = root / FROZEN_CONFIG
    frozen = read_json(config_path) if config_path is not None else None
    if frozen is not None and frozen["schema_version"] != 1:
        raise ValueError("Unsupported frozen configuration version")
    default_test_seeds = frozen["checkpoint_seeds"] if frozen is not None else [2026, 2027, 2028]
    seeds = args.seed or (default_test_seeds if args.phase == "test" else [2026])
    if frozen is not None and any(seed not in default_test_seeds for seed in seeds):
        parser.error("Requested seed is absent from the frozen configuration")
    if len(set(seeds)) != len(seeds):
        parser.error("Checkpoint seeds must be unique")
    if args.phase == "dev" and len(seeds) != 1:
        parser.error("Dev evaluation accepts one checkpoint seed")
    if args.output is None:
        suffix = "test" if args.phase == "test" else "pilot"
        args.output = Path(f"results/rpg_behavior_rerank_{suffix}.json")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not torch.cuda.is_available():
        parser.error("A CUDA device is required for this pilot")
    sys.path.insert(0, str(root / "src"))
    from helpers.FrozenSeqReader import FrozenSeqReader
    from models.sequential.RPG import RPG
    from utils.unified_topk import blocked_items, mask_history, rank_items

    report = {
        "phase": args.phase,
        "split_seed": SPLIT_SEED,
        "device": torch.cuda.get_device_name(),
        "top_k": TOP_K,
        "behavior_history": RECENT,
        "history_decay": 0.8,
        "prior_mass": PRIOR_MASS,
        "reliability_mass": RELIABILITY_MASS,
        "candidate_normalization": "Per-query z-score over eligible Top-100 candidates",
        "fusion": "standardized RPG score + weight * standardized behavior score",
        "behavior": "Weighted log predictive lift above training popularity, shrunk by pair count",
        "time_aware": "Train adjacent pairs are forward for strictly increasing time and half-weight in both directions for equal time",
        "datasets": {},
    }
    pilot = read_json(args.locked_pilot) if args.locked_pilot else None
    if pilot is not None:
        if pilot["phase"] != "dev":
            raise ValueError("Frozen weights must come from the validation phase")
        for key in ("split_seed", "top_k", "behavior_history", "history_decay",
                    "prior_mass", "reliability_mass", "candidate_normalization", "fusion", "time_aware"):
            if report[key] != pilot[key]:
                raise ValueError(f"Frozen pilot setting differs: {key}")
        report["locked_pilot_sha256"] = json_digest(args.locked_pilot)
    if frozen is not None:
        for key, value in frozen["protocol"].items():
            if report[key] != value:
                raise ValueError(f"Frozen test setting differs: {key}")
        report["locked_pilot_sha256"] = frozen["validation_pilot_sha256"]
        report["frozen_config_sha256"] = json_digest(config_path)
    if args.phase == "dev":
        report["checkpoint_seed"] = seeds[0]
    else:
        report["checkpoint_seeds"] = seeds
    ranking_api = blocked_items, mask_history, rank_items
    for dataset in args.dataset:
        locked_settings = frozen if frozen is not None else pilot
        locked = None if locked_settings is None else locked_settings["datasets"][dataset]
        per_seed = {
            str(seed): run_dataset(
                dataset, root, seed, FrozenSeqReader, RPG, ranking_api, locked=locked, phase=args.phase,
                frozen=frozen,
            )
            for seed in seeds
        }
        report["datasets"][dataset] = next(iter(per_seed.values())) if args.phase == "dev" else summarize_test(per_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
