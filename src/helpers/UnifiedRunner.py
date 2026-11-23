import hashlib
import json
import logging
from itertools import islice
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from helpers.BaseRunner import BaseRunner
from utils import utils
from utils.unified_topk import (
    blocked_items,
    graph_topk_one,
    mask_history,
    metric_values,
    rank_items,
)


class UnifiedRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser = BaseRunner.parse_runner_args(parser)
        parser.add_argument(
            "--decode", choices=["full", "graph"], default="full"
        )
        parser.add_argument("--item_chunk", type=int, default=512)
        parser.add_argument("--eval_limit", type=int, default=0)
        parser.add_argument("--graph_edges", type=int, default=50)
        parser.add_argument("--graph_beams", type=int, default=10)
        parser.add_argument("--graph_steps", type=int, default=2)
        parser.add_argument("--decode_seed", type=int, default=2026)
        return parser

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        if args.item_chunk < 1 or args.eval_batch_size < 1:
            raise ValueError("item_chunk 和 eval_batch_size 必须大于 0")
        if args.eval_limit < 0:
            raise ValueError("eval_limit 必须非负")
        if args.history_max < 1:
            raise ValueError("history_max 必须大于 0")
        if args.train and args.decode != "full":
            raise ValueError("训练期间使用 full 评估选择 checkpoint")

    def eval_termination(self, criterion):
        # 连续 early_stop 个 epoch 没有严格改善时停止。
        best_index = max(
            range(len(criterion)), key=criterion.__getitem__
        )
        return len(criterion) - 1 - best_index >= self.early_stop

    @torch.no_grad()
    def evaluate(self, dataset, topks, metrics):
        model = dataset.model
        corpus = dataset.corpus
        phase = dataset.phase
        model.eval()

        frame = corpus.data_df[phase]
        frame = frame.loc[frame["position"] > 0].copy()
        if self.args.eval_limit:
            frame = frame.head(self.args.eval_limit)
        frame = frame.reset_index(drop=True)

        if frame.empty:
            raise ValueError(f"{phase} 没有可评估样本")

        is_rpg = callable(getattr(model, "token_logits", None))
        if self.args.decode == "graph" and not is_rpg:
            raise ValueError("graph 模式只用于 RPG")
        if is_rpg and not callable(getattr(model, "score_items", None)):
            raise ValueError("当前 RPG 缺少 score_items() 接口")

        # 总是输出完整的 5/10 指标，保证训练验证和最终评估一致。
        ks = sorted(set(topks) | {5, 10})
        max_k = max(ks)
        if corpus.n_items - 1 < max_k:
            raise ValueError("真实物品数小于 topk")

        device = next(model.parameters()).device

        if self.args.decode == "graph":
            # 每次评估在当前权重上建图，避免载入 checkpoint 后使用旧图。
            model.build_graph(self.args.graph_edges)

        totals = {
            f"{metric}@{k}": 0.0
            for k in ks for metric in ("HR", "NDCG")
        }
        visited = []
        instance_ids = []

        for start in range(0, len(frame), self.eval_batch_size):
            rows = frame.iloc[start:start + self.eval_batch_size]
            histories = []
            history_times = []
            forbidden = []

            for row in rows.itertuples(index=False):
                complete_history = corpus.user_his[
                    int(row.user_id)
                ][:int(row.position)]

                forbidden.append(blocked_items(
                    [item for item, _ in complete_history],
                    row.item_id,
                ))

                recent = complete_history[-model.history_max:]
                histories.append(torch.tensor(
                    [item for item, _ in recent], dtype=torch.long
                ))
                history_times.append(torch.tensor(
                    [timestamp for _, timestamp in recent],
                    dtype=torch.long,
                ))

            batch_size = len(rows)
            targets = torch.tensor(
                rows["item_id"].tolist(),
                dtype=torch.long, device=device,
            )
            feed = {
                "user_id": torch.tensor(
                    rows["user_id"].tolist(),
                    dtype=torch.long, device=device,
                ),
                "history_items": pad_sequence(
                    histories, batch_first=True
                ).to(device),
                "history_times": pad_sequence(
                    history_times, batch_first=True
                ).to(device),
                "lengths": torch.tensor(
                    [len(history) for history in histories],
                    dtype=torch.long, device=device,
                ),
                "batch_size": batch_size,
                "phase": phase,
            }

            log_probs = None
            if is_rpg:
                log_probs = model.token_logits(feed).log_softmax(dim=-1)

            if self.args.decode == "graph":
                predictions = []
                for index, row_id in enumerate(rows["row_id"]):
                    # 由样本 ID 决定随机种子，改变评估 batch size
                    # 不会改变该样本的初始搜索节点。
                    seed = (
                        self.args.decode_seed + int(row_id)
                    ) % (2 ** 63 - 1)

                    result, count = graph_topk_one(
                        model,
                        log_probs[index:index + 1],
                        forbidden[index],
                        max_k,
                        self.args.graph_beams,
                        self.args.graph_steps,
                        seed,
                    )
                    predictions.append(result)
                    visited.append(count)
                predictions = torch.stack(predictions)

            else:
                best_ids = torch.empty(
                    batch_size, 0, dtype=torch.long, device=device
                )
                best_scores = torch.empty(
                    batch_size, 0, device=device
                )

                for first in range(
                    1, corpus.n_items, self.args.item_chunk
                ):
                    ids = torch.arange(
                        first,
                        min(first + self.args.item_chunk, corpus.n_items),
                        device=device,
                    )

                    if is_rpg:
                        scores = model.score_items(log_probs, ids)
                    else:
                        candidate_feed = dict(feed)
                        candidate_feed["item_id"] = ids.unsqueeze(0).expand(
                            batch_size, -1
                        )
                        scores = model(candidate_feed)["prediction"]

                    if not torch.isfinite(scores).all():
                        raise ValueError("模型输出包含 NaN 或 Inf")

                    scores = mask_history(ids, scores, forbidden)
                    expanded_ids = ids.unsqueeze(0).expand(batch_size, -1)
                    best_ids, best_scores = rank_items(
                        torch.cat([best_ids, expanded_ids], dim=1),
                        torch.cat([best_scores, scores], dim=1),
                        max_k,
                    )

                predictions = best_ids

            values = metric_values(predictions, targets, ks)
            for name, value in values.items():
                totals[name] += value.sum().item()
            instance_ids.extend(int(x) for x in rows["row_id"])

        result = {
            name: value / len(frame) for name, value in totals.items()
        }
        row_hash = hashlib.sha256(
            json.dumps(instance_ids).encode("utf-8")
        ).hexdigest()

        record = {
            "model": type(model).__name__,
            "dataset": corpus.dataset,
            "dataset_hash": corpus.dataset_hash,
            "phase": phase,
            "decode": self.args.decode,
            "metrics": result,
            "examples": len(frame),
            "instance_hash": row_hash,
            "smoke_test": bool(
                self.args.eval_limit or getattr(self.args, "max_train_batches", 0)
            ),
            "arguments": vars(self.args),
        }
        if visited:
            record["mean_visited_items"] = sum(visited) / len(visited)

        path = Path(self.args.log_file).with_suffix(
            f".{phase}.{self.args.decode}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                record, ensure_ascii=False, indent=2, default=str
            ) + "\n",
            encoding="utf-8",
        )
        return result


class RPGTrainingRunner(UnifiedRunner):
    @staticmethod
    def parse_training_args(parser):
        parser.add_argument("--warmup_steps", type=int, default=10000)
        parser.add_argument("--max_train_batches", type=int, default=0)
        parser.set_defaults(
            epoch=150, early_stop=20, optimizer="AdamW",
            main_metric="NDCG@10", topk="5,10",
            num_workers=0, eval_batch_size=32,
        )
        return parser

    def train(self, data_dict):
        from transformers.optimization import get_cosine_schedule_with_warmup

        args = self.args
        model = data_dict["dev"].model
        dataset = data_dict["train"]
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=dataset.collate_batch,
            pin_memory=bool(self.pin_memory),
        )
        batches = (
            min(len(loader), args.max_train_batches)
            if args.max_train_batches else len(loader)
        )
        if batches == 0:
            raise ValueError("RPG requires at least one training window")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.l2
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=self.epoch * batches,
        )
        logging.info(
            "RPG train: windows=%d targets=%d batches/epoch=%d device=%s "
            "optimizer=AdamW warmup=%d grad_clip=1.0",
            len(dataset), dataset.target_count, batches,
            model.device, args.warmup_steps,
        )

        best = -float("inf")
        best_epoch = 0
        for epoch in range(1, self.epoch + 1):
            model.train()
            loss_sum = 0.0
            for batch in islice(loader, batches):
                batch = utils.batch_to_gpu(batch, model.device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.training_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                loss_sum += loss.item()

            metrics = self.evaluate(data_dict["dev"], self.topk, self.metrics)
            score = metrics[self.main_metric]
            improved = score > best
            if improved:
                best = score
                best_epoch = epoch
                self._save_best(model)
            logging.info(
                "Epoch %d loss=%.6f lr=%.6g dev=(%s)%s",
                epoch, loss_sum / batches, scheduler.get_last_lr()[0],
                utils.format_metric(metrics), " *" if improved else "",
            )
            if self.early_stop > 0 and epoch - best_epoch >= self.early_stop:
                break

        logging.info("Best Iter(dev)=%d %s=%.8f", best_epoch, self.main_metric, best)
        model.load_model()

    @staticmethod
    def _save_best(model):
        path = Path(model.model_path)
        temporary = path.with_name(path.name + ".tmp")
        try:
            model.save_model(str(temporary))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
