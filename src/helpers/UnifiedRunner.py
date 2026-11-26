import hashlib
import json
import logging
import time
from itertools import islice
from pathlib import Path

import numpy as np
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
        # 多时间尺度历史集成（test-time augmentation）。
        parser.add_argument(
            "--history_tta",
            choices=["none", "crops", "relative"], default="none",
        )
        parser.add_argument(
            "--crop_lengths", default="50,35,20,10",
            help="逗号分隔的截断长度，仅在 --history_tta crops 时生效",
        )
        parser.add_argument(
            "--crop_ratios", default="1.0,0.6,0.3",
            help="逗号分隔的历史比例，仅在 --history_tta relative 时生效",
        )
        parser.add_argument(
            "--tta_aggregate", choices=["log", "prob"], default="log"
        )
        parser.add_argument(
            "--tta_include_full", type=int, choices=[0, 1], default=1,
            help="是否在 crop_lengths 之外额外加入完整的 history_max 视图",
        )
        parser.add_argument(
            "--digit_norm", choices=["none", "zscore"], default="none",
            help="对每个数字位跨码本的分布做标准化后再求和",
        )
        parser.add_argument(
            "--digit_weight", choices=["none", "confidence"], default="none",
            help="按数字位的预测置信度对其贡献加权",
        )
        parser.add_argument(
            "--mcd_samples", type=int, default=0,
            help="推理期 dropout 的随机前向次数；0 表示关闭",
        )
        parser.add_argument(
            "--tta_min_history", type=int, default=0,
            help="历史长度达到该值才做多视图/随机集成；0 表示不过滤",
        )
        parser.add_argument(
            "--logprob_scale", type=float, default=1.0,
            help="对聚合后的对数概率整体缩放（等价于后验温度缩放）",
        )
        parser.add_argument(
            "--pop_beta", type=float, default=0.0,
            help="训练集物品流行度先验的权重；0 表示关闭",
        )
        parser.add_argument(
            "--pop_length_tau", type=float, default=0.0,
            help="流行度权重按 exp(-history_len/tau) 衰减；0 表示不衰减",
        )
        parser.add_argument(
            "--rerank_pop_weight", type=float, default=0.0,
            help="在 top-N 候选内用 z-score 融合流行度；0 表示关闭",
        )
        parser.add_argument(
            "--rerank_topk", type=int, default=100,
            help="z-score 流行度重排的候选池大小",
        )
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

        self.crop_lengths = []
        self.crop_ratios = []
        if args.history_tta == "crops":
            try:
                self.crop_lengths = sorted(
                    {int(x) for x in str(args.crop_lengths).split(",") if x}
                )
            except ValueError as error:
                raise ValueError(
                    f"无法解析 crop_lengths: {args.crop_lengths}"
                ) from error
            if not self.crop_lengths or min(self.crop_lengths) < 1:
                raise ValueError("crop_lengths 必须为正整数")
            self.crop_lengths = [
                min(length, args.history_max) for length in self.crop_lengths
            ]
        elif args.history_tta == "relative":
            try:
                self.crop_ratios = sorted(
                    {float(x) for x in str(args.crop_ratios).split(",") if x},
                    reverse=True,
                )
            except ValueError as error:
                raise ValueError(
                    f"无法解析 crop_ratios: {args.crop_ratios}"
                ) from error
            if not self.crop_ratios or not all(
                0 < ratio <= 1 for ratio in self.crop_ratios
            ):
                raise ValueError("crop_ratios 必须位于 (0, 1]")
        if args.mcd_samples < 0:
            raise ValueError("mcd_samples 必须非负")
        if args.tta_min_history < 0:
            raise ValueError("tta_min_history 必须非负")
        if args.logprob_scale <= 0:
            raise ValueError("logprob_scale 必须大于 0")
        if args.pop_length_tau < 0:
            raise ValueError("pop_length_tau 必须非负")
        if args.rerank_pop_weight < 0:
            raise ValueError("rerank_pop_weight 必须非负")
        if args.rerank_topk < 1:
            raise ValueError("rerank_topk 必须大于 0")

    def _build_views(self, histories):
        if self.args.history_tta == "none":
            return [list(histories)]
        if self.args.history_tta == "relative":
            views = []
            for ratio in self.crop_ratios:
                crops = []
                for history in histories:
                    length = max(
                        1, min(
                            len(history),
                            int(round(ratio * len(history))),
                        )
                    )
                    crops.append(history[-length:])
                views.append(crops)
            return views

        if self.args.tta_include_full:
            limits = sorted(
                set(self.crop_lengths + [self.args.history_max])
            )
        else:
            limits = sorted(set(self.crop_lengths))
        return [
            [h[-limit:] if limit > 0 else h for h in histories]
            for limit in limits
        ]

    @torch.no_grad()
    def _aggregate_views(
        self, model, histories, device, views, allow_dropout
    ):
        """对给定视图集合做前向并聚合，返回逐位对数概率与前向次数。"""
        # mcd_samples > 1 时打开推理期 dropout，做多次随机前向。
        passes = (
            self.args.mcd_samples
            if allow_dropout and self.args.mcd_samples > 1 else 1
        )
        was_training = model.training
        if passes > 1:
            model.train()

        accumulated = None
        seen = set()
        used = 0
        try:
            for crops in views:
                lengths = torch.tensor(
                    [len(crop) for crop in crops], dtype=torch.long
                )
                key = tuple(lengths.tolist())
                if key in seen:
                    continue
                seen.add(key)

                feed = {
                    "history_items": pad_sequence(
                        crops, batch_first=True
                    ).to(device),
                    "lengths": lengths.to(device),
                }
                for _ in range(passes):
                    log_probs = model.token_logits(
                        feed
                    ).log_softmax(dim=-1)
                    if self.args.tta_aggregate == "prob":
                        log_probs = log_probs.exp()
                    accumulated = (
                        log_probs if accumulated is None
                        else accumulated + log_probs
                    )
                    used += 1
        finally:
            if passes > 1:
                model.train(was_training)

        accumulated = accumulated / used
        if self.args.tta_aggregate == "prob":
            accumulated = accumulated.clamp_min(1e-12).log()
        return accumulated, used

    @torch.no_grad()
    def _history_log_probs(self, model, histories, device):
        """按历史长度门控：长历史做多视图/随机集成，短历史单次前向。"""
        count = len(histories)
        min_len = self.args.tta_min_history
        if min_len > 0:
            long_index = [
                i for i, h in enumerate(histories) if len(h) >= min_len
            ]
            short_index = [
                i for i, h in enumerate(histories) if len(h) < min_len
            ]
        else:
            long_index = list(range(count))
            short_index = []

        if not short_index:
            accumulated, used = self._aggregate_views(
                model, histories, device,
                self._build_views(histories), True,
            )
        elif not long_index:
            accumulated, used = self._aggregate_views(
                model, histories, device, [list(histories)], False,
            )
        else:
            accumulated = torch.zeros(
                count, model.digits, model.codebook_size,
                dtype=torch.float32, device=device,
            )
            long_histories = [histories[i] for i in long_index]
            result, used = self._aggregate_views(
                model, long_histories, device,
                self._build_views(long_histories), True,
            )
            accumulated[long_index] = result
            short_histories = [histories[i] for i in short_index]
            result, _ = self._aggregate_views(
                model, short_histories, device,
                [list(short_histories)], False,
            )
            accumulated[short_index] = result

        if self.args.digit_norm == "zscore":
            centered = accumulated - accumulated.mean(
                dim=-1, keepdim=True
            )
            scale = accumulated.std(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            accumulated = centered / scale

        if self.args.digit_weight == "confidence":
            # 对每个样本、每个数字位，用该位最大概率作为可信度，
            # 权重均值归一为 1，避免改变整体尺度。
            confidence = accumulated.exp().max(dim=-1).values
            weight = confidence / confidence.mean(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            accumulated = accumulated * weight.unsqueeze(-1)
        accumulated = accumulated * self.args.logprob_scale
        self.last_views_used = used
        return accumulated

    @staticmethod
    def _row_standardize(values, valid):
        """在每行的有效候选内做 z-score；无效位置返回 0。"""
        values = values.masked_fill(~valid, 0.0)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
        mean = values.sum(dim=1, keepdim=True) / count
        centered = (values - mean) * valid
        variance = (centered * centered).sum(
            dim=1, keepdim=True
        ) / count
        return centered / variance.sqrt().clamp_min(1e-12)

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
        if self.args.rerank_pop_weight and self.args.decode != "full":
            raise ValueError("rerank_pop_weight 目前只支持 full 解码")

        # 总是输出完整的 5/10 指标，保证训练验证和最终评估一致。
        ks = sorted(set(topks) | {5, 10})
        max_k = max(ks)
        if corpus.n_items - 1 < max_k:
            raise ValueError("真实物品数小于 topk")
        if self.args.rerank_pop_weight and self.args.rerank_topk < max_k:
            raise ValueError("rerank_topk 不能小于最大 topk")

        device = next(model.parameters()).device

        item_prior = None
        item_pop = None
        if self.args.pop_beta or self.args.rerank_pop_weight:
            counts = np.zeros(corpus.n_items, dtype=np.float64)
            values = corpus.data_df["train"]["item_id"].value_counts()
            counts[values.index.to_numpy()] = values.to_numpy()
            pop = np.log1p(counts)
            if self.args.pop_beta:
                body = pop[1:]
                standardized = (pop - body.mean()) / (body.std() + 1e-12)
                item_prior = torch.tensor(
                    standardized, dtype=torch.float32, device=device
                )
            if self.args.rerank_pop_weight:
                item_pop = torch.tensor(
                    pop, dtype=torch.float32, device=device
                )

        graph_build_seconds = 0.0
        if self.args.decode == "graph":
            # 每次评估在当前权重上建图，避免载入 checkpoint 后使用旧图。
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            build_start = time.perf_counter()
            model.build_graph(self.args.graph_edges)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            graph_build_seconds = time.perf_counter() - build_start

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eval_start = time.perf_counter()

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
                log_probs = self._history_log_probs(
                    model, histories, device
                )

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
                pool_k = (
                    max(max_k, self.args.rerank_topk)
                    if self.args.rerank_pop_weight else max_k
                )
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
                        if item_prior is not None:
                            beta = torch.full(
                                (batch_size,), self.args.pop_beta,
                                device=device,
                            )
                            if self.args.pop_length_tau > 0:
                                beta = beta * torch.exp(
                                    -feed["lengths"].float()
                                    / self.args.pop_length_tau
                                )
                            scores = scores + (
                                beta[:, None] * item_prior[ids][None, :]
                            )
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
                        pool_k,
                    )

                if self.args.rerank_pop_weight:
                    # 只在 top-N 候选内融合：z(RPG) + λ z(流行度)。
                    valid = best_ids.gt(0) & torch.isfinite(best_scores)
                    z_score = self._row_standardize(
                        best_scores, valid
                    )
                    z_pop = self._row_standardize(
                        item_pop[best_ids], valid
                    )
                    fused = (
                        z_score + self.args.rerank_pop_weight * z_pop
                    ).masked_fill(~valid, -torch.inf)
                    order = fused.argsort(
                        dim=1, descending=True
                    )[:, :max_k]
                    predictions = best_ids.gather(1, order)
                    predictions = predictions.masked_fill(
                        ~torch.isfinite(fused.gather(1, order)), 0
                    )
                else:
                    predictions = best_ids

            values = metric_values(predictions, targets, ks)
            for name, value in values.items():
                totals[name] += value.sum().item()
            instance_ids.extend(int(x) for x in rows["row_id"])

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_seconds = time.perf_counter() - eval_start

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
            "history_tta": self.args.history_tta,
            "crop_lengths": (
                self.crop_lengths
                if self.args.history_tta == "crops" else []
            ),
            "crop_ratios": (
                self.crop_ratios
                if self.args.history_tta == "relative" else []
            ),
            "tta_aggregate": self.args.tta_aggregate,
            "digit_norm": self.args.digit_norm,
            "digit_weight": self.args.digit_weight,
            "mcd_samples": self.args.mcd_samples,
            "tta_min_history": self.args.tta_min_history,
            "tta_views_used": getattr(self, "last_views_used", 0),
            "logprob_scale": self.args.logprob_scale,
            "pop_beta": self.args.pop_beta,
            "pop_length_tau": self.args.pop_length_tau,
            "rerank_pop_weight": self.args.rerank_pop_weight,
            "rerank_topk": self.args.rerank_topk,
            "metrics": result,
            "examples": len(frame),
            "evaluation_seconds": evaluation_seconds,
            "graph_build_seconds": graph_build_seconds,
            "device": str(device),
            "torch_version": torch.__version__,
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
