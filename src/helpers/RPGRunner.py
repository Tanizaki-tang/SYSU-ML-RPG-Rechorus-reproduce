import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from helpers.BaseRunner import BaseRunner
from utils import utils


class RPGRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser = BaseRunner.parse_runner_args(parser)
        parser.add_argument(
            "--decode", choices=["full", "graph"], default="full"
        )
        parser.add_argument("--graph_edges", type=int, default=50)
        parser.add_argument("--graph_beams", type=int, default=50)
        parser.add_argument("--graph_steps", type=int, default=3)
        return parser

    def __init__(self, args):
        super().__init__(args)
        self.decode = args.decode
        self.graph_edges = args.graph_edges
        self.graph_beams = args.graph_beams
        self.graph_steps = args.graph_steps
        self.seed = args.random_seed
        self.log_file = args.log_file

        if args.save_final_results:
            raise ValueError(
                "请设置 --save_final_results 0；"
                "RPGRunner 会自行保存指标 JSON"
            )
        if self.decode == "graph" and args.train:
            raise ValueError(
                "训练请使用 --decode full；"
                "最佳模型训练完成后用 --train 0 --load 1 --decode graph"
            )

    @staticmethod
    def synchronize(model):
        if model.codes.is_cuda:
            torch.cuda.synchronize(model.codes.device)

    @torch.no_grad()
    def evaluate(self, dataset, topks, metrics):
        model = dataset.model
        model.eval()

        max_k = max(topks)
        if model.item_num - 1 < max_k:
            raise ValueError("物品数小于请求的 topk")

        # 只排除推荐时已经发生的历史交互。
        # 这里使用完整历史，而不是模型截断后的最近 history_max 条。
        excluded = []
        for user, position in zip(
            dataset.data["user_id"], dataset.data["position"]
        ):
            history = dataset.corpus.user_his[int(user)][:int(position)]
            excluded.append(list({int(item) for item, _ in history}))

        build_seconds = 0.0
        if self.decode == "graph":
            self.synchronize(model)
            build_start = time.perf_counter()
            model.build_graph(self.graph_edges)
            self.synchronize(model)
            build_seconds = time.perf_counter() - build_start

        generator = torch.Generator(device=model.codes.device)
        generator.manual_seed(self.seed)

        loader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=dataset.collate_batch,
        )

        ranks = []
        visited_counts = []
        offset = 0

        self.synchronize(model)
        start = time.perf_counter()

        for batch in loader:
            batch = utils.batch_to_gpu(batch, model.device)
            log_probs = model.token_logits(batch).log_softmax(dim=-1)
            size = len(log_probs)
            blocked = excluded[offset:offset + size]

            if self.decode == "graph":
                predictions, visited = model.graph_topk(
                    log_probs,
                    blocked,
                    max_k,
                    beams=self.graph_beams,
                    steps=self.graph_steps,
                    generator=generator,
                )
                visited_counts.extend(visited)
            else:
                candidates = torch.arange(
                    1, model.item_num, device=model.codes.device
                )
                scores = model.score_items(log_probs, candidates)
                for row, history in enumerate(blocked):
                    if history:
                        indices = torch.tensor(
                            history, device=scores.device
                        ) - 1
                        scores[row, indices] = -torch.inf

                values, indices = scores.topk(max_k, dim=1)
                predictions = indices + 1
                predictions = predictions.masked_fill(
                    ~torch.isfinite(values), 0
                )

            targets = batch["target_item"]
            matches = predictions.eq(targets[:, None])
            found = matches.any(dim=1)
            rank = matches.long().argmax(dim=1) + 1
            rank = torch.where(
                found, rank, torch.full_like(rank, max_k + 1)
            )
            ranks.extend(rank.cpu().tolist())
            offset += size

        self.synchronize(model)
        elapsed = time.perf_counter() - start

        ranks = np.asarray(ranks)
        if len(ranks) == 0:
            raise ValueError("评估集为空")

        result = {}
        for k in topks:
            hits = ranks <= k
            for metric in metrics:
                if metric == "HR":
                    value = hits.mean()
                elif metric == "NDCG":
                    value = (hits / np.log2(ranks + 1)).mean()
                else:
                    raise ValueError(f"不支持的指标 {metric}")
                result[f"{metric}@{k}"] = float(value)

        record = {
            "phase": dataset.phase,
            "decode": self.decode,
            "metrics": result,
            "examples": len(ranks),
            "evaluation_seconds": elapsed,
            "graph_build_seconds": build_seconds,
            "graph_edges": self.graph_edges,
            "graph_beams": self.graph_beams,
            "graph_steps": self.graph_steps,
            "seed": self.seed,
            "candidate_policy": "all items excluding prior history",
        }
        if visited_counts:
            record["mean_visited_items"] = float(
                np.mean(visited_counts)
            )

        output = Path(self.log_file).with_suffix(
            f".{dataset.phase}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logging.info(
            "Evaluation %s/%s: %.2f seconds; graph build %.2f seconds",
            dataset.phase, self.decode, elapsed, build_seconds,
        )
        return result