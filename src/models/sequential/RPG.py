from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from models.BaseModel import SequentialModel


class RPG(SequentialModel):
    reader = "SeqReader"
    runner = "RPGRunner"
    extra_log_args = ["emb_size", "temperature"]

    @staticmethod
    def parse_model_args(parser):
        parser = SequentialModel.parse_model_args(parser)
        parser.add_argument("--sid_file", default="semantic_ids.npy")
        parser.add_argument("--emb_size", type=int, default=128)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--temperature", type=float, default=0.07)
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        if self.history_max <= 0:
            raise ValueError("RPG 要求 history_max > 0")
        if args.emb_size % args.n_heads:
            raise ValueError("emb_size 必须能被 n_heads 整除")
        if args.temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        sid_path = Path(args.path) / args.dataset / args.sid_file
        codes = np.load(sid_path, allow_pickle=False)

        if codes.ndim != 2 or codes.shape[0] != self.item_num:
            raise ValueError(
                f"语义 ID 行数应等于 corpus.n_items={self.item_num}，"
                f"实际形状为 {codes.shape}"
            )
        if codes.min() < 0 or codes.max() >= 256:
            raise ValueError("当前实现要求每位 token 编码在 0..255")

        self.digits = codes.shape[1]
        self.codebook_size = 256
        self.temperature = args.temperature

        self.register_buffer(
            "codes", torch.tensor(codes, dtype=torch.long)
        )
        offsets = torch.arange(self.digits) * self.codebook_size
        self.register_buffer("offsets", offsets)

        config = GPT2Config(
            vocab_size=self.digits * self.codebook_size + 1,
            n_positions=self.history_max,
            n_embd=args.emb_size,
            n_layer=args.n_layers,
            n_head=args.n_heads,
            n_inner=args.emb_size * 4,
            activation_function="gelu_new",
            resid_pdrop=args.dropout,
            embd_pdrop=args.dropout,
            attn_pdrop=args.dropout,
            use_cache=False,
        )
        self.backbone = GPT2Model(config)

        # 每个预测头实现 h + SiLU(W_j h + b_j)。
        self.heads = nn.ModuleList(
            nn.Linear(args.emb_size, args.emb_size)
            for _ in range(self.digits)
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.adjacency = None

    def token_logits(self, feed):
        history = feed["history_items"].long()
        lengths = feed["lengths"].long()
        valid = history.ne(0)

        # 每个码本使用独立的 token 区间，0 为 padding。
        tokens = self.codes[history] + self.offsets + 1
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        embeddings = self.backbone.wte(tokens).mean(dim=-2)
        embeddings = embeddings * valid.unsqueeze(-1)

        hidden = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=valid.long(),
            use_cache=False,
        ).last_hidden_state
        batch_rows = torch.arange(len(history), device=history.device)
        state = hidden[batch_rows, lengths - 1]

        projected = torch.stack(
            [state + F.silu(head(state)) for head in self.heads],
            dim=1,
        )
        projected = F.normalize(projected, dim=-1)

        token_embeddings = self.backbone.wte.weight[1:].reshape(
            self.digits, self.codebook_size, -1
        )
        token_embeddings = F.normalize(token_embeddings, dim=-1)

        # [batch, digits, 256]
        return torch.einsum(
            "bmd,mcd->bmc", projected, token_embeddings
        ) / self.temperature

    def score_items(self, log_probs, item_ids):
        """item_ids 可以为 [candidates] 或 [batch, candidates]。"""
        if item_ids.ndim == 1:
            item_ids = item_ids.unsqueeze(0).expand(
                log_probs.shape[0], -1
            )
        item_codes = self.codes[item_ids]
        scores = log_probs.new_zeros(item_ids.shape)
        for digit in range(self.digits):
            scores = scores + log_probs[:, digit].gather(
                1, item_codes[:, :, digit]
            )
        return scores

    def forward(self, feed):
        logits = self.token_logits(feed)
        log_probs = logits.log_softmax(dim=-1)
        return {
            "prediction": self.score_items(log_probs, feed["item_id"]),
            "token_logits": logits,
            "target_codes": self.codes[feed["target_item"]],
        }

    def loss(self, output):
        return F.cross_entropy(
            output["token_logits"].reshape(-1, self.codebook_size),
            output["target_codes"].reshape(-1),
        )

    @torch.no_grad()
    def build_graph(self, edges=50, chunk_size=256):
        """依据训练后的 token embedding 相似度建立物品邻接表。"""
        self.eval()
        count = self.item_num - 1
        edges = min(edges, count)
        if edges < 1:
            raise ValueError("edges 必须大于 0")

        token_embeddings = self.backbone.wte.weight[1:].reshape(
            self.digits, self.codebook_size, -1
        )
        token_embeddings = F.normalize(token_embeddings, dim=-1)
        similarities = torch.bmm(
            token_embeddings, token_embeddings.transpose(1, 2)
        )

        all_codes = self.codes[1:]
        adjacency = torch.zeros(
            self.item_num, edges,
            dtype=torch.long, device=self.codes.device,
        )

        # 只分配 chunk_size × n_items 的相似度块。
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            block = token_embeddings.new_zeros(stop - start, count)
            for digit in range(self.digits):
                left = all_codes[start:stop, digit]
                right = all_codes[:, digit]
                block += similarities[digit][
                    left[:, None], right[None, :]
                ]
            adjacency[start + 1:stop + 1] = (
                block.topk(edges, dim=1).indices + 1
            )

        self.adjacency = adjacency

    @torch.no_grad()
    def graph_topk(
        self, log_probs, excluded, topk,
        beams=50, steps=3, generator=None,
    ):
        if self.adjacency is None:
            raise RuntimeError("需要先调用 build_graph")
        if beams < topk or steps < 1:
            raise ValueError("要求 beams >= topk 且 steps >= 1")

        results = []
        visited_counts = []
        device = log_probs.device

        for row in range(len(log_probs)):
            # 允许初始采样重复，再去重。
            beam = torch.randint(
                1, self.item_num,
                (beams,),
                generator=generator,
                device=device,
            ).unique()

            blocked = torch.as_tensor(
                excluded[row], dtype=torch.long, device=device
            )
            visited = beam
            probabilities = log_probs[row:row + 1]

            for _ in range(steps):
                candidates = torch.cat([
                    beam, self.adjacency[beam].reshape(-1)
                ]).unique()
                visited = torch.cat([visited, candidates]).unique()

                scores = self.score_items(
                    probabilities, candidates
                )[0]
                scores = scores.masked_fill(
                    torch.isin(candidates, blocked), -torch.inf
                )
                keep = min(beams, candidates.numel())
                beam = candidates[scores.topk(keep).indices]

            scores = self.score_items(probabilities, beam)[0]
            scores = scores.masked_fill(
                torch.isin(beam, blocked), -torch.inf
            )
            order = scores.argsort(descending=True)
            order = order[torch.isfinite(scores[order])]
            selected = beam[order[:topk]]

            # 候选不足时用 padding 补齐，评估中不会命中正样本。
            output = torch.zeros(topk, dtype=torch.long, device=device)
            output[:selected.numel()] = selected
            results.append(output)
            visited_counts.append(visited.numel())

        return torch.stack(results), visited_counts

    class Dataset(SequentialModel.Dataset):
        def actions_before_epoch(self):
            # MTP 损失直接使用目标 token，无需采样训练负物品。
            pass

        def _get_feed_dict(self, index):
            user = int(self.data["user_id"][index])
            target = int(self.data["item_id"][index])
            position = int(self.data["position"][index])
            history = self.corpus.user_his[user][:position]
            history = history[-self.model.history_max:]

            return {
                "user_id": user,
                "item_id": np.array([target], dtype=np.int64),
                "target_item": target,
                "history_items": np.array(
                    [item for item, _ in history], dtype=np.int64
                ),
                "lengths": len(history),
            }