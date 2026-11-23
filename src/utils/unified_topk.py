import torch


def blocked_items(history, target):
    """完整历史过滤，允许当前目标属于重复交互。"""
    return set(history) - {int(target)}


def rank_items(ids, scores, k):
    """
    按分数降序、物品 ID 升序排序。
    ids 为 [C] 或 [B,C]，scores 为 [B,C]。
    输入的真实候选应已经去重。
    """
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(scores.shape[0], -1)

    order = ids.argsort(dim=1, stable=True)
    ids = ids.gather(1, order)
    scores = scores.gather(1, order)

    order = scores.argsort(
        dim=1, descending=True, stable=True
    )[:, :k]
    ids = ids.gather(1, order)
    scores = scores.gather(1, order)

    ids = ids.masked_fill(~torch.isfinite(scores), 0)
    return ids, scores


def mask_history(ids, scores, blocked):
    scores = scores.clone()
    for row, history in enumerate(blocked):
        if history:
            forbidden = torch.tensor(
                sorted(history),
                dtype=torch.long,
                device=scores.device,
            )
            scores[row, torch.isin(ids, forbidden)] = -torch.inf
    return scores


def metric_values(predictions, targets, ks):
    """
    返回逐样本指标，最后由调用者等权平均。
    每个样本恰好一个目标物品。
    """
    matches = predictions.eq(targets[:, None])
    rank = matches.long().argmax(dim=1) + 1
    rank = torch.where(
        matches.any(dim=1),
        rank,
        torch.full_like(rank, predictions.shape[1] + 1),
    )

    result = {}
    for k in ks:
        hit = rank <= k
        result[f"HR@{k}"] = hit.double()
        result[f"NDCG@{k}"] = (
            hit.double() / torch.log2(rank.double() + 1)
        )
    return result


@torch.no_grad()
def graph_topk_one(
    model, log_probs, blocked, k, beams, steps, seed
):
    """
    单样本图搜索。
    model 需要提供 item_num、adjacency、score_items()。
    log_probs 已由一次序列编码计算得到。
    """
    count = model.item_num - 1
    if not (k <= beams <= count) or steps < 1:
        raise ValueError("要求 k <= beams <= 物品数，steps >= 1")

    device = log_probs.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    # 使用局部随机数发生器，不改变训练使用的全局随机状态。
    beam = torch.empty(0, dtype=torch.long, device=device)
    while beam.numel() < beams:
        sampled = torch.randint(
            1, model.item_num,
            (beams,),
            generator=generator,
            device=device,
        )
        beam = torch.cat([beam, sampled]).unique()[:beams]

    visited = beam.clone()

    for _ in range(steps):
        if beam.numel() == 0:
            break

        candidates = torch.cat([
            beam,
            model.adjacency[beam].reshape(-1),
        ]).unique()

        # padding 永远不是候选节点。
        candidates = candidates[
            (candidates > 0) & (candidates < model.item_num)
        ]
        visited = torch.cat([visited, candidates]).unique()

        scores = model.score_items(log_probs, candidates)
        if not torch.isfinite(scores).all():
            raise ValueError("图搜索出现非有限原始分数")

        scores = mask_history(candidates, scores, [blocked])
        selected, _ = rank_items(candidates, scores, beams)
        beam = selected[0]
        beam = beam[beam > 0]

    output = torch.zeros(k, dtype=torch.long, device=device)
    output[:min(k, beam.numel())] = beam[:k]
    return output, int(visited.numel())


def self_test():
    # 并列分数时按物品 ID 升序。
    ids = torch.tensor([4, 2, 3, 1])
    scores = torch.tensor([[0.5, 0.5, 0.5, 0.1]])
    predicted, _ = rank_items(ids, scores, 3)
    assert predicted.tolist() == [[2, 3, 4]]

    # 目标可重复推荐，未来物品不在过去历史里。
    assert blocked_items([1, 2, 1], 1) == {2}
    candidates = torch.tensor([1, 2, 3, 4])
    scores = mask_history(
        candidates, torch.ones(1, 4), [blocked_items([1, 2], 3)]
    )
    predicted, _ = rank_items(candidates, scores, 4)
    assert predicted.tolist() == [[3, 4, 0, 0]]

    # 排名为 2 的目标，以及完全未返回的目标。
    values = metric_values(
        torch.tensor([[2, 3, 4], [4, 0, 0]]),
        torch.tensor([3, 5]),
        [1, 3],
    )
    assert values["HR@1"].tolist() == [0, 0]
    assert values["HR@3"].tolist() == [1, 0]
    assert abs(
        values["NDCG@3"][0].item()
        - 1 / 1.584962500721156
    ) < 1e-12

    # 分块合并与一次性排序一致。
    ids = torch.arange(1, 21)
    scores = torch.ones(2, 20)
    expected = rank_items(ids, scores, 10)[0]
    best_ids = torch.empty(2, 0, dtype=torch.long)
    best_scores = torch.empty(2, 0)
    for start in range(0, 20, 3):
        current = ids[start:start + 3].expand(2, -1)
        best_ids, best_scores = rank_items(
            torch.cat([best_ids, current], dim=1),
            torch.cat([
                best_scores, scores[:, start:start + 3]
            ], dim=1),
            10,
        )
    assert torch.equal(expected, best_ids)

    # 完全连通图应返回同分数规则下的正确结果。
    class ToyModel:
        item_num = 5
        adjacency = torch.arange(1, 5).repeat(5, 1)

        def score_items(self, probabilities, item_ids):
            return probabilities[:, item_ids - 1]

    result, _ = graph_topk_one(
        ToyModel(),
        torch.tensor([[1.0, 1.0, 0.5, 0.2]]),
        blocked={1},
        k=2,
        beams=2,
        steps=2,
        seed=2026,
    )
    assert result.tolist() == [2, 3]
    print("PASS: unified ranking, masking, metrics and graph search")


if __name__ == "__main__":
    self_test()