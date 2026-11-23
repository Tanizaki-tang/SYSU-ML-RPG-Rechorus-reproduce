import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument(
        "--model_name", required=True,
        choices=["RPG", "SASRec", "GRU4Rec"],
    )
    initial.add_argument("--gpu", default="0")
    selected, _ = initial.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = selected.gpu

    import torch
    from helpers.FrozenSeqReader import FrozenSeqReader
    from helpers.UnifiedRunner import RPGTrainingRunner, UnifiedRunner
    from utils import utils

    module = importlib.import_module(
        f"models.sequential.{selected.model_name}"
    )
    model_class = getattr(module, selected.model_name)

    parser = argparse.ArgumentParser(parents=[initial])
    parser.add_argument("--train", type=int, choices=[0, 1], default=0)
    parser.add_argument("--load", type=int, choices=[0, 1], default=0)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--log_file", required=True)
    parser.add_argument(
        "--eval_phase", choices=["dev", "test", "both"], default="dev"
    )

    parser = FrozenSeqReader.parse_data_args(parser)
    parser = UnifiedRunner.parse_runner_args(parser)
    parser = model_class.parse_model_args(parser)
    is_rpg = selected.model_name == "RPG"
    if is_rpg:
        parser = RPGTrainingRunner.parse_training_args(parser)
    args = parser.parse_args()

    if is_rpg and args.train:
        if args.optimizer != "AdamW" or args.main_metric != "NDCG@10":
            parser.error("RPG training uses AdamW and selects checkpoints by NDCG@10")
        if args.epoch < 1 or args.batch_size < 1:
            parser.error("epoch and batch_size must be positive")
        if args.warmup_steps < 0 or args.max_train_batches < 0:
            parser.error("warmup_steps and max_train_batches must be nonnegative")
        if args.max_train_batches and (not args.eval_limit or args.eval_phase != "dev"):
            parser.error("a capped training run requires --eval_limit and --eval_phase dev")
    if args.test_epoch != -1:
        parser.error("本入口通过 --eval_phase 选择最终评估集合，test_epoch 使用 -1")
    if args.train and args.decode == "graph":
        parser.error("请先 full 训练，再加载 checkpoint 执行 graph 评估")
    if not args.model_path:
        parser.error("请明确设置 --model_path")
    if args.train and not args.load and Path(args.model_path).exists():
        parser.error("该 checkpoint 已存在，请指定新的实验文件名")

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(
                args.log_file, mode="w", encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    utils.init_seed(args.random_seed)
    args.device = torch.device(
        "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    )

    corpus = FrozenSeqReader(args)
    model = model_class(args, corpus).to(args.device)
    runner = RPGTrainingRunner(args) if is_rpg and args.train else UnifiedRunner(args)

    if is_rpg:
        for method in ("token_logits", "score_items", "build_graph"):
            if not callable(getattr(model, method, None)):
                raise TypeError(
                    f"当前 RPG 缺少 {method}()，"
                    "请核对是否为前面提供的 ReChorus 适配版本"
                )

    if args.load:
        # 沿用你当前模型自己的加载逻辑。
        model.load_model()

    data = {
        phase: SimpleNamespace(
            model=model, corpus=corpus, phase=phase
        )
        for phase in ("dev", "test")
    }

    if args.train:
        if is_rpg:
            data["train"] = model_class.PackedDataset(corpus, model.history_max)
        else:
            data["train"] = model_class.Dataset(model, corpus, "train")
            data["train"].prepare()
        if len(data["train"]) == 0:
            raise ValueError("没有可训练的历史序列")
        runner.train(data)

    phases = (
        ("dev", "test")
        if args.eval_phase == "both"
        else (args.eval_phase,)
    )
    for phase in phases:
        result = runner.evaluate(
            data[phase], runner.topk, runner.metrics
        )
        logging.info(
            "%s FINAL %s %s",
            args.model_name, phase, result,
        )

    if args.eval_limit:
        logging.info("SMOKE TEST: 当前只评估指定数量的样本")
    if not args.train and not args.load:
        logging.info("RANDOM INITIALIZATION: 当前模型未训练且未加载权重")


if __name__ == "__main__":
    main()
