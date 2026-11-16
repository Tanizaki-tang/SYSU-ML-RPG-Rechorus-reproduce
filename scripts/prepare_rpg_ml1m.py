import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/ML1M_RPG")
    parser.add_argument("--digits", type=int, default=16)
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.digits <= 0 or args.pca_dim % args.digits:
        raise ValueError("pca_dim 必须能被 digits 整除")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    faiss.omp_set_num_threads(4)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "ml-1m.zip"

    if not archive.exists():
        print("Downloading MovieLens-1M...")
        urllib.request.urlretrieve(
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            archive,
        )

    with zipfile.ZipFile(archive) as z:
        ratings = pd.read_csv(
            io.BytesIO(z.read("ml-1m/ratings.dat")),
            sep="::",
            engine="python",
            names=["user_id", "original_id", "rating", "time"],
        )
        movies = pd.read_csv(
            io.BytesIO(z.read("ml-1m/movies.dat")),
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["original_id", "title", "genres"],
        )

    original_ids = np.sort(ratings["original_id"].unique())
    item_mapping = {
        int(original): index + 1
        for index, original in enumerate(original_ids)
    }
    ratings["item_id"] = ratings["original_id"].map(item_mapping)

    # 用原始行号稳定处理相同时间戳。
    ratings["row_order"] = np.arange(len(ratings))
    ratings = ratings.sort_values(
        ["user_id", "time", "row_order"]
    ).reset_index(drop=True)

    groups = ratings.groupby("user_id", sort=False)
    position = groups.cumcount()
    count = groups["item_id"].transform("size")

    masks = {
        "train": position < count - 2,
        "dev": position == count - 2,
        "test": position == count - 1,
    }

    for phase, mask in masks.items():
        frame = ratings.loc[
            mask, ["user_id", "item_id", "time"]
        ].copy()
        frame.to_csv(out / f"{phase}.csv", sep="\t", index=False)
        print(phase, len(frame))

    # 保证文本第 i 行对应 ReChorus 的物品 i+1。
    metadata = movies.set_index("original_id").loc[original_ids].copy()
    metadata["item_id"] = np.arange(1, len(metadata) + 1)
    metadata["text"] = (
        "Title: " + metadata["title"].fillna("")
        + ". Genres: "
        + metadata["genres"].fillna("").str.replace("|", ", ", regex=False)
    )
    metadata.reset_index().to_csv(
        out / "items.tsv", sep="\t", index=False
    )

    encoder_name = "sentence-transformers/sentence-t5-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)
    vectors = encoder.encode(
        metadata["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    train_items = np.sort(
        ratings.loc[masks["train"], "item_id"].unique()
    )
    train_rows = train_items - 1

    if len(train_rows) < max(256, args.pca_dim):
        raise ValueError("训练物品数不足以拟合当前 PCA 和 256 大小的码本")

    pca = PCA(
        n_components=args.pca_dim,
        whiten=True,
        svd_solver="full",
    )
    pca.fit(vectors[train_rows])
    features = np.ascontiguousarray(
        pca.transform(vectors), dtype="float32"
    )

    index = faiss.index_factory(
        args.pca_dim,
        f"OPQ{args.digits},IVF1,PQ{args.digits}x8",
        faiss.METRIC_INNER_PRODUCT,
    )
    print("Training OPQ...")
    index.train(np.ascontiguousarray(features[train_rows]))
    index.add(features)

    ivf = faiss.extract_index_ivf(index)
    lists = ivf.invlists
    size = lists.list_size(0)

    # IVF1 只有一个倒排列表。8-bit PQ 每一位占一个字节。
    stored_ids = faiss.rev_swig_ptr(
        lists.get_ids(0), size
    ).copy()
    stored_codes = faiss.rev_swig_ptr(
        lists.get_codes(0), size * lists.code_size
    ).copy().reshape(size, lists.code_size)

    if size != len(metadata) or lists.code_size != args.digits:
        raise RuntimeError("FAISS 输出形状与预期不一致")

    codes = np.zeros(
        (len(metadata) + 1, args.digits), dtype=np.int64
    )
    codes[stored_ids + 1] = stored_codes

    np.save(out / "semantic_ids.npy", codes)
    joblib.dump(pca, out / "pca.joblib")
    faiss.write_index(index, str(out / "opq.index"))

    config = {
        "dataset": "MovieLens-1M",
        "split": "per-user leave-two-out",
        "feedback": "all ratings treated as implicit interactions",
        "encoder": encoder_name,
        "pca_dim": args.pca_dim,
        "digits": args.digits,
        "codebook_size": 256,
        "seed": args.seed,
        "n_items": len(metadata),
        "quantizer_fit": "training items only",
    }
    (out / "preprocess.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    unique_codes = len(np.unique(codes[1:], axis=0))
    print("semantic_ids shape:", codes.shape)
    print("unique semantic IDs:", unique_codes)
    print("items:", len(metadata))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()