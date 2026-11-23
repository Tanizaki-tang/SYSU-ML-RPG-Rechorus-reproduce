import argparse
import csv
import getpass
import hashlib
import html
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from filelock import FileLock


MODEL = "text-embedding-3-large"
DIMENSION = 3072
PCA_DIMENSION = 512
MAX_INPUT_TOKENS = 8191
TEXT_VERSION = "metadata_text_v1"
FIELDS = (
    "title", "price", "brand",
    "feature", "categories", "description",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            indent=2, allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def atomic_bytes(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def atomic_array(path, array):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def save_json(path, value):
    atomic_bytes(path, json_bytes(value))


def save_fixed(path, content):
    if path.exists():
        require(
            path.read_bytes() == content,
            f"已有缓存与当前配置不一致：{path}",
        )
    else:
        atomic_bytes(path, content)


def flatten(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(
            part for part in (flatten(x) for x in value) if part
        )
    if isinstance(value, dict):
        return ", ".join(
            f"{key}: {flatten(value[key])}" for key in sorted(value)
        )
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_text(metadata):
    if "genres" in metadata:
        title = flatten(metadata.get("title"))
        genres = flatten(metadata.get("genres"))
        parts = []
        if title:
            parts.append("Title: " + title)
        if genres:
            parts.append("Genres: " + genres)
        return ". ".join(parts)

    parts = [flatten(metadata.get(field)) for field in FIELDS]
    return ". ".join(part for part in parts if part)


def prepare(folder, work):
    manifest = json.loads(
        (folder / "manifest.json").read_text(encoding="utf-8")
    )
    for filename, expected in manifest["files"].items():
        require(
            digest_file(folder / filename) == expected,
            f"冻结文件发生变化：{filename}",
        )

    # 与 FrozenSeqReader 的 dataset_hash 算法一致。
    dataset_hash = digest_bytes(
        json.dumps(
            manifest["files"], sort_keys=True
        ).encode("utf-8")
    )
    mapping = json.loads(
        (folder / "item2id.json").read_text(encoding="utf-8")
    )
    count = len(mapping)
    require(
        sorted(mapping.values()) == list(range(1, count + 1)),
        "物品映射必须连续且从 1 开始",
    )

    records = [
        json.loads(line)
        for line in (
            folder / "item_metadata.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda row: row["item_id"])
    require(
        [row["item_id"] for row in records]
        == list(range(1, count + 1)),
        "元数据物品 ID 缺失、重复或越界",
    )

    with (folder / "train.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        training_items = {
            int(row["item_id"])
            for row in csv.DictReader(stream, delimiter="\t")
        }
    require(
        training_items <= set(range(1, count + 1)),
        "训练集包含目录外物品",
    )

    encoding = tiktoken.get_encoding("cl100k_base")
    prepared = []
    missing_ids = []
    truncated_ids = []

    for row in records:
        item_id = int(row["item_id"])
        require(
            mapping[row["original_id"]] == item_id,
            f"物品映射错位：{item_id}",
        )
        require(
            bool(row["seen_in_train"]) == (item_id in training_items),
            f"seen_in_train 与训练集不一致：{item_id}",
        )

        text = make_text(row["metadata"])
        if not text:
            text = "Item metadata unavailable."
            missing_ids.append(item_id)

        tokens = encoding.encode(text, disallowed_special=())
        original_length = len(tokens)
        if original_length > MAX_INPUT_TOKENS:
            truncated_ids.append(item_id)
        tokens = tokens[:MAX_INPUT_TOKENS]

        prepared.append({
            "item_id": item_id,
            "original_id": row["original_id"],
            "text": text,
            "original_token_count": original_length,
            # API 使用这些确切的 token，避免截断边界的歧义。
            "input_tokens": tokens,
        })

    text_bytes = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in prepared
        ) + "\n"
    ).encode("utf-8")
    save_fixed(work / "texts.jsonl", text_bytes)

    binding = {
        "dataset_hash": dataset_hash,
        "model": MODEL,
        "dimensions": DIMENSION,
        "text_version": TEXT_VERSION,
        "tokenizer": "cl100k_base",
        "max_input_tokens": MAX_INPUT_TOKENS,
        "texts_sha256": digest_bytes(text_bytes),
    }
    save_fixed(work / "binding.json", json_bytes(binding))

    summary = {
        **binding,
        "items": count,
        "training_items": len(training_items),
        "input_tokens_total": sum(
            len(row["input_tokens"]) for row in prepared
        ),
        "missing_metadata_item_ids": missing_ids,
        "truncated_item_ids": truncated_ids,
    }
    save_fixed(work / "text_summary.json", json_bytes(summary))
    print(json.dumps({
        "items": count,
        "training_items": len(training_items),
        "input_tokens_total": summary["input_tokens_total"],
        "missing_metadata_items": len(missing_ids),
        "truncated_items": len(truncated_ids),
    }, indent=2))

    return prepared, training_items, binding


def verify_artifacts(work, report_name):
    report = json.loads(
        (work / report_name).read_text(encoding="utf-8")
    )
    for filename, expected in report["files"].items():
        require(
            digest_file(work / filename) == expected,
            f"缓存文件校验失败：{filename}",
        )
    return report


def embed(prepared, binding, work, limit):
    from openai import OpenAI

    database = sqlite3.connect(work / "embeddings.sqlite")
    try:
        database.execute(
            "CREATE TABLE IF NOT EXISTS config "
            "(id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
        )
        database.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(item_id INTEGER PRIMARY KEY, vector BLOB NOT NULL)"
        )
        database.execute(
            "CREATE TABLE IF NOT EXISTS requests "
            "(time TEXT, request_id TEXT, model TEXT, tokens INTEGER)"
        )

        config = json_bytes(binding).decode("utf-8")
        existing = database.execute(
            "SELECT value FROM config WHERE id=1"
        ).fetchone()
        if existing:
            require(existing[0] == config, "SQLite 缓存绑定不一致")
        else:
            database.execute(
                "INSERT INTO config VALUES (1, ?)", (config,)
            )
        database.commit()

        completed = {
            row[0] for row in database.execute(
                "SELECT item_id FROM embeddings"
            )
        }
        selected = prepared[:limit] if limit else prepared
        pending = [
            row for row in selected if row["item_id"] not in completed
        ]
        print("Already cached:", len(completed))
        print("Pending this run:", len(pending))

        client = None
        if pending:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                key = getpass.getpass("OpenAI API key: ")
            client = OpenAI(
                api_key=key,
                base_url="https://openrouter.ai/api/v1",
                max_retries=5,
                timeout=120.0,
            )

        cursor = 0
        while cursor < len(pending):
            batch = []
            token_count = 0
            while cursor < len(pending) and len(batch) < 64:
                row = pending[cursor]
                length = len(row["input_tokens"])
                if batch and token_count + length > 50000:
                    break
                batch.append(row)
                token_count += length
                cursor += 1

            response = client.embeddings.create(
                model=f"openai/{MODEL}",
                dimensions=DIMENSION,
                encoding_format="float",
                input=[row["input_tokens"] for row in batch],
            )
            entries = sorted(response.data, key=lambda entry: entry.index)
            require(
                [entry.index for entry in entries] == list(range(len(batch))),
                "API 返回的向量数量或索引不完整",
            )

            vectors = np.asarray(
                [entry.embedding for entry in entries],
                dtype="<f4",
            )
            require(
                vectors.shape == (len(batch), DIMENSION),
                "API 向量维度不正确",
            )
            require(
                np.isfinite(vectors).all(),
                "API 向量包含 NaN 或 Inf",
            )

            # 整批向量与请求记录在同一事务中提交。
            with database:
                database.executemany(
                    "INSERT INTO embeddings VALUES (?, ?)",
                    [
                        (row["item_id"], vectors[index].tobytes())
                        for index, row in enumerate(batch)
                    ],
                )
                database.execute(
                    "INSERT INTO requests VALUES (?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        getattr(response, "_request_id", None),
                        response.model,
                        int(response.usage.total_tokens),
                    ),
                )
            print(
                f"Cached batch of {len(batch)} items; "
                f"processed {cursor}/{len(pending)}",
                flush=True,
            )

        stored = database.execute(
            "SELECT item_id, vector FROM embeddings ORDER BY item_id"
        ).fetchall()
        if len(stored) != len(prepared):
            print("Partial cache saved. Run embed without --limit to finish.")
            return

        require(
            [row[0] for row in stored] == list(range(1, len(prepared) + 1)),
            "SQLite 中的物品索引不完整",
        )
        vectors = np.stack([
            np.frombuffer(blob, dtype="<f4")
            for _, blob in stored
        ])
        require(
            vectors.shape == (len(prepared), DIMENSION)
            and np.isfinite(vectors).all(),
            "缓存向量不完整",
        )
        atomic_array(work / "embeddings.npy", vectors)

        request_records = database.execute(
            "SELECT time, request_id, model, tokens FROM requests"
        ).fetchall()
        save_json(work / "embedding_report.json", {
            "binding": binding,
            "shape": list(vectors.shape),
            "requests": request_records,
            "files": {
                "embeddings.npy": digest_file(work / "embeddings.npy")
            },
        })
        print("EMBEDDINGS COMPLETE:", vectors.shape)
    finally:
        database.close()


def quantize(prepared, training_items, binding, work, folder, digits):
    import faiss
    import joblib
    import sklearn
    from sklearn.decomposition import PCA

    report = verify_artifacts(work, "embedding_report.json")
    require(report["binding"] == binding, "Embedding 数据绑定不一致")

    pca_report_path = work / "pca_report.json"
    if pca_report_path.exists():
        pca_report = verify_artifacts(work, "pca_report.json")
        require(
            pca_report["embedding_sha256"]
            == report["files"]["embeddings.npy"],
            "PCA 与 embedding 缓存不一致",
        )
        features = np.load(work / "features_512.npy", allow_pickle=False)
    else:
        vectors = np.load(work / "embeddings.npy", allow_pickle=False)
        require(
            min(vectors.shape) >= PCA_DIMENSION,
            "物品数量不足以拟合 512 维 PCA",
        )
        pca = PCA(
            n_components=PCA_DIMENSION,
            whiten=True,
            svd_solver="full",
        )
        features = np.ascontiguousarray(
            pca.fit_transform(vectors), dtype="float32"
        )
        require(
            np.isfinite(features).all()
            and np.all(pca.explained_variance_ > 1e-12),
            "PCA 存在退化维度，请检查物品文本和向量",
        )
        atomic_array(work / "features_512.npy", features)
        temporary = work / "pca.joblib.tmp"
        joblib.dump(pca, temporary)
        os.replace(temporary, work / "pca.joblib")
        save_json(pca_report_path, {
            "embedding_sha256": report["files"]["embeddings.npy"],
            "fit_items": "all catalog items",
            "components": PCA_DIMENSION,
            "whiten": True,
            "solver": "full",
            "sklearn_version": sklearn.__version__,
            "files": {
                name: digest_file(work / name)
                for name in ("features_512.npy", "pca.joblib")
            },
        })

    features = np.ascontiguousarray(features, dtype="float32")
    require(features.shape == (len(prepared), PCA_DIMENSION), "PCA 形状错误")

    train_rows = np.array(sorted(training_items), dtype=np.int64) - 1
    require(len(train_rows) >= 256, "训练物品不足 256 个")
    faiss.omp_set_num_threads(4)

    for length in digits:
        require(
            length in (4, 8, 16, 32, 64),
            "digits 应取 4、8、16、32、64",
        )
        output = folder / f"semantic_ids_m{length}.npy"
        metadata_path = folder / f"semantic_ids_m{length}.json"

        if metadata_path.exists():
            verify_sid(folder, length, binding)
            print("REUSED:", output.name)
            continue

        specification = f"OPQ{length},IVF1,PQ{length}x8"
        index = faiss.index_factory(
            PCA_DIMENSION, specification, faiss.METRIC_INNER_PRODUCT
        )
        print("Training:", specification, flush=True)
        index.train(np.ascontiguousarray(features[train_rows]))
        index.add(features)

        ivf = faiss.extract_index_ivf(index)
        lists = ivf.invlists
        size = lists.list_size(0)
        require(
            size == len(prepared) and lists.code_size == length,
            "FAISS 编码数量或长度不符合预期",
        )

        stored_ids = faiss.rev_swig_ptr(
            lists.get_ids(0), size
        ).copy()
        stored_codes = faiss.rev_swig_ptr(
            lists.get_codes(0), size * lists.code_size
        ).copy().reshape(size, length)

        require(
            np.array_equal(np.sort(stored_ids), np.arange(size)),
            "FAISS 返回的物品索引不完整",
        )

        # FAISS 行索引 0 对应真实物品 ID 1。
        codes = np.zeros((size + 1, length), dtype=np.int64)
        codes[stored_ids + 1] = stored_codes
        atomic_array(output, codes)

        index_path = work / f"opq_m{length}.index"
        temporary = index_path.with_suffix(".tmp")
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)

        unique_count = len(np.unique(codes[1:], axis=0))
        save_json(metadata_path, {
            "binding": binding,
            "digits": length,
            "codebook_size": 256,
            "shape": list(codes.shape),
            "quantizer": specification,
            "quantizer_fit": "training items only",
            "faiss_version": faiss.__version__,
            "faiss_threads": 4,
            "faiss_initialization": "library defaults",
            "embedding_sha256": report["files"]["embeddings.npy"],
            "features_sha256": digest_file(work / "features_512.npy"),
            "index_sha256": digest_file(index_path),
            "sid_sha256": digest_file(output),
            "unique_semantic_ids": unique_count,
            "extra_items_sharing_ids": size - unique_count,
        })
        verify_sid(folder, length, binding)
        print("SAVED:", output.name, codes.shape)


def verify_sid(folder, length, binding):
    work = folder / "semantic_openai_v1"
    metadata = json.loads(
        (folder / f"semantic_ids_m{length}.json").read_text(encoding="utf-8")
    )
    require(metadata["binding"] == binding, "语义 ID 数据绑定不一致")
    output = folder / f"semantic_ids_m{length}.npy"
    require(
        digest_file(output) == metadata["sid_sha256"],
        "语义 ID 文件哈希不一致",
    )
    require(
        digest_file(work / "embeddings.npy") == metadata["embedding_sha256"]
        and digest_file(work / "features_512.npy") == metadata["features_sha256"]
        and digest_file(work / f"opq_m{length}.index") == metadata["index_sha256"],
        "语义 ID 的上游产物不一致",
    )
    codes = np.load(output, allow_pickle=False)
    count = len(json.loads(
        (folder / "item2id.json").read_text(encoding="utf-8")
    ))
    require(codes.shape == (count + 1, length), "语义 ID 形状错误")
    require(np.issubdtype(codes.dtype, np.integer), "语义 ID 必须是整数")
    require(np.all(codes[0] == 0), "padding 行必须为 0")
    require(codes.min() >= 0 and codes.max() < 256, "编码超出 0..255")
    print("VERIFIED:", output.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument(
        "--stage", required=True,
        choices=["prepare", "embed", "quantize", "verify"],
    )
    parser.add_argument("--digits", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    require(args.limit >= 0, "limit 必须非负")

    folder = Path(args.dataset_dir).resolve()
    work = folder / "semantic_openai_v1"
    work.mkdir(parents=True, exist_ok=True)

    # 同一数据集的生成过程串行执行，退出后自动释放锁。
    with FileLock(str(work / ".build.lock"), timeout=0):
        prepared, training_items, binding = prepare(folder, work)
        if args.stage == "embed":
            embed(prepared, binding, work, args.limit)
        elif args.stage == "quantize":
            quantize(
                prepared, training_items, binding,
                work, folder, args.digits,
            )
        elif args.stage == "verify":
            for length in args.digits:
                verify_sid(folder, length, binding)


if __name__ == "__main__":
    main()
