import argparse
import ast
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


UPSTREAM_COMMIT = "c164ec4303cc20ddcfbd1b57de366a481811d1e5"
GROCERY = "Grocery_and_Gourmet_Food"
PHASES = ("train", "dev", "test")
KEYS = ["user_id", "item_id", "time"]

GITHUB_RAW = (
    "https://raw.githubusercontent.com/THUwangcy/ReChorus/"
    + UPSTREAM_COMMIT
)
AMAZON_RAW = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            print("Downloading:", url, flush=True)
            request = urllib.request.Request(
                url, headers={"User-Agent": "RPG-course-reproduction"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            os.replace(temporary, path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return path


def read_records(path):
    # Amazon 2014 metadata 存在 Python 字典文本格式。
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def make_mapping(values):
    return {
        original: index + 1
        for index, original in enumerate(sorted(set(values)))
    }


def map_frame(frame, users, items):
    frame = frame.copy()
    frame["user_id"] = frame["user_raw"].map(users)
    frame["item_id"] = frame["item_raw"].map(items)
    require(
        frame[["user_id", "item_id"]].notna().all().all(),
        "存在无法映射的用户或物品",
    )
    frame["user_id"] = frame["user_id"].astype("int64")
    frame["item_id"] = frame["item_id"].astype("int64")
    return frame


def build_grocery(raw_dir):
    sources = []

    def fetch(name, url):
        path = download(url, raw_dir / name)
        sources.append({
            "name": name, "url": url, "sha256": sha256(path)
        })
        return path

    reviews_path = fetch(
        "reviews_Grocery_and_Gourmet_Food_5.json.gz",
        AMAZON_RAW + "/reviews_Grocery_and_Gourmet_Food_5.json.gz",
    )
    metadata_path = fetch(
        "meta_Grocery_and_Gourmet_Food.json.gz",
        AMAZON_RAW + "/meta_Grocery_and_Gourmet_Food.json.gz",
    )

    rows = []
    for order, record in enumerate(read_records(reviews_path)):
        rows.append((
            str(record["reviewerID"]),
            str(record["asin"]),
            int(record["unixReviewTime"]),
            order,
        ))

    frame = pd.DataFrame(
        rows, columns=["user_raw", "item_raw", "time", "source_order"]
    )
    frame = frame.drop_duplicates(
        ["user_raw", "item_raw", "time"], keep="first"
    )
    users = make_mapping(frame["user_raw"])
    items = make_mapping(frame["item_raw"])
    frame = map_frame(frame, users, items)

    # 直接导入框架现成切分，忽略其中的抽样负物品列。
    official_parts = []
    for phase in PHASES:
        path = fetch(
            f"rechorus_{phase}.csv",
            f"{GITHUB_RAW}/data/{GROCERY}/{phase}.csv",
        )
        part = pd.read_csv(path, sep="\t", usecols=KEYS)
        part = part.astype("int64")
        part["split"] = phase
        official_parts.append(part)

    official = pd.concat(official_parts, ignore_index=True)
    require(
        not official.duplicated(KEYS).any(),
        "Grocery 官方切分存在跨集合重复交互",
    )

    joined = frame.merge(
        official,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    require(
        joined["_merge"].eq("both").all(),
        "原始 Grocery 与固定版本 ReChorus 切分不一致，"
        "请检查原始数据来源和映射",
    )
    joined = joined.drop(columns="_merge")

    fields = [
        "title", "price", "brand", "feature",
        "categories", "description",
    ]
    metadata = {}
    for record in read_records(metadata_path):
        original = str(record["asin"])
        if original in items and original not in metadata:
            metadata[original] = {
                key: record[key] for key in fields if key in record
            }

    protocol = {
        "dataset": "Amazon Reviews 2014 Grocery 5-core",
        "split": "fixed ReChorus train/dev/test",
        "upstream_commit": UPSTREAM_COMMIT,
        "feedback": "review interactions",
        "mapping_order": "original ID string ascending",
    }
    return joined, users, items, metadata, protocol, sources


def build_movielens(raw_dir):
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    path = download(url, raw_dir / "ml-1m.zip")
    sources = [{
        "name": path.name, "url": url, "sha256": sha256(path)
    }]

    with zipfile.ZipFile(path) as archive:
        ratings = pd.read_csv(
            io.BytesIO(archive.read("ml-1m/ratings.dat")),
            sep="::",
            engine="python",
            names=["user_raw", "item_raw", "rating", "time"],
            dtype={"user_raw": str, "item_raw": str},
        )
        movies = pd.read_csv(
            io.BytesIO(archive.read("ml-1m/movies.dat")),
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["item_raw", "title", "genres"],
            dtype={"item_raw": str},
            keep_default_na=False,
        )

    ratings["source_order"] = range(len(ratings))
    positive = ratings.loc[ratings["rating"] >= 4].copy()

    # 迭代至正反馈子图中所有用户和物品都至少有 5 条交互。
    while True:
        user_count = positive.groupby("user_raw")["item_raw"].transform("size")
        item_count = positive.groupby("item_raw")["user_raw"].transform("size")
        filtered = positive.loc[
            (user_count >= 5) & (item_count >= 5)
        ].copy()
        if len(filtered) == len(positive):
            break
        positive = filtered

    require(len(positive) > 0, "MovieLens 5-core 过滤后为空")

    # notebook 的日期原点来自过滤后的全部评分记录。
    retained = ratings.loc[
        ratings["user_raw"].isin(positive["user_raw"].unique())
        & ratings["item_raw"].isin(positive["item_raw"].unique())
    ]
    origin_day = int((retained["time"] // 86400).min())
    positive["day"] = positive["time"] // 86400 - origin_day
    maximum_day = int(positive["day"].max())
    boundary1 = int(maximum_day * 0.8)
    boundary2 = int(maximum_day * 0.9)

    positive["split"] = "test"
    positive.loc[positive["day"] <= boundary2, "split"] = "dev"
    positive.loc[positive["day"] <= boundary1, "split"] = "train"

    training = positive.loc[positive["split"] == "train"]
    known_users = set(training["user_raw"])
    known_items = set(training["item_raw"])

    before = positive.groupby("split").size().to_dict()
    positive = positive.loc[
        positive["user_raw"].isin(known_users)
        & positive["item_raw"].isin(known_items)
    ].copy()
    after = positive.groupby("split").size().to_dict()

    users = make_mapping(positive["user_raw"])
    items = make_mapping(positive["item_raw"])
    frame = map_frame(positive, users, items)

    metadata = {}
    for row in movies.itertuples(index=False):
        if row.item_raw in items:
            metadata[row.item_raw] = {
                "title": row.title,
                "genres": row.genres.split("|"),
            }

    protocol = {
        "dataset": "MovieLens-1M",
        "positive_rating_threshold": 4,
        "filter": "iterative positive-interaction user/item 5-core",
        "split": "calendar-day span boundaries 80% and 90%",
        "timezone": "UTC",
        "origin_day_since_unix_epoch": origin_day,
        "boundary_days": [boundary1, boundary2],
        "evaluation_users_items": "must occur in training",
        "mapping_order": "original ID string ascending",
        "upstream_commit": UPSTREAM_COMMIT,
        "removed_eval_rows": {
            phase: int(before.get(phase, 0) - after.get(phase, 0))
            for phase in ("dev", "test")
        },
    }
    return frame, users, items, metadata, protocol, sources


def validate_and_write(directory, built):
    frame, users, items, metadata, protocol, sources = built

    require(not frame.duplicated(KEYS).any(), "存在重复的用户-物品-时间交互")
    require(
        set(frame["split"]) == set(PHASES),
        "训练、验证、测试中至少一个集合为空",
    )

    # 定义所有后续 Reader 必须遵守的交互顺序。
    frame = frame.sort_values(
        ["time", "user_id", "source_order"], kind="mergesort"
    ).reset_index(drop=True)
    frame["row_id"] = range(1, len(frame) + 1)
    frame["history_length"] = frame.groupby("user_id", sort=False).cumcount()

    phase_number = frame["split"].map({"train": 0, "dev": 1, "test": 2})
    previous_phase = phase_number.groupby(frame["user_id"]).cummax()
    require(
        phase_number.eq(previous_phase).all(),
        "同一用户出现训练/验证/测试时间顺序交叉",
    )
    evaluation = frame["split"].ne("train")
    require(
        frame.loc[evaluation, "history_length"].gt(0).all(),
        "存在无历史的验证或测试交互",
    )

    training = frame.loc[frame["split"] == "train"]
    training_items = set(training["item_id"])
    stats = {
        "users": len(users),
        "items": len(items),
        "interactions": len(frame),
        "training_items": len(training_items),
        "missing_metadata_items": sum(
            not metadata.get(original) for original in items
        ),
        "splits": {},
    }

    columns = KEYS + ["row_id", "history_length"]
    for phase in PHASES:
        part = frame.loc[frame["split"] == phase, columns]
        part.to_csv(
            directory / f"{phase}.csv",
            sep="\t", index=False, lineterminator="\n",
        )
        stats["splits"][phase] = {
            "interactions": len(part),
            "users": int(part["user_id"].nunique()),
            "items": int(part["item_id"].nunique()),
            "usable_sequence_examples": int(
                part["history_length"].gt(0).sum()
            ),
        }

    write_json(directory / "user2id.json", users)
    write_json(directory / "item2id.json", items)

    # 每行对应一个真实物品，顺序为 item_id=1..N。
    with (directory / "item_metadata.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for original, item_id in sorted(
            items.items(), key=lambda pair: pair[1]
        ):
            record = {
                "item_id": item_id,
                "original_id": original,
                "seen_in_train": item_id in training_items,
                "metadata": metadata.get(original, {}),
            }
            stream.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True,
                allow_nan=False,
            ) + "\n")

    protocol.update({
        "history_order": "time ascending, then row_id ascending within user",
        "history_length": "all earlier retained interactions",
        "model_history_limit": 50,
        "evaluation": "rolling next-item, full catalog",
        "filter_policy": "past history, with current target eligible",
        "metrics": ["HR@5", "HR@10", "NDCG@5", "NDCG@10"],
        "sampled_evaluation_negatives": False,
        "padding_item_id": 0,
    })
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "stats.json", stats)

    manifest = {
        "format_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "pandas_version": pd.__version__,
        "script_sha256": sha256(Path(__file__)),
        "sources": sources,
        "files": {
            path.name: sha256(path)
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
    }
    write_json(directory / "manifest.json", manifest)
    return stats


def verify(directory):
    manifest_path = directory / "manifest.json"
    require(manifest_path.is_file(), f"找不到 {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        require(path.is_file(), f"冻结文件缺失：{path}")
        require(sha256(path) == expected, f"冻结文件已改变：{path}")

    print("VERIFIED:", directory)
    return json.loads(
        (directory / "stats.json").read_text(encoding="utf-8")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/frozen_v1")
    parser.add_argument("--raw", default="data/raw_freeze_v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    output = Path(args.out).resolve()
    raw = Path(args.raw).resolve()
    output.mkdir(parents=True, exist_ok=True)

    builders = {
        GROCERY: build_grocery,
        "MovieLens_1M": build_movielens,
    }
    for name, builder in builders.items():
        target = output / name
        if args.verify or target.exists():
            stats = verify(target)
        else:
            # 完整生成后才发布为正式目录，中断时可重新执行。
            staging = Path(tempfile.mkdtemp(
                prefix=f".{name}-building-", dir=output
            ))
            try:
                built = builder(raw / name)
                validate_and_write(staging, built)
                require(not target.exists(), f"输出目录已存在：{target}")
                staging.rename(target)
                stats = verify(target)
            except Exception:
                print("本次未完成，临时目录：", staging)
                raise

        print(name)
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()