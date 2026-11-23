import hashlib
import json
from pathlib import Path

import pandas as pd

from helpers.BaseReader import BaseReader


class FrozenSeqReader(BaseReader):
    def _read_data(self):
        folder = Path(self.prefix) / self.dataset
        manifest = json.loads(
            (folder / "manifest.json").read_text(encoding="utf-8")
        )

        for name, expected in manifest["files"].items():
            path = folder / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"冻结文件发生变化：{path}")

        # 只由冻结内容决定，便于三个模型比较数据版本。
        self.dataset_hash = hashlib.sha256(
            json.dumps(
                manifest["files"], sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

        required = [
            "user_id", "item_id", "time", "row_id", "history_length"
        ]
        frames = []
        for phase in ("train", "dev", "test"):
            frame = pd.read_csv(
                folder / f"{phase}.csv",
                sep="\t",
                usecols=required,
                dtype="int64",
            )
            frame["split"] = phase
            frames.append(frame)

        complete = pd.concat(frames, ignore_index=True)
        if complete["row_id"].duplicated().any():
            raise ValueError("row_id 跨集合重复")

        complete = complete.sort_values(
            ["user_id", "time", "row_id"],
            kind="mergesort",
        ).reset_index(drop=True)

        complete["position"] = complete.groupby(
            "user_id", sort=False
        ).cumcount()

        if not complete["position"].eq(
            complete["history_length"]
        ).all():
            raise ValueError("Reader 计算的历史与冻结记录不一致")

        phase_order = complete["split"].map({
            "train": 0, "dev": 1, "test": 2
        })
        previous = phase_order.groupby(
            complete["user_id"]
        ).cummax()
        if not phase_order.eq(previous).all():
            raise ValueError("同一用户的切分时间顺序交叉")

        self.n_users = int(complete["user_id"].max()) + 1
        self.n_items = int(complete["item_id"].max()) + 1

        if sorted(complete["item_id"].unique()) != list(
            range(1, self.n_items)
        ):
            raise ValueError("物品 ID 必须连续且从 1 开始")

        self.user_his = {}
        for row in complete.itertuples(index=False):
            self.user_his.setdefault(int(row.user_id), []).append(
                (int(row.item_id), int(row.time))
            )

        self.data_df = {}
        for phase in ("train", "dev", "test"):
            frame = complete.loc[
                complete["split"].eq(phase)
            ].drop(columns="split")
            frame = frame.sort_values("row_id").reset_index(drop=True)

            if phase != "train" and frame["position"].eq(0).any():
                raise ValueError(f"{phase} 中存在无历史样本")

            self.data_df[phase] = frame

        self.all_df = complete.drop(columns="split")