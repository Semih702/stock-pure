from __future__ import annotations

import os

import pandas as pd
import pandas.api.types as ptypes
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# from your_time_module import time_features  # <- define/import if you use timeenc == 1


class DatasetETTHour(Dataset):
    """
    ETT dataset loader.
    size = [seq_len, label_len, pred_len]
    features: "S" (single), "M" (multi), "MS" (multi->single)
    """

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: list[int] | tuple[int, int, int] | None = None,
        features: str = "S",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
    ) -> None:
        # lengths [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = int(size[0]), int(size[1]), int(size[2])

        assert flag in {"train", "val", "test"}
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScaler()

        self._read_data()

    def _read_data(self) -> None:
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Split borders (12 months train, 4 months val, 4 months test)
        # Subtract seq_len so every split can yield at least one window.
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]  # skip date
            df_data = df_raw.loc[:, cols_data]
        else:  # "S"
            df_data = df_raw.loc[:, [self.target]]

        if self.scale:
            train_slice = slice(border1s[0], border2s[0])  # train region only
            self.scaler.fit(df_data.iloc[train_slice].to_numpy())
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        # time stamps for the current split (copy to avoid SettingWithCopy)
        df_stamp = df_raw.loc[b1 : b2 - 1, ["date"]].copy()

        # Convert to datetime *before* using .dt
        date_series = pd.to_datetime(df_stamp["date"], errors="coerce", utc=False)
        df_stamp["date"] = date_series

        # Help both runtime and Pylance: ensure dtype is datetime-like
        if not ptypes.is_datetime64_any_dtype(df_stamp["date"]):
            raise TypeError("df_stamp['date'] must be datetime64[ns] after to_datetime")

        # Now .dt is a DatetimeProperties object, so month/day/... are known
        df_stamp["month"] = df_stamp["date"].dt.month  # type: ignore
        df_stamp["day"] = df_stamp["date"].dt.day  # type: ignore
        df_stamp["weekday"] = df_stamp["date"].dt.weekday  # type: ignore
        df_stamp["hour"] = df_stamp["date"].dt.hour  # type: ignore
        data_stamp = df_stamp.drop(columns=["date"]).to_numpy()

        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
