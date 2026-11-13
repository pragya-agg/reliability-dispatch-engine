"""LaDe pickup data: loading, label, and leakage-safe courier-history features."""
import glob
import os

import numpy as np
import pandas as pd

RAW_GLOB = "data/raw/lade/pickup/*.parquet"
CACHE = "data/lade_features.parquet"
YEAR = "2022"  # raw timestamps have no year; only differences between them are used
TIME_COLS = ["accept_time", "time_window_start", "time_window_end", "pickup_time"]
KEY_MULT = 10**8  # courier_id * KEY_MULT + seconds gives one sortable key per (courier, time)

JOB_FEATURES = ["city", "aoi_type", "slack_h", "window_len_h", "start_gap_h", "accept_hour"]
COURIER_FEATURES = ["past_n", "past_late_rate", "open_jobs"]


def load_pickups() -> pd.DataFrame:
    cols = ["order_id", "city", "courier_id", "aoi_type", *TIME_COLS]
    df = pd.concat(
        [pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(RAW_GLOB))],
        ignore_index=True,
    )
    # Courier ids are only unique inside a city (13 ids occur in two cities), so re-key them.
    df["courier_id"] = df.groupby(["city", "courier_id"], sort=True).ngroup()
    for c in TIME_COLS:
        df[c] = pd.to_datetime(YEAR + "-" + df[c], format="%Y-%m-%d %H:%M:%S")
    # Accepted after the deadline: trivially late, and not a dispatchable decision.
    df = df[df.time_window_end >= df.accept_time]
    return df.sort_values("accept_time").reset_index(drop=True)


def _seconds(df: pd.DataFrame, col: str) -> np.ndarray:
    return (df[col] - pd.Timestamp(f"{YEAR}-01-01")).dt.total_seconds().to_numpy().astype(np.int64)


def add_label_and_job_features(df: pd.DataFrame) -> pd.DataFrame:
    hours = lambda a, b: (df[a] - df[b]).dt.total_seconds() / 3600
    df["late"] = (df.pickup_time > df.time_window_end).astype(np.int8)
    df["slack_h"] = hours("time_window_end", "accept_time")
    df["window_len_h"] = hours("time_window_end", "time_window_start")
    df["start_gap_h"] = hours("time_window_start", "accept_time")
    df["accept_hour"] = df.accept_time.dt.hour + df.accept_time.dt.minute / 60
    df["city"] = df.city.astype("category")
    df["aoi_type"] = df.aoi_type.astype("category")
    return df


def add_courier_history(df: pd.DataFrame) -> pd.DataFrame:
    """Per job, use only this courier's outcomes already known at accept time."""
    cid = df.courier_id.to_numpy(np.int64)
    acc = cid * KEY_MULT + _seconds(df, "accept_time")
    pick = cid * KEY_MULT + _seconds(df, "pickup_time")
    late = df.late.to_numpy(np.int64)

    order = np.argsort(pick, kind="stable")
    pick_sorted = pick[order]
    late_cum = np.concatenate([[0], np.cumsum(late[order])])
    acc_sorted = np.sort(acc)

    courier_start = cid * KEY_MULT
    start_p = np.searchsorted(pick_sorted, courier_start, "left")
    start_a = np.searchsorted(acc_sorted, courier_start, "left")
    # 'left': pickups strictly before accept time; a job's own pickup is never >= counted.
    end_p = np.searchsorted(pick_sorted, acc, "left")
    end_a = np.searchsorted(acc_sorted, acc, "right")

    n_done = end_p - start_p
    n_late = late_cum[end_p] - late_cum[start_p]
    df["past_n"] = n_done
    df["past_late_rate"] = np.where(n_done > 0, n_late / np.maximum(n_done, 1), np.nan)
    df["open_jobs"] = (end_a - start_a - 1) - n_done
    return df


def load_features(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    df = add_courier_history(add_label_and_job_features(load_pickups()))
    keep = ["order_id", "courier_id", "accept_time", "late", *JOB_FEATURES, *COURIER_FEATURES]
    df = df[keep]
    df.to_parquet(CACHE)
    return df


if __name__ == "__main__":
    f = load_features(use_cache=False)
    print(f.shape, "late rate", f.late.mean().round(4))
    print(f[COURIER_FEATURES].describe().round(3))
