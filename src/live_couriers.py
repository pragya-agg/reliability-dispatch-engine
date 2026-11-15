"""Live risk monitor for human couriers: replays one real day of LaDe-P pickups minute by minute.

Everything shown is recorded data: which courier accepted which job, when, where, the time
window and when it was actually picked up. The model's late-risk score for each job comes from
the test period (the model never trained on these days). The monitor does not re-dispatch jobs,
because the outcome of giving a job to a different courier is not recorded anywhere.
"""
import glob
import os

import numpy as np
import pandas as pd

PRED = "data/lade_test_predictions.parquet"
RAW = "data/raw/lade/pickup/pickup_{code}-*.parquet"
CODE = {"Shanghai": "sh", "Hangzhou": "hz", "Chongqing": "cq", "Yantai": "yt", "Jilin": "jl"}
FLAG_SHARE = 0.10   # flag the riskiest 10% of jobs (threshold from the whole test period, no labels used)


def _minutes(ts: pd.Series, day: pd.Timestamp) -> np.ndarray:
    return ((ts - day).dt.total_seconds() // 60).astype(int).to_numpy()


def risk_threshold() -> float:
    return float(pd.read_parquet(PRED, columns=["p_late"]).p_late.quantile(1 - FLAG_SHARE))


def load_day(city: str) -> pd.DataFrame:
    """The busiest test-period day for this city, with locations and times. Cached after first use."""
    cache = f"data/live_couriers_{CODE[city]}.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    pred = pd.read_parquet(PRED)
    pred = pred[pred.city == city]
    days = pred.accept_time.dt.normalize()
    day = days.value_counts().idxmax()
    pred = pred[days == day]
    raw = pd.read_parquet(glob.glob(RAW.format(code=CODE[city]))[0],
                          columns=["order_id", "lng", "lat", "time_window_end", "pickup_time"])
    raw = raw[raw.order_id.isin(pred.order_id)]
    for c in ("time_window_end", "pickup_time"):
        raw[c] = pd.to_datetime("2022-" + raw[c], format="%Y-%m-%d %H:%M:%S")
    d = pred.merge(raw, on="order_id")
    d["day"] = day
    d = d.sort_values("accept_time").reset_index(drop=True)
    d.to_parquet(cache)
    return d


class CourierMonitor:
    def __init__(self, city: str, threshold: float):
        d = load_day(city)
        self.city, self.thr, self.day = city, threshold, d.day.iloc[0]
        self.d = d
        self.acc = _minutes(d.accept_time, self.day)
        self.pick = _minutes(d.pickup_time, self.day)
        self.due = _minutes(d.time_window_end, self.day)
        self.late = d.late.to_numpy().astype(int)
        self.risk = d.p_late.to_numpy()
        self.flag = self.risk >= threshold
        self.courier = d.courier_id.to_numpy()
        self.n = len(d)
        self.now = int(self.acc.min()) - 1
        self.end = int(min(self.pick.max(), 2 * 1440)) + 1   # stop after two days; a few pickups happen days later
        lo, hi = np.percentile(d.lng, [1, 99]), np.percentile(d.lat, [1, 99])
        self.bounds = {"lng": [float(lo[0]), float(lo[1])], "lat": [float(hi[0]), float(hi[1])]}
        self.track = d.p_history.to_numpy()

    def meta(self) -> dict:
        d = self.d
        return {"city": self.city, "day": str(self.day.date()), "jobs": self.n, "threshold": round(self.thr, 4),
                "base_rate": round(float(self.late.mean()), 4), "start": int(self.now), "end": self.end,
                "bounds": self.bounds, "couriers": int(d.courier_id.nunique()),
                "points": {"lng": d.lng.round(5).tolist(), "lat": d.lat.round(5).tolist()}}

    def step(self, minutes: int = 1) -> dict:
        t0, t1 = self.now, min(self.now + minutes, self.end)
        self.now = t1
        ev = []
        for j in np.nonzero((self.acc > t0) & (self.acc <= t1))[0]:
            ev.append({"type": "accept", "j": int(j), "t": int(self.acc[j]), "courier": int(self.courier[j]),
                       "risk": round(float(self.risk[j]), 4), "flag": bool(self.flag[j]), "due": int(self.due[j]),
                       "track": round(float(self.track[j]), 4), "open_jobs": int(self.d.open_jobs.iloc[j])})
        for j in np.nonzero(self.late.astype(bool) & (self.due > t0) & (self.due <= t1))[0]:
            ev.append({"type": "overdue", "j": int(j), "t": int(self.due[j]), "courier": int(self.courier[j]),
                       "flag": bool(self.flag[j]), "risk": round(float(self.risk[j]), 4)})
        for j in np.nonzero((self.pick > t0) & (self.pick <= t1))[0]:
            ev.append({"type": "pickup", "j": int(j), "t": int(self.pick[j]), "courier": int(self.courier[j]),
                       "late": bool(self.late[j]), "flag": bool(self.flag[j]), "late_by": int(self.pick[j] - self.due[j])})
        ev.sort(key=lambda e: e["t"])
        return {"now": self.now, "finished": self.now >= self.end, "events": ev, "stats": self.stats()}

    def stats(self) -> dict:
        t = self.now
        accepted = self.acc <= t
        picked = self.pick <= t
        open_ = accepted & ~picked
        late_known = (picked & self.late.astype(bool)) | (open_ & (self.due < t))
        flagged = accepted & self.flag
        caught = late_known & self.flag
        top = pd.DataFrame({"courier": self.courier[open_], "risk": self.risk[open_], "flag": self.flag[open_],
                            "overdue": self.due[open_] < t})
        busiest = (top.groupby("courier").agg(open=("risk", "size"), flagged=("flag", "sum"), overdue=("overdue", "sum"),
                                              max_risk=("risk", "max"))
                   .sort_values(["flagged", "overdue", "max_risk"], ascending=False).head(12).reset_index())
        known = picked | (open_ & (self.due < t))       # outcome already known: picked up, or window already missed
        return {"known": int(known.sum()), "flagged_known": int((known & self.flag).sum()),
                "accepted": int(accepted.sum()), "open": int(open_.sum()), "picked": int(picked.sum()),
                "late": int(late_known.sum()), "flagged": int(flagged.sum()), "caught": int(caught.sum()),
                "overdue_open": int((open_ & (self.due < t)).sum()),
                "watchlist": busiest.astype({"flagged": int, "overdue": int}).round(4).to_dict("records")}
