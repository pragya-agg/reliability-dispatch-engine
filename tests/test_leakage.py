"""Courier history must use only pickups finished strictly before the job is accepted."""
import numpy as np
import pandas as pd

from src.lade_data import add_courier_history


def test_courier_history_uses_only_the_past():
    rng = np.random.default_rng(0)
    n = 400
    accept = pd.Timestamp("2022-06-01") + pd.to_timedelta(rng.integers(0, 5 * 86400, n), unit="s")
    df = pd.DataFrame({"courier_id": rng.integers(0, 6, n), "accept_time": accept,
                       "pickup_time": accept + pd.to_timedelta(rng.integers(60, 6 * 3600, n), unit="s"),
                       "late": rng.integers(0, 2, n)})
    out = add_courier_history(df.copy())
    for i in range(n):
        past = df[(df.courier_id == df.courier_id[i]) & (df.pickup_time < df.accept_time[i])]
        accepted = df[(df.courier_id == df.courier_id[i]) & (df.accept_time <= df.accept_time[i])]
        assert out.past_n[i] == len(past)
        assert out.open_jobs[i] == len(accepted) - 1 - len(past)
        if len(past):
            assert np.isclose(out.past_late_rate[i], past.late.mean())
        else:
            assert np.isnan(out.past_late_rate[i])
