import glob
import sys

import numpy as np
import pandas as pd

# Westbound upper deck of the Richmond-San Rafael bridge. Two lanes, both
# loops observed, and no ramps between the two stations, so every car past the
# upstream detector reaches the downstream one.
UPSTREAM = 421260
DOWNSTREAM = 421258

COLS = ["ts", "id", "district", "fwy", "dir", "lane_type", "length", "samples",
        "pct_obs", "flow", "occ", "speed"]
for _n in range(1, 9):
    COLS += [f"l{_n}_samples", f"l{_n}_flow", f"l{_n}_occ", f"l{_n}_speed",
             f"l{_n}_obs"]

# The raw product is 30 second samples with no quality flags, and unlike the
# 5 minute rollup its speeds are measured rather than estimated.
RAW_COLS = ["ts", "id"] + [f"l{n}_{q}" for n in range(1, 9)
                           for q in ("flow", "occ", "speed")]

# One time step is 5 minutes throughout, matching the units in the report, so
# k and lambda stay comparable between the two products.
STEP_MINUTES = 5.0


def segment_length(upstream: int = UPSTREAM, downstream: int = DOWNSTREAM):
    meta = pd.read_csv(glob.glob("d04_text_meta_*.txt")[0], sep="\t")
    pm = meta.set_index("ID").Abs_PM
    return abs(pm[downstream] - pm[upstream])


# Pulls the two stations out of the full district file and caches them, since
# the district file is 150 MB and only a few hundred rows of it matter.
def stations_5min(date: str, ids=(UPSTREAM, DOWNSTREAM)):
    cache = f"pems_{date}_stations.csv"
    try:
        return pd.read_csv(cache, parse_dates=["ts"])
    except FileNotFoundError:
        pass

    matches = glob.glob(f"d04_text_station_5min_{date}.txt*")
    assert matches, f"no station file for {date}"
    df = pd.read_csv(matches[0], names=COLS, header=None, usecols=range(52),
                     low_memory=False, parse_dates=["ts"])
    df = df[df.id.isin(ids)]
    df.to_csv(cache, index=False)
    return df


# The raw file is ~700 MB, so filter it in chunks and cache the two stations.
def stations_raw(date: str, ids=(UPSTREAM, DOWNSTREAM)):
    cache = f"pems_raw_{date}_stations.csv"
    try:
        return pd.read_csv(cache, parse_dates=["ts"])
    except FileNotFoundError:
        pass

    matches = glob.glob(f"d04_text_station_raw_{date}.txt*")
    assert matches, f"no raw file for {date}"
    keep = []
    for chunk in pd.read_csv(matches[0], names=RAW_COLS, header=None,
                             usecols=range(26), low_memory=False,
                             parse_dates=["ts"], chunksize=1_000_000):
        keep.append(chunk[chunk.id.isin(ids)])
    df = pd.concat(keep)
    df.to_csv(cache, index=False)
    return df


# Vehicles in the segment, from q = rho * v. Flow is counted per sample, so
# bins_per_hour converts it to vehicles per hour before dividing by mph.
def vehicles(flow: pd.Series, speed: pd.Series, length: float,
             bins_per_hour: float = 12):
    return flow * bins_per_hour / speed * length


def _segment(df: pd.DataFrame, bins_per_hour: float, start: str, end: str):
    L = segment_length()
    up = df[df.id == UPSTREAM].set_index("ts").sort_index()
    down = df[df.id == DOWNSTREAM].set_index("ts").sort_index()

    # Arrivals are quoted per 5 minutes and t is counted in 5 minute steps,
    # whatever the sampling interval of the underlying product.
    per_step = bins_per_hour / 12
    dt = 12 / bins_per_hour

    out = pd.DataFrame({
        "A_1": up.l1_flow * per_step,
        "A_2": up.l2_flow * per_step,
        "N_1": vehicles(down.l1_flow, down.l1_speed, L, bins_per_hour),
        "N_2": vehicles(down.l2_flow, down.l2_speed, L, bins_per_hour),
    })

    # PeMS reports feed outages as exact zeros rather than as missing data. Left
    # in, they look like the road emptying and refilling within one time step.
    # A zero flow sample also leaves the speed undefined, so N is meaningless.
    dead = ((up.l1_flow == 0) | (up.l2_flow == 0) |
            (down.l1_flow == 0) | (down.l2_flow == 0) |
            (down.l1_speed == 0) | (down.l2_speed == 0))
    out = out[~dead]

    out = out.between_time(start, end).dropna()
    out.insert(0, "t", np.arange(len(out), dtype=float) * dt)
    return out


def load_segment(date: str, start: str = "05:00", end: str = "08:00"):
    return _segment(stations_5min(date), 12, start, end)


def load_segment_raw(date: str, start: str = "05:00", end: str = "08:00"):
    return _segment(stations_raw(date), 120, start, end)


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026_08_03"
    print(f"segment {UPSTREAM} -> {DOWNSTREAM}, {segment_length():.3f} mi, {date}")
    for name, loader in [("5 minute", load_segment), ("30 second", load_segment_raw)]:
        try:
            seg = loader(date)
        except AssertionError as e:
            print(f"\n{name}: {e}")
            continue
        print(f"\n{name}: {len(seg)} points")
        print(seg.describe().loc[["mean", "std", "min", "max"]].to_string())
