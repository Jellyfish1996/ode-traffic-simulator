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


def segment_length(upstream: int = UPSTREAM, downstream: int = DOWNSTREAM):
    meta = pd.read_csv(glob.glob("d04_text_meta_*.txt")[0], sep="\t")
    pm = meta.set_index("ID").Abs_PM
    return abs(pm[downstream] - pm[upstream])


# Pulls the two stations out of the full district file and caches them, since
# the raw file is 150 MB and only a few hundred rows of it matter.
def raw_stations(date: str, ids=(UPSTREAM, DOWNSTREAM)):
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


# Vehicles in the segment, from q = rho * v. Flow is per 5 minutes, so the
# factor of 12 converts it to vehicles per hour before dividing by mph.
def vehicles(flow: pd.Series, speed: pd.Series, length: float):
    return flow * 12 / speed * length


def load_segment(date: str, start: str = "05:00", end: str = "08:00"):
    df = raw_stations(date)
    L = segment_length()

    up = df[df.id == UPSTREAM].set_index("ts").sort_index()
    down = df[df.id == DOWNSTREAM].set_index("ts").sort_index()

    out = pd.DataFrame({
        # Arrivals onto each lane of the segment, in vehicles per 5 minutes.
        "A_1": up.l1_flow,
        "A_2": up.l2_flow,
        # Cars currently in each lane of the segment.
        "N_1": vehicles(down.l1_flow, down.l1_speed, L),
        "N_2": vehicles(down.l2_flow, down.l2_speed, L),
    })

    # PeMS reports feed outages as exact zeros rather than as missing data. Left
    # in, they look like the road emptying and refilling within one time step.
    dead = ((up.l1_flow == 0) & (up.l2_flow == 0)) | \
           ((down.l1_flow == 0) & (down.l2_flow == 0))
    out = out[~dead]

    out = out.between_time(start, end).dropna()
    # One time step is one 5 minute sample, matching the units used in the report.
    out.insert(0, "t", np.arange(len(out), dtype=float))
    return out


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2026_08_03"
    seg = load_segment(date)
    print(f"segment {UPSTREAM} -> {DOWNSTREAM}, {segment_length():.3f} mi, {date}")
    print(seg.describe().to_string())
    print(seg.head(12).to_string())
