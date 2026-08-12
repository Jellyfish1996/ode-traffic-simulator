import glob

import numpy as np
import pandas as pd

from load_pems import COLS, vehicles

# Eastbound lower deck of the Richmond-San Rafael bridge, the same span the rest
# of the report uses. These are the two stations cited in paper.tex; the
# westbound pair in load_pems.py mirrors them across the same piers.
#
# Note the deck has three lanes, not two. Model 4 is a two lane model, so lanes
# 1 and 2 are fitted and lane 3 is carried alongside for reference only.
UPSTREAM = 421237
DOWNSTREAM = 421238

# From the metadata: |73.222 - 71.146|. The report's 5.71 mi is not this pair.
SEGMENT_MILES = 2.076


# Same caching trick as the westbound loader: the district file is large and
# only two stations of it matter.
def stations_5min(date: str, ids=(UPSTREAM, DOWNSTREAM)):
    cache = f"pems_eb_{date}_stations.csv"
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


def load_segment(date: str, start: str = "15:00", end: str = "18:00"):
    df = stations_5min(date)
    up = df[df.id == UPSTREAM].set_index("ts").sort_index()
    down = df[df.id == DOWNSTREAM].set_index("ts").sort_index()
    L = SEGMENT_MILES

    out = pd.DataFrame({
        "A_1": up.l1_flow,
        "A_2": up.l2_flow,
        "N_1": vehicles(down.l1_flow, down.l1_speed, L),
        "N_2": vehicles(down.l2_flow, down.l2_speed, L),
        # Lane 3 is not part of the two lane model, only reported.
        "A_3": up.l3_flow,
        "N_3": vehicles(down.l3_flow, down.l3_speed, L),
    })

    # PeMS writes feed outages as exact zeros, which read as the road emptying
    # and refilling inside one time step. A zero flow sample also leaves the
    # speed undefined, so N is meaningless there.
    dead = ((up.l1_flow == 0) | (up.l2_flow == 0) |
            (down.l1_flow == 0) | (down.l2_flow == 0) |
            (down.l1_speed == 0) | (down.l2_speed == 0))

    out = out[~dead].between_time(start, end).dropna()
    out.insert(0, "t", np.arange(len(out), dtype=float))
    return out


if __name__ == "__main__":
    for date in ("2026_08_03", "2026_01_06"):
        seg = load_segment(date)
        print(f"\n{date}: {len(seg)} points, {SEGMENT_MILES} mi")
        print(seg.describe().loc[["mean", "std", "min", "max"]].round(1).to_string())
