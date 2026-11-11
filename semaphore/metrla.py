"""METR-LA loading, profiling, cleaning and sensor -> controller-input mapping (Part C).

METR-LA (Li et al., DCRNN, ICLR 2018): average speed (mph) from 207 loop
detectors on Los Angeles County freeways, 5-minute intervals, 2012-03-01 to
2012-06-27. A reading of exactly 0 means "no data" in this dataset.

The file is an old pandas HDF5 ("fixed" format, pandas 0.15). Modern pandas
cannot read it with read_hdf, so it is decoded with h5py directly.

---------------------------------------------------------------------------
Sensor-to-input mapping  [OURS - document and defend in the report]
---------------------------------------------------------------------------
METR-LA has *speed* only, no counts and no queues. The simulator generates
queues itself (closed loop: the controller's actions change the queues), so
what the sensors must supply is the *arrival process* at the grid boundary,
plus an independent sensor-side queue estimate used as a sanity check.

1. Speed drop -> congestion index (per sensor, per 5 min)
       x = v / v_f                    v_f = sensor's 95th-percentile speed
       c = clip((x_db - x) / (x_db - x_jam), 0, 1)
   with a dead band x_db = 0.85: in free flow, speed carries almost no
   information about flow (a first attempt with Greenshields' q = k v gave
   *higher* demand at 3 am than at noon, because loops read slightly lower
   night-time speeds). Only a real speed drop is treated as signal.

2. Congestion index -> arrival rate on the grid entry fed by that sensor
       lambda = lam_ff + (lam_peak - lam_ff) * c        (veh/h/lane)
   lam_ff = uncongested urban approach demand, lam_peak = rush-hour demand.
   The temporal *shape* (when rush hours start, how long they last, which
   direction peaks AM vs PM, incidents) is the real-data content; the two
   levels are assumptions.

3. Speed drop -> queue (Greenshields density above critical density)
       k = k_j (1 - x),   Q = max(0, k - k_j/2) * L_section       (veh/lane)
   i.e. vehicles stored on the sensor's section beyond what free-flowing
   traffic would hold. Reported next to the simulated queues; it is *not*
   fed to the controller (the controller sees the simulator's queues, as a
   stop-line camera would).

Known weak points: freeway sensors stand in for arterial entries; night and
midday get the same uncongested demand; Greenshields is a crude freeway fit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MappingConfig:
    k_jam: float = 185.0         # veh/mi/lane (~8.7 m per vehicle)
    vf_percentile: float = 95.0  # free-flow speed per sensor
    section_mi: float = 0.5      # length of freeway represented by one sensor
    x_deadband: float = 0.85     # v/v_f above this = no congestion signal
    x_jam: float = 0.25          # v/v_f at or below this = full congestion
    lam_ff_vphpl: float = 250.0  # uncongested approach demand, veh/h/lane
    lam_peak_vphpl: float = 750.0  # demand at full congestion, veh/h/lane
    smooth_steps: int = 3        # 3 x 5 min rolling mean
    max_gap_interp: int = 6      # interpolate gaps up to 30 min
    min_completeness: float = 0.92  # ~6 % of timestamps are network-wide outages
    congested_p05_mph: float = 45.0


def load_metrla(path) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = df.columns.astype(str)
        return df
    import h5py
    with h5py.File(path, "r") as f:
        g = f["df"]
        cols = [c.decode() for c in g["axis0"][:]]
        idx = pd.to_datetime(g["axis1"][:], unit="ns")
        vals = g["block0_values"][:]
    df = pd.DataFrame(vals, index=idx, columns=cols)
    df.index.name = "timestamp"
    return df


def profile(df: pd.DataFrame) -> dict:
    missing = (df == 0) | df.isna()
    step = pd.Series(df.index).diff().mode()[0]
    per_sensor = missing.mean()
    valid = df.where(~missing)
    return {
        "time_start": str(df.index[0]),
        "time_end": str(df.index[-1]),
        "n_timesteps": int(df.shape[0]),
        "n_sensors": int(df.shape[1]),
        "sampling_interval": str(step),
        "irregular_steps": int((pd.Series(df.index).diff().dropna() != step).sum()),
        "units": "mph (average speed per 5 min); 0 = missing",
        "missing_frac_total": float(missing.values.mean()),
        "missing_frac_sensor_median": float(per_sensor.median()),
        "missing_frac_sensor_max": float(per_sensor.max()),
        "sensors_over_10pct_missing": int((per_sensor > 0.10).sum()),
        "timesteps_all_sensors_missing": int(missing.all(axis=1).sum()),
        "speed_min_valid": float(np.nanmin(valid.values)),
        "speed_max_valid": float(np.nanmax(valid.values)),
        "speed_mean_valid": float(np.nanmean(valid.values)),
    }


def clean(df: pd.DataFrame, mc: MappingConfig = MappingConfig()):
    """zeros -> NaN; short gaps linear; long gaps -> sensor's time-of-week median."""
    x = df.replace(0.0, np.nan)
    missing0 = x.isna()
    x = x.interpolate(method="time", limit=mc.max_gap_interp, limit_area="inside")
    tow = x.index.dayofweek * 288 + x.index.hour * 12 + x.index.minute // 5
    med = x.groupby(tow).median()
    fill = med.reindex(tow).set_index(x.index)
    x = x.fillna(fill)
    x = x.ffill().bfill()
    x = x.clip(lower=1.0, upper=70.0)
    report = {
        "missing_before": float(missing0.values.mean()),
        "filled_by_interpolation": float((missing0 & df.replace(0.0, np.nan)
                                          .interpolate(method="time", limit=mc.max_gap_interp,
                                                       limit_area="inside").notna()).values.mean()),
        "missing_after": float(x.isna().values.mean()),
    }
    report["filled_by_time_of_week_median"] = report["missing_before"] - report["filled_by_interpolation"]
    return x, missing0, report


def choose_sensors(df_raw: pd.DataFrame, n: int, mc: MappingConfig = MappingConfig()):
    """Most complete sensors that actually show weekday congestion."""
    miss = ((df_raw == 0) | df_raw.isna()).mean()
    valid = df_raw.replace(0.0, np.nan)
    wk = valid[valid.index.dayofweek < 5]
    p05 = wk.quantile(0.05)
    ok = (1 - miss >= mc.min_completeness) & (p05 < mc.congested_p05_mph)
    cand = pd.DataFrame({"complete": 1 - miss, "p05": p05})[ok]
    cand = cand.sort_values(["complete", "p05"], ascending=[False, True])
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} sensors pass the selection criteria")
    return list(cand.index[:n]), cand


def choose_days(df_raw: pd.DataFrame, sensors, n_days=2):
    """Tue-Thu days with the most complete data for the chosen sensors."""
    miss = ((df_raw[sensors] == 0) | df_raw[sensors].isna()).mean(axis=1)
    by_day = miss.groupby(miss.index.normalize()).mean()
    by_day = by_day[by_day.index.dayofweek.isin([1, 2, 3])]
    counts = miss.groupby(miss.index.normalize()).size()
    by_day = by_day[counts.reindex(by_day.index) == 288]
    return [d for d in by_day.sort_values(kind="stable").index[:n_days]]


def speed_to_demand(speed: pd.DataFrame, vf: pd.Series, mc: MappingConfig = MappingConfig()):
    """Returns (lambda veh/h/lane, congestion index c, sensor queue Q veh/lane)."""
    x = (speed / vf).clip(0.0, 1.0)
    c = ((mc.x_deadband - x) / (mc.x_deadband - mc.x_jam)).clip(0.0, 1.0)
    c = c.rolling(mc.smooth_steps, min_periods=1, center=True).mean()
    lam = mc.lam_ff_vphpl + (mc.lam_peak_vphpl - mc.lam_ff_vphpl) * c
    k = mc.k_jam * (1 - x)
    Q = (k - mc.k_jam / 2).clip(lower=0) * mc.section_mi
    return lam, c, Q


def free_flow_speed(clean_df: pd.DataFrame, mc: MappingConfig = MappingConfig()):
    return clean_df.quantile(mc.vf_percentile / 100.0)


def to_per_second(lam_5min: pd.DataFrame, lanes: int) -> np.ndarray:
    """(T5, n) veh/h/lane at 5-min steps -> (n, T5*300) veh/s per approach."""
    a = lam_5min.values.T * lanes / 3600.0
    return np.repeat(a, 300, axis=1)
