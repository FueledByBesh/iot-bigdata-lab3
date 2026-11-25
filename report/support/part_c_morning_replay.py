"""Reproduce the Part C viewer's traffic and measure only its playback window."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import sys
import time
import itertools

BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['MPLCONFIGDIR'] = str(BASE / 'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from semaphore import report as R, metrla as M
from semaphore.config import DEFAULT
from semaphore.controllers import FixedController, FNNController, webster_plan, expected_side_flows
from semaphore.fnn import FNN
from semaphore.sim import GridSim

SEED = 11
DATA = ROOT / 'data/kaggle/METR-LA.h5'
visual_path = ROOT / 'visualize.py'
tree = ast.parse(visual_path.read_text())
definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
               and n.name in ('LoggedFNN', 'scene_c', '_hms')]
namespace = dict(np=np, R=R, DEFAULT=DEFAULT, FNN=FNN, GridSim=GridSim,
                 FixedController=FixedController, FNNController=FNNController,
                 webster_plan=webster_plan, expected_side_flows=expected_side_flows)
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(visual_path), 'exec'), namespace)

class ReplayArrivals:
    def __init__(self, a):
        self.a = a
    def poisson(self, rates):
        assert rates.shape == self.a.shape
        return self.a.copy()

payload = {'meta': {'visualization_sha256': hashlib.sha256(visual_path.read_bytes()).hexdigest(),
                    'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    'model_sha256': hashlib.sha256((ROOT/'models/fnn3_trained.npz').read_bytes()).hexdigest(),
                    'seed': SEED}, 'controllers': []}

for key, label in [('fixed', 'Fixed'), ('naive', 'Naive'), ('coord', 'Coord')]:
    started = time.time()
    sc = namespace['scene_c'](key, str(DATA))
    t0, t1 = sc['t0'], sc['t1']
    assert (t0, t1) == (21600, 39600)
    # Match simulate()'s RNG shape, then exclude its final extra second from metrics.
    arrival_duration = min(sc['duration'], sc['t1'] + 1)
    a = np.random.default_rng(SEED).poisson(sc['rates'][:, :arrival_duration])[:, :t1]
    arrival_hash = hashlib.sha256(a.tobytes()).hexdigest()
    if 'arrivals_sha256' in payload['meta']:
        assert arrival_hash == payload['meta']['arrivals_sha256']
    else:
        payload['meta'].update(arrivals_sha256=arrival_hash,
                               window_start_s=t0, window_end_s=t1,
                               clock_start=sc['clock'](t0), clock_end=sc['clock'](t1),
                               boundary_arrivals_window=int(a[:, t0:t1].sum()),
                               grid=[sc['rows'], sc['cols']],
                               arrival_draw_duration=arrival_duration)
        # Recover the same selected day and raw profile for the input chart.
        raw = M.load_metrla(DATA)
        sensors, _ = M.choose_sensors(raw, 12)
        day = M.choose_days(raw, sensors, 2)[0]
        visible = raw.loc[day + pd.Timedelta(seconds=t0):
                          day + pd.Timedelta(seconds=t1-300), sensors[:4]].replace(0, np.nan)
        payload['meta'].update(dataset_start=str(raw.index[0]), dataset_end=str(raw.index[-1]),
                               day=str(day.date()), sensors=sensors)
        payload['input'] = {
            'sample_times_s': (t0 + np.arange((t1-t0)//300)*300).tolist(),
            'demand_vphpl': (sc['rates'][:, t0:t1:300] * 3600 / sc['cfg'].road.lanes).tolist(),
            'raw_speed_mph': [[None if pd.isna(v) else float(v) for v in visible[s]] for s in sensors[:4]],
            'raw_sensor_ids': sensors[:4],
        }
        del raw
    print(f'{label}: selected {payload["meta"]["day"]}, 06:00–11:00; replaying midnight prefix', flush=True)
    sim = GridSim(sc['rows'], sc['cols'], sc['make'], sc['cfg'], seed=SEED,
                  exit_caps=sc['exit_caps'], record_timeseries=False)
    sim.np_rng = ReplayArrivals(a)
    all_links = [L for n in sim.nodes for L in sim.links[n]]
    previous = []
    passed_waits, all_queued, queues_only, buffers, hours = [], [], [], [], []
    bin_waits = [[] for _ in range((t1-t0)//300)]
    qsum = 0

    def observe(sm, t):
        global previous, qsum
        # Snapshot each timestamp before transfers can overwrite Vehicle.t_stop.
        if t >= t0:
            for L, vehicles in previous:
                still_here = set(itertools.chain(L.queue, L.transit))
                for vehicle, stop_time in vehicles:
                    if stop_time <= t and vehicle not in still_here:
                        wait = t - stop_time
                        passed_waits.append(wait)
                        bin_waits[(t-t0)//300].append(wait)
            q = sum(len(L.queue) for L in all_links)
            qsum += q
            if (t-t0) % sc['every'] == 0:
                buf = sum(len(L.buffer) for L in sm.entries)
                hours.append(t/3600)
                queues_only.append(q)
                buffers.append(buf)
                all_queued.append(q+buf)
        previous = [(L, [(v, int(v.t_stop)) for v in itertools.chain(L.queue, L.transit)])
                    for L in all_links]
        if t and t % 3600 == 0:
            print(f'  {label} {t//3600:02d}:00', flush=True)

    m = sim.run(t1, sc['rates'], warmup=t0, hooks=[observe])
    censored = [t1-v.t_stop for L in all_links for v in L.queue if v.t_stop >= t0]
    recovered = np.asarray(passed_waits+censored)
    np.testing.assert_array_equal(np.sort(recovered), np.sort(m['_waits']))
    np.testing.assert_allclose(qsum / ((t1-t0) * 4 * len(sim.nodes)), m['avg_queue'], rtol=0, atol=1e-12)
    ctrl = {'name': label,
            'metrics': {k: float(m[k]) for k in ('avg_queue','avg_wait','p95_wait','throughput_vph',
                                               'blocked_green_s','worst_node_wait','avg_delay_all','unfinished')},
            'hours': hours, 'queued_total': all_queued, 'stopline_queue': queues_only, 'boundary_buffers': buffers,
            'wait_bin_centres_hours': ((t0 + np.arange(len(bin_waits))*300 + 150)/3600).tolist(),
            'wait_mean_5min': [float(np.mean(v)) if v else None for v in bin_waits],
            'wait_p95_5min': [float(np.percentile(v,95)) if v else None for v in bin_waits],
            'passages_per_bin': [len(v) for v in bin_waits],
            'completed_passages': len(passed_waits), 'censored_waits': len(censored),
            'peak_queued_total': max(all_queued), 'wall_seconds': time.time()-started}
    if key == 'coord':
        ds = [v['x'][3] for values in sc['log'].values() for v in values if t0 <= v['t'] < t1]
        ctrl['D_ge_0_5_fraction'] = float(np.mean(np.asarray(ds) >= .5))
    payload['controllers'].append(ctrl)
    (BASE/'part_c_morning_results.json').write_text(json.dumps(payload, indent=2))
    print(label, ctrl['metrics'], f'validated {len(recovered)} waits; {time.time()-started:.1f}s', flush=True)
    del sc, sim, previous, a

fixed_wait = payload['controllers'][0]['metrics']['avg_wait']
with (BASE/'part_c_morning_metrics.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['controller','avg_queue','avg_wait','p95_wait','wait_reduction_vs_fixed_pct',
                     'throughput_vph','worst_node_wait','blocked_green_s','peak_queued_including_buffers'])
    for c in payload['controllers']:
        m = c['metrics']
        writer.writerow([c['name'],m['avg_queue'],m['avg_wait'],m['p95_wait'],
                         100*(1-m['avg_wait']/fixed_wait),m['throughput_vph'],m['worst_node_wait'],
                         m['blocked_green_s'],c['peak_queued_total']])
print(BASE/'part_c_morning_results.json', flush=True)
