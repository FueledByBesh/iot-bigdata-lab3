"""Reproduce Part C teacher samples and compare with Part A synthetic labels.

Run from the repository root: .venv/bin/python report/support/reconstruct_part_c_teacher.py
Uses the project data and seed; does not modify either trained model.
"""
from pathlib import Path
from dataclasses import replace
import csv
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from semaphore import metrla as M
from semaphore.config import DEFAULT
from semaphore.oracle import collect_on_rates
from semaphore.fnn import FNN
from semaphore.ga import mse

OUT = ROOT / 'report/support'
if '--use-temp-cache' in sys.argv:
    z = np.load('/private/tmp/lab3_metr_teacher_set.npz')
    X, y = z['X'], z['y']
    sensors = ['717608', '767542', '718066', '767351']
else:
    raw = M.load_metrla(ROOT / 'data/kaggle/METR-LA.h5')
    mc = M.MappingConfig()
    clean, _, _ = M.clean(raw, mc)
    sensors, _ = M.choose_sensors(raw, 12, mc)
    eval_day, train_day = M.choose_days(raw, sensors, 2)
    assert str(train_day.date()) == '2012-03-14'
    vf = M.free_flow_speed(clean[sensors], mc)
    train_speeds = clean.loc[train_day:train_day + pd.Timedelta('23:55:00'), sensors]
    lam, _, _ = M.speed_to_demand(train_speeds, vf, mc)
    rates = M.to_per_second(lam[sensors[:4]], DEFAULT.road.lanes)[:, :86400]
    cfg = replace(DEFAULT, road=replace(DEFAULT.road, turn_probs=(1.0, 0.0, 0.0)))
    X, y = collect_on_rates(cfg, rates, 86400, seed=DEFAULT.seed)
assert len(y) == 5671
net = FNN.load(ROOT / 'models/fnn3_retuned_real.npz')
error = float(mse(net, net.genome[None], X, y)[0])
with open(ROOT / 'results/tables/c_summary.json') as f:
    recorded_error = json.load(f)['retune']['train_E']
assert abs(error - recorded_error) < 1e-12
np.savez(OUT / 'part_c_teacher_reconstructed.npz', X=X, y=y)

synthetic = np.load(ROOT / 'models/training_set.npz')
ys = synthetic['y']
def stats(a):
    return {'n': int(len(a)), 'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'zero_count': int(np.sum(a == 0)), 'zero_percent': float(100*np.mean(a == 0)),
            'counts': {str(int(k)):int(v) for k, v in zip(*np.unique(a, return_counts=True))}}
summary = {'real': stats(y), 'synthetic': stats(ys),
           'provenance': {'real_day':'2012-03-14','real_duration_s':86400,
                          'seed':DEFAULT.seed,'sensor_ids':sensors[:4],
                          'reconstructed_train_E':error,'recorded_train_E':recorded_error,
                          'note':'Different scenario populations; not paired situations.'},
           'examples': []}
for target in (0, 2, 8, 14, 20, 30):
    idxs = np.flatnonzero(y == target)
    i = int(idxs[len(idxs)//2])
    summary['examples'].append({'sample_index':i,'qg':float(X[i,0]), 'qr':float(X[i,1]),
                                'AV':float(X[i,2]), 'teacher_s':float(y[i]),
                                'real_fnn_s':float(net(*X[i]))})
(OUT/'part_c_teacher_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
with open(OUT/'part_c_teacher_distribution.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['extension_s','synthetic_count','synthetic_percent','metr_count','metr_percent'])
    for t in range(0,31,2):
        writer.writerow([t,int((ys==t).sum()),100*float((ys==t).mean()),int((y==t).sum()),100*float((y==t).mean())])
print('Reconstructed',len(y),'labels; training error',error, flush=True)
