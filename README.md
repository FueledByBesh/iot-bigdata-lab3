# Lab 3 — Smart Semaphore Control: from isolated intersection to city-scale network

Python re-implementation of _Control of Traffic Light in Isolated Intersections Using Fuzzy Neural
Network and Genetic Algorithm_ (course presentation), extended to a coordinated 3×3 grid (Part B) and
driven by real METR-LA loop-detector data from Kaggle (Part C).

## Quick start

```bash
pip install -r requirements.txt
# put the Kaggle download in data/kaggle/  (METR-LA.h5 + adj_METR-LA.pkl, see "Data" below)
python main.py              # Parts A, B, C  (~22 min; the GA sweep is ~15 of that)
python main.py --quick      # smoke run, ~1 min; writes to results/quick/, never touches the full results
python main.py --part B     # one part only (B and C reuse models/ produced by A)
python main.py --fresh      # regenerate the GA training set instead of using models/training_set.npz
python visualize.py         # live window: watch the controller manage traffic (see below)
python tests/test_core.py   # or: python -m pytest -q
python scripts/clean_metrla.py   # standalone cleaning script -> data/processed/
```

Everything is seeded (`semaphore/config.py: Config.seed`), so a rerun gives the same tables.

## Layout

| Path                                  | What it is                                                                                                                                  | Lab part      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `semaphore/config.py`                 | every parameter, marked **[PAPER]** (from the deck) or **[OURS]** (our choice)                                                              | all           |
| `semaphore/fuzzy.py`                  | membership functions, 48-rule base (paper's 6 rules reproduced exactly), readable Mamdani-product / height-defuzzification controller       | A.1           |
| `semaphore/fnn.py`                    | 5-layer FNN (fuzzification → rule → normalisation → defuzzification → output), initialised from the rule base; 4-input extension for Part B | A.2, B.3      |
| `semaphore/ga.py`                     | real-coded GA, paper parameters, fitness `f = 1/(1+E)`, stop `(1−f) ≤ ε`                                                                    | A.3           |
| `semaphore/oracle.py`                 | rollout "teacher" that produces the target extensions the GA's squared error is measured against                                            | A.3           |
| `semaphore/sim.py`                    | mesoscopic simulator for an R×C grid (1×1 = isolated intersection): finite link storage → spillback, Poisson or data-driven arrivals        | A.4, B.1      |
| `semaphore/controllers.py`            | fixed-cycle (Webster) baseline, FNN controller, demand propagation for per-node Webster plans                                               | A.5, B        |
| `semaphore/metrla.py`                 | METR-LA loader (h5py), profiling, cleaning, sensor selection, speed-drop → arrival-rate / queue mapping                                     | C.1, C.2      |
| `experiments/part_{a,b,c}.py`         | the experiments; each writes its own figures and tables                                                                                     | A, B, C       |
| `scripts/clean_metrla.py`             | standalone data-cleaning script                                                                                                             | C deliverable |
| `visualize.py`                        | live window: simulator + what the controller sees, which rules fire, what it decides                                                        | all           |
| `tests/test_core.py`                  | paper rules reproduced, untrained FNN == FLC, blocked-downstream rule, GA improves, simulator sanity                                        | —             |
| `results/figures/`, `results/tables/` | all outputs (`.png`, `.csv`, `.md`, `.json`)                                                                                                | deliverables  |
| `models/`                             | trained genomes (`fnn3_trained.npz`, `fnn4_coordinated.npz`, `fnn3_retuned_real.npz`) and cached training set                               | —             |

## Data

- **Dataset:** METR-LA on Kaggle, <https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset>
  (archive with `METR-LA.h5`, 57,038,056 bytes, and `adj_METR-LA.pkl`; file dates 2024-02-24).
- **Exact file used:** `METR-LA.h5`, SHA-256
  `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`
  (byte-identical to the original DCRNN release of METR-LA).
- **Source of the data:** Li, Yu, Shahabi, Liu, _Diffusion Convolutional Recurrent Neural Network:
  Data-Driven Traffic Forecasting_, ICLR 2018.
- The file is a pandas-0.15 HDF5; modern pandas cannot open it with `read_hdf`, so
  `metrla.load_metrla` decodes it with `h5py`.

## How the pieces fit

```
Part A   Poisson arrivals ─► 1×1 sim ─► states ─► rollout teacher ─► (x, Δt*) samples
                                                                 │
                         FLC rule base ─► FNN initial weights ─► GA (squared error) ─► trained FNN
Part B   trained FNN copied to all 9 nodes (naive)            ─┐
         + 4th fuzzy input D = neighbour's receiving-link       ├─► 3×3 sim (spillback, turns)
           occupancy (coordinated)                             ─┘
Part C   METR-LA speed ─► clean ─► speed drop ─► arrival rates ─► same sims, nothing else changed
```

## Design decisions you need to own

The submission rules say the rule design, GA configuration, coordination mechanism and the Part C
sensor-to-input mapping must be **your** decisions and explained in your own words. The code makes a
concrete, working choice for each so it runs, and each is a single parameter or function you can change.
Go through this table, change what you disagree with, and make sure you can defend what you keep.

| Decision              | What the code does                                                                                                                                    | Why (short)                                                                                                       | Where                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Input units           | all inputs **per lane**, critical (larger) approach of each phase                                                                                     | same controller works for any lane count and every grid node                                                      | `controllers.FNNController.features`                  |
| AV definition         | vehicles ≤ 10 s of free-flow travel from the stop line                                                                                                | the paper says "approaching vehicles" without a zone; 10 s = one sampling period                                  | `config.RoadConfig.detector_horizon_s`                |
| MF shapes             | trapezoids with shoulders at the ends, triangles inside; queue universe 0–20 veh/lane, AV 0–6                                                         | the paper gives term names only; 20 veh/lane ≈ storage of a 200 m link                                            | `fuzzy.py`                                            |
| Rule base             | 4×4×3 = 48 rules from `level = round(0.6·qg + 0.3·AV − 1.5·qr + 1)`                                                                                   | simplest monotone rule that reproduces all 6 paper rules exactly (asserted)                                       | `fuzzy.base_rule_level`                               |
| Inference             | product AND, height (centre-of-sets) defuzzification                                                                                                  | identical to the FNN's layers, so untrained FNN = FLC (tested)                                                    | `fuzzy.FuzzyController`                               |
| How Δt is used        | asked at g_min, then every 10 s while green ("rolling" extension); total ≤ e_max                                                                      | uses the paper's sampling time _and_ e_max                                                                        | `sim.py`, step 3                                      |
| GA genome             | 48 rule consequents + 11 MF-centre shifts (±15 % of universe)                                                                                         | "FNN weights" = layer-4 weights; shifts let the GA move term boundaries                                           | `fnn.py`                                              |
| GA operators          | tournament (k=3), BLX-0.25 crossover (p=0.5 per pair), Gaussian mutation (p=0.01 per gene), elitism 2                                                 | real-coded GA fits continuous weights; paper probabilities kept                                                   | `ga.py`, `config.GAConfig`                            |
| GA targets            | rollout teacher: try Δt ∈ {0,2,…,30}, simulate 150 s ahead over 4 Poisson futures, take the cheapest                                                  | the paper says "sample values" and "squared error" but not where samples come from                                | `oracle.py`                                           |
| Training demand       | 40 random scenarios, 100–850 veh/h/lane, total ≤ 1500                                                                                                 | must include saturated states, otherwise the paper's "qr VL → Z" rules stay untouched (see findings)              | `config.OracleConfig`, `oracle.collect_training_set`  |
| Fixed baseline        | Webster optimum cycle per intersection, clamped to the deck's 40–120 s                                                                                | a fair, tuned fixed-time plan, not a strawman                                                                     | `controllers.webster_plan`                            |
| Coordination          | **shared fuzzy input** D = occupancy of the link the green phase discharges into; Free keeps the trained consequent, Congested halves it, Blocked → 0 | spillback is the main failure of the naive copy; this is the paper's own "adjacent intersection" future-work item | `fuzzy.D`, `fuzzy.D_FACTOR`, `FNN.expand_to_4_inputs` |
| Grid                  | 200 m links, 50 km/h, 3 lanes, 7.5 m jam spacing → 80 veh storage; turns 80/10/10                                                                     | short blocks make spillback reachable; turns mix corridors                                                        | `config.RoadConfig`                                   |
| Part B scenarios      | balanced (undersaturated), rush hour (1.9× peak), incident (eastbound exits capped at 900 veh/h for 30 min)                                           | show _when_ the isolated controller breaks, not only that it can                                                  | `experiments/part_b.scenarios`                        |
| Sensor selection      | 12 sensors with ≥ 92 % data that show weekday congestion (5th-pct speed < 45 mph); most complete first                                                | real rush hours must be in the input                                                                              | `metrla.choose_sensors`                               |
| Day                   | most complete Tue–Thu (evaluation), second most complete (retuning)                                                                                   | retune and evaluate on different days to avoid in-sample results                                                  | `metrla.choose_days`                                  |
| Speed → arrivals      | congestion index from speed drop with a dead band at 0.85·v_f; λ = 250 + 500·c veh/h/lane                                                             | Greenshields flow gave higher 3 am than noon demand on this data; only real drops are used                        | `metrla.speed_to_demand`                              |
| Speed → queue         | Greenshields density above critical × 0.5 mi section                                                                                                  | the "queue from speed drops" estimate; used as a check, not fed to the controller                                 | `metrla.speed_to_demand`                              |
| Synthetic counterpart | constant Poisson rate = that sensor's daily mean                                                                                                      | same vehicle count, so the comparison isolates the input _distribution_                                           | `experiments/part_c.py`                               |

## Results

All numbers below come from `python main.py` (full run, ~22 min) and are in `results/tables/`.
They are what _this_ code produced; interpreting them is the report's job.

### Part A — isolated intersection (`a_*.png`, `a_*.md`)

- Untrained FNN output equals the plain fuzzy controller to machine precision (`fnn_flc_max_diff = 0`).
- Teacher produced 4,862 labelled decision states (80/20 train/validation split).
- GA with the paper's parameters: E fell from 0.084 to 0.033 (validation 0.037), RMS error ≈ 5.5 s.
  **The paper's stop condition (1 − f) ≤ 0.01 was never reached in 200 generations**; the teacher's
  labels are noisy (argmin over sampled futures), so E plateaus above 0.01. 29 of 48 rules ended up in
  a different linguistic output term than the initial rule base (`a_rule_base_48.md`).
- Sensitivity (`a_ga_sweep.md`, 3 seeds × 150 generations): population 50 / 220 / 500 → validation E
  0.0415 / 0.0381 / 0.0375 at ~10 / 45 / 102 s per run (diminishing returns past 220). Mutation
  0.001 / 0.01 / 0.05 / 0.1 → 0.0408 / 0.0381 / 0.0369 / 0.0363: with only 59 genes, the paper's 0.01
  mutates < 1 gene per child and the search is mutation-starved.
- Average wait per vehicle, s (5 seeds × 2 h; `a_baseline_comparison.md`):

| scenario (veh/h/lane N-S / E-W) | Fixed (Webster) | FLC (paper rules, untrained) | FNN-GA   |
| ------------------------------- | --------------- | ---------------------------- | -------- |
| low (200/150)                   | 10.5            | 11.1                         | **9.9**  |
| medium (400/300)                | 12.1            | 12.8                         | **11.6** |
| high (550/450)                  | 16.5            | 28.0                         | **14.1** |
| very high (650/550)             | 21.8            | 176.9                        | **17.9** |
| asymmetric (600/150)            | 10.4            | 13.5                         | **10.3** |

The untrained rule base collapses near saturation. Rules like "qg VL, qr VL → Z" end every green
at g_min, so lost time eats capacity (one "high" seed also blew up, hence the large error bar). An
earlier training set capped at lower demand left those rules untouched and the trained FNN still
failed at "very high"; including saturated demand in training fixed it (see the decision table).

### Part B — 3×3 network (`b_*.png`, `b_network_comparison.md`)

| scenario  | controller   | avg wait (s) | p95 wait (s) | worst node (s) | green wasted on full downstream (s/node-h) |
| --------- | ------------ | ------------ | ------------ | -------------- | ------------------------------------------ |
| balanced  | Fixed        | 19.1         | 27           | 22.5           | 0                                          |
|           | naive FNN-GA | 10.1         | 26           | 10.6           | 0                                          |
|           | coordinated  | 10.1         | 26           | 10.6           | 0                                          |
| rush hour | Fixed        | 29.0         | 85           | 33.9           | 0                                          |
|           | naive FNN-GA | 23.2         | 63           | 27.5           | 2                                          |
|           | coordinated  | 23.1         | 62           | 28.6           | 0                                          |
| incident  | Fixed        | 58.2         | 254          | 76.9           | 644                                        |
|           | naive FNN-GA | 39.4         | 161          | 54.3           | 632                                        |
|           | coordinated  | **34.1**     | **100**      | **39.1**       | **449**                                    |

- Without congestion the naive copy works well; D stays "Free", so coordinated = naive exactly.
- What goes wrong with the naive copy is **spillback**. During the incident it keeps extending green
  into links that cannot accept vehicles (632 s of wasted green per intersection-hour) because its
  own green queue looks long. The shared input D cuts green when the receiving link is full: p95 wait
  −38 %, worst intersection −28 %, and the network drains faster (`b_queue_timeseries.png`).
- Neither adaptive version builds green waves: stops per trip stay ≈ 2.3–2.8 for all controllers in
  a 3-node crossing (`stops_per_trip` column). Offsets would be needed for that.
- Scaling (`b_scaling_benchmark.md`): simulating one network-hour takes 0.2 s (3×3), 2.1 s (10×10)
  and 9.1 s (20×20). A GA whose fitness is a network simulation (220 × 200 evaluations) would take
  ~2.6 h at 3×3 and ~111 h at 20×20, before multiple seeds, with 23,600 genes if every node had its
  own FNN. Input for the B.5 discussion.

### Part C — METR-LA (`c_*.png`, `c_*.md`, `summary_queue_wait.png`)

- Profile (`c_data_profile.md`): 2012-03-01 → 2012-06-27, 5-min steps, 207 sensors × 34,272 steps,
  mph. 8.1 % of readings missing, including 2,148 timestamps (~7.5 days) where _all_ sensors are
  missing. Cleaning filled 1.2 % by interpolation and 6.9 % by time-of-week medians.
- 12 sensors selected (`c_selected_sensors.md`). Evaluation day Tue 2012-03-13; retuning day Wed 2012-03-14.
- Average wait per stop (s); synthetic = constant Poisson at the same daily mean, so the vehicle
  count is identical:

| scale  | controller                         | synthetic | real METR-LA               |
| ------ | ---------------------------------- | --------- | -------------------------- |
| single | Fixed (Webster on daily mean)      | 11.6      | **45.8** (p95 147)         |
| single | FNN-GA                             | 11.2      | 15.5 (p95 47)              |
| single | FNN-GA retuned on another real day | 11.0      | 15.3                       |
| 3×3    | Fixed                              | 13.4      | **26.2** (worst node 36.3) |
| 3×3    | naive FNN-GA                       | 10.0      | 11.9                       |
| 3×3    | coordinated                        | 10.0      | 12.0                       |

- **What changed:** real arrivals cost every controller something, but the fixed plan pays most.
  A plan sized for the daily mean cannot absorb the 06:30–09:30 rush (`c_day_3x3_network.png`).
  The adaptive controllers absorb it, with ~20 % (network) to ~40 % (single) more wait than under synthetic inputs.
- **Rules that never fired on synthetic traffic now fire:** 19 of 48, all with L/VL queues
  (`c_rule_firing_single_intersection.png`, `c_rule_firing_single.md`). The 99th-percentile green
  queue rises from 1.7 to 18.3 veh/lane (`c_input_distributions.png`), and 0.3 % of decisions exceed
  the 20 veh/lane universe. The membership functions were never exercised in that range on
  synthetic data.
- The coordination input barely activated on this day (D ≥ 0.5 in 0.2 % of decisions): the real
  peak did not cause spillback at these demand levels, so coordinated ≈ naive.
- Retuning on a different real day (same GA parameters, `c_retuned_rules.md`) changed the
  high-queue rules most (e.g. L/L/H from 27.8 s to 2.2 s; L/VL/\* up by ~10 s) and gained ~1–2 %.
  Most of the GA's synthetic training already covered the needed range.
- Sanity check (`c_queue_check.png`): the sensor's speed-drop queue estimate and the simulated
  stop-line queue peak in the same window (06:30–09:00). The simulated queue is smaller, as expected
  for an adaptive signal versus a congested freeway section.

## Live visualisation (`visualize.py`)

```bash
python visualize.py                      # Part A: isolated intersection, trained FNN-GA, very high demand
python visualize.py --part B             # Part B: 3x3 grid, exit incident, coordinated FNN-GA
python visualize.py --part B --incident-start 5 --incident-end 20  # custom incident interval
python visualize.py --part B --no-incident  # same Part B traffic, unrestricted exits
python visualize.py --part C             # Part C: 3x3 grid, real METR-LA morning rush 06:00-11:00
python visualize.py --controller flc     # A: fnn | flc | fixed      B, C: coord | naive | fixed
python visualize.py --part B --snapshot shot.png --at 1900   # save a frame (e.g. for the report)
```

The scene is simulated first (1-10 s), then played back in a window (needs `sudo apt install python3-tk`).

Part B's incident is scheduled: by default it limits each eastbound exit to
900 veh/h from minute 5 to minute 35 of playback (simulation minutes 30-60).
Use `--incident-start 5 --incident-end 20` for playback minutes 5-20. Both values
are minutes on the window's clock, with `0 <= start < end <= 60`; omitted bounds
keep their defaults. Fractional minutes are rounded to simulation seconds.
The one-hour playback window stays fixed, so the same seed gives the same
incoming traffic when the incident interval changes. Use `--no-incident` to
disable the restriction; it cannot be combined with custom incident times.
These options work with `--controller coord`, `naive`, or `fixed`. They only
affect visualization scenarios, do not change `main.py` experiments or model
files, and require no re-training.

- **Left:** the simulator. Every vehicle is drawn at its real position (grey = moving, orange = queued),
  the bar at each stop line is that approach's signal, a red-shaded link is full (spillback), and red
  `+N` counts cars that cannot enter. The blue box marks the intersection being inspected.
- **Right:** that intersection's controller. _What it sees_: qg, qr, AV (and D) with their fuzzy
  memberships. _What it thinks_: the strongest firing rules with their weights and consequents.
  _What it does_: the extension dt and what the signal does with it. Below that: the signal history
  for the last 5 min and the queue on each approach for the last 10 min.
- **Keys:** space = pause, up/down = speed, r = restart. Click any intersection to inspect its controller.

## Limitations

- Mesoscopic point-queue model: no lane changing, no queue length in metres along the link beyond
  link storage, permissive turns without conflicts, no pedestrians (as in the paper).
- The teacher is a one-step rollout with a heuristic continuation policy, not an optimal controller;
  its labels are noisy, which is why the GA's ε = 0.01 stop is rarely reached.
- The coordinated controller is not re-trained by the GA: its D-dependent consequents are set by rule
  (Free ×1, Congested ×0.5, Blocked ×0). A natural next step is a GA on the network-level fitness.
- The Part C fixed-cycle baseline is one Webster plan for the daily mean demand. A city would use
  time-of-day plans, which would narrow (not close) the gap to the adaptive controllers.
- METR-LA is freeway speed, used as a proxy for arterial entry demand. Its timing and shape are real;
  the absolute demand levels (`lam_ff`, `lam_peak`) are assumptions.

## References

- Course presentation: _Control of Traffic Light in Isolated Intersections Using Fuzzy Neural Network and Genetic Algorithm_ (source of the controller design and the [PAPER] parameters).
- Li Y., Yu R., Shahabi C., Liu Y. (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ICLR. (METR-LA)
- METR-LA dataset on Kaggle: <https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset>
- Webster F. V. (1958). Traffic Signal Settings. Road Research Technical Paper 39, HMSO.
- Greenshields B. D. (1935). A Study of Traffic Capacity. Highway Research Board Proceedings 14.
- Eshelman L. J., Schaffer J. D. (1993). Real-coded genetic algorithms and interval-schemata (BLX-α crossover).
- Libraries: NumPy, pandas, Matplotlib, h5py.
