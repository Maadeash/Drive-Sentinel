# Prior results — kept as evidence, not as working code

Nothing here is on the v5 code path. It is retained because documented claims
cite it, and deleting the evidence behind a claim leaves the claim unsupported.

## `v4/`

Checkpoints and reports from the v4 pipeline (227 hand-crafted features, 5-fold
grouped CV). Cited in the root `README.md` for the v4-vs-v5 comparison, and in
`docs/accuracy_ceiling.md`.

The relevant v4 figure is **0.6620 window / 0.6605 per-recording**, on a 5-fold
split with all six sensors — including the load cell and torque shaft that a
PYNQ-Z2 cannot acquire.

## `v5_sensor_ablation/`

The sensor-availability ablation that motivated the v5 rescope: which accuracy
is reachable from each *physically obtainable* sensor group. Its finding was
that 58 of the 227 v4 features came from rig-only instruments, and that current
+ vibration alone matched or beat the full set.

## `v4_caches_other_machine/`

Two feature caches — `ds_features_cache.parquet` and
`ds_envelope_cache.parquet` — built on a **different machine, from a more
complete copy of the dataset**: 2,559 recordings against the 2,546 present here.
The extra 13 are the 12 missing KI14 runs plus the corrupt KA08 file.

They are useless to v5 (wrong feature space entirely) but are **not
reproducible on this machine**, which is the only reason they survive. They are
87 MB.

**If the KI14 gap is ever closed from the original source, delete this
directory** — it exists solely as a record that those recordings once existed in
a copy this project could read. See `docs/data_notes.md` for the full
reconciliation.
