# DriveSentinel-FPGA -- Sensor Availability Ablation (v5)

> Native 3-class (KB23/24/27 removed from train and test).
> Identical folds (`ds_folds_v4.json`), identical models, identical protocol
> as `results_v4a_3class.md` Section C. Only the feature subset changes.

## Which physical sensor each configuration needs

| Config | Sensors | Features | Buildable on the proposed PYNQ-Z2 BOM? |
|---|---|---|---|
| A | 2x phase current | 118 | **yes** |
| B | + temperature | | yes, one extra transducer |
| C | + vibration (accelerometer + charge amp) | | only with the 'future scope' sensor |
| D | force + torque + shaft speed only | | **no** -- load cell and torque shaft are rig-only |
| E | everything (what results_v4 reports) | | **no** |

## Results

| Config | Model | Features | Accuracy | Macro-F1 | Per-recording acc | Majority baseline |
|---|---|---|---|---|---|---|
| A_current_only | RF | 118 | 0.6177 +/- 0.1401 | 0.5938 | 0.6277 +/- 0.1425 | 0.4150 |
| A_current_only | MLP | 118 | 0.4889 +/- 0.0938 | 0.3986 | 0.4841 +/- 0.1094 | 0.4150 |
| B_current_temp | RF | 120 | 0.6225 +/- 0.1307 | 0.5977 | 0.6231 +/- 0.1337 | 0.4150 |
| B_current_temp | MLP | 120 | 0.4597 +/- 0.0963 | 0.3702 | 0.4687 +/- 0.0976 | 0.4150 |
| C_current_vibration | RF | 169 | 0.6419 +/- 0.0468 | 0.6328 | 0.6429 +/- 0.0512 | 0.4150 |
| C_current_vibration | MLP | 169 | 0.7063 +/- 0.1127 | 0.6706 | 0.7161 +/- 0.1182 | 0.4150 |
| D_mechanical_only | RF | 58 | 0.5080 +/- 0.1396 | 0.4653 | 0.5117 +/- 0.1557 | 0.4150 |
| D_mechanical_only | MLP | 58 | 0.5340 +/- 0.0681 | 0.4958 | 0.5483 +/- 0.0645 | 0.4150 |
| E_all_sensors | RF | 227 | 0.6496 +/- 0.0721 | 0.6389 | 0.6491 +/- 0.0846 | 0.4150 |
| E_all_sensors | MLP | 227 | 0.6620 +/- 0.0956 | 0.6270 | 0.6605 +/- 0.0906 | 0.4150 |

## Per-fold accuracy

| Config | Model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|---|
| A_current_only | RF | 0.8081 | 0.6157 | 0.5411 | 0.4042 | 0.7196 |
| A_current_only | MLP | 0.4810 | 0.3185 | 0.5325 | 0.5116 | 0.6006 |
| B_current_temp | RF | 0.8229 | 0.5492 | 0.5666 | 0.4561 | 0.7177 |
| B_current_temp | MLP | 0.5651 | 0.3464 | 0.5347 | 0.3409 | 0.5114 |
| C_current_vibration | RF | 0.7060 | 0.6669 | 0.6449 | 0.5644 | 0.6272 |
| C_current_vibration | MLP | 0.8011 | 0.6363 | 0.6706 | 0.5564 | 0.8670 |
| D_mechanical_only | RF | 0.6504 | 0.3441 | 0.3913 | 0.4608 | 0.6934 |
| D_mechanical_only | MLP | 0.5347 | 0.4103 | 0.5521 | 0.6185 | 0.5546 |
| E_all_sensors | RF | 0.7647 | 0.5711 | 0.6926 | 0.5819 | 0.6379 |
| E_all_sensors | MLP | 0.7265 | 0.6012 | 0.5983 | 0.5655 | 0.8183 |
