# Dataset Runs — SOLAQUA

> Reference: Aquaculture robotics public datasets
>
> - Title: Aquaculture robotics public datasets
> - Created by: Sveinung.ohrem@sintef.no
> - Date created: 2024-06-24 13:29
> - Date modified: 2025-04-02 18:04
> - Customer organization: (not specified)
> - Source / dataset page: https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6

This document describes the dataset runs collected during sea trials. It lists the available data types, the recorded dataset runs with timestamps and actions performed, details about the net-following experiments, abbreviations, and environmental / cage metadata.

## Data features (datasets)

The dataset contains ROV and environmental data collected during full-scale fish farm trials. Available data types include:

- IMU
- Gyroscope
- DVL (Doppler Velocity Log)
- USBL
- Multibeam sonar
- Ping360 sonar
- Mono camera
- Stereo camera
- Depth
- Pressure
- Temperature


## Dataset runs — filename (timestamp) and action performed

Calibration runs (stereo camera):

| Filename (timestamp) | Action performed |
|---|---|
| 2024-08-20_13-39-34 | Calibration of stereo camera |
| 2024-08-20_13-40-35 | Calibration of stereo camera |
| 2024-08-20_13-42-51 | Calibration of stereo camera |

Manual-control runs:

| Filename (timestamp) | Action performed |
|---|---|
| 2024-08-20_13-55-34 | Manual control - shallow |
| 2024-08-20_13-57-42 | Manual control - shallow |
| 2024-08-20_14-16-05 | Manual control - deeper |
| 2024-08-20_14-22-12 | Manual control - deeper |
| 2024-08-20_14-24-35 | Manual control - deeper |
| 2024-08-20_14-31-29 | Manual control - shallow |


Net-following session (continuous):

| Filename (timestamp) | Action performed |
|---|---|
| 2024-08-20_14-34-07 — 2024-08-20_18-52-15 | Net following (various distances, depths and velocities) |

The net following experiments were performed with varying net distances, depths and velocities as described below.


## Abbreviations

- D0: Initial desired distance to net [m]
- D1: Final desired distance to net [m]
- Z: Depth [m]
- V: Net-relative velocity [m/s], horizontal
- Q: Heading-angle offset from net [deg]


## Net-following sets (somewhat inconsistent in net distance)

These runs include net following but show variations in the desired/achieved net distance.

| Filename | D0 | D1 | Z | V | Q |
|---|---:|---:|---:|---:|---:|
| 2024-08-20_14-34-07 | 1.5 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_14-36-22 | 1.5 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_14-38-37 | 2.0 | 2.0 | 2 | 0.2 | 0 |
| 2024-08-20_14-49-47 | 2.0 | 2.0 | 2 | 0.2 | 0 |
| 2024-08-20_14-54-52 | 2.0 | 2.0 | 2 | 0.2 | 0 |
| 2024-08-20_14-57-38 | 2.0 | 1.1 | 2 | 0.2 | 0 |
| 2024-08-20_15-00-24 | 1.5 | 1.5 | 5 | 0.2 | 0 |
| 2024-08-20_15-05-53 | 1.0 | 1.5 | 5 | 0.2 | 0 |
| 2024-08-20_15-09-34 | 1.5 | x | 5 | 0.2 | 0 |
| 2024-08-20_15-12-51 | 1.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_15-14-40 | 1.4 | 1.9 | 5 | 0.1 | 0 |
| 2024-08-20_15-18-27 | 1.4 | 1.4 | 5 | 0.3 | 0 |
| 2024-08-20_15-20-29 | 1.4 | 1.4 | 5 | 0.3 | 0 |

Notes: for 2024-08-20_15-09-34, `x` indicates the final distance alternated between 1.8, 2.1 and 1.1 m during the run.


## Net-following sets (consistent net distance)

These runs show consistent net distance settings.

| Filename | D0 | D1 | Z | V | Q |
|---|---:|---:|---:|---:|---:|
| 2024-08-20_16-34-34 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-37-15 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-39-23 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-43-25 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-45-21 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-51-57 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-47-54 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_16-54-36 | 1.0 | 1.5 | 2 | 0.1 | 0 |
| 2024-08-20_16-57-46 | 1.0 | 1.5 | 2 | 0.1 | 0 |
| 2024-08-20_17-02-00 | 1.0 | 1.5 | 2 | 0.1 | 0 |
| 2024-08-20_17-04-52 | 0.5 | 1.0 | 2 | 0.1 | 0 |
| 2024-08-20_17-08-14 | 0.5 | 1.0 | 2 | 0.1 | 0 |
| 2024-08-20_17-11-14 | 0.5 | 1.0 | 2 | 0.1 | 0 |
| 2024-08-20_17-14-36 | 1.0 | 1.5 | 2 | 0.3 | 0 |
| 2024-08-20_17-22-40 | 1.0 | 1.5 | 2 | 0.3 | 0 |
| 2024-08-20_17-31-58 | 1.0 | 1.5 | 2 | 0.3 | 0 |
| 2024-08-20_17-34-52 | 1.0 | 1.5 | 2 | 0.3 | 0 |
| 2024-08-20_17-37-08 | 1.0 | 1.5 | 2 | 0.3 | 0 |
| 2024-08-20_17-39-32 | 1.0 | 1.5 | 5 | 0.2 | 0 |
| 2024-08-20_17-40-54 | 1.0 | 1.5 | 5 | 0.2 | 0 |
| 2024-08-20_17-47-49 | 0.5 | 1.0 | 5 | 0.2 | 0 |
| 2024-08-20_17-50-22 | 0.5 | 1.0 | 5 | 0.2 | 0 |
| 2024-08-20_17-53-06 | 0.5 | 1.0 | 5 | 0.2 | 0 |
| 2024-08-20_17-55-40 | 1.0 | 1.5 | 5 | 0.2 | 0 |
| 2024-08-20_17-57-55 | 0.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_18-01-46 | 0.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_18-05-42 | 0.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_18-07-47 | 0.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_18-09-52 | 0.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_18-12-20 | 0.5 | 1.0 | 5 | 0.1 | 0 |
| 2024-08-20_18-38-53 | 0.5 | 1.0 | 5 | 0.2 | 0 |
| 2024-08-20_18-41-02 | 0.5 | 1.0 | 5 | 0.2 | 0 |
| 2024-08-20_18-47-40 | 1.0 | 1.0 | 2 | 0.2 | 10 |
| 2024-08-20_18-53-59 | 1.0 | 1.0 | 2 | 0.2 | 10 |
| 2024-08-20_18-50-22 | 1.0 | 1.5 | 2 | 0.2 | 0 |
| 2024-08-20_18-52-15 | 1.0 | 1.5 | 2 | 0.2 | 0 |


## Additional NFH (Net Following Horizontal) datasets (other date)

These sets were gathered on a different date and include measurements from two DVLs (Waterlinked A50 and Nortek Nucleus 1000).

- 2024-08-22_14-06-43 — NFH, 2 m depth. Dist. 0.5 m to 1 m. 0.2 m/s speed
- 2024-08-22_14-29-05 — NFH, 2 m depth. Dist. 0.6 m to 0.8 m. 0.1 m/s speed
- 2024-08-22_14-47-39 — NFH, 2 m depth. Dist. 0.6 m (steady). 0.1 m/s speed
- 2024-08-22_14-48-39 — NFH, 2 m depth. Dist. 0.6 m (steady). 0.1 m/s speed

The following run contains heading changes and direction changes during net following:

- 2024-08-22_14-50-14 — NFH, 2 m depth, dist. 0.6 m, 0.1 m/s speed; includes changes in heading offset and direction.


## ROS messages and auxiliary files

- `SOLAQUA_msg_files` contains most (if not all) relevant ROS messages recorded for the datasets. Use these for message-level replay and full-scope analysis.


## Environmental data (approx.)

- Waves: N/A
- Current: 0.04–0.2 m/s
- Wind: 6 m/s
- Air temperature: 14 °C
- Weather: Rain


## Biomass data (approx.)

- Number of fish: ~188,000
- Average weight: ~3,000 g


## Fish cage data (approx.)

- Cage submerged height: (not provided)
- Cage diameter (surface): 50 m
- Cage circumference: 157 m
- Net mesh grid size: 27.5 mm × 27.5 mm
- Parts of the net show biofouling


## Notes

- Use the `by_bag` CSV exports to find sensor timestamps for each recorded dataset.
- Net following experiments include different control parameters and environmental conditions — check the per-run CSV logs for exact time series of control setpoints and sensor measurements.


---

Document created/updated: 2025-10-13
