# Phase 1: Project setup and data preparation

**Status:** Done
**Branch:** `phase-1-data-prep`
**Script:** `prepare_data.py`
**Output:** `data/claims.csv` (not committed)

## Goal

Build a claims dataset that a model can learn from. No public dataset of real compensation claims exists, so the claims and their labels are derived from a real flight-delay dataset.

## What was done

1. Created the project folder, a Python virtual environment and installed the packages.
2. Downloaded `flights.csv` from the Kaggle dataset "2015 Flight Delays and Cancellations" (`usdot/flight-delays`) by hand and placed it in `data/`.
3. Wrote and ran `prepare_data.py`.
4. Checked the printed summary.

## What `prepare_data.py` does

| Step | Action | Why |
|---|---|---|
| 1 | Read only the 13 columns needed | The raw file has 31 columns and is large |
| 2 | Keep flights with arrival delay of 180 minutes or more that were not cancelled | Matches the 3-hour rule in EU261-style compensation |
| 3 | Build the label `approved` from the cause of delay | 1 if airline delay plus late-aircraft delay is at least weather plus security delay, otherwise 0 |
| 4 | Drop rows where only air-traffic delay is recorded | The cause is ambiguous |
| 5 | Flip about 5% of labels at random | Imitates inconsistent human agents |
| 6 | Convert miles to km and add the compensation tier | Rules layer: EUR 250 up to 1,500 km, EUR 400 up to 3,500 km, EUR 600 above |
| 7 | Keep only features known at claim time | Prevents data leakage |
| 8 | Sample up to 50,000 rows, sort by month and save | Keeps the dataset light |

## Output columns

`month`, `day_of_week`, `hour`, `airline`, `origin`, `destination`, `distance_km`, `delay_minutes`, `tier_eur`, `approved`

The cause-of-delay columns are used to build the label and then dropped.

## Results

**Rows saved:** 44,196

**Label balance**

| approved | Share |
|---|---|
| 1 (airline fault) | 86.5% |
| 0 (weather or security) | 13.5% |

**Rows per month**

| Month | Rows | Month | Rows |
|---|---|---|---|
| 1 | 3,479 | 7 | 4,131 |
| 2 | 3,905 | 8 | 3,756 |
| 3 | 3,745 | 9 | 2,063 |
| 4 | 3,123 | 10 | 2,268 |
| 5 | 4,017 | 11 | 3,047 |
| 6 | 5,608 | 12 | 5,054 |

## Findings

- **The labels are imbalanced.** About 86.5% of claims are approved, which is outside the 30-70% range I first expected. This is a real property of the data: most delays of three hours or more are caused by the airline or by late aircraft. It means accuracy is a misleading metric, because a model that approves everything scores about 86.5%. Later phases use PR-AUC, look at the reject class separately, and check calibration.
- **Every month has data.** This allows a time-based split: train on January-September (33,827 rows) and test on October-December (10,369 rows).
- **Leakage is prevented.** The output file contains no cause-of-delay columns, so the model has to guess from route, airline, season, time of day, distance and delay length.

## Limitations

- The data is US 2015 flights. US flights fall under US DOT rules, not EU261, so EU261-style rules are applied as a simulated policy.
- The labels are derived from delay causes, not from real claim decisions. They stand in for what an agent would decide.
- The 5% label noise is artificial.
- The ambiguous air-traffic-only rows were dropped, which slightly changes the distribution.

## Problems met along the way

- The Kaggle command line tool was not installed, and the API token file was not found. The dataset was downloaded manually instead.
- Some commands were first run in Command Prompt instead of PowerShell, so `$HOME` was not expanded.
- The first run of the script failed because `flights.csv` was not yet in the `data/` folder. Once it was copied in, the script ran successfully.

## Next

Phase 2: train a baseline logistic regression and a LightGBM model with `train.py`, using the time-based split, and evaluate them with PR-AUC and a calibration table.