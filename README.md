# Clear Skies Ahead: Upstate NY Flight Delay Prediction

Predicting flight delays across Upstate New York airports using **PySpark ML** on a merged **BTS + NOAA** dataset (2015 to 2025). Built for **IST 418: Big Data Analytics** at Syracuse University.

The pipeline answers two questions with **Gradient Boosted Trees**:

| Question | Model | Type | Label |
|----------|-------|------|-------|
| *Will this flight be delayed?* | `GBTClassifier` | Binary | `DepDel15` / `ArrDel15` |
| *How many minutes late?* | `GBTRegressor` | Regression | `ArrDelayMinutes` |

Each model is trained twice, once from the **origin** airport's perspective and once from the **destination** airport's perspective.

---

## Table of Contents

- [Mental Model](#mental-model)
- [Data](#data)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Known Issues and Things to Review](#known-issues-and-things-to-review)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Mental Model

```
   BTS flight records ──┐
                        ├──► merged parquet ──► clean + feature engineer ──► GBT model ──► metrics + plots
   NOAA daily weather ──┘        (per airport)         (PySpark)              (Spark ML)
```

The core idea: a flight's outcome depends on **when** it flies (hour, day, month, season), **how far** (distance, scheduled elapsed time), the **carrier**, and the **weather** at the airport that day (temperature, precipitation, snow, wind, fog, thunder, ice). Weather comes from NOAA daily summaries joined to each flight by station and date.

---

## Data

**Coverage:** 2015 to 2025, filtered to flights touching Upstate NY and Northeast regional airports.

**Regional airports of interest:** `SYR`, `ROC`, `BUF`, `ALB`, `BGM`, `ITH` (plus `ELM`, `PWM`, `BTV`, `AVP`, `MDT`, `ABE` for the peer comparison plots).

**Sources:**

| Source | What it provides |
|--------|------------------|
| **BTS** (Bureau of Transportation Statistics) | On-time performance: schedules, delays, delay causes, distance, carrier |
| **NOAA** (GHCN daily) | Daily weather per station: temp, precipitation, snow, wind, and weather-type flags |

**Data files (parquet):**

| File | Purpose |
|------|---------|
| `df_mergedorigin.parquet` | Merged BTS + NOAA keyed on the **origin** airport (classifier 3a, regressor 4a) |
| `df_mergeddest.parquet` | Merged BTS + NOAA keyed on the **destination** airport (classifier 3b, regressor 4b) |
| `planes.parquet` | Full flight records used for the final poster plots |
| `weather.parquet` | Raw NOAA daily weather used for the final poster plots |

> **Note:** The merged files carry 100+ columns. `COLS_TO_KEEP` in the pipeline trims this down to the modeling set before any training runs.

---

## Repository Structure

```
Upstate-NY-Flight-Delays/
├── Flight_Delay.py            # Full annotated notebook export (Parts 1 to 5)
├── flight_delay_pipeline.py   # Refactored, production-style pipeline
├── Data/
│   ├── df_mergedorigin.parquet
│   ├── df_mergeddest.parquet
│   ├── planes.parquet
│   └── weather.parquet
└── README.md
```

**Two entry points, two purposes:**

| File | Use it when you want to... |
|------|----------------------------|
| `Flight_Delay.py` | Read the full story: setup, exploration, both classifiers, both regressors, and poster-ready plots, all commented (Colab / notebook layout). |
| `flight_delay_pipeline.py` | Run the modeling cleanly. Origin and destination share one `run_classifier()` and one `run_regressor()`, so there is no copy-paste drift. |

---

## Setup

**Requirements**

- Python 3.9+
- Java 8 or 11 (required by Spark)
- The packages below

```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn pyarrow
```

**Clone**

```bash
git clone https://github.com/<your-username>/Upstate-NY-Flight-Delays.git
cd Upstate-NY-Flight-Delays
```

---

## Running the Pipeline

### Option A: refactored pipeline (recommended)

1. Open `flight_delay_pipeline.py` and update the two path constants in the **CONFIG** section to point at your parquet files (see [Configuration](#configuration)).
2. Run it:

```bash
python flight_delay_pipeline.py
```

This runs, in order:

- **Origin classifier** (`DepDel15`) and **destination classifier** (`ArrDel15`)
- **Origin regressor** and **destination regressor**, each with hyperparameter tuning via `TrainValidationSplit`

Each stage prints its metrics and shows a confusion matrix (classifiers) or feature importances (regressors).

### Option B: full notebook export

`Flight_Delay.py` is laid out as sequential notebook cells. Run **Part 1 (Setup)** first, then run the parts you want. It also produces the **Part 5** poster plots. It was written for Google Colab (`pip install pyspark`, Drive mount), so adjust those cells if you run locally.

---

## Configuration

All paths live in one place. In `flight_delay_pipeline.py`:

```python
ORIGIN_CSV_PATH = r"...\Data\df_mergedorigin.parquet"
DEST_CSV_PATH   = r"...\Data\df_mergeddest.parquet"
```

Two toggles worth knowing:

| Flag | Default | Effect |
|------|---------|--------|
| `DROP_LEAKY_FEATURES` | `False` | When `True`, drops delay-cause columns from the regressor so it forecasts from weather and time only (see below) |
| `tune` (arg to `run_regressor`) | `True` | Runs the grid search over `maxIter`, `stepSize`, `maxDepth` |

The pipeline auto-detects the `ELEVATION` vs `Elevation` column casing and filters every feature list to columns that actually exist, so a missing column will not crash the run.

---

## Known Issues and Things to Review

> Read this section before trusting any regression number.

**1. Data leakage in the regressor (important).**
`CarrierDelay`, `WeatherDelay`, `NASDelay`, and `LateAircraftDelay` are **components** of the delay being predicted (`ArrDelayMinutes`). Leaving them in inflates R-squared and is not a true forecast. It is kept in by default only to match the original coursework output. For an honest weather-and-time model, set `DROP_LEAKY_FEATURES = True`.

**2. Case sensitivity.**
The original notebook set `spark.sql.caseSensitive = true`, which is what made the `ELEVATION` / `Elevation` mismatch matter. The refactored pipeline leaves it at the default (`false`) and standardizes the column name instead.

**3. The refactor fixed a real bug.**
In the original, the "destination" regressor loaded the destination file but never retrained, so its results were actually the origin model scored on origin data. `run_regressor(DEST_PATH)` now retrains end to end, so destination numbers are genuinely destination numbers.

---

## Results

Populate this table after a run. The regressor is reported both as baseline and tuned.

| Metric | Origin | Destination |
|--------|--------|-------------|
| Classifier AUC | — | — |
| Classifier F1 | — | — |
| Regressor RMSE (min) | — | — |
| Regressor MAE (min) | — | — |
| Regressor R-squared | — | — |
| Operational Tier Accuracy | — | — |

**How to read these:**

- **RMSE** is average error in minutes, penalizing large misses more heavily.
- **MAE** is the plain-English "off by about X minutes on average."
- **R-squared** is the share of delay variance the model explains (closer to 1.0 is better).
- **Operational Tier Accuracy** asks whether the model put a flight in the right severity bucket (minor / moderate / significant / severe), which is the most operationally useful signal.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Compute | Apache Spark (PySpark) |
| Modeling | Spark ML: `GBTClassifier`, `GBTRegressor`, `StringIndexer`, `VectorAssembler`, `Pipeline` |
| Tuning | `ParamGridBuilder`, `TrainValidationSplit`, `CrossValidator` |
| Evaluation | `BinaryClassificationEvaluator`, `MulticlassClassificationEvaluator`, `RegressionEvaluator`, scikit-learn confusion matrix |
| Visualization | Matplotlib, Seaborn |
| Data format | Parquet |

---

## Author

**Philip Cheuk** — B.S. Applied Data Analytics, Syracuse University iSchool
IST 418: Big Data Analytics, Fall 2026

---

*Academic project. Data from public BTS and NOAA sources.*
