# IST 418 Final Project

**Pipeline:** Upstate NY flight delays predicted with PySpark on a merged BTS + NOAA dataset (2015–2025).

## Contents

1. **Setup** — pip install, all imports, Spark session, Drive mount, **all data paths defined as constants**
2. **Prototype & Exploration** — original data-loading and prototype regressor
3. **GBT Classifiers** — does a flight get delayed? (binary)
    - 3a. Origin airport — `DepDel15`
    - 3b. Destination airport — `ArrDel15`
4. **GBT Regressors** — *how much* does it get delayed? (minutes)
    - 4a. Origin airport (full pipeline)
    - 4b. Destination airport (path swap + dest-specific results only)
5. **Final Plots** — poster-ready visualizations

Every section below assumes the Setup cell has run. No section reinstalls PySpark, re-imports libraries, or hardcodes a Drive path — change the path constants in Setup once and every section follows.

---

# Part 1 · Setup

Single setup cell: install PySpark, import every library used anywhere in the notebook, mount Drive, build the SparkSession, and define **all data paths as constants** so each section below references the constant instead of a hardcoded string. To move data files, edit the path constants here once.

# ───────────────────────────────────────────────────────────────────────────
# IST 418 Final Project — Master Setup
# Run this cell once. Every part below depends on it.
# ───────────────────────────────────────────────────────────────────────────

# 1. Install PySpark (Colab needs this every fresh runtime)
! pip install pyspark -q

# 2. Imports — consolidated from all six source notebooks
from google.colab import drive

# Standard data libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import HPacker

# PySpark core
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import (
    col, when, hour, to_timestamp, lpad, concat, lit,
    count, isnan, row_number, sum as spark_sum,
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window

# PySpark ML — classification side
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)

# PySpark ML — regression side
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import (
    CrossValidator, ParamGridBuilder, TrainValidationSplit,
)

# sklearn — for the confusion matrix display
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 3. Mount Google Drive
drive.mount('/content/drive')

# 4. Build SparkSession (8 GB driver/executor — the classifier setup needs it)
spark = (
    SparkSession.builder
    .appName("IST418FinalProject")
    .config("spark.driver.memory",   "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.caseSensitive", "true")

# 5. Centralized data paths — adjust these to match your Drive layout
ORIGIN_CSV_PATH       = "/content/drive/MyDrive/418 Final Folder/df_mergedorigin.csv"          # Part 3a classifier
DEST_CSV_PATH         = "/content/drive/MyDrive/418 Final Folder/df_mergeddest.csv"            # Part 3b classifier
ORIGIN_REGRESSOR_PATH = "/content/drive/MyDrive/418 Final Folder/df_mergedorigin.csv"   # Part 4a regressor
DEST_REGRESSOR_PATH   = "/content/drive/MyDrive/418 Final Folder/df_mergeddest.csv"     # Part 4b regressor
PLANES_PARQUET        = "/content/drive/MyDrive/418 Final Folder/planes.parquet"   # Part 5 plots
WEATHER_PARQUET       = "/content/drive/MyDrive/418 Final Folder/weather.parquet"  # Part 5 plots
RAW_DATA_DIR          = "/content/drive/MyDrive/418 Final Folder/"   # Part 2 raw data

print("Setup complete — SparkSession, imports, and data paths are ready.")

---

# Part 2 · Prototype & Exploration

The original prototype/exploration code: an early end-to-end GBT regressor and null-checking pass. Kept for completeness; the production pipelines live in Parts 3 and 4.

## **Setup** ##
# Read in Reporting Carrier dataset
flight_df = spark.read.csv(
    f"{RAW_DATA_DIR}df_master.csv",
    header=True,
    inferSchema=True
)
## cleaning airplane flight dataset to only include upstate NY airports

df = pd.read_csv(f"{RAW_DATA_DIR}df_master.csv")

regional_airports = ['SYR', 'ITH', 'ROC', 'ALB', 'BUF', 'BGM']

df_filtered = df[
    (df['Origin'].isin(regional_airports)) |
    (df['Dest'].isin(regional_airports))
]

df_filtered.to_csv('flights_regional.csv', index=False)
print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(df_filtered)}")
# Define the base path for your Google Drive files
drive_path = "/content/drive/MyDrive/418 Final Folder/" # Corrected path with space

# 1. Load and join
planes = spark.read.parquet(f"{drive_path}planes.parquet")
weather = spark.read.parquet(f"{drive_path}weather.parquet")
if 'cancelled' in weather.columns:
    weather = weather.drop('cancelled')

df = planes.join(weather, on="DATE", how="inner")

# 2. Derive dep_hour from CRSDepTime
df = df.withColumn("CRSDepTimeStr", lpad(col("CRSDepTime").cast("string"), 4, "0"))
df = df.withColumn("dep_hour", col("CRSDepTimeStr").substr(1, 2).cast("int"))

# 3. Select columns and drop nulls
label_col = "DepDelay"
numeric_features = [
    "CarrierDelay", "LateAircraftDelay", "NASDelay", "WeatherDelay",
    "dep_hour", "avg_wind_speed", "max_temp", "Month",
    "precipitation", "min_temp", "Distance", "DayOfWeek"
]
categorical_features = ["Dest", "Reporting_Airline", "Origin"]

df = df.select(numeric_features + categorical_features + [label_col]).na.drop()

# 4. Index categoricals
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_features
]

# 5. Assemble all features
all_features = numeric_features + [f"{c}_idx" for c in categorical_features]
assembler = VectorAssembler(inputCols=all_features, outputCol="features")

# 6. GBT
gbt = GBTRegressor(featuresCol="features", labelCol=label_col, maxIter=20, maxBins=64)

pipeline = Pipeline(stages=indexers + [assembler, gbt])
train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)

# 7. Extract top 7 feature importances
gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()

fi_df = (pd.DataFrame({"feature": all_features, "importance": importances})
           .sort_values("importance", ascending=False)
           .head(7)
           .reset_index(drop=True))

print(fi_df)

# 8. Plot matching your style
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
        color="steelblue", edgecolor="white")
ax.set_title("GBTRegressor — Top 7 Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
print('Checking for missing values:')
for col_name in df.columns:
    null_count = df.filter(col(col_name).isNull()).count()
    if null_count > 0:
        print(f"Column '{col_name}': {null_count} missing values")
if df.na.drop().count() == df.count():
    print("No missing values found in the dataset.")
else:
    print(f"Total rows before dropping nulls: {df.count()}")
    print(f"Total rows after dropping nulls: {df.na.drop().count()}")
---

# Part 3 · GBT Classifiers — Will the Flight Be Delayed?

Two binary classifiers using `GBTClassifier`. The structure is the same; the differences are the data file, the target label, and the time column.

| | Origin model | Destination model |
|---|---|---|
| Path constant | `ORIGIN_CSV_PATH` | `DEST_CSV_PATH` |
| Target label | `DepDel15` | `ArrDel15` |
| Time column | `CRSDepTime` | `CRSArrTime` |

## 3a · Origin Airport — `DepDel15`

### Load Data

Read the pre-merged BTS + NOAA origin dataset from Google Drive into a Spark DataFrame. This file contains flights that departed from Upstate New York airports (SYR, ROC, ALB, BUF, etc.), so the weather features reflect conditions at the departure airport. The dataset has to be in the user's "MyDrive".
# Read in Reporting Carrier dataset
flight_df = spark.read.csv(
    ORIGIN_CSV_PATH,
    header=True,
    inferSchema=True
)
Preview the first 5 rows to confirm the data loaded correctly and inspect column names and values.
flight_df.show(5)
---
## Column Selection

The raw merged dataset has over 100 columns. This step trims it down to only the columns needed for modeling flight schedule info, delay labels, and weather features from NOAA. Keeping only relevant columns reduces processing time and makes sure we're getting rid of irrelevant columns that have missing values.
#Only keep necessary columns for pipeline
cols_to_keep = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "DATE",
    "Reporting_Airline", "DOT_ID_Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "OriginCityName", "Dest", "DestCityName",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepartureDelayGroups", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
    "ArrDel15", "ArrivalDelayGroups", "Cancelled", "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "ELEVATION", "avg_wind_speed", "peak_gust_time", "precipitation",
    "snowfall", "snow_depth", "max_temp", "min_temp", "fog", "thunder",
    "ice_sleet", "hail", "glaze_rime"
]

final_df = flight_df.select(cols_to_keep)
---
## Feature Engineering and Data Cleaning

This section prepares the data for the ML pipeline:

1. **Cast all columns to string** to safely detect and replace literal `"NA"` strings (these come from the raw BTS/NOAA CSVs and are not real nulls that PySpark can detect by default).
2. **Replace `"NA"` strings with `null`** so PySpark handles them correctly in next steps.
3. **Cast the target label** (`DepDel15`) to `DoubleType` as required by PySpark ML classifiers. Note: this notebook predicts *departure* delays (`DepDel15`) rather than arrival delays, since the weather data is joined at the origin airport.
4. **Extract departure hour** from `CRSDepTime` as a numeric feature.
5. **Encode binary weather conditions** (`fog`, `thunder`, etc.) from string booleans (`"True"`/`"False"`) to `0.0`/`1.0` doubles.
6. **Cast all numeric features** to `DoubleType`.
7. **Fill nulls** in binary and precipitation columns with `0.0` (absence of weather event).
8. **Add a class weight column** to penalize false negatives — delayed flights (`DepDel15 == 1`) are weighted 2x to help the model take delays more seriously.
9. **Drop rows** with any remaining nulls in feature or label columns so the pipeline doesn't break on missing data.
# FEATURE ENGINEERING AND PIPELINE

# Force ALL columns to string so we can catch literal "NA" values
final_df = final_df.select([
    col(c).cast("string").alias(c)
    for c in final_df.columns
])

# Replace ALL "NA" strings with null globally
final_df = final_df.select([
    when(col(c) == "NA", None).otherwise(col(c)).alias(c)
    for c in final_df.columns
])

# Create the label column
final_df = final_df.withColumn("DepDel15", col("DepDel15").cast(DoubleType()))

# Time features
final_df = final_df.withColumn("CRSDepTime", col("CRSDepTime").cast(IntegerType()))

final_df = final_df.withColumn(
    "dep_hour",
    (col("CRSDepTime") / 100).cast("int")
)

#Feature columns
feature_cols = [
    "dep_hour",
    "DayOfWeek",
    "Month",
    "Distance",
    "CRSElapsedTime",
    "ELEVATION",
    "max_temp",
    "min_temp",
    "precipitation",
    "snowfall",
    "snow_depth",
    "avg_wind_speed",
    "fog",
    "thunder",
    "ice_sleet",
    "hail",
    "glaze_rime",
]

# Handle binary strings first
binary_cols = ["fog", "thunder", "ice_sleet", "hail", "glaze_rime"]

for c in binary_cols:
    final_df = final_df.withColumn(
        c,
        when(col(c).isin("TRUE", "True", "true", "1"), 1.0)
        .when(col(c).isin("FALSE", "False", "false", "0"), 0.0)
        .otherwise(None)
        .cast(DoubleType())
    )

# Cast features to double
for c in feature_cols:
    if c not in binary_cols:
        final_df = final_df.withColumn(c, col(c).cast(DoubleType()))

# Handle null values by changing them to 0 (won't affect data because these are binary variables)
final_df = final_df.fillna({
    "fog": 0.0, "thunder": 0.0, "ice_sleet": 0.0, "hail": 0.0, "glaze_rime": 0.0,
    "precipitation": 0.0,
    "avg_wind_speed": 0.0,
})

delay_count = final_df.filter(col("DepDel15") == 1).count()
total_count = final_df.count()
delay_rate = delay_count / total_count
print(f"Delay rate: {delay_rate:.3f}")

final_df = final_df.withColumn(
    "classWeight",
    when(col("DepDel15") == 1, 2.0)
    .otherwise(1.0)
)

final_df = final_df.dropna(subset=feature_cols + ["DepDel15", "Reporting_Airline"])
---
## Model Pipeline: GBTClassifier

This cell builds and runs the full ML pipeline:

1. **Null check** — confirms no nulls remain in the feature columns after cleaning.
2. **StringIndexer (airline)** — converts the `Reporting_Airline` string codes (e.g., `"AA"`, `"DL"`) into numeric indices that the model can use. `handleInvalid='keep'` ensures unseen airlines at test time don't cause errors.
3. **StringIndexer (origin)** — same encoding for the `Origin` airport code.
4. **VectorAssembler** — combines all numeric and indexed features into a single `features` vector column, which is the required input format for PySpark ML models.
5. **Train/test split** — 80% of the data is used for training, 20% for evaluation.
6. **GBTClassifier** — the main model. Gradient Boosted Trees build an ensemble of decision trees sequentially, each one correcting the errors of the previous. Key hyperparameters:
   - `maxIter=50`: number of boosting rounds
   - `maxDepth=4`: controls tree complexity
   - `stepSize=0.1`: learning rate (how much each new tree contributes)
   - `weightCol='classWeight'`: uses the 2x weight on delayed flights
7. **Pipeline** — chains the indexers, assembler, and model into a single object so the same transformations are applied consistently at both train and test time.
8. **Fit and transform** — trains the pipeline on `train_df` and generates predictions on `test_df`.
# Null check
print("Null check after cleaning:")
final_df.select([
    col(c).isNull().cast("int").alias(c)
    for c in feature_cols + ["DepDel15"]
]).groupBy().sum().show()

#Encode airlines and origins
airline_indexer = StringIndexer(
    inputCol="Reporting_Airline",
    outputCol="airline_idx",
    handleInvalid="keep",
)

origin_indexer = StringIndexer(
    inputCol="Origin",
    outputCol="origin_idx",
    handleInvalid="keep"
)

#Vector assembler
assembler = VectorAssembler(
    inputCols=feature_cols + ["airline_idx", "origin_idx"],
    outputCol="features",
    handleInvalid="skip",
)

#Split data for training and test sets
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train_df.count()}, Test: {test_df.count()}")
train_df.select("Reporting_Airline").show(5)
train_df.select("Reporting_Airline").distinct().show()

#Model
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="DepDel15",
    weightCol="classWeight",
    maxIter=50,
    maxDepth=4,
    stepSize=0.1,
    seed=42,
)

pipeline = Pipeline(stages=[
    airline_indexer,
    origin_indexer,
    assembler,
    gbt,
])

model = pipeline.fit(train_df)
predictions = model.transform(test_df)
---
## Evaluation

### AUC-ROC

Evaluate the classifier using **Area Under the ROC Curve (AUC)**. AUC measures how well the model separates delayed from on-time flights across all classification thresholds, a score of 1.0 is perfect, 0.5 is random guessing.
#Evaluate the model using AUC

evaluator = BinaryClassificationEvaluator(
    labelCol="DepDel15",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print("AUC:", auc)
### Prediction Distribution

Count how many flights the model predicted as on-time (`0.0`) vs. delayed (`1.0`). This sanity check confirms the model is actually predicting both classes and not collapsing to a single output.
predictions.groupBy("prediction").count().show()
### Sample Predictions vs. Actuals

Display a sample of actual labels (`DepDel15`) alongside model predictions (`prediction`) to do a quick visual spot-check of where the model gets it right and wrong.
predictions.select("DepDel15", "prediction").show(50)
### Additional Null Check: Temperature and Snow Columns

Extra validation step to confirm that the temperature and snow columns have no nulls or NaN values remaining after cleaning. These NOAA weather features are critical predictors, so any missing values here would silently drop rows during the pipeline.
final_df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in ["max_temp", "min_temp", "snowfall", "snow_depth"]
]).show()
### Confusion Matrix

Visualize the full breakdown of true positives, true negatives, false positives, and false negatives. The confusion matrix makes it easy to see how the model performs on each class, especially important here because false negatives (predicting on-time when a flight is actually delayed) are more costly than false positives.
pred_pdf = predictions.select("DepDel15", "prediction").toPandas()

cm = confusion_matrix(pred_pdf["DepDel15"], pred_pdf["prediction"])
ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Delayed"]).plot(cmap="Blues")
plt.title("Flight Delay Prediction — Confusion Matrix")
plt.show()
### F1 Score

Compute the **F1 Score**. F1 is a better metric than raw accuracy here because the dataset is imbalanced (~16% delayed flights). A high F1 score means the model is doing well at both catching actual delays and not over-predicting delays.
f1_eval = MulticlassClassificationEvaluator(
    labelCol="DepDel15", predictionCol="prediction", metricName="f1"
)
print(f"F1 Score: {f1_eval.evaluate(predictions):.4f}")
---

## 3b · Destination Airport — `ArrDel15`

_Source: `IST418_DestProjectCode_Classifier.ipynb`._

### Load Data

Read the pre-merged BTS + NOAA destination dataset from Google Drive into a Spark DataFrame. The dataset has to be in the user's "MyDrive".
# Read in Reporting Carrier dataset
flight_df = spark.read.csv(
    DEST_CSV_PATH,
    header=True,
    inferSchema=True
)
Preview the first 5 rows to confirm the data loaded correctly and inspect column names and values.
flight_df.show(5)
---
## Column Selection

The raw merged dataset has over 100 columns. This step trims it down to only the columns needed for modeling: flight schedule info, delay labels, and weather features from NOAA. Keeping only relevant columns reduces memory usage and speeds up processing.
#Only keep necessary columns for pipeline
cols_to_keep = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "DATE",
    "Reporting_Airline", "DOT_ID_Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "OriginCityName", "Dest", "DestCityName",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepartureDelayGroups", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
    "ArrDel15", "ArrivalDelayGroups", "Cancelled", "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "Elevation", "avg_wind_speed", "peak_gust_time", "precipitation",
    "snowfall", "snow_depth", "max_temp", "min_temp", "fog", "thunder",
    "ice_sleet", "hail", "glaze_rime"
]

final_df = flight_df.select(cols_to_keep)
---
## Feature Engineering and Data Cleaning

This section prepares the data for the ML pipeline:

1. **Cast all columns to string** to safely detect and replace literal `"NA"` strings (these come from the raw BTS/NOAA CSVs and are not real nulls that PySpark can detect by default).
2. **Replace `"NA"` strings with `null`** so PySpark handles them correctly in downstream steps.
3. **Cast the target label** (`ArrDel15`) to `DoubleType` as required by PySpark ML classifiers.
4. **Extract departure hour** from `CRSArrTime` as a numeric feature.
5. **Encode binary weather conditions** (`fog`, `thunder`, etc.) from string booleans (`"True"`/`"False"`) to `0.0`/`1.0` doubles.
6. **Cast all numeric features** to `DoubleType`.
7. **Fill nulls** in binary and precipitation columns with `0.0` (absence of weather event).
8. **Add a class weight column** to penalize false negatives — delayed flights (`ArrDel15 == 1`) are weighted 2x to help the model take delays more seriously.
9. **Drop rows** with any remaining nulls in feature or label columns so the pipeline doesn't break on missing data.
# FEATURE ENGINEERING AND PIPELINE

# Force ALL columns to string so we can catch literal "NA" values
final_df = final_df.select([
    col(c).cast("string").alias(c)
    for c in final_df.columns
])

# Replace ALL "NA" strings with null globally
final_df = final_df.select([
    when(col(c) == "NA", None).otherwise(col(c)).alias(c)
    for c in final_df.columns
])

# Create the label column
final_df = final_df.withColumn("ArrDel15", col("ArrDel15").cast(DoubleType()))

# Time features
final_df = final_df.withColumn("CRSArrTime", col("CRSArrTime").cast(IntegerType()))

final_df = final_df.withColumn(
    "dep_hour",
    (col("CRSArrTime") / 100).cast("int")
)

#Feature columns
feature_cols = [
    "dep_hour",
    "DayOfWeek",
    "Month",
    "Distance",
    "CRSElapsedTime",
    "Elevation",
    "max_temp",
    "min_temp",
    "precipitation",
    "snowfall",
    "snow_depth",
    "avg_wind_speed",
    "fog",
    "thunder",
    "ice_sleet",
    "hail",
    "glaze_rime",
]

# Handle binary strings first
binary_cols = ["fog", "thunder", "ice_sleet", "hail", "glaze_rime"]

for c in binary_cols:
    final_df = final_df.withColumn(
        c,
        when(col(c).isin("TRUE", "True", "true", "1"), 1.0)
        .when(col(c).isin("FALSE", "False", "false", "0"), 0.0)
        .otherwise(None)
        .cast(DoubleType())
    )

# Cast features to double
for c in feature_cols:
    if c not in binary_cols:
        final_df = final_df.withColumn(c, col(c).cast(DoubleType()))

# Handle null values by changing them to 0 (won't affect data because these are binary variables)
final_df = final_df.fillna({
    "fog": 0.0, "thunder": 0.0, "ice_sleet": 0.0, "hail": 0.0, "glaze_rime": 0.0,
    "precipitation": 0.0,
    "avg_wind_speed": 0.0,
})

delay_count = final_df.filter(col("ArrDel15") == 1).count()
total_count = final_df.count()
delay_rate = delay_count / total_count
print(f"Delay rate: {delay_rate:.3f}")

final_df = final_df.withColumn(
    "classWeight",
    when(col("ArrDel15") == 1, 2.0)
    .otherwise(1.0)
)

final_df = final_df.dropna(subset=feature_cols + ["ArrDel15", "Reporting_Airline"])
---
## Model Pipeline: GBTClassifier

This cell builds and runs the full ML pipeline:

1. **Null check** — confirms no nulls remain in the feature columns after cleaning.
2. **StringIndexer (airline)** — converts the `Reporting_Airline` string codes (e.g., `"AA"`, `"DL"`) into numeric indices that the model can use. `handleInvalid='keep'` ensures unseen airlines at test time don't cause errors.
3. **StringIndexer (origin)** — same encoding for the `Origin` airport code.
4. **VectorAssembler** — combines all numeric and indexed features into a single `features` vector column, which is the required input format for PySpark ML models.
5. **Train/test split** — 80% of the data is used for training, 20% for evaluation.
6. **GBTClassifier** — the main model. Gradient Boosted Trees build an ensemble of decision trees sequentially, each one correcting the errors of the previous. Key hyperparameters:
   - `maxIter=50`: number of boosting rounds
   - `maxDepth=4`: controls tree complexity
   - `maxBins=64`: number of bins for continuous features
   - `stepSize=0.1`: learning rate (how much each new tree contributes)
   - `weightCol='classWeight'`: uses the 2x weight on delayed flights
7. **Pipeline** — chains the indexers, assembler, and model into a single object so the same transformations are applied consistently at both train and test time.
8. **Fit and transform** — trains the pipeline on `train_df` and generates predictions on `test_df`.
# Null check
print("Null check after cleaning:")
final_df.select([
    col(c).isNull().cast("int").alias(c)
    for c in feature_cols + ["ArrDel15"]
]).groupBy().sum().show()

#Encode airlines and origins
airline_indexer = StringIndexer(
    inputCol="Reporting_Airline",
    outputCol="airline_idx",
    handleInvalid="keep",
)

origin_indexer = StringIndexer(
    inputCol="Origin",
    outputCol="origin_idx",
    handleInvalid="keep"
)

#Vector assembler
assembler = VectorAssembler(
    inputCols=feature_cols + ["airline_idx", "origin_idx"],
    outputCol="features",
    handleInvalid="skip",
)

#Split data for training and test sets
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

print(f"Train: {train_df.count()}, Test: {test_df.count()}")
train_df.select("Reporting_Airline").show(5)
train_df.select("Reporting_Airline").distinct().show()

#Model
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="ArrDel15",
    weightCol="classWeight",
    maxIter=50,
    maxDepth=4,
    maxBins=64,
    stepSize=0.1,
    seed=42,
)

pipeline = Pipeline(stages=[
    airline_indexer,
    origin_indexer,
    assembler,
    gbt,
])

model = pipeline.fit(train_df)
predictions = model.transform(test_df)
---
## Evaluation

### AUC-ROC

Evaluate the classifier using **Area Under the ROC Curve (AUC)**. AUC measures how well the model separates delayed from on-time flights across all classification thresholds a score of 1.0 is perfect, 0.5 is random guessing.
#Evaluate the model using AUC

evaluator = BinaryClassificationEvaluator(
    labelCol="ArrDel15",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print("AUC:", auc)
### Prediction Distribution

Count how many flights the model predicted as on-time (`0.0`) vs. delayed (`1.0`). This sanity check confirms the model is actually predicting both classes and not collapsing to a single output.
predictions.groupBy("prediction").count().show()
### Sample Predictions vs. Actuals

Display a sample of actual labels (`ArrDel15`) alongside model predictions (`prediction`) to do a quick visual spot-check of where the model gets it right and wrong.
predictions.select("ArrDel15", "prediction").show(20)
### Additional Null Check: Temperature and Snow Columns

Extra validation step to confirm that the temperature and snow columns have no nulls or NaN values remaining after cleaning. These NOAA weather features are critical predictors, so any missing values here would silently drop rows during the pipeline.
final_df.select([
    count(when(col(c).isNull() | isnan(c), c)).alias(c)
    for c in ["max_temp", "min_temp", "snowfall", "snow_depth"]
]).show()
### Confusion Matrix

Visualize the full breakdown of true positives, true negatives, false positives, and false negatives. The confusion matrix makes it easy to see how the model performs on each class, especially important here because false negatives (predicting on-time when a flight is actually delayed) are more costly than false positives.
pred_pdf = predictions.select("ArrDel15", "prediction").toPandas()

cm = confusion_matrix(pred_pdf["ArrDel15"], pred_pdf["prediction"])
ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Delayed"]).plot(cmap="Blues")
plt.title("Flight Delay Prediction — Confusion Matrix")
plt.show()
### F1 Score

Compute the **F1 Score**, the harmonic mean of precision and recall. F1 is a better metric than raw accuracy here because the dataset is imbalanced (~20% delayed flights). A high F1 score means the model is doing well at both catching actual delays and not over-predicting delays.
f1_eval = MulticlassClassificationEvaluator(
    labelCol="ArrDel15", predictionCol="prediction", metricName="f1"
)
print(f"F1 Score: {f1_eval.evaluate(predictions):.4f}")
---

# Part 4 · GBT Regressors — How Many Minutes Late?

Two `GBTRegressor` pipelines that predict actual delay duration in minutes (training only on flights where `ArrDel15 == 1`). The Origin and Destination versions run **the exact same code** — they only differ by which CSV they consume — so the full pipeline appears once under 4a, and 4b shows only the data-path swap plus dest-specific result outputs.

## 4a · Origin Airport — Full Pipeline

_Source: `gbt_regressor_origindelay.ipynb`._

## 2. Load the Combined Dataset

This is the pre-joined BTS + NOAA dataset covering 2015–2025 for the Upstate NY region.
# Update this path to wherever your combined CSV/parquet lives
DATA_PATH = ORIGIN_REGRESSOR_PATH

df_raw = spark.read.csv( DATA_PATH, header=True, inferSchema=True, nullValue="NA")
df_raw = df_raw.withColumn("ArrDel15", F.col("ArrDel15").cast("int")) \
               .withColumn("ArrDelayMinutes", F.col("ArrDelayMinutes").cast("double"))

print(f"Total rows: {df_raw.count():,}")
print(f"Total columns: {len(df_raw.columns)}")
df_raw.printSchema()
## 3. Exploratory Look at the Target Variable
# Distribution of ArrDelayMinutes on delayed flights only
df_raw.filter(F.col("ArrDel15") == 1) \
      .select("ArrDelayMinutes") \
      .describe() \
      .show()
# Quick look at delay categories to understand the range
df_raw.filter(F.col("ArrDel15") == 1) \
      .groupBy(
          F.when(F.col("ArrDelayMinutes") < 30, "15-30 min")
           .when(F.col("ArrDelayMinutes") < 60, "30-60 min")
           .when(F.col("ArrDelayMinutes") < 120, "1-2 hours")
           .otherwise("2+ hours")
           .alias("delay_bucket")
      ) \
      .count() \
      .orderBy("delay_bucket") \
      .show()
## 4. Filter to Delayed Flights Only

The regressor is only meaningful on flights that are actually delayed. Training on all flights would teach the model to predict near-zero values for most rows, which isn't useful.
# Keep only delayed flights (ArrDel15 == 1) and drop rows with null target
df_delayed = df_raw.filter(F.col("ArrDel15") == 1) \
                   .dropna(subset=["ArrDelayMinutes"])

# Rename target to 'label' — required by Spark ML
df_delayed = df_delayed.withColumnRenamed("ArrDelayMinutes", "label")

print(f"Delayed flight rows: {df_delayed.count():,}")
## 5. Preprocessing & Feature Engineering

Same pattern as labs: handle nulls → engineer features → StringIndex categoricals → VectorAssemble.
# ── 5a. Fill nulls ──────────────────────────────────────────────────────────
# Delay cause columns are null for on-time flights; since we filtered to delayed
# flights some causes may still be 0 — fill any remaining nulls with 0
cause_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
df_clean = df_delayed.fillna(0, subset=cause_cols)

# Fill weather nulls with 0 (calm/clear conditions are a safe default)
weather_cols = ["avg_wind_speed", "precipitation", "avg_temp",
                "max_temp", "min_temp", "multiday_precip",
                "snowfall", "snow_depth"]
df_clean = df_clean.fillna(0, subset=weather_cols)

print("Nulls filled.")
# ── 5b. Extract temporal features from departure time ───────────────────────
# CRSDepTime is stored as HHMM integer (e.g. 1430 = 2:30 PM)
# We need departure hour as a numeric feature
df_clean = df_clean.withColumn("dep_hour", (F.col("CRSDepTime") / 100).cast("int"))

# Season: winter/spring/summer/fall encoded as 1-4
df_clean = df_clean.withColumn(
    "season",
    F.when(F.col("Month").isin([12, 1, 2]), 1)   # Winter
     .when(F.col("Month").isin([3, 4, 5]),  2)    # Spring
     .when(F.col("Month").isin([6, 7, 8]),  3)    # Summer
     .otherwise(4)                                 # Fall
)

# Weekend flag: 1 if Saturday or Sunday
df_clean = df_clean.withColumn(
    "is_weekend",
    F.when(F.col("DayOfWeek").isin([6, 7]), 1).otherwise(0)
)

print("Temporal features created.")
# ── 5c. Define feature lists ─────────────────────────────────────────────────
# Categorical columns that need StringIndexer
CATEGORICAL_COLS = ["Reporting_Airline", "Origin", "Dest"]

# Numeric features going directly into VectorAssembler
NUMERIC_FEATURES = [
    "Month", "DayOfWeek", "dep_hour", "season", "is_weekend", "Distance",
    "avg_wind_speed", "precipitation", "avg_temp",
    "max_temp", "min_temp", "snowfall", "snow_depth",
    "CarrierDelay", "WeatherDelay", "NASDelay", "LateAircraftDelay",
]

# Indexed versions of categorical cols (StringIndexer output names)
INDEXED_COLS = [c + "_idx" for c in CATEGORICAL_COLS]

# All features going into the assembler
ALL_FEATURES = NUMERIC_FEATURES + INDEXED_COLS

print(f"Total features: {len(ALL_FEATURES)}")
print(ALL_FEATURES)
## 6. Train / Test Split
train, test = df_clean.randomSplit([0.7, 0.3], seed=1458)

print(f"Training rows:  {train.count():,}")
print(f"Test rows:      {test.count():,}")
## 7. Build the Pipeline

Same pattern from class: StringIndexer(s) → VectorAssembler → GBTRegressor, all wrapped in a `Pipeline`.
# ── 7a. StringIndexers for each categorical column ───────────────────────────
# handleInvalid='keep' so unseen categories in test set don't crash the model
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid="keep")
    for col in CATEGORICAL_COLS
]

# ── 7b. VectorAssembler ───────────────────────────────────────────────────────
assembler = VectorAssembler(
    inputCols=ALL_FEATURES,
    outputCol="features",
    handleInvalid="skip"   # skip any remaining null rows at assembly time
)

# ── 7c. GBTRegressor ─────────────────────────────────────────────────────────
# Three key tuning parameters from lecture:
#   maxIter  = number of trees (more trees = stronger model, more compute)
#   stepSize = shrinkage / learning rate (smaller = more conservative learning)
#   maxDepth = depth of each tree (controls complexity of individual learners)
gbt_reg = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxIter=100,        # 100 trees — good starting point per lecture guidance
    stepSize=0.1,       # default shrinkage
    maxDepth=5,         # default tree depth
    maxBins=64,
    seed=1458
)

# ── 7d. Assemble pipeline ─────────────────────────────────────────────────────
pipeline = Pipeline(stages=indexers + [assembler, gbt_reg])

print("Pipeline built. Stages:", [type(s).__name__ for s in pipeline.getStages()])
## 8. Train the Model
# Check how many rows survive each step
print("df_delayed:    ", df_delayed.count())
print("df_clean:      ", df_clean.count())

# Find which feature columns still have nulls after filling

RAW_CHECK_COLS = NUMERIC_FEATURES + CATEGORICAL_COLS  # use original names, not _idx versions

null_counts = df_clean.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in RAW_CHECK_COLS  # fixed: was ALL_FEATURES which included _idx cols that don't exist yet
])
null_counts.show(truncate=False)

# Drop any remaining null rows then split
df_clean = df_clean.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_COLS + ["label"])

train, test = df_clean.randomSplit([0.7, 0.3], seed=1458)
print(f"Training rows:  {train.count():,}")
print(f"Test rows:      {test.count():,}")
print("Training GBTRegressor...")
model = pipeline.fit(train)
print("Training complete.")
## 9. Evaluate the Model

Standard regression metrics: **RMSE** (root mean squared error) and **R²** (coefficient of determination).
predictions = model.transform(test)

evaluator_rmse = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

rmse = evaluator_rmse.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)
mae  = evaluator_mae.evaluate(predictions)

print(f"GBTRegressor Results")
print(f"--------------------")
print(f"RMSE : {rmse:.2f} minutes")
print(f"MAE  : {mae:.2f} minutes")
print(f"R²   : {r2:.4f}")
# Preview sample predictions vs. actual
predictions.select("Origin", "Dest", "Reporting_Airline", "label", "prediction") \
           .orderBy(F.col("label").desc()) \
           .show(20, truncate=False)
# Get total row count and show 20 rows from the middle
total = predictions.count()
middle = total // 2

predictions.select("Origin", "Dest", "Reporting_Airline", "label", "prediction") \
           .orderBy(F.col("label").desc()) \
           .limit(middle + 10) \
           .orderBy(F.col("label").asc()) \
           .limit(20) \
           .show(truncate=False)
## 10. Feature Importance

GBT gives us feature importance scores out of the box — this tells us which features drove the predictions the most.
# Extract the GBTRegressor model from the pipeline
gbt_model = model.stages[-1]

# Pair feature names with importances and sort descending
feature_importances = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": gbt_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print("Top 15 Feature Importances:")
print(feature_importances.head(15).to_string(index=False))
# Optional: plot feature importances

top15 = feature_importances.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top15["feature"][::-1], top15["importance"][::-1])
plt.xlabel("Importance")
plt.title("GBTRegressor — Top 15 Feature Importances")
plt.tight_layout()
plt.show()
# Extract the GBTRegressor model from the pipeline
gbt_model = model.stages[-1]

# Pair feature names with importances
feature_importances = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": gbt_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)

# Filter to only weather features
weather_features = [
    "avg_wind_speed", "precipitation", "avg_temp",
    "max_temp", "min_temp", "snowfall", "snow_depth"
]

weather_importances = feature_importances[
    feature_importances["feature"].isin(weather_features)
].sort_values("importance", ascending=False)

print("Weather Feature Importances:")
print(weather_importances.to_string(index=False))

# Plot weather features only

plt.figure(figsize=(8, 5))
plt.barh(weather_importances["feature"][::-1], weather_importances["importance"][::-1])
plt.xlabel("Importance")
plt.title("GBTRegressor — Weather Feature Importances")
plt.tight_layout()
plt.show()
## 11. Hyperparameter Tuning with CrossValidator

Same approach from Lab 07 — use `ParamGridBuilder` to tune the three lecture parameters: `maxIter`, `stepSize`, `maxDepth`.
# Build a fresh pipeline with a GBT that has no locked-in params
gbt_tune = GBTRegressor(
    featuresCol="features",
    labelCol="label",
    maxBins=64,
    seed=1458
)

tune_pipeline = Pipeline(stages=indexers + [assembler, gbt_tune])

# Parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(gbt_tune.maxIter,  [50, 100]) \
    .addGrid(gbt_tune.stepSize, [0.05, 0.1]) \
    .addGrid(gbt_tune.maxDepth, [4, 5]) \
    .build()

# Use TrainValidationSplit instead of CrossValidator (8 models instead of 24)
tvs = TrainValidationSplit(
    estimator=tune_pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator_rmse,
    trainRatio=0.7,
    seed=1458
)

# Sample 30% of training data to find best params faster
train_sample = train.sample(fraction=0.3, seed=1458)
print(f"Tuning on {train_sample.count():,} rows (30% sample)...")
print(f"Running TVS over {len(param_grid)} parameter combinations...")

cv_model = tvs.fit(train_sample)
print("Tuning complete.")
# Evaluate best model from CV
best_predictions = cv_model.transform(test)

best_rmse = evaluator_rmse.evaluate(best_predictions)
best_r2   = evaluator_r2.evaluate(best_predictions)
best_mae  = evaluator_mae.evaluate(best_predictions)

print(f"Best Model (after CV tuning)")
print(f"----------------------------")
print(f"RMSE : {best_rmse:.2f} minutes")
print(f"MAE  : {best_mae:.2f} minutes")
print(f"R²   : {best_r2:.4f}")

# Show best params
best_gbt = cv_model.bestModel.stages[-1]
print(f"\nBest params:")
print(f"  maxIter  = {best_gbt.getMaxIter()}")
print(f"  stepSize = {best_gbt.getStepSize()}")
print(f"  maxDepth = {best_gbt.getMaxDepth()}")
## 12. Business Metric: Operational Delay Buckets

Beyond RMSE/R², we want to evaluate how well the model performs from an **operational standpoint**. Airlines classify delays into tiers that trigger different response protocols — predicting the right tier is what actually matters for crew scheduling and rebooking.
def delay_bucket(col_name):
    """Convert delay minutes column into operational tier labels."""
    return (
        F.when(F.col(col_name) < 30,  "minor (<30 min)")
         .when(F.col(col_name) < 60,  "moderate (30-60 min)")
         .when(F.col(col_name) < 120, "significant (1-2 hrs)")
         .otherwise("severe (2+ hrs)")
    )

best_predictions = best_predictions.withColumn("actual_bucket",     delay_bucket("label")) \
                                   .withColumn("predicted_bucket",  delay_bucket("prediction"))

# How often do we predict the right operational tier?
tier_accuracy = best_predictions.filter(
    F.col("actual_bucket") == F.col("predicted_bucket")
).count() / best_predictions.count()

print(f"Operational Tier Accuracy: {tier_accuracy:.2%}")

# Breakdown: where do we get it right vs wrong?
best_predictions.groupBy("actual_bucket", "predicted_bucket") \
                .count() \
                .orderBy("actual_bucket", "predicted_bucket") \
                .show(20, truncate=False)
## 13. Summary

| Metric | Baseline Model | Tuned Model |
|--------|---------------|-------------|
| RMSE (minutes) | — | — |
| MAE (minutes) | — | — |
| R² | — | — |
| Operational Tier Accuracy | — | — |

*(Fill in after running)*

**Interpretation guidance:**
- **RMSE** tells us the average error in minutes — if RMSE = 22, our predictions are off by about 22 minutes on average (penalizes large errors more than MAE does)
- **MAE** is easier to explain to stakeholders — "on average our prediction is X minutes off"
- **R²** tells us how much variance the model explains; closer to 1.0 is better
- **Operational Tier Accuracy** is the most business-relevant metric — gets at whether we'd make the right operational decision, not just whether the number is close
---

## 4b · Destination Airport — Path Swap & Dest-Specific Results

Pipeline code is byte-identical to 4a. Below: the data-path change, then only the cells whose outputs differ from the origin run so you can compare model performance.

_Source: `gbt_regressor_destdelay.ipynb`._

### Data path swap

# Update this path to wherever your combined CSV/parquet lives
DATA_PATH = DEST_REGRESSOR_PATH

df_raw = spark.read.csv( DATA_PATH, header=True, inferSchema=True, nullValue="NA")
df_raw = df_raw.withColumn("ArrDel15", F.col("ArrDel15").cast("int")) \
               .withColumn("ArrDelayMinutes", F.col("ArrDelayMinutes").cast("double"))

print(f"Total rows: {df_raw.count():,}")
print(f"Total columns: {len(df_raw.columns)}")
df_raw.printSchema()
### Target variable on the dest dataset

# Distribution of ArrDelayMinutes on delayed flights only
df_raw.filter(F.col("ArrDel15") == 1) \
      .select("ArrDelayMinutes") \
      .describe() \
      .show()
# Quick look at delay categories to understand the range
df_raw.filter(F.col("ArrDel15") == 1) \
      .groupBy(
          F.when(F.col("ArrDelayMinutes") < 30, "15-30 min")
           .when(F.col("ArrDelayMinutes") < 60, "30-60 min")
           .when(F.col("ArrDelayMinutes") < 120, "1-2 hours")
           .otherwise("2+ hours")
           .alias("delay_bucket")
      ) \
      .count() \
      .orderBy("delay_bucket") \
      .show()
### Row counts after preprocessing & RMSE / R²

# Check how many rows survive each step
print("df_delayed:    ", df_delayed.count())
print("df_clean:      ", df_clean.count())

# Find which feature columns still have nulls after filling

RAW_CHECK_COLS = NUMERIC_FEATURES + CATEGORICAL_COLS  # use original names, not _idx versions

null_counts = df_clean.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in RAW_CHECK_COLS  # fixed: was ALL_FEATURES which included _idx cols that don't exist yet
])
null_counts.show(truncate=False)

# Drop any remaining null rows then split
df_clean = df_clean.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_COLS + ["label"])

train, test = df_clean.randomSplit([0.7, 0.3], seed=1458)
print(f"Training rows:  {train.count():,}")
print(f"Test rows:      {test.count():,}")
predictions = model.transform(test)

evaluator_rmse = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="rmse"
)

evaluator_r2 = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="r2"
)

evaluator_mae = RegressionEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="mae"
)

rmse = evaluator_rmse.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)
mae  = evaluator_mae.evaluate(predictions)

print(f"GBTRegressor Results")
print(f"--------------------")
print(f"RMSE : {rmse:.2f} minutes")
print(f"MAE  : {mae:.2f} minutes")
print(f"R²   : {r2:.4f}")
### Sample predictions and per-airline error

# Preview sample predictions vs. actual
predictions.select("Origin", "Dest", "Reporting_Airline", "label", "prediction") \
           .orderBy(F.col("label").desc()) \
           .show(20, truncate=False)
# Get total row count and show 20 rows from the middle
total = predictions.count()
middle = total // 2

predictions.select("Origin", "Dest", "Reporting_Airline", "label", "prediction") \
           .orderBy(F.col("label").desc()) \
           .limit(middle + 10) \
           .orderBy(F.col("label").asc()) \
           .limit(20) \
           .show(truncate=False)
### Feature importance (dest model)

# Extract the GBTRegressor model from the pipeline
gbt_model = model.stages[-1]

# Pair feature names with importances and sort descending
feature_importances = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": gbt_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)

print("Top 15 Feature Importances:")
print(feature_importances.head(15).to_string(index=False))
# Optional: plot feature importances

top15 = feature_importances.head(15)

plt.figure(figsize=(10, 6))
plt.barh(top15["feature"][::-1], top15["importance"][::-1])
plt.xlabel("Importance")
plt.title("GBTRegressor — Top 15 Feature Importances")
plt.tight_layout()
plt.show()
# Extract the GBTRegressor model from the pipeline
gbt_model = model.stages[-1]

# Pair feature names with importances
feature_importances = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": gbt_model.featureImportances.toArray()
}).sort_values("importance", ascending=False)

# Filter to only weather features
weather_features = [
    "avg_wind_speed", "precipitation", "avg_temp",
    "max_temp", "min_temp", "snowfall", "snow_depth"
]

weather_importances = feature_importances[
    feature_importances["feature"].isin(weather_features)
].sort_values("importance", ascending=False)

print("Weather Feature Importances:")
print(weather_importances.to_string(index=False))

# Plot weather features only

plt.figure(figsize=(8, 5))
plt.barh(weather_importances["feature"][::-1], weather_importances["importance"][::-1])
plt.xlabel("Importance")
plt.title("GBTRegressor — Weather Feature Importances")
plt.tight_layout()
plt.show()
### Best model from CrossValidator (dest)

# Evaluate best model from CV
best_predictions = cv_model.transform(test)

best_rmse = evaluator_rmse.evaluate(best_predictions)
best_r2   = evaluator_r2.evaluate(best_predictions)
best_mae  = evaluator_mae.evaluate(best_predictions)

print(f"Best Model (after CV tuning)")
print(f"----------------------------")
print(f"RMSE : {best_rmse:.2f} minutes")
print(f"MAE  : {best_mae:.2f} minutes")
print(f"R²   : {best_r2:.4f}")

# Show best params
best_gbt = cv_model.bestModel.stages[-1]
print(f"\nBest params:")
print(f"  maxIter  = {best_gbt.getMaxIter()}")
print(f"  stepSize = {best_gbt.getStepSize()}")
print(f"  maxDepth = {best_gbt.getMaxDepth()}")
### Operational delay-bucket evaluation (dest)

def delay_bucket(col_name):
    """Convert delay minutes column into operational tier labels."""
    return (
        F.when(F.col(col_name) < 30,  "minor (<30 min)")
         .when(F.col(col_name) < 60,  "moderate (30-60 min)")
         .when(F.col(col_name) < 120, "significant (1-2 hrs)")
         .otherwise("severe (2+ hrs)")
    )

best_predictions = best_predictions.withColumn("actual_bucket",     delay_bucket("label")) \
                                   .withColumn("predicted_bucket",  delay_bucket("prediction"))

# How often do we predict the right operational tier?
tier_accuracy = best_predictions.filter(
    F.col("actual_bucket") == F.col("predicted_bucket")
).count() / best_predictions.count()

print(f"Operational Tier Accuracy: {tier_accuracy:.2%}")

# Breakdown: where do we get it right vs wrong?
best_predictions.groupBy("actual_bucket", "predicted_bucket") \
                .count() \
                .orderBy("actual_bucket", "predicted_bucket") \
                .show(20, truncate=False)
## 13. Summary

| Metric | Baseline Model | Tuned Model |
|--------|---------------|-------------|
| RMSE (minutes) | — | — |
| MAE (minutes) | — | — |
| R² | — | — |
| Operational Tier Accuracy | — | — |

*(Fill in after running)*

**Interpretation guidance:**
- **RMSE** tells us the average error in minutes — if RMSE = 22, our predictions are off by about 22 minutes on average (penalizes large errors more than MAE does)
- **MAE** is easier to explain to stakeholders — "on average our prediction is X minutes off"
- **R²** tells us how much variance the model explains; closer to 1.0 is better
- **Operational Tier Accuracy** is the most business-relevant metric — gets at whether we'd make the right operational decision, not just whether the number is close
---

# Part 5 · Final Plots

Poster-ready visualizations. Project_Plots' original setup cell has been split: imports/Spark/paths moved to the master Setup cell at the top; only the plot-specific data load, color palette, helpers, and joined `df` remain here.

_Source: `Project_Plots.ipynb`._

## **Setup**
# Update this path to wherever the data files are on your google drive.
flight_data  = spark.read.parquet(PLANES_PARQUET)
weather_data = spark.read.parquet(WEATHER_PARQUET)

if "Month" not in flight_data.columns and "DATE" in flight_data.columns:
    flight_data = flight_data.withColumn("Month", F.month(F.col("DATE")))

flight_data = flight_data.cache()
weather_data = weather_data.cache()

regional_codes = ["SYR", "ROC", "BUF", "ALB", "BGM", "ITH", "ELM",
                  "PWM", "BTV", "SBY", "AVP", "MDT", "ABE"]
hub_codes      = ["ATL", "ORD", "DFW", "JFK", "LAX", "DEN", "CLT",
                  "EWR", "SFO", "IAH", "PHX", "MIA", "SEA", "BOS"]
ne_regional    = ["SYR", "ROC", "BUF", "ALB", "BGM", "ITH",
                  "PWM", "BTV", "AVP", "MDT", "ABE", "ELM"]

TEAL          = "#1ABC9C"
MIDNIGHT_BLUE = "#2C3E50"
SUNSET_ORANGE = "#E67E22"
BURNT_ORANGE  = "#935116"
BRIGHT_RED    = "#E74C3C"
SKY_BLUE      = "#5DADE2"
GOLDENROD     = "#F1C40F"
PEER_GREY     = "#7F8C8D"
LIGHT_GREY    = "#BDC3C7"
DEEP_BLUE     = "#154360"
ROYAL_BLUE    = "#3498DB"

sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.edgecolor":   "#CCCCCC",
    "axes.labelcolor":  MIDNIGHT_BLUE,
    "axes.titlecolor":  MIDNIGHT_BLUE,
    "axes.titleweight": "bold",
    "font.family":      "DejaVu Sans",
})

def add_titles(ax, title, subtitle, caption, xlab, ylab):
    ax.set_title(title, fontsize=14, fontweight="bold",
                 color=MIDNIGHT_BLUE, loc="left", pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=10, color="#555555", style="italic")
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylab, fontsize=11)
    ax.figure.text(0.01, 0.01, caption, fontsize=8, color="#777777", style="italic")

weather_per_date = weather_data.groupBy("DATE").count().agg(F.max("count")).first()[0]
print(f"Max weather rows per DATE: {weather_per_date}")

df = (
    flight_data.join(weather_data, on="DATE", how="inner")
    .filter(F.col("precipitation").isNotNull() & F.col("DepDelay").isNotNull())
    .withColumn("dep_hour", F.floor(F.col("CRSDepTime") / 100).cast("int"))
)
## **Monthly Delay Trends - Syracuse vs. Northeast Regional Peers**
I picked a line chart because it shows change over time, which is what monthly data is all about. Putting Syracuse in blue against the grey peer airports lets the reader spot SYR's pattern at a glance. The bubbles sized by flight volume add a second layer of info without needing a separate plot, so you can see both "how bad were the delays" and "how busy was the airport" at the same time.
# @title
def make_p1():
    p1_data = (
        flight_data
        .filter(F.col("Origin").isin(ne_regional))
        .filter(F.col("DepDelay").isNotNull())
        .filter(F.col("Month").isNotNull())
        .groupBy("Origin", "Month")
        .agg(
            F.avg("DepDelay").alias("Avg_Delay"),
            F.count(F.lit(1)).alias("Flight_Count"),
        )
        .filter(F.col("Flight_Count") >= 30)
        .toPandas()
        .sort_values(["Origin", "Month"])
    )

    syr = p1_data[p1_data["Origin"] == "SYR"].sort_values("Month")
    peers = p1_data[p1_data["Origin"] != "SYR"]

    fig, (ax, ax_legend) = plt.subplots(
        2, 1, figsize=(11, 8.5),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.40},
    )
    ax_legend.axis("off")

    for _, grp in peers.groupby("Origin"):
        ax.plot(grp["Month"], grp["Avg_Delay"],
                color=PEER_GREY, linewidth=0.8, alpha=0.7)

    ax.plot(syr["Month"], syr["Avg_Delay"],
            color=SKY_BLUE, linewidth=2.4, zorder=5)

    cmap = sns.blend_palette(["#F8C471", "#BA4A00"], as_cmap=True)
    fc_min, fc_max = syr["Flight_Count"].min(), syr["Flight_Count"].max()
    sizes = np.interp(syr["Flight_Count"], (fc_min, fc_max), (60, 220))
    ax.scatter(syr["Month"], syr["Avg_Delay"],
               s=sizes, c=syr["Flight_Count"], cmap=cmap,
               alpha=0.9, edgecolor="white", linewidth=0.8, zorder=6)

    for _, row in syr.iterrows():
        ax.annotate(f"{row['Avg_Delay']:.1f}",
                    xy=(row["Month"], row["Avg_Delay"]),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold",
                    color=MIDNIGHT_BLUE)

    last = syr.iloc[-1]
    ax.annotate("SYR", xy=(last["Month"], last["Avg_Delay"]),
                xytext=(10, 0), textcoords="offset points",
                color=SKY_BLUE, fontweight="bold", fontsize=12, va="center")

    lo = int(np.ceil(fc_min / 500) * 500)
    hi = int(np.floor(fc_max / 500) * 500)
    breaks = np.linspace(lo, hi, 3).astype(int)
    norm = plt.Normalize(vmin=fc_min, vmax=fc_max)

    title_y = 0.65
    bubble_y = 0.65
    label_y = 0.10

    ax_legend.text(0.02, title_y, "SYR Flight Volume",
                   ha="left", va="center",
                   fontsize=10, fontweight="bold",
                   transform=ax_legend.transAxes)

    bubble_xs = [0.22, 0.32, 0.42]
    for x, b in zip(bubble_xs, breaks):
        size = np.interp(b, (fc_min, fc_max), (60, 220))
        color = cmap(norm(b))
        ax_legend.scatter(x, bubble_y, s=size, color=[color],
                          edgecolor="white", linewidth=0.8,
                          transform=ax_legend.transAxes)
        ax_legend.text(x, label_y, f"{b:,}",
                       ha="center", va="center", fontsize=9,
                       transform=ax_legend.transAxes)

    ax_legend.text(0.98, title_y,
                   "Data Source: BTS Airline On-Time Data (planes.parquet)",
                   ha="right", va="center", fontsize=9,
                   color="#7F8C8D", style="italic",
                   transform=ax_legend.transAxes)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.margins(x=0.04, y=0.15)

    add_titles(
        ax,
        title="Monthly Delay Trends: Syracuse vs. Regional Peers",
        subtitle="SYR (blue) highlighted against other Northeast regional airports (grey)",
        caption="",
        xlab="Month",
        ylab="Average Departure Delay (minutes)",
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.06)
    return fig

_ = make_p1()
## **Root Cause of Delays**
A stacked bar chart works here because I'm comparing parts of a whole (the four delay causes) across two groups. Using proportions instead of raw minutes makes the comparison fair, since major hubs handle way more flights. The stacked layout makes it easy to see which cause dominates at each airport type without forcing the reader to do math.
# @title
def make_p2():
    CAUSE_COLORS = {
        "Security": "#1F3A68",
        "Weather": "#E87722",
        "Airspace Traffic and Mild Weather": "#F2C641",
        "Airline at Fault": "#7FB3D5",
    }

    def _delay_cause_long(base, group_cols):
        """
        Aggregate delay causes for the given grouping and return a pandas
        DataFrame with each cause as a proportion of total delay minutes.
        """
        causes = ["Weather", "Security",
                  "Airspace Traffic and Mild Weather", "Airline at Fault"]

        agg = base.groupBy(*group_cols).agg(
            F.sum("WeatherDelay").alias("Weather"),
            F.sum("SecurityDelay").alias("Security"),
            F.sum("NASDelay").alias("Airspace Traffic and Mild Weather"),
            (F.sum("CarrierDelay") + F.sum("LateAircraftDelay")).alias("Airline at Fault"),
        )

        total = sum(F.col(c) for c in causes)
        proportions = [
            F.when(total == 0, F.lit(0.0))
             .otherwise(F.col(c) / total)
             .alias(c)
            for c in causes
        ]
        return agg.select(*group_cols, *proportions).toPandas()

    base = (
        flight_data
        .filter(
            F.col("WeatherDelay").isNotNull()
            & F.col("CarrierDelay").isNotNull()
            & F.col("Origin").isin(regional_codes + hub_codes)
        )
        .withColumn(
            "Airport_Type",
            F.when(F.col("Origin").isin(regional_codes), F.lit("Regional"))
             .otherwise(F.lit("Major Hub"))
        )
    )

    pdf = _delay_cause_long(base, ["Airport_Type"])
    pdf = pdf.set_index("Airport_Type").loc[["Major Hub", "Regional"]].reset_index()

    p2_stack = ["Security",
                "Weather",
                "Airspace Traffic and Mild Weather",
                "Airline at Fault"]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    bottoms = np.zeros(len(pdf))
    for cause in p2_stack:
        ax.bar(pdf["Airport_Type"], pdf[cause], bottom=bottoms,
               color=CAUSE_COLORS[cause], edgecolor="white",
               linewidth=0.6, label=cause, width=0.6)
        bottoms += pdf[cause].values

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["Security", "Airline at Fault",
                    "Airspace Traffic and Mild Weather", "Weather"]
    order = [labels.index(c) for c in legend_order]
    ax.legend([handles[i] for i in order],
              [labels[i] for i in order],
              title="Root Cause of Delay",
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=4)

    add_titles(
        ax,
        title="Why Regional Airports Are Different: Root Cause of Delays",
        subtitle="Comparing the proportion of delay causes between regional and major hub airports",
        caption="Data Source: BTS Airline On-Time Data (planes.parquet)",
        xlab="Airport Category",
        ylab="Proportion of Total Delay Minutes",
    )
    fig.tight_layout()
    return fig

_ = make_p2()
## **Baseline vs Adverse Weather - SYR Departure Delays**
I went with a simple bar chart because the question is direct: how much worse are delays in bad weather versus normal days? Bars are the clearest way to show "this is bigger than that." Coloring the baseline differently from the adverse conditions creates a visual anchor, so the reader's eye knows what to compare everything else against.
# @title
def make_p3():
    syr_weather = weather_data.filter(F.col("NAME").contains("SYRACUSE"))
    syr_flights = flight_data.filter(F.col("Origin") == "SYR")

    joined = (
        syr_flights.join(syr_weather, on="DATE", how="inner")
        .filter(F.col("DepDelayMinutes").isNotNull())
        .withColumn(
            "condition",
            F.when(F.col("thunder") == 1, "Thunder")
             .when(F.col("snowfall") > 1, "Snow")
             .when(F.col("fog") == 1, "Fog")
             .when(F.col("precipitation") > 0.25, "Heavy rain")
             .when(F.col("avg_wind_speed") > 15, "High wind")
             .otherwise("Baseline"),
        )
    )

    pdf = (
        joined.groupBy("condition")
              .agg(
                  F.avg("DepDelayMinutes").alias("avg_delay"),
                  F.count(F.lit(1)).alias("n_flights"),
              )
              .toPandas()
    )

    order = ["Baseline", "Fog", "High wind", "Heavy rain", "Snow", "Thunder"]
    pdf["condition"] = pd.Categorical(pdf["condition"], categories=order, ordered=True)
    pdf = pdf.sort_values("condition").reset_index(drop=True)
    pdf["is_baseline"] = pdf["condition"] == "Baseline"

    fig, ax = plt.subplots(figsize=(11, 7.5))

    ADVERSE_YELLOW = "#E5B82E"
    bar_colors = [ROYAL_BLUE if b else ADVERSE_YELLOW for b in pdf["is_baseline"]]
    bars = ax.bar(
        pdf["condition"].astype(str),
        pdf["avg_delay"],
        width=0.7,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
    )

    y_max = pdf["avg_delay"].max()
    for bar, avg, n in zip(bars, pdf["avg_delay"], pdf["n_flights"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{avg:.1f} min\n(n={n:,})",
            ha="center",
            va="bottom",
            fontsize=9,
            color=MIDNIGHT_BLUE,
        )

    ax.set_ylim(0, y_max * 1.18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_titles(
        ax,
        title="Flight Departure Delays at SYR",
        subtitle="Baseline (clear & calm) days vs days with adverse weather",
        caption="Data Source: Merged Flight & Weather Data",
        xlab="",
        ylab="Average Departure Delay (minutes)",
    )
    fig.tight_layout()
    return fig

_ = make_p3()
## **GBTRegressor: Feature Importance for DepDelay**
A horizontal bar chart is the standard for feature importance because the labels are long and read better sideways. Sorting by importance puts the most useful predictors on top, which is what readers care about. Showing only the top 7 keeps it focused, since the rest had low scores and would just add noise to the chart.
numeric_features = [
    "CarrierDelay", "LateAircraftDelay", "NASDelay", "WeatherDelay",
    "dep_hour", "avg_wind_speed", "max_temp", "Month",
    "precipitation", "min_temp", "Distance", "DayOfWeek",
]
categorical_features = ["Dest", "Reporting_Airline", "Origin"]
label_col = "DepDelay"

df_model = (
    df.select(*numeric_features, *categorical_features, label_col)
      .dropna()
)

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

train_df = train_df.cache()
train_df.count()

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_features
]
indexed_cols = [f"{c}_idx" for c in categorical_features]
feature_cols = numeric_features + indexed_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

gbt = GBTRegressor(
    featuresCol="features",
    labelCol=label_col,
    maxIter=20,
    maxBins=64,
    seed=42,
)

pipeline = Pipeline(stages=indexers + [assembler, gbt])
model    = pipeline.fit(train_df)

train_df.unpersist()

gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()

importance_df = (
    pd.DataFrame({"Feature": feature_cols, "Gain": importances})
      .sort_values("Gain", ascending=False)
      .reset_index(drop=True)
)
top7 = importance_df.head(7)
print(top7)

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")

top7_plot = top7.iloc[::-1]
ax.barh(top7_plot["Feature"], top7_plot["Gain"],
        color="#F1C40F", edgecolor="black")

ax.set_title("GBTRegressor: Top 7 Feature Importances",
             color="black", fontsize=14, pad=15)
ax.set_xlabel("Importance", color="black", fontsize=12)
ax.set_ylabel("")

ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xlim(0, 0.5)

ax.tick_params(colors="black")
for spine in ax.spines.values():
    spine.set_color("black")
ax.grid(axis="x", color="black", alpha=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()