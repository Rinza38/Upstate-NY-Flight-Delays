"""
IST 418 Final Project - Flight Delay Prediction (refactored)
============================================================
Upstate NY flight delays modeled with PySpark on a merged BTS + NOAA
dataset (2015-2025).

WHAT CHANGED VS THE ORIGINAL NOTEBOOK
-------------------------------------
1. The origin and destination CLASSIFIERS were near-identical copies. They
   now share ONE function: run_classifier(). You call it twice, once per path.
2. The origin and destination REGRESSORS were also copies. They now share
   run_regressor(). This also fixes a real bug: in the original, section 4b
   loaded the destination CSV but never re-ran filtering / cleaning / training,
   so every "destination" regressor result was actually the ORIGIN model
   evaluated on a fresh split of origin data. Calling run_regressor(DEST_PATH)
   retrains end to end, so the dest numbers are now genuinely dest numbers.
3. The elevation column is auto-detected ("ELEVATION" vs "Elevation") because
   the two source CSVs used different casing and spark.sql.caseSensitive was on.
4. Feature lists are filtered to columns that actually exist before use, so a
   missing column (e.g. avg_temp, multiday_precip) no longer crashes the run.
5. Style cleanup: consistent F.* usage, no dead re-imports, config centralized.

THINGS TO REVIEW BEFORE YOU TRUST THE RESULTS (not silently changed)
--------------------------------------------------------------------
* DATA LEAKAGE in the regressor. CarrierDelay, WeatherDelay, NASDelay and
  LateAircraftDelay are COMPONENTS of the delay you are predicting
  (ArrDelayMinutes). Feeding them in inflates R-squared and is not a real
  forecast. Set DROP_LEAKY_FEATURES = True to model delay from weather/time
  only. Left False here to match your original output.
* spark.sql.caseSensitive is left OFF (default). The original set it True,
  which is what forced the ELEVATION/Elevation mismatch to matter.

Run order in Colab: run SETUP once, then call the functions in main().
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# confusion matrix (only needed if you plot it)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ============================================================================
# 2. CONFIG - edit paths here once; everything else follows
# ============================================================================
ORIGIN_CSV_PATH = "/content/drive/MyDrive/418 Final Folder/df_mergedorigin.csv"
DEST_CSV_PATH   = "/content/drive/MyDrive/418 Final Folder/df_mergeddest.csv"

# Columns pulled from the raw merged CSV (>100 cols) before modeling.
COLS_TO_KEEP = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "DATE",
    "Reporting_Airline", "DOT_ID_Reporting_Airline", "Flight_Number_Reporting_Airline",
    "Origin", "OriginCityName", "Dest", "DestCityName",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepartureDelayGroups", "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
    "ArrDel15", "ArrivalDelayGroups", "Cancelled", "CRSElapsedTime", "ActualElapsedTime",
    "AirTime", "Distance", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "avg_wind_speed", "peak_gust_time", "precipitation",
    "snowfall", "snow_depth", "max_temp", "min_temp", "fog", "thunder",
    "ice_sleet", "hail", "glaze_rime",
]  # elevation handled separately by _elevation_col()

# Weather flags stored as string booleans in the raw data.
BINARY_COLS = ["fog", "thunder", "ice_sleet", "hail", "glaze_rime"]

# Base classifier feature set (elevation appended at runtime if present).
CLASSIFIER_NUMERIC = [
    "dep_hour", "DayOfWeek", "Month", "Distance", "CRSElapsedTime",
    "max_temp", "min_temp", "precipitation", "snowfall", "snow_depth",
    "avg_wind_speed", "fog", "thunder", "ice_sleet", "hail", "glaze_rime",
]

# Regressor feature sets.
REGRESSOR_CATEGORICAL = ["Reporting_Airline", "Origin", "Dest"]
REGRESSOR_NUMERIC = [
    "Month", "DayOfWeek", "dep_hour", "season", "is_weekend", "Distance",
    "avg_wind_speed", "precipitation", "avg_temp",
    "max_temp", "min_temp", "snowfall", "snow_depth",
    "CarrierDelay", "WeatherDelay", "NASDelay", "LateAircraftDelay",
]
# Columns that leak the label; toggle to remove them (see module docstring).
LEAKY_COLS = ["CarrierDelay", "WeatherDelay", "NASDelay", "LateAircraftDelay"]
DROP_LEAKY_FEATURES = False


# ============================================================================
# 3. SPARK SESSION
# ============================================================================
def build_spark():
    """Create the SparkSession used by every stage below."""
    spark = (
        SparkSession.builder
        .appName("IST418FinalProject")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    # NOTE: caseSensitive intentionally left at its default (False).
    return spark


# ============================================================================
# 4. SHARED CLEANING HELPERS
# ============================================================================
def _replace_na_strings(df):
    """Cast every column to string, then turn the literal text "NA" into null."""
    df = df.select([F.col(c).cast("string").alias(c) for c in df.columns])
    df = df.select([
        F.when(F.col(c) == "NA", None).otherwise(F.col(c)).alias(c)
        for c in df.columns
    ])
    return df


def _encode_binaries(df, cols=BINARY_COLS):
    """Convert string booleans ("True"/"False"/"1"/"0") to 1.0 / 0.0 doubles."""
    for c in cols:
        if c in df.columns:
            df = df.withColumn(
                c,
                F.when(F.col(c).isin("TRUE", "True", "true", "1"), 1.0)
                 .when(F.col(c).isin("FALSE", "False", "false", "0"), 0.0)
                 .otherwise(None)
                 .cast(DoubleType()),
            )
    return df


def _elevation_col(df):
    """Return whichever elevation column casing exists in this DataFrame."""
    for name in ("ELEVATION", "Elevation", "elevation"):
        if name in df.columns:
            return name
    return None


def _existing(df, cols):
    """Keep only the requested columns that are actually present."""
    present = set(df.columns)
    return [c for c in cols if c in present]


# ============================================================================
# 5. CLASSIFIER - "will this flight be delayed?" (binary)
# ============================================================================
def run_classifier(spark, csv_path, label_col, time_col, positive_rate_label=""):
    """
    Train + evaluate one GBT delay classifier.

    Origin model : run_classifier(spark, ORIGIN_CSV_PATH, "DepDel15", "CRSDepTime")
    Dest   model : run_classifier(spark, DEST_CSV_PATH,   "ArrDel15", "CRSArrTime")

    Returns (fitted_pipeline_model, predictions_df).
    """
    print(f"\n=== CLASSIFIER: {label_col} from {csv_path} ===")

    flight_df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = flight_df.select(_existing(flight_df, COLS_TO_KEEP + ["ELEVATION", "Elevation"]))

    # Detect elevation casing and standardize to "ELEVATION".
    elev = _elevation_col(df)
    if elev and elev != "ELEVATION":
        df = df.withColumnRenamed(elev, "ELEVATION")
        elev = "ELEVATION"

    # Clean literal "NA" -> null, then encode binaries.
    df = _replace_na_strings(df)
    df = _encode_binaries(df)

    # Label + time-derived features.
    df = df.withColumn(label_col, F.col(label_col).cast(DoubleType()))
    df = df.withColumn(time_col, F.col(time_col).cast(IntegerType()))
    df = df.withColumn("dep_hour", (F.col(time_col) / 100).cast("int"))

    # Build the feature list, adding elevation only if it exists.
    numeric_features = list(CLASSIFIER_NUMERIC)
    if elev:
        numeric_features.append("ELEVATION")
    numeric_features = _existing(df, numeric_features)

    # Cast non-binary numeric features to double.
    for c in numeric_features:
        if c not in BINARY_COLS:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # Weather absence == 0 for flags / precip / wind.
    df = df.fillna({
        "fog": 0.0, "thunder": 0.0, "ice_sleet": 0.0, "hail": 0.0, "glaze_rime": 0.0,
        "precipitation": 0.0, "avg_wind_speed": 0.0,
    })

    # Class imbalance report + 2x weight on the (rarer) delayed class.
    total = df.count()
    delayed = df.filter(F.col(label_col) == 1).count()
    print(f"Delay rate {positive_rate_label}: {delayed / total:.3f}")
    df = df.withColumn("classWeight", F.when(F.col(label_col) == 1, 2.0).otherwise(1.0))

    df = df.dropna(subset=numeric_features + [label_col, "Reporting_Airline"])

    # Pipeline: index airline + origin, assemble, GBT.
    indexers = [
        StringIndexer(inputCol="Reporting_Airline", outputCol="airline_idx", handleInvalid="keep"),
        StringIndexer(inputCol="Origin", outputCol="origin_idx", handleInvalid="keep"),
    ]
    assembler = VectorAssembler(
        inputCols=numeric_features + ["airline_idx", "origin_idx"],
        outputCol="features",
        handleInvalid="skip",
    )
    gbt = GBTClassifier(
        featuresCol="features", labelCol=label_col, weightCol="classWeight",
        maxIter=50, maxDepth=4, maxBins=64, stepSize=0.1, seed=42,
    )
    pipeline = Pipeline(stages=indexers + [assembler, gbt])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train_df.count()}, Test: {test_df.count()}")

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    _report_classifier(predictions, label_col)
    return model, predictions


def _report_classifier(predictions, label_col):
    """Print AUC, F1, prediction distribution, and confusion matrix."""
    auc = BinaryClassificationEvaluator(
        labelCol=label_col, metricName="areaUnderROC"
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)
    print(f"AUC: {auc:.4f}")
    print(f"F1 : {f1:.4f}")

    predictions.groupBy("prediction").count().show()

    pred_pdf = predictions.select(label_col, "prediction").toPandas()
    cm = confusion_matrix(pred_pdf[label_col], pred_pdf["prediction"])
    ConfusionMatrixDisplay(cm, display_labels=["On-Time", "Delayed"]).plot(cmap="Blues")
    plt.title(f"Flight Delay Prediction - Confusion Matrix ({label_col})")
    plt.show()


# ============================================================================
# 6. REGRESSOR - "how many minutes late?" (delayed flights only)
# ============================================================================
def run_regressor(spark, csv_path, tune=True):
    """
    Train + evaluate one GBT delay-duration regressor end to end.

    Origin: run_regressor(spark, ORIGIN_CSV_PATH)
    Dest  : run_regressor(spark, DEST_CSV_PATH)   # now genuinely retrains

    Returns (fitted_model, best_model_or_None, test_predictions).
    """
    print(f"\n=== REGRESSOR from {csv_path} ===")

    df_raw = spark.read.csv(csv_path, header=True, inferSchema=True, nullValue="NA")
    df_raw = (
        df_raw
        .withColumn("ArrDel15", F.col("ArrDel15").cast("int"))
        .withColumn("ArrDelayMinutes", F.col("ArrDelayMinutes").cast("double"))
    )
    print(f"Total rows: {df_raw.count():,} | columns: {len(df_raw.columns)}")

    # Regressor only makes sense on flights that were actually delayed.
    df = (
        df_raw.filter(F.col("ArrDel15") == 1)
              .dropna(subset=["ArrDelayMinutes"])
              .withColumnRenamed("ArrDelayMinutes", "label")
    )
    print(f"Delayed flight rows: {df.count():,}")

    # Fill delay-cause and weather nulls with 0 (only for columns present).
    fill_zero = _existing(df, LEAKY_COLS + ["SecurityDelay"] + [
        "avg_wind_speed", "precipitation", "avg_temp",
        "max_temp", "min_temp", "multiday_precip", "snowfall", "snow_depth",
    ])
    df = df.fillna(0, subset=fill_zero)

    # Temporal features.
    df = df.withColumn("dep_hour", (F.col("CRSDepTime") / 100).cast("int"))
    df = df.withColumn(
        "season",
        F.when(F.col("Month").isin([12, 1, 2]), 1)
         .when(F.col("Month").isin([3, 4, 5]), 2)
         .when(F.col("Month").isin([6, 7, 8]), 3)
         .otherwise(4),
    )
    df = df.withColumn(
        "is_weekend", F.when(F.col("DayOfWeek").isin([6, 7]), 1).otherwise(0)
    )

    # Feature lists, filtered to what exists and (optionally) de-leaked.
    numeric = _existing(df, REGRESSOR_NUMERIC)
    if DROP_LEAKY_FEATURES:
        numeric = [c for c in numeric if c not in LEAKY_COLS]
        print(f"Leaky columns removed: {LEAKY_COLS}")
    categorical = _existing(df, REGRESSOR_CATEGORICAL)
    indexed = [c + "_idx" for c in categorical]
    all_features = numeric + indexed

    df = df.dropna(subset=numeric + categorical + ["label"])
    train, test = df.randomSplit([0.7, 0.3], seed=1458)
    print(f"Training rows: {train.count():,} | Test rows: {test.count():,}")

    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in categorical
    ]
    assembler = VectorAssembler(inputCols=all_features, outputCol="features", handleInvalid="skip")
    gbt = GBTRegressor(
        featuresCol="features", labelCol="label",
        maxIter=100, stepSize=0.1, maxDepth=5, maxBins=64, seed=1458,
    )
    pipeline = Pipeline(stages=indexers + [assembler, gbt])

    print("Training GBTRegressor...")
    model = pipeline.fit(train)
    predictions = model.transform(test)
    _report_regressor(predictions, "Baseline")

    _feature_importance(model, all_features)

    best_model = None
    if tune:
        best_model = _tune_regressor(pipeline, indexers, assembler, train, test)

    return model, best_model, predictions


def _report_regressor(predictions, tag):
    """Print RMSE / MAE / R-squared for a set of predictions."""
    def metric(name):
        return RegressionEvaluator(labelCol="label", predictionCol="prediction",
                                   metricName=name).evaluate(predictions)
    print(f"\n{tag} GBTRegressor Results")
    print("-" * 28)
    print(f"RMSE : {metric('rmse'):.2f} minutes")
    print(f"MAE  : {metric('mae'):.2f} minutes")
    print(f"R2   : {metric('r2'):.4f}")


def _feature_importance(model, all_features, top_n=15):
    """Print the top-N GBT feature importances."""
    gbt_model = model.stages[-1]
    fi = (
        pd.DataFrame({"feature": all_features,
                      "importance": gbt_model.featureImportances.toArray()})
          .sort_values("importance", ascending=False)
          .reset_index(drop=True)
    )
    print(f"\nTop {top_n} feature importances:")
    print(fi.head(top_n).to_string(index=False))
    return fi


def _tune_regressor(pipeline, indexers, assembler, train, test):
    """TrainValidationSplit over maxIter / stepSize / maxDepth on a 30% sample."""
    gbt_tune = GBTRegressor(featuresCol="features", labelCol="label", maxBins=64, seed=1458)
    tune_pipeline = Pipeline(stages=indexers + [assembler, gbt_tune])

    grid = (
        ParamGridBuilder()
        .addGrid(gbt_tune.maxIter, [50, 100])
        .addGrid(gbt_tune.stepSize, [0.05, 0.1])
        .addGrid(gbt_tune.maxDepth, [4, 5])
        .build()
    )
    tvs = TrainValidationSplit(
        estimator=tune_pipeline, estimatorParamMaps=grid,
        evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"),
        trainRatio=0.7, seed=1458,
    )

    sample = train.sample(fraction=0.3, seed=1458)
    print(f"\nTuning on {sample.count():,} rows over {len(grid)} combos...")
    cv_model = tvs.fit(sample)

    best_predictions = cv_model.transform(test)
    _report_regressor(best_predictions, "Tuned")
    _operational_accuracy(best_predictions)

    best = cv_model.bestModel.stages[-1]
    print(f"Best params: maxIter={best.getMaxIter()}, "
          f"stepSize={best.getStepSize()}, maxDepth={best.getMaxDepth()}")
    return cv_model


def _operational_accuracy(predictions):
    """Business metric: how often we predict the correct delay tier."""
    def bucket(col_name):
        return (
            F.when(F.col(col_name) < 30, "minor (<30 min)")
             .when(F.col(col_name) < 60, "moderate (30-60 min)")
             .when(F.col(col_name) < 120, "significant (1-2 hrs)")
             .otherwise("severe (2+ hrs)")
        )
    preds = (
        predictions
        .withColumn("actual_bucket", bucket("label"))
        .withColumn("predicted_bucket", bucket("prediction"))
    )
    correct = preds.filter(F.col("actual_bucket") == F.col("predicted_bucket")).count()
    print(f"Operational Tier Accuracy: {correct / preds.count():.2%}")


# ============================================================================
# 7. ORCHESTRATION
# ============================================================================
def main():
    spark = build_spark()

    # --- Part 3: classifiers ---
    run_classifier(spark, ORIGIN_CSV_PATH, "DepDel15", "CRSDepTime", "(origin)")
    run_classifier(spark, DEST_CSV_PATH, "ArrDel15", "CRSArrTime", "(dest)")

    # --- Part 4: regressors (dest now retrains correctly) ---
    run_regressor(spark, ORIGIN_CSV_PATH, tune=True)
    run_regressor(spark, DEST_CSV_PATH, tune=True)


if __name__ == "__main__":
    main()
