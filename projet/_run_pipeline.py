"""DE2 Final Project — Pipeline runner (Track A: CS:GO/HLTV)."""
import yaml, pathlib, datetime, time, json, os, io, sys, csv, statistics

# ── Spark ─────────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

with open("de2_project_config.yml") as f:
    CFG = yaml.safe_load(f)

spark = (SparkSession.builder
    .appName("de2-project")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
print("Spark:", spark.version)
print("UI:", spark.sparkContext.uiWebUrl)

for key in ["bronze", "silver", "gold", "streaming_sink", "streaming_checkpoint",
            "streaming_landing", "inverted_index", "models", "llm_ready", "proof"]:
    p = CFG["paths"].get(key, "")
    if p:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def dir_size_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for dp, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(dp, fn)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def capture_plan(df):
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    df.explain(extended=True)
    sys.stdout = old
    return buf.getvalue()

metrics_rows = []

def log_metric(stage, task, phase, name, value,
               shuffle_r="", shuffle_w="", elapsed="", notes=""):
    metrics_rows.append([
        RUN_ID, stage, task, phase, name,
        round(float(value), 4) if value != "" else "",
        shuffle_r, shuffle_w,
        round(float(elapsed), 1) if elapsed != "" else "",
        notes,
        datetime.datetime.now().isoformat()
    ])

RUN_ID = "r" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PROOF_DIR = CFG["paths"]["proof"]
print("Run ID:", RUN_ID)

# ════════════════════════════════════════════════════════════════════════════
# 1. BRONZE — raw landing
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 1. BRONZE ===")
BRONZE = CFG["paths"]["bronze"]
RAW_CSV = CFG["paths"]["raw_csv"]
t0 = time.time()

df_raw = (spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(RAW_CSV))

raw_count = df_raw.count()
elapsed = (time.time() - t0) * 1000
df_raw.write.mode("overwrite").option("header", "true").csv(BRONZE)

raw_mb    = dir_size_mb(RAW_CSV)
bronze_mb = dir_size_mb(BRONZE)

print(f"Raw rows   : {raw_count:,}")
print(f"Source CSV : {raw_mb:.2f} MB")
print(f"Bronze CSV : {bronze_mb:.2f} MB")
print(f"Elapsed    : {elapsed:.0f} ms")
df_raw.printSchema()

log_metric("ETL", "bronze_landing", "baseline", "row_count", raw_count, elapsed=elapsed)
log_metric("ETL", "bronze_landing", "baseline", "size_mb",   bronze_mb)

# ════════════════════════════════════════════════════════════════════════════
# 2. SILVER — cleaning, typing, schema contracts
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 2. SILVER ===")
SILVER = CFG["paths"]["silver"]
t0 = time.time()

df_b = (spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv(BRONZE))

df_silver = (df_b
    .withColumn("date",         F.to_date("date", "yyyy-MM-dd"))
    .withColumn("result_1",     F.col("result_1").cast(T.IntegerType()))
    .withColumn("result_2",     F.col("result_2").cast(T.IntegerType()))
    .withColumn("map_winner",   F.col("map_winner").cast(T.IntegerType()))
    .withColumn("starting_ct",  F.col("starting_ct").cast(T.IntegerType()))
    .withColumn("ct_1",         F.col("ct_1").cast(T.IntegerType()))
    .withColumn("t_2",          F.col("t_2").cast(T.IntegerType()))
    .withColumn("t_1",          F.col("t_1").cast(T.IntegerType()))
    .withColumn("ct_2",         F.col("ct_2").cast(T.IntegerType()))
    .withColumn("event_id",     F.col("event_id").cast(T.IntegerType()))
    .withColumn("match_id",     F.col("match_id").cast(T.LongType()))
    .withColumn("rank_1",       F.col("rank_1").cast(T.IntegerType()))
    .withColumn("rank_2",       F.col("rank_2").cast(T.IntegerType()))
    .withColumn("map_wins_1",   F.col("map_wins_1").cast(T.IntegerType()))
    .withColumn("map_wins_2",   F.col("map_wins_2").cast(T.IntegerType()))
    .withColumn("match_winner", F.col("match_winner").cast(T.IntegerType()))
    .withColumn("year",         F.year("date"))
    .withColumn("month",        F.month("date"))
    .withColumn("score_diff",   F.abs(F.col("result_1") - F.col("result_2")))
    .dropna(subset=["date", "team_1", "team_2", "_map", "match_id"])
    .dropDuplicates(["match_id", "_map"])
    .filter(F.col("result_1").isNotNull() & F.col("result_2").isNotNull()))

df_silver.cache()
silver_count = df_silver.count()
elapsed = (time.time() - t0) * 1000
df_silver.write.mode("overwrite").parquet(SILVER)
silver_mb = dir_size_mb(SILVER)

null_dates = df_silver.filter(F.col("date").isNull()).count()
print(f"Silver rows    : {silver_count:,}")
print(f"Silver Parquet : {silver_mb:.2f} MB")
print(f"Compression    : {silver_mb / raw_mb:.1%} of source CSV")
print(f"Null dates     : {null_dates} (contract OK)")
print(f"Elapsed        : {elapsed:.0f} ms")
df_silver.printSchema()

pathlib.Path(f"{PROOF_DIR}/plan_silver.txt").write_text(capture_plan(df_silver))
log_metric("ETL", "silver_cleaning", "baseline", "row_count",       silver_count, elapsed=elapsed)
log_metric("ETL", "silver_cleaning", "baseline", "size_mb",         silver_mb)
log_metric("ETL", "silver_cleaning", "baseline", "null_violations",  null_dates)
df_silver.unpersist()

# ════════════════════════════════════════════════════════════════════════════
# 3. GOLD — analytics tables
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 3. GOLD ===")
GOLD    = CFG["paths"]["gold"]
PART_BY = CFG["layout"]["partition_by"]
t0 = time.time()

df_s = spark.read.parquet(SILVER)

# Gold Q1 — team performance
team_stats = (df_s
    .groupBy("team_1", "year")
    .agg(
        F.count("*").alias("maps_played"),
        F.sum(F.when(F.col("map_winner") == 1, 1).otherwise(0)).alias("maps_won"),
        F.avg("rank_1").alias("avg_rank"),
        F.avg("score_diff").alias("avg_score_diff"),
        F.avg("result_1").alias("avg_score")
    )
    .withColumn("win_rate", F.col("maps_won") / F.col("maps_played"))
    .withColumnRenamed("team_1", "team_name"))
team_stats.write.mode("overwrite").partitionBy("year").parquet(f"{GOLD}/team_stats")

# Gold Q2 — map stats
map_stats = (df_s
    .groupBy("_map", "year", "month")
    .agg(
        F.count("*").alias("matches_played"),
        F.avg("result_1").alias("avg_score_t1"),
        F.avg("result_2").alias("avg_score_t2"),
        F.avg("score_diff").alias("avg_score_diff"),
        F.countDistinct("event_id").alias("events_count")
    ))
map_stats.write.mode("overwrite").partitionBy(*PART_BY).parquet(f"{GOLD}/map_stats")

# Gold Q3 — match results enriched
match_results = (df_s
    .withColumn("winner_name",  F.when(F.col("map_winner") == 1, F.col("team_1")).otherwise(F.col("team_2")))
    .withColumn("loser_name",   F.when(F.col("map_winner") == 1, F.col("team_2")).otherwise(F.col("team_1")))
    .withColumn("winner_score", F.when(F.col("map_winner") == 1, F.col("result_1")).otherwise(F.col("result_2")))
    .withColumn("loser_score",  F.when(F.col("map_winner") == 1, F.col("result_2")).otherwise(F.col("result_1")))
    .select("match_id", "_map", "date", "year", "month", "team_1", "team_2",
            "result_1", "result_2", "winner_name", "loser_name",
            "winner_score", "loser_score", "rank_1", "rank_2",
            "score_diff", "event_id"))
match_results.write.mode("overwrite").partitionBy(*PART_BY).parquet(f"{GOLD}/match_results")

elapsed_gold = (time.time() - t0) * 1000
gold_mb = dir_size_mb(GOLD)

ts_count = team_stats.count()
ms_count = map_stats.count()
mr_count = match_results.count()
print(f"team_stats rows   : {ts_count:,}")
print(f"map_stats rows    : {ms_count:,}")
print(f"match_results rows: {mr_count:,}")
print(f"Gold total        : {gold_mb:.2f} MB")
print(f"Elapsed           : {elapsed_gold:.0f} ms")

pathlib.Path(f"{PROOF_DIR}/plan_etl_silver_to_gold.txt").write_text(capture_plan(match_results))
log_metric("ETL", "silver_to_gold", "baseline", "size_mb",      gold_mb,   elapsed=elapsed_gold)
log_metric("ETL", "silver_to_gold", "baseline", "team_rows",    ts_count)
log_metric("ETL", "silver_to_gold", "baseline", "map_rows",     ms_count)
log_metric("ETL", "silver_to_gold", "baseline", "match_rows",   mr_count)

# ════════════════════════════════════════════════════════════════════════════
# 4. STREAMING
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 4. STREAMING ===")
LANDING    = CFG["paths"]["streaming_landing"]
STREAM_OUT = CFG["paths"]["streaming_sink"]
STREAM_CHK = CFG["paths"]["streaming_checkpoint"]

df_s = spark.read.parquet(SILVER)
df_land = (df_s
    .withColumn("event_timestamp",
        F.to_timestamp(F.col("date").cast("string"), "yyyy-MM-dd"))
    .select("match_id", "event_timestamp", "_map",
            "score_diff", "rank_1", "rank_2", "year"))

df_land.repartition(5).write.mode("overwrite").option("header", "true").csv(LANDING)
print(f"Landing CSV files written to {LANDING}")

stream_schema = T.StructType([
    T.StructField("match_id",        T.LongType()),
    T.StructField("event_timestamp", T.TimestampType()),
    T.StructField("_map",            T.StringType()),
    T.StructField("score_diff",      T.IntegerType()),
    T.StructField("rank_1",          T.IntegerType()),
    T.StructField("rank_2",          T.IntegerType()),
    T.StructField("year",            T.IntegerType()),
])

df_stream = (spark.readStream
    .schema(stream_schema)
    .option("header", "true")
    .option("maxFilesPerTrigger", CFG["streaming"]["max_files_per_trigger"])
    .csv(LANDING))

windowed = (df_stream
    .withWatermark("event_timestamp", CFG["streaming"]["watermark"])
    .groupBy(
        F.window("event_timestamp", CFG["streaming"]["window_duration"]),
        "_map")
    .agg(
        F.count("*").alias("match_count"),
        F.avg("score_diff").alias("avg_score_diff"),
        F.avg("rank_1").alias("avg_rank")))

query = (windowed.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", STREAM_OUT)
    .option("checkpointLocation", STREAM_CHK)
    .trigger(processingTime=CFG["streaming"]["trigger_interval"])
    .start())

print("Streaming started, waiting 90 s ...")
query.awaitTermination(timeout=90)
progress = query.lastProgress or {}
print(json.dumps(progress, indent=2, default=str))

pathlib.Path(f"{PROOF_DIR}/query_progress.json").write_text(
    json.dumps(progress, indent=2, default=str))

rows_sec = float(progress.get("processedRowsPerSecond") or 0)
num_rows = int(progress.get("numInputRows") or 0)
log_metric("Streaming", "window_agg", "baseline", "processedRowsPerSecond", rows_sec)
log_metric("Streaming", "window_agg", "baseline", "numInputRows", num_rows)
query.stop()

stream_mb = dir_size_mb(STREAM_OUT)
print(f"Streaming output: {stream_mb:.3f} MB")
log_metric("Streaming", "window_agg", "baseline", "output_size_mb", stream_mb)
print("Streaming done.")

# ════════════════════════════════════════════════════════════════════════════
# 5. TEXT PIPELINE — inverted index
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 5. TEXT PIPELINE ===")
TEXT_OUT   = CFG["paths"]["inverted_index"]
stop_words = set(CFG["text"]["stop_words"])
min_len    = CFG["text"]["min_token_length"]

df_s = spark.read.parquet(SILVER)

corpus = (df_s
    .withColumn("doc_id",
        F.concat(F.col("match_id").cast("string"), F.lit("_"), F.col("_map")))
    .withColumn("text",
        F.concat_ws(" ",
            F.col("team_1"), F.col("team_2"), F.col("_map"),
            F.col("result_1").cast("string"),
            F.col("result_2").cast("string")))
    .select("doc_id", "text", "date", "_map", "team_1", "team_2"))

corpus_count = corpus.count()
print(f"Corpus documents: {corpus_count:,}")

tokens = (corpus
    .withColumn("text_clean",
        F.regexp_replace(F.lower(F.col("text")), r"[^a-z0-9\s]", " "))
    .withColumn("token", F.explode(F.split(F.col("text_clean"), r"\s+")))
    .filter(F.length(F.col("token")) >= min_len)
    .filter(~F.col("token").isin(list(stop_words))))

inverted_index = (tokens
    .groupBy("token")
    .agg(
        F.collect_set("doc_id").alias("doc_ids"),
        F.count("*").alias("freq"))
    .orderBy(F.desc("freq")))

t0 = time.time()
inverted_index.write.mode("overwrite").parquet(TEXT_OUT)
elapsed_idx = (time.time() - t0) * 1000
unique_terms = spark.read.parquet(TEXT_OUT).count()
print(f"Unique terms: {unique_terms:,} | Build time: {elapsed_idx:.0f} ms")
inverted_index.show(10, truncate=False)

# Storage footprint
corpus_csv_path     = "outputs/project/corpus_csv"
corpus_parquet_path = "outputs/project/corpus_parquet"
pathlib.Path(corpus_csv_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(corpus_parquet_path).mkdir(parents=True, exist_ok=True)
corpus.write.mode("overwrite").option("header", "true").csv(corpus_csv_path)
corpus.write.mode("overwrite").parquet(corpus_parquet_path)

csv_mb  = dir_size_mb(corpus_csv_path)
parq_mb = dir_size_mb(corpus_parquet_path)
idx_mb  = dir_size_mb(TEXT_OUT)
ratio   = parq_mb / csv_mb if csv_mb > 0 else 0

print(f"Corpus CSV     : {csv_mb:.2f} MB")
print(f"Corpus Parquet : {parq_mb:.2f} MB ({ratio:.1%} of CSV)")
print(f"Index Parquet  : {idx_mb:.2f} MB")

pathlib.Path(f"{PROOF_DIR}/plan_index_build.txt").write_text(capture_plan(inverted_index))
log_metric("Text", "build_index", "baseline", "unique_terms",      unique_terms,   elapsed=elapsed_idx)
log_metric("Text", "build_index", "baseline", "index_mb",          idx_mb)
log_metric("Text", "build_index", "baseline", "csv_mb",            csv_mb)
log_metric("Text", "build_index", "baseline", "parquet_mb",        parq_mb)
log_metric("Text", "build_index", "baseline", "compression_ratio", ratio)

# Query latency
idx = spark.read.parquet(TEXT_OUT)
idx.cache()
idx.count()

query_terms = CFG["text"]["query_terms"]
print("\nQuery latency benchmark:")
max_lat = 0
for term in query_terms:
    t0 = time.time()
    rows = idx.filter(F.col("token") == term).collect()
    lat_ms = (time.time() - t0) * 1000
    doc_count = len(rows[0]["doc_ids"]) if rows else 0
    max_lat = max(max_lat, lat_ms)
    print(f"  '{term}': {lat_ms:.1f} ms, {doc_count} docs")
    log_metric("Text", "query_latency", "baseline", f"latency_ms_{term}", lat_ms)

print(f"Max latency: {max_lat:.1f} ms")
print(f"SLO <= {CFG['slo']['text_query_latency_ms_max']} ms:",
      "PASS" if max_lat <= CFG["slo"]["text_query_latency_ms_max"] else "FAIL")
log_metric("Text", "query_latency", "baseline", "max_latency_ms", max_lat)
pathlib.Path(f"{PROOF_DIR}/plan_query.txt").write_text(
    capture_plan(idx.filter(F.col("token") == query_terms[0])))

# ════════════════════════════════════════════════════════════════════════════
# 6. CLUSTERING — KMeans sweep
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 6. CLUSTERING ===")
cl_cfg    = CFG["iterative"]["clustering"]
FEAT_COLS = cl_cfg["feature_cols"]
K_VALUES  = cl_cfg["k_values"]
SEEDS     = cl_cfg["seeds"]
MIN_MAPS  = cl_cfg["min_maps_played"]

df_s = spark.read.parquet(SILVER)

team_feat = (df_s
    .groupBy("team_1")
    .agg(
        F.count("*").alias("maps_played"),
        F.avg(F.when(F.col("map_winner") == 1, 1).otherwise(0)).alias("win_rate"),
        F.avg("rank_1").alias("avg_rank"),
        F.avg("score_diff").alias("avg_score_diff"),
        F.avg("result_1").alias("avg_score"),
        F.countDistinct("_map").alias("maps_variety"),
        F.countDistinct("event_id").alias("events_played")
    )
    .filter(F.col("maps_played") >= MIN_MAPS)
    .withColumnRenamed("team_1", "team_name"))

team_feat.cache()
n_teams = team_feat.count()
print(f"Teams eligible for clustering: {n_teams}")

assembler    = VectorAssembler(inputCols=FEAT_COLS, outputCol="raw_features", handleInvalid="skip")
scaler       = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
df_assembled = assembler.transform(team_feat)
scaler_model = scaler.fit(df_assembled)
df_scaled    = scaler_model.transform(df_assembled)
df_scaled.cache()
df_scaled.count()

evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")
cluster_results = []
t0_sweep = time.time()

for k in K_VALUES:
    sil_scores = []
    for seed in SEEDS:
        km    = KMeans(featuresCol="scaled_features", k=k, seed=seed, maxIter=30)
        model = km.fit(df_scaled)
        preds = model.transform(df_scaled)
        sil   = evaluator.evaluate(preds)
        sil_scores.append(sil)
        print(f"  k={k}, seed={seed}: silhouette={sil:.4f}")
    mean_sil = statistics.mean(sil_scores)
    std_sil  = statistics.stdev(sil_scores) if len(sil_scores) > 1 else 0.0
    cluster_results.append({"k": k, "mean_sil": mean_sil, "std_sil": std_sil})
    print(f"k={k}: mean={mean_sil:.4f}  std={std_sil:.4f}")
    log_metric("Iterative", "kmeans_sweep", "baseline",
               f"silhouette_k{k}", mean_sil, notes=f"std={std_sil:.4f}")

elapsed_sweep = (time.time() - t0_sweep) * 1000
best   = max(cluster_results, key=lambda x: x["mean_sil"])
best_k = best["k"]
print(f"\nBest k={best_k} silhouette={best['mean_sil']:.4f} (sweep {elapsed_sweep:.0f} ms)")
print(f"SLO >= {CFG['slo']['iterative_quality_min']}:",
      "PASS" if best["mean_sil"] >= CFG["slo"]["iterative_quality_min"] else "FAIL")

log_metric("Iterative", "kmeans_sweep", "baseline", "best_k",          best_k)
log_metric("Iterative", "kmeans_sweep", "baseline", "best_silhouette", best["mean_sil"], elapsed=elapsed_sweep)

# Final model
best_model  = KMeans(featuresCol="scaled_features", k=best_k, seed=42, maxIter=30).fit(df_scaled)
final_preds = best_model.transform(df_scaled)
MODELS_PATH = CFG["paths"]["models"]
best_model.write().overwrite().save(f"{MODELS_PATH}/kmeans_k{best_k}")

final_preds.select(
    "team_name", "win_rate", "avg_rank", "maps_played",
    "avg_score_diff", "maps_variety", "prediction"
).write.mode("overwrite").parquet(f"{GOLD}/team_clusters")

print("\nCluster composition:")
(final_preds
    .groupBy("prediction")
    .agg(
        F.count("*").alias("n_teams"),
        F.round(F.avg("win_rate"), 3).alias("avg_win_rate"),
        F.round(F.avg("avg_rank"), 1).alias("avg_rank"),
        F.round(F.avg("maps_played"), 1).alias("avg_maps_played")
    ).orderBy("prediction").show())

# Partitioning experiment
print("--- Partitioning experiment ---")
t0 = time.time(); final_preds.count(); t_default = (time.time() - t0) * 1000
df_reparted = final_preds.repartition(best_k, "prediction")
df_reparted.write.mode("overwrite").parquet(f"{GOLD}/team_clusters_opt")
t0 = time.time(); spark.read.parquet(f"{GOLD}/team_clusters_opt").count(); t_opt = (time.time() - t0) * 1000
print(f"Default partitioning  : {t_default:.0f} ms")
print(f"Repartitioned by pred : {t_opt:.0f} ms")
log_metric("Iterative", "partitioning", "before", "count_ms", t_default)
log_metric("Iterative", "partitioning", "after",  "count_ms", t_opt)
pathlib.Path(f"{PROOF_DIR}/plan_clustering.txt").write_text(capture_plan(final_preds))

# ════════════════════════════════════════════════════════════════════════════
# 7. LLM DATA READINESS
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 7. LLM DATA READINESS ===")
LLM_OUT = CFG["paths"]["llm_ready"]
llm_cfg = CFG["llm"]
MIN_LEN = llm_cfg["min_text_length"]

df_s = spark.read.parquet(SILVER)

df_text = (df_s
    .withColumn("winner_team",
        F.when(F.col("map_winner") == 1, F.col("team_1")).otherwise(F.col("team_2")))
    .withColumn("loser_team",
        F.when(F.col("map_winner") == 1, F.col("team_2")).otherwise(F.col("team_1")))
    .withColumn("winner_score",
        F.when(F.col("map_winner") == 1, F.col("result_1")).otherwise(F.col("result_2")))
    .withColumn("loser_score",
        F.when(F.col("map_winner") == 1, F.col("result_2")).otherwise(F.col("result_1")))
    .withColumn("text",
        F.concat_ws(" ",
            F.lit("On"), F.col("date").cast("string"),
            F.lit("in CS:GO event"), F.col("event_id").cast("string"),
            F.lit(","), F.col("winner_team"),
            F.lit("defeated"), F.col("loser_team"),
            F.lit("on"), F.col("_map"),
            F.lit("with a score of"), F.col("winner_score").cast("string"),
            F.lit("to"), F.col("loser_score").cast("string"),
            F.lit(". Team 1 world ranking:"), F.col("rank_1").cast("string"),
            F.lit(", Team 2 world ranking:"), F.col("rank_2").cast("string"),
            F.lit(".")))
    .withColumn("doc_id",
        F.concat(F.col("match_id").cast("string"), F.lit("_"), F.col("_map")))
    .withColumn("source",       F.lit(llm_cfg["source"]))
    .withColumn("version",      F.lit(llm_cfg["version"]))
    .withColumn("curated_at",   F.current_timestamp())
    .withColumn("content_hash", F.xxhash64("text")))

total_before = df_text.count()

df_llm = (df_text
    .filter(F.col("text").isNotNull())
    .filter(F.length("text") >= MIN_LEN)
    .dropDuplicates(["content_hash"])
    .select("doc_id", "text", "source", "version", "curated_at",
            "content_hash", "date", "_map", "winner_team", "loser_team", "event_id"))

total_after   = df_llm.count()
quality_ratio = total_after / total_before if total_before > 0 else 0
df_llm.write.mode("overwrite").parquet(LLM_OUT)
llm_mb = dir_size_mb(LLM_OUT)

print(f"Before filters  : {total_before:,}")
print(f"After filters   : {total_after:,}")
print(f"Quality ratio   : {quality_ratio:.2%}")
print(f"Output size     : {llm_mb:.2f} MB")
df_llm.show(3, truncate=80)
print(f"SLO >= {CFG['slo']['llm_quality_pass_ratio_min']:.0%}:",
      "PASS" if quality_ratio >= CFG["slo"]["llm_quality_pass_ratio_min"] else "FAIL")

pathlib.Path(f"{PROOF_DIR}/plan_llm_curation.txt").write_text(capture_plan(df_llm))
log_metric("LLM_prep", "curate", "baseline", "total_before",  total_before)
log_metric("LLM_prep", "curate", "baseline", "curated_rows",  total_after)
log_metric("LLM_prep", "curate", "baseline", "quality_ratio", quality_ratio)
log_metric("LLM_prep", "curate", "baseline", "output_mb",     llm_mb)

data_card = {
    "name": "HLTV CS:GO Match Descriptions",
    "source": llm_cfg["source"], "version": llm_cfg["version"],
    "size_rows": total_after, "size_mb": round(llm_mb, 3),
    "schema": {
        "doc_id": "string (match_id + map)", "text": f"match description >= {MIN_LEN} chars",
        "source": "HLTV_CSGO", "content_hash": "xxhash64 for dedup",
        "date": "match date", "_map": "CS:GO map name",
        "winner_team": "team that won the map", "event_id": "tournament identifier"
    },
    "quality_filters": [f"min_text_length={MIN_LEN}", "content_hash dedup", "null text dropped"],
    "intended_use": "LLM fine-tuning or RAG on CS:GO competitive match history"
}
pathlib.Path(f"{PROOF_DIR}/data_card.json").write_text(json.dumps(data_card, indent=2))

# ════════════════════════════════════════════════════════════════════════════
# 8. COMPACTION EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════
print("\n=== 8. COMPACTION ===")
mr_path_default = f"{GOLD}/match_results"
mr_path_compact = f"{GOLD}/match_results_compact"

t0 = time.time(); spark.read.parquet(mr_path_default).count(); t_before = (time.time() - t0) * 1000
files_before = sum(1 for _, _, fs in os.walk(mr_path_default) for f in fs if f.endswith(".parquet"))

(spark.read.parquet(mr_path_default)
    .coalesce(2)
    .write.mode("overwrite").partitionBy(*PART_BY).parquet(mr_path_compact))

t0 = time.time(); spark.read.parquet(mr_path_compact).count(); t_after = (time.time() - t0) * 1000
files_after    = sum(1 for _, _, fs in os.walk(mr_path_compact) for f in fs if f.endswith(".parquet"))
size_before_mb = dir_size_mb(mr_path_default)
size_after_mb  = dir_size_mb(mr_path_compact)

print(f"Before: {files_before} files  {size_before_mb:.2f} MB  {t_before:.0f} ms")
print(f"After : {files_after} files  {size_after_mb:.2f} MB  {t_after:.0f} ms")
print(f"Scan gain: {(t_before - t_after) / t_before * 100:.1f}%")

pathlib.Path(f"{PROOF_DIR}/plan_compaction_before.txt").write_text(
    capture_plan(spark.read.parquet(mr_path_default)))
pathlib.Path(f"{PROOF_DIR}/plan_compaction_after.txt").write_text(
    capture_plan(spark.read.parquet(mr_path_compact)))

log_metric("ETL", "compaction", "before", "scan_ms",    t_before)
log_metric("ETL", "compaction", "before", "file_count", files_before)
log_metric("ETL", "compaction", "after",  "scan_ms",    t_after)
log_metric("ETL", "compaction", "after",  "file_count", files_after)

# ════════════════════════════════════════════════════════════════════════════
# WRITE METRICS CSV
# ════════════════════════════════════════════════════════════════════════════
METRICS_FILE = "project_metrics_log.csv"
header = ["run_id", "stage", "task", "phase", "metric_name",
          "metric_value", "shuffle_read_bytes", "shuffle_write_bytes",
          "elapsed_ms", "notes", "timestamp"]

with open(METRICS_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(metrics_rows)

print(f"\nMetrics written: {len(metrics_rows)} rows -> {METRICS_FILE}")
for row in metrics_rows:
    print(f"  {row[1]:12s} | {row[2]:25s} | {row[4]:30s} = {row[5]}")

spark.stop()
print("\n=== Pipeline complete ===")
