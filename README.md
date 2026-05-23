# DE2 — Data Engineering II · Lab Assignments
### ESIEE Paris · 2025–2026

> **Authors :** Florian Kenzoua & Daryl Coddeville  
> **Course :** Data Engineering II 
> **Track :** Esport — CS:GO Match Results (Track A)  
> **Dataset :** `results.csv` — ~45 773 professional CS:GO matches, 2015–2020

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Assignment 1 — Streaming Pipeline](#assignment-1--streaming-pipeline)
4. [Assignment 2 — Inverted Index](#assignment-2--inverted-index)
5. [Assignment 3 — Graph Processing (PageRank)](#assignment-3--graph-processing-pagerank)
6. [Repository Structure](#repository-structure)
7. [Environment & Dependencies](#environment--dependencies)

---

## Project Overview

These three lab assignments form a progressive exploration of **data-intensive workloads** using **Apache Spark (PySpark)**, applied to a unified CS:GO esport dataset. Each lab focuses on a different paradigm:

| Lab | Paradigm | Key Concept |
|-----|----------|-------------|
| Assignment 1 | Spark Structured Streaming | Windowed aggregations, watermarks, Parquet sinks |
| Assignment 2 | Batch Processing | Inverted index, NLP pipeline, storage format comparison |
| Assignment 3 | Graph Processing | PageRank, partitioning strategies, convergence analysis |

---

## Dataset

The dataset (`results.csv`) contains CS:GO professional match results from 2015 to 2020. Each row represents **one map played** within a match.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Date of the match |
| `team_1` / `team_2` | string | Team names |
| `_map` | string | Map played (Dust2, Inferno, Mirage, …) |
| `result_1` / `result_2` | int | Rounds won by each team |
| `map_winner` | int | 1 = team_1 won, 2 = team_2 won |
| `rank_1` / `rank_2` | int | World rankings |
| `match_id` | int | Unique match identifier |
| `event_id` | int | Tournament event identifier |

---

## Assignment 1 — Streaming Pipeline

**Notebook :** `assignment1_esiee_completed.ipynb`

### Objective

Build a complete **Spark Structured Streaming** pipeline that ingests CS:GO match data in real time, aggregates it in tumbling windows, and persists results to Parquet.

### Pipeline Architecture

```
results.csv ──► JSON line files (simulator) ──► Spark Structured Streaming
                                                          │
                                                withWatermark + window()
                                                          │
                                                Parquet sink (append mode)
                                                          │
                                                Metrics log CSV
```

### Key Methods & Technologies

**Stream Simulation**
- A background thread (`threading.Thread`) reads `results.csv` in chunks of 200 rows and writes one JSON-lines file every 3 seconds into `streaming_lab/input_stream/`, simulating a live event feed.
- Each batch record is injected with an `event_timestamp` field set to the current UTC time to serve as the Spark event-time column.

**Schema Definition & ReadStream**
```python
event_schema = StructType([
    StructField("event_timestamp", TimestampType(), False),
    StructField("team_1",          StringType(),   True),
    StructField("map",             StringType(),   True),
    # ...
])

raw_stream = spark.readStream.schema(event_schema)
    .option("maxFilesPerTrigger", 2)
    .json(str(STREAM_INPUT))
```

**Windowed Aggregation with Watermark**
- **Watermark :** `30 seconds` — allows Spark to safely drop state for windows older than the delay.
- **Window :** 2-minute tumbling window grouped by map.
- Metrics computed per window:

| Metric | Description |
|--------|-------------|
| `map_plays` | Total number of times the map was played |
| `avg_rounds_team1` / `avg_rounds_team2` | Average round scores |
| `team1_wins` / `team2_wins` | Win counts per side |
| `avg_rank_diff` | Average absolute ranking difference between opponents |

```python
windowed_agg = (
    raw_stream
    .withWatermark("event_timestamp", "30 seconds")
    .groupBy(F.window(F.col("event_timestamp"), "2 minutes"), F.col("map"))
    .agg(
        F.count("*").alias("map_plays"),
        F.round(F.avg("result_1"), 2).alias("avg_rounds_team1"),
        F.sum(F.when(F.col("map_winner") == 1, 1).otherwise(0)).alias("team1_wins"),
        F.round(F.avg(F.abs(F.col("rank_1") - F.col("rank_2"))), 2).alias("avg_rank_diff"),
    )
)
```

**Parquet Sink**
- Output mode: `append` — only finalized (past-watermark) windows are written.
- Trigger: `processingTime = "10 seconds"` (baseline).
- Checkpoint directory for exactly-once fault-tolerance guarantee.

**Performance Optimisation**

Two configurations were benchmarked:

| Parameter | Baseline | Optimised |
|-----------|----------|-----------|
| `spark.sql.shuffle.partitions` | 4 | 2 |
| `maxFilesPerTrigger` | 2 | 5 |
| `processingTime` trigger | 10 s | 5 s |

### Results

| Metric | Baseline | Optimised | Δ |
|--------|----------|-----------|---|
| Avg input rows / batch | 388.9 | 1 000.0 | **+157 %** |
| Avg processed rows/s | 140.3 | 421.2 | **+200 %** |
| Avg trigger execution (ms) | 3 114.6 | 2 587.4 | **−17 %** |
| Avg add-batch time (ms) | 1 820.3 | 1 145.6 | **−37 %** |

### Esport Insights

- **Inferno** was the most played map (908 occurrences), followed by Nuke (773) and Mirage (753).
- Average round scores hovered around 13–14 vs 12–13, consistent with the 16-round win format in professional play.
- **Team 1 held a slight structural advantage** across all maps (win rate: 52–55 %), likely because the higher-seeded team (listed as `team_1` in HLTV data) tends to pick its best map in the veto.
- **Vertigo** showed the highest team_1 win rate (55.3 %), consistent with its status as a newer map where preparation was unevenly distributed.

### Lessons Learned

- Watermark + `append` mode requires patience: a window is emitted only after `window_end + watermark_delay` passes in event-time, so the first Parquet files appear after several minutes — expected and correct behavior.
- Checkpoint is **mandatory** for stateful aggregations; without it, Spark cannot recover window state across restarts.
- For small datasets, reducing `shuffle.partitions` has a disproportionately positive impact because the bottleneck is task scheduling overhead, not data shuffling.

---

## Assignment 2 — Inverted Index

**Notebook :** `assignment2_esiee_completed.ipynb`

### Objective

Build a **distributed inverted index** over the CS:GO corpus using PySpark's DataFrame API, persist it in two formats (Parquet and CSV), and benchmark query latency and storage footprint.

### Pipeline Architecture

```
results.csv
    │
    ▼  Corpus ingestion
doc_id | text  (team_1 + team_2 + _map)
    │
    ▼  Text normalisation
lowercase → regex tokenise → stop-word removal
    │
    ▼  Inverted index
token | doc_ids (array) | freq (long)
    │
    ├──► Parquet  outputs/lab2/inverted_index/
    └──► CSV      outputs/lab2/inverted_index_csv/
    │
    ▼  Query latency benchmark + storage footprint comparison
```

### Key Methods

**Corpus Ingestion**

Each row becomes one document. The text field concatenates the three textual columns:
```
text = team_1 + " " + team_2 + " " + _map
```
Document ID (`doc_id`) = `match_id` cast to string (unique per match).

**Text Normalisation**

| Step | Operation | PySpark API |
|------|-----------|-------------|
| 1 | Lowercase | `F.lower()` |
| 2 | Tokenise | `F.split()` with regex `[\s\W]+` |
| 3 | Explode | `F.explode()` → one token per row |
| 4 | Remove empty tokens | `F.length() > 0` filter |
| 5 | Remove stop-words | Custom English stop-word set |

Stop-words were handled without UDFs using `F.col("token").isin(STOP_WORDS)` — avoiding Python serialisation overhead.

**Index Construction**

```python
inverted_index = (
    tokens_clean
    .groupBy("token")
    .agg(
        F.collect_list("doc_id").alias("doc_ids"),  # posting list
        F.count("*").alias("freq")                  # corpus-wide term frequency
    )
    .orderBy(F.desc("freq"))
)
```

**Dual Persistence (Parquet vs CSV)**

Because CSV does not natively support array types, `doc_ids` was serialised as a pipe-separated string before writing to CSV:
```python
index_for_csv = inverted_index.withColumn(
    "doc_ids", F.array_join(F.col("doc_ids"), "|")
)
```

**Query Latency Benchmark**

Six terms were queried across both formats, covering different frequency ranges:

| Frequency | Terms |
|-----------|-------|
| High | `navi`, `dust2`, `inferno` |
| Medium | `liquid`, `astralis` |
| Low | `rugratz` |

### Results

**Corpus Statistics**

| Metric | Value |
|--------|-------|
| Documents (unique match IDs) | ~45 773 |
| Tokens before stop-word removal | ~137 000+ |
| Vocabulary reduction | ~significant reduction via stop-words + single-char filter |

**Storage Footprint**

Parquet's columnar Snappy compression yields a significantly smaller footprint — the `token` column (low-cardinality strings) compresses very efficiently. The `doc_ids` array is stored natively without pipe-serialisation overhead.

**Query Latency Observation**

On this small corpus (~1 645 unique index terms), Parquet was not systematically faster than CSV. This is expected: the fixed overhead of Parquet columnar decoding (`ColumnarToRow`) exceeds the gain from predicate pushdown at small scale. The latency advantage of Parquet materialises at millions of rows, where row-group pruning becomes decisive.

### Lessons Learned

- `collect_list` ordering is **non-deterministic** across Spark runs; if reproducibility is required, a `sort_array` step should follow the aggregation.
- Stop-word removal has a limited impact on a CS:GO corpus because team names rarely overlap with common English words — the main vocabulary reduction comes from removing single-character tokens and punctuation artefacts.
- `explain("formatted")` is the right tool to verify that Parquet `PushedFilters` are active on a token lookup.
- Parquet's advantage over CSV at this scale is primarily **storage size**, not query latency.

---

## Assignment 3 — Graph Processing (PageRank)

**Notebook :** `assignment3_esiee_FIXED.ipynb`

### Objective

Build a **team interaction graph** from the CS:GO match dataset and apply a custom **PageRank** implementation in PySpark to rank teams by competitive prestige. Two partitioning strategies are compared and convergence is analysed.

### Graph Definition

| Element | Definition |
|---------|------------|
| **Vertices** | Unique team names (`id` = row_number, `name` = team name) |
| **Edges** | Undirected pairs of teams that played against each other, weighted by match frequency |

```python
# Normalise edges to undirected
edges_norm = edges_raw.withColumn('src_norm', F.least('src', 'dst'))
                      .withColumn('dst_norm', F.greatest('src', 'dst'))

# Aggregate by pair, count matches as weight
edges = edges_norm.groupBy('src', 'dst').agg(F.count('*').alias('weight'))
```

**Graph properties:**
- Vertices: unique teams in the dataset
- Edges: team pairs with their match count as edge weight
- Graph density reported at build time

### PageRank Implementation

A custom weighted PageRank was implemented iteratively in PySpark (no GraphFrames dependency):

```
rank(v) = (1 - d) / N + d × Σ [ rank(u) × weight(u→v) / out_degree(u) ]
```

Parameters: damping factor `d = 0.85`, tolerance `tol = 0.0001`, max iterations = 20.

**Critical fix — Memory Management:**

Iterative Spark jobs accumulate a growing DAG (lineage) which causes `java.lang.OutOfMemoryError` on the driver after several iterations. Two fixes were applied:

1. **Lineage truncation via `localCheckpoint()`** — materialises the DataFrame and truncates the execution plan at each iteration, preventing unbounded DAG growth.
2. **Driver memory increase** — `spark.driver.memory = 6g` at session creation.

```python
# At the end of each iteration:
ranks = ranks.localCheckpoint()  # breaks lineage, forces materialisation
```

### Partitioning Experiment

Two strategies were compared:

| Strategy | Configuration |
|----------|--------------|
| Default | Spark default partitioning |
| Custom | `edges_df.repartition(8, 'src')` — co-locates edges by source node |

Co-locating edges by `src` reduces shuffle volume during the join with rank scores at each iteration, since all edges from the same source are on the same partition.

### Convergence Analysis

Four plots were generated and saved to `outputs/lab3/convergence_analysis.png`:

- **Per-iteration time** (default vs custom)
- **Convergence delta** on a log scale (with tolerance threshold line)
- **Cumulative time** across iterations
- **Bar comparison** of per-iteration times

### Results

**Top CS:GO teams by PageRank** (based on the 2015–2020 dataset):

The PageRank score reflects not only win count but also the competitive prestige of opponents faced. Teams with high scores played frequently against other top-ranked teams, creating a self-reinforcing centrality in the match graph.

**Partitioning impact:**

| Config | Total time | Mean time / iter |
|--------|-----------|-----------------|
| Default | reported in metrics log | reported in metrics log |
| Custom (`repartition(8, 'src')`) | reported in metrics log | reported in metrics log |

The custom partitioning reduces shuffle overhead on the `src`-keyed join inside each PageRank iteration.

### Lessons Learned

- **Lineage accumulation is the main pitfall** in iterative Spark algorithms. `localCheckpoint()` is the correct solution in a local/single-node context; `checkpoint()` with a shared HDFS directory is preferred in production cluster settings.
- Edge normalisation (undirected graph) matters: without `least()`/`greatest()` normalisation, each match generates two directed edges, artificially inflating degree counts.
- Graph density is very low for this dataset (most team pairs never met), making the adjacency matrix extremely sparse — which is why PageRank converges relatively fast.

---

## Repository Structure

```
de2-labs-esiee/
│
├── README.md                          ← this file
│
├── assignment1_esiee_completed.ipynb  ← Lab 1: Streaming Pipeline
├── assignment2_esiee_completed.ipynb  ← Lab 2: Inverted Index
├── assignment3_esiee_FIXED.ipynb      ← Lab 3: Graph Processing
│
├── data/
│   └── TrackA_CSGO/
│       └── results.csv                ← source dataset (not committed if large)
│
├── streaming_lab/                     ← Lab 1 runtime outputs
│   ├── input_stream/                  ← simulated JSON event files
│   ├── output_parquet/                ← Parquet sink (baseline)
│   ├── output_parquet_opt/            ← Parquet sink (optimised)
│   ├── checkpoint/                    ← Spark checkpoint (baseline)
│   ├── checkpoint_opt/                ← Spark checkpoint (optimised)
│   └── lab1_metrics_log.csv           ← performance metrics
│
├── outputs/
│   ├── lab2/
│   │   ├── inverted_index/            ← Parquet inverted index
│   │   ├── inverted_index_csv/        ← CSV inverted index
│   │   └── ...
│   └── lab3/
│       ├── top_teams.csv              ← Top 50 teams by PageRank
│       ├── metrics_default.csv        ← Per-iteration metrics (default partitioning)
│       ├── metrics_custom.csv         ← Per-iteration metrics (custom partitioning)
│       └── convergence_analysis.png   ← Convergence plots
│
├── proof/                             ← Spark execution plans (txt files)
│   ├── plan_query.txt
│   ├── plan_index_build.txt
│   └── plan_graph.txt
│
├── lab2_metrics_log.csv               ← Lab 2 metrics (corpus + query + footprint)
└── lab3_metrics_log.csv               ← Lab 3 metrics (graph + PageRank)
```

---

## Environment & Dependencies

**Python :** 3.9+  
**Apache Spark :** 3.x (local mode, `local[*]`)  
**Key libraries :**

```
pyspark
pandas
matplotlib
numpy<2
```

**Spark session configuration (common baseline) :**

```python
spark = (
    SparkSession.builder
    .appName("de2-esiee")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)
```

**Running the notebooks :**

```bash
# Install dependencies
pip install pyspark pandas matplotlib "numpy<2"

# Launch Jupyter
jupyter notebook

# Open and run cells in order:
# 1. assignment1_esiee_completed.ipynb
# 2. assignment2_esiee_completed.ipynb
# 3. assignment3_esiee_FIXED.ipynb
```

> **Note :** Place `results.csv` at `data/TrackA_CSGO/results.csv` relative to the notebook directory before running any notebook.

---

*ESIEE Paris — Florian Kenzoua & Daryl Coddeville — 2025–2026*
