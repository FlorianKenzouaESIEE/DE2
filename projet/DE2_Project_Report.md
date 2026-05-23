# DE2 — Final Project Report

**Track A — Esports / CS:GO (HLTV match results)**  
**Authors:** Coddeville Daryl — Kenzoua Florian  
**Date:** May 2026 | ESIEE — Data Engineering II (Badr TAJINI)

---

## 1. Use-case and Dataset

**Problem statement.** Competitive CS:GO generates thousands of professional match records per year. The goal is to build a reproducible, production-grade data pipeline that transforms raw HLTV match results into curated analytics and LLM-ready text, enabling downstream use cases such as team performance dashboards, map-meta analysis, and match narrative generation.

**Target user.** Esports analysts and data scientists who need clean, partitioned datasets to query team rankings, win-rate trends, and map-specific statistics over time.

**Dataset.**
| Attribute | Value |
|---|---|
| Source | HLTV (CS:GO professional match results) |
| File | `results.csv` |
| Size | 3.48 MB |
| Rows | **45 773** map-results |
| Columns | 19 (date, team_1, team_2, _map, result_1/2, map_winner, ranks, event_id, match_id, …) |
| Natural key | `(match_id, _map)` — one row per map played in a match |
| Time range | 2018 – 2022 |
| Known issues | 2 duplicate content-hash rows (dropped in LLM prep); all 45 773 rows pass Silver schema contracts |

**Track selected.** Track A — Esports. We use the HLTV CS:GO dataset covering professional matches across 5000+ teams and 200+ events.

---

## 2. System and SLOs

**Hardware.** Single machine (WSL2 / Linux, ~16 GB RAM, `local[*]` Spark).

**Spark config.**
```
spark.sql.shuffle.partitions = 8
spark.driver.memory          = 4g
spark.sql.adaptive.enabled   = true
Spark version                = 4.1.1
```

**SLOs and thresholds.**

| SLO | Target | Observed | Status |
|---|---|---|---|
| Streaming trigger latency | ≤ 30 s (10 s trigger) | 10 s trigger, 5 micro-batches in ~50 s total | PASS |
| Text query latency | ≤ 2 000 ms per term | max 127 ms (natus) | **PASS** |
| Clustering silhouette | ≥ 0.25 | 0.3725 (k=3) | **PASS** |
| Pipeline latency (full run) | ≤ 10 min | ~8 min (including 90 s streaming) | **PASS** |
| Storage reduction (Parquet/CSV) | ≤ 60 % | 34.2 % (corpus) | **PASS** |
| LLM quality pass ratio | ≥ 80 % | 100.0 % | **PASS** |

**Design choices.**
- Bronze kept as CSV to preserve raw, immutable landing.
- Silver and Gold written as Parquet with `year`/`month` partitioning for predicate pushdown.
- Inverted index chosen over TF-IDF: sufficient for exact-token lookup, lower storage overhead.
- KMeans chosen over graph PageRank: team feature vectors are well-defined and directly answer the "team tier" question; the bipartite match graph is too sparse for meaningful PageRank convergence.

---

## 3. Batch ETL Pipeline Design

### 3.1 Bronze (landing)

Raw CSV landed immutably at `outputs/project/bronze` — 45 773 rows, 3.48 MB. All columns remain StringType. Plan saved to `proof/project/plan_silver.txt`.

### 3.2 Silver (cleaning)

Schema contracts applied:

| Column | Target type | Contract |
|---|---|---|
| date | DateType (yyyy-MM-dd) | NOT NULL |
| result_1, result_2 | IntegerType | NOT NULL |
| match_id | LongType | NOT NULL (natural key) |
| _map | StringType | NOT NULL (natural key) |
| rank_1, rank_2 | IntegerType | nullable |

Derived columns added: `year`, `month`, `score_diff = abs(result_1 − result_2)`.  
Deduplication on `(match_id, _map)`.  
**Result:** 45 773 rows, 0 null violations, 1.06 MB Parquet (30.5 % of CSV).

### 3.3 Gold (analytics)

Three tables, all partitioned by `(year, month)`:

| Table | Rows | Partition | Size |
|---|---|---|---|
| `team_stats` | 2 005 | year | included in 4.19 MB total |
| `map_stats` | 393 | year, month | included in 4.19 MB total |
| `match_results` | 45 773 | year, month | included in 4.19 MB total |
| `team_clusters` | 787 | — | included in 4.19 MB total |

**Lineage:**
```
results.csv (3.48 MB)
    └─► bronze/ (CSV, raw, immutable)
            └─► silver/ (Parquet 1.06 MB, typed, 45 773 rows, 0 violations)
                    ├─► gold/team_stats/     (Parquet, partitioned by year)
                    ├─► gold/map_stats/      (Parquet, partitioned by year/month)
                    ├─► gold/match_results/  (Parquet, partitioned by year/month)
                    └─► gold/team_clusters/  (Parquet, KMeans output)
```

EXPLAIN FORMATTED plan for silver→gold saved to `proof/project/plan_etl_silver_to_gold.txt`.

---

## 4. Streaming Ingestion

**Source type.** File source (CSV), simulated by writing Silver data partitioned into 5 CSV files at `data/project/landing/`.

**Schema.** Explicit StructType with `event_timestamp: TimestampType` derived from the `date` column.

**Window and watermark.**
```
Watermark    : event_timestamp, 10 minutes
Window       : 5 minutes (tumbling)
Aggregation  : count(*), avg(score_diff), avg(rank_1)  GROUP BY window, _map
```

**Output mode.** `append` — Parquet sink at `outputs/project/streaming`.

**Trigger.** `processingTime = "10 seconds"`, `maxFilesPerTrigger = 1`.

**Evidence.** `query.lastProgress` JSON saved to `proof/project/query_progress.json`.

| Metric | Observed |
|---|---|
| Output rows (streaming Parquet) | **7 742** windowed aggregation rows |
| Output size | **0.14 MB** Parquet |
| Micro-batches | 5 (one per landing file) |
| Total processing time | ≤ 90 s |

**Note.** All 5 landing files were present at stream start; they were processed sequentially (one per 10 s trigger). The processedRowsPerSecond in lastProgress is 0 because it was captured after the query had already completed its final trigger.

---

## 5. Text Processing

**Corpus.** 45 773 documents, one per `(match_id, _map)`. Each document: `"{team_1} {team_2} {_map} {result_1} {result_2}"`.

**Pipeline.**
1. Lowercase + `regexp_replace(r"[^a-z0-9\s]", " ")` to remove punctuation.
2. Split on whitespace, explode to (doc_id, token) pairs.
3. Filter: length ≥ 2, not in 15-word stop-list.
4. Group by token → `collect_set(doc_id)`, `count(*)`.
5. Write inverted index as Parquet to `outputs/project/text/`.

**Storage footprint.**

| Format | Size (MB) | Ratio vs CSV |
|---|---|---|
| Corpus CSV | 3.51 | 1.0× (baseline) |
| Corpus Parquet | 1.20 | **34.2 %** ✅ (target ≤ 60 %) |
| Inverted index Parquet | 0.80 | — |
| **Unique terms** | **1 676** | — |

**Query latency benchmark** (warmed cache):

| Term | Latency (ms) | Matching docs |
|---|---|---|
| natus | 127.3 | (top teams) |
| fnatic | 40.3 | (top teams) |
| mirage | 43.0 | (map name) |
| dust2 | 43.6 | (map name) |
| inferno | 37.2 | (map name) |
| **Max** | **127.3** | **SLO ≤ 2 000 ms ✅** |

Plans saved: `proof/project/plan_index_build.txt`, `proof/project/plan_query.txt`.

---

## 6. Iterative Workload — KMeans Clustering

**Choice.** Clustering. Rationale: CS:GO match data naturally expresses team quality as a multi-dimensional feature vector (win rate, average rank, score differential). KMeans groups teams into performance tiers, which is more meaningful than graph PageRank on this dataset.

**Features.** 5 features after StandardScaler: `win_rate`, `avg_rank`, `avg_score_diff`, `avg_score`, `maps_variety`. Teams with fewer than 5 maps played are excluded → **787 teams eligible**.

**Sweep configuration** (3 seeds × 4 k-values = 12 runs):

| k | Mean silhouette | Std (seed stability) |
|---|---|---|
| 3 | **0.3725** ← best | 0.0114 |
| 5 | 0.3616 | 0.0122 |
| 8 | 0.3356 | 0.0126 |
| 12 | 0.3062 | 0.0144 |

**Best k = 3** — silhouette 0.3725 ≥ 0.25 **✅ SLO PASS**. Low std (< 0.015) confirms geometric stability across seeds.

**Cluster interpretation:**

| Cluster | Teams | Avg win rate | Avg rank | Avg maps played | Tier |
|---|---|---|---|---|---|
| 0 | 168 | 26.2 % | 160.6 | 13.7 | Low-ranked / inactive |
| 1 | 325 | 57.8 % | 70.9 | 116.0 | Top-tier / high-activity |
| 2 | 294 | 45.7 % | 141.5 | 15.8 | Mid-tier |

**Partitioning experiment:**

| Configuration | Scan time (ms) |
|---|---|
| Default (random) partitioning | 41 ms |
| `repartition(3, "prediction")` | 106 ms |

The repartitioning is slower because the dataset is small (787 rows) and the shuffle cost exceeds the scan benefit. At scale (millions of rows), co-locating predictions would reduce downstream aggregation shuffle.

Plans saved to `proof/project/plan_clustering.txt`.

---

## 7. LLM Data Readiness

**Text fields.** Each row becomes a structured natural-language description:
> *"On 2020-03-18 in CS:GO event 5151, Natus Vincere defeated North on Nuke with a score of 16 to 10. Team 1 world ranking: 1, Team 2 world ranking: 23."*

Average description length: ~140 characters (well above 100-char threshold).

**Quality filters.**
1. Drop null text.
2. Minimum length ≥ 100 characters.
3. Content-hash deduplication (`xxhash64`).

**Output schema.** `doc_id, text, source, version, curated_at, content_hash, date, _map, winner_team, loser_team, event_id`.

**Quality metrics.**

| Metric | Value |
|---|---|
| Records before filters | 45 773 |
| Records after filters | **45 771** (2 exact duplicates removed) |
| Quality pass ratio | **100.0 %** ✅ (target ≥ 80 %) |
| Output size | **2.90 MB** Parquet |

**Data card** saved to `proof/project/data_card.json`.

---

## 8. Physical Design & Optimization

**Partitioning strategy.** All Gold tables partitioned by `(year, month)`. Team stats partitioned by `year` only (months irrelevant for yearly aggregates). Enables predicate pushdown on date-range queries — a query on `year=2021` scans only the relevant partition directory.

**Compaction.** The default Gold `match_results` write creates many small part files (one per Spark task). After `coalesce(2)` per partition:

| Metric | Before | After | Gain |
|---|---|---|---|
| File count | 424 | 66 | −84.4 % |
| Scan time (ms) | 376 | 117 | **−68.7 %** |
| Size (MB) | 4.01 | 1.36 | −66.1 % |

**Exchange optimization.** Setting `spark.sql.shuffle.partitions=8` (down from default 200) avoids over-partitioning on a ~45 K-row dataset, reducing shuffle write overhead in Gold aggregations.

Plans saved: `proof/project/plan_compaction_before.txt`, `proof/project/plan_compaction_after.txt`.

---

## 9. Results and Limits

**Gains vs SLOs — summary.**

| SLO | Target | Observed | Status |
|---|---|---|---|
| Streaming trigger latency | ≤ 30 s | 10 s trigger, 5 batches | ✅ PASS |
| Text query latency | ≤ 2 000 ms | 127 ms (max) | ✅ PASS |
| Clustering silhouette | ≥ 0.25 | 0.3725 (k=3) | ✅ PASS |
| Pipeline latency (full run) | ≤ 10 min | ~8 min | ✅ PASS |
| Storage reduction (Parquet/CSV) | ≤ 60 % | 34.2 % | ✅ PASS |
| LLM quality pass ratio | ≥ 80 % | 100.0 % | ✅ PASS |

**Pipeline coherence.** The five components form a coherent end-to-end pipeline:
- Bronze → Silver → Gold: immutable landing, typed and deduplicated, partitioned analytics.
- Streaming: same Silver schema consumed as a file-source stream with windowed aggregation.
- Text: Silver corpus tokenized into an inverted index for fast exact-token lookup.
- Clustering: Silver team-level aggregates fed into KMeans for performance tier segmentation.
- LLM Prep: Silver match rows transformed into prose descriptions with quality filters.

**Limits.**
- Dataset is 45 773 rows (below the 10M-row guideline). All SLOs pass comfortably; scale testing would reveal real bottlenecks.
- Streaming simulation uses pre-written files; true end-to-end latency depends on upstream producer throughput.
- Inverted index uses `collect_set`: memory-bound at driver for very large corpora (> 100 M documents).
- KMeans assumes spherical clusters; teams form a long tail of low-activity clubs that may not cluster well. Silhouette of 0.37 at k=3 confirms acceptable but not exceptional separation.
- Partitioning by `prediction` did not improve scan on small data (787 rows); gain expected at scale.

**Future work.** Integrate real-time HLTV feed via Kafka; add TF-IDF alongside inverted index for ranked retrieval; extend clustering to detect team win-rate drift over seasons; add schema evolution handling for future CSV format changes.
