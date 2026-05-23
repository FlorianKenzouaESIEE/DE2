# DE2 — Generative AI Usage Declaration

**Project:** DE2 Final Project — Track A (CS:GO/HLTV)  
**Authors:** Coddeville Daryl - Kenzoua Florian
**Date:** May 2026

---

## How we used generative AI

We used **Claude (Anthropic)** via the Claude Code CLI as a coding assistant throughout this project.

### Tasks assisted by GenAI

| Task | GenAI contribution | Human validation |
|---|---|---|
| Notebook skeleton and cell structure | Generated full pipeline code for all 5 components | Reviewed logic, tested locally, adjusted parameters |
| Config file design | Proposed YAML structure, SLO values, CS:GO-specific query terms | Validated against project brief requirements |
| Report structure | Drafted section outlines and metric tables | Filled in measured values after running the pipeline |
| Regex patterns for text tokenization | Suggested `[^a-z0-9\s]` normalization pattern | Verified output quality on sample documents |
| Schema contracts (Silver cell) | Generated all `.cast()` and `.dropna()` calls | Cross-checked against raw CSV column types |
| Streaming schema definition | Suggested explicit `StructType` for file source | Tested against actual landing CSV output |

### What GenAI did NOT do

- **Run the pipeline.** All execution, output verification, and metrics collection were done by the team on the local machine.
- **Choose the dataset.** We selected Track A (Esports/CS:GO) independently.
- **Choose the iterative workload path.** We chose clustering (vs. graph) based on our assessment of the data.
- **Fill in measured values.** All numeric results in the report and metrics log come from actual Spark runs.

### Assessment

GenAI significantly accelerated boilerplate code generation (schema casting, file I/O, helper functions) and helped structure the project according to the brief. All generated code was reviewed, understood, and validated by the team before submission. The intellectual contribution of understanding PySpark internals, designing the pipeline architecture, and interpreting results remained entirely ours.
