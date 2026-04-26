# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**CareMap** — an agentic healthcare intelligence system for India. It ingests ~9,253 facility records from a CSV dataset, runs an LLM-powered extraction pipeline to enrich them, stores the results in Databricks Delta Lake (Bronze → Silver → Gold), and serves a FastAPI backend with a natural-language query interface and a geographic map dashboard.

The system targets deployment on **Databricks Apps** (Free Edition). Spark and MLflow are required at runtime; they are not available locally.

## Running the App

```bash
# Install dependencies
pip install fastapi uvicorn pydantic mlflow

# Run locally (Spark will be unavailable — query endpoint will fail, static pages will load)
python app.py
# Serves at http://localhost:8000
```

For full functionality, deploy to Databricks Apps:
1. Upload `app.py`, `query_agent.py`, `index.html`, `map.html` to a Databricks App (Custom type).
2. Set entry point: `app.py`.

## Running the Data Pipeline

```bash
# Requires openai, pydantic, pandas, openpyxl, mlflow
pip install openai pydantic pandas openpyxl mlflow

# Set your OpenAI API key in data_extraction_pipeline.py (line 35) before running
python data_extraction_pipeline.py
# Outputs go to caremap_output/
```

## Architecture

### Data Pipeline (`data_extraction_pipeline.py`)

Standalone script that processes the raw CSV through three layers:

- **Bronze**: Raw CSV load (`VF_Hackathon_Dataset_India_Large.csv`)
- **Silver**: Cleaning, JSON-array parsing, completeness scoring
- **Gold**: LLM extraction (OpenAI `gpt-4o-mini`) per facility → structured `ExtractedCapability` + `TrustScore` (0–100 across 5 dimensions: completeness, consistency, verifiability, recency, digital presence)

Also produces a **medical desert analysis** by state × specialty. Outputs: `gold_facilities_enriched.csv`, `desert_analysis.csv`, `facilities_map.geojson`.

### Query Agent (`query_agent.py`)

Imported by `app.py`. Requires a live Spark session and a Databricks MLflow deploy client pointing to `databricks-meta-llama-3-3-70b-instruct`.

**5-step pipeline per query** (in `query_healthcare()`):
1. `parse_query()` — LLM call #1: natural language → structured JSON filters (state, city, ICU, emergency, etc.)
2. `hybrid_search()` — SQL-only search against `workspace.gold.facilities_enriched` JOIN `workspace.silver.facilities_clean`; ordered by `trust_score DESC`
3. `format_results()` — builds row-level citations from extracted availability/capabilities
4. `validate_top_results()` — LLM call #2: batch validate top 5 claims against source text (VERIFIED / DISPUTED / INSUFFICIENT_TEXT)
5. `re_extract_with_feedback()` — LLM call #3, only triggered for DISPUTED results; also `check_medical_standards()` (rule-based, no LLM) runs on all results

MLflow tracking: every query run is logged under `/Users/yashlammarr@gmail.com/query_agent_runs`.

### FastAPI Backend (`app.py`)

- Serves `index.html` (query UI) at `/` and `map.html` (map dashboard) at `/map`
- `/query` (POST): runs full agent pipeline, calls `format_for_frontend()` to reshape results for the UI
- `/map/data` (GET): queries Spark for all facilities with coordinates
- `/deserts/data` (GET): queries `workspace.gold.desert_analysis` for medical desert heatmap data
- `/stats` (GET): aggregate counts across the Gold table
- `/facility/{name}` (GET): full profile for a single facility

`SPARK_AVAILABLE` flag guards all Spark-dependent endpoints — if import fails, static HTML still loads.

### Delta Lake Table Schema

| Table | Key columns |
|---|---|
| `workspace.silver.facilities_clean` | `name`, `address_stateOrRegion`, `address_city`, `facilityTypeId`, `latitude`, `longitude`, `specialties`, `searchable_text` |
| `workspace.gold.facilities_enriched` | `name`, `trust_score`, `trust_score_breakdown`, `confidence_level`, `extracted_availability`, `extracted_capabilities`, `extracted_equipment`, `extracted_staff`, `contradictions`, `extracted_bed_count` |
| `workspace.gold.desert_analysis` | `state`, `specialty`, `desert_severity`, `centroid_lat`, `centroid_lng` |
| `workspace.gold.state_summary` | `state`, `max_desert_severity`, `critical_specialty_count` |

Tables are joined on `name` throughout; `extracted_availability` and `extracted_capabilities` are JSON strings stored in the table and parsed with `json.loads()` at query time.

## Key Design Decisions

- **SQL-only search**: Vector/semantic search was deferred for the hackathon. `hybrid_search()` builds a plain SQL query with conditional WHERE clauses.
- **Trust score adjustment**: DISPUTED results get `trust_score × 0.85`; self-corrected results get `× 0.90`. These adjusted values are shown as `trust_score_display` in the frontend.
- **LLM endpoints**: The extraction pipeline uses OpenAI (`gpt-4o-mini`). The query agent uses Databricks-hosted Llama 3.3 70B via MLflow deployments. These are separate clients — don't conflate them.
- **`_call_llm()` strips markdown fences**: The LLM sometimes wraps JSON in ` ```json ``` ` blocks; `_call_llm()` strips them before `json.loads()`.
