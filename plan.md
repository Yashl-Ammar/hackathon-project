# Agentic Healthcare Intelligence System
## Implementation Plan of Action

---

## 1. System Architecture Overview

The system follows a **Medallion Architecture** (Bronze → Silver → Gold) on Databricks, with an agentic intelligence layer on top and a visual frontend for end users. All components run within the Databricks ecosystem on serverless compute.

```
┌─────────────────────────────────────────┐
│         FRONTEND LAYER                  │
│   Query UI  │  Map Dashboard            │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         AGENT ORCHESTRATION             │
│  Extraction │ Trust  │ Query │ Validator │
│  Agent      │ Scorer │ Agent │ Agent    │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         PLATFORM SERVICES               │
│  MLflow 3  │ Vector Search │ Unity Cat. │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         DATA LAYER (Delta Lake)         │
│  Bronze    │  Silver       │  Gold      │
│  (Raw)     │  (Cleaned)    │  (Enriched)│
└─────────────────────────────────────────┘
```

---

## 2. Full Tech Stack

### Data Platform
| Tool | Purpose |
|---|---|
| Databricks Free Edition | Unified compute and notebook environment |
| Delta Lake | Open table format — stores Bronze/Silver/Gold data |
| Apache Spark | Distributed processing of 10,000 records |
| Unity Catalog | Data governance, lineage, access control |
| Databricks Workflows / Jobs | Orchestrate the full batch pipeline |

### AI & Agent Layer
| Tool | Purpose |
|---|---|
| Databricks Agent Bricks | Build, evaluate, and serve AI agents |
| Claude Sonnet (Anthropic API) | Primary LLM for unstructured text extraction |
| LangChain | Agent orchestration, tool calling, chain-of-thought |
| MLflow 3 | Trace every agent step — inputs, outputs, token cost |
| Mosaic AI Vector Search | Semantic similarity search across 10k records |

### Backend
| Tool | Purpose |
|---|---|
| Python 3.11+ | Primary language across all layers |
| FastAPI | REST API for query interface |
| Pydantic v2 | Data validation and schema enforcement (Virtue Foundation Schema) |
| Databricks Apps | Host and serve the FastAPI backend natively on Databricks |

### Frontend
| Tool | Purpose |
|---|---|
| React 18 + Vite | Query interface and dashboard shell |
| Tailwind CSS | Styling |
| Leaflet.js / React-Leaflet | Interactive India facility map |
| Recharts / Chart.js | Charts (desert severity, trust score distributions) |
| Axios | API communication |

### Alternative (Faster for Hackathon)
| Tool | Purpose |
|---|---|
| Streamlit | All-in-one Python frontend — faster to build |
| Folium | Python-native map rendering |
| Plotly Express | Charts within Streamlit |

---

## 3. Data Architecture — Medallion Pattern

### Bronze Layer (Raw Ingestion)
- Source: `VF_Hackathon_Dataset_India_Large.xlsx` / CSV
- Action: Load as-is into Delta Lake table `healthcare.bronze.facilities_raw`
- Schema: All 41 original columns preserved
- No transformations — raw fidelity maintained

### Silver Layer (Cleaned & Standardised)
Table: `healthcare.silver.facilities_clean`

Transformations applied:
- Parse JSON-stringified arrays in `specialties`, `procedure`, `equipment`, `capability`
- Normalise `facilityTypeId` and `operatorTypeId` casing
- Fill `address_stateOrRegion` and `address_zipOrPostcode` nulls where inferrable from lat/long (reverse geocoding)
- Compute `data_completeness_score` (0–1) per record based on populated fields
- Flag `has_unstructured_text` boolean (any of description/capability/procedure non-null)
- Flatten and deduplicate `specialties` into a standard controlled vocabulary

### Gold Layer (Enriched & Scored)
Table: `healthcare.gold.facilities_enriched`

Fields added by the AI extraction pipeline:
- `extracted_capabilities`: Structured JSON of extracted medical capabilities
- `extracted_equipment`: Normalised equipment list from free text
- `extracted_staff`: Inferred doctor counts and specialties from text
- `extracted_availability`: 24/7 status, emergency readiness
- `trust_score`: 0–100 composite score
- `trust_score_breakdown`: JSON with sub-scores per dimension
- `contradictions`: List of detected claim contradictions
- `confidence_level`: high / medium / low / insufficient_data
- `embedding`: 1536-dim vector embedding for semantic search

---

## 4. Implementation Phases

---

### Phase 1 — Data Foundation (Target: 3 hours)

**Goal:** Get all 10,000 records into Delta Lake and clean them.

**Tasks:**
1. Load CSV into Databricks notebook using `spark.read.csv()`
2. Write Bronze table to Unity Catalog (`CREATE TABLE IF NOT EXISTS`)
3. Build Silver transformation notebook:
   - Parse all JSON array columns with `from_json()`
   - Normalise enums and strip whitespace
   - Compute `data_completeness_score`
4. Register both tables in Unity Catalog with column-level descriptions
5. Run data quality checks — row counts, null rates per column, lat/long bounds

**Output:** Two clean, versioned Delta tables registered in Unity Catalog.

---

### Phase 2 — NLP Extraction Agent (Target: 6 hours)

**Goal:** Extract structured capabilities from unstructured text fields.

**Architecture:**
```
Silver Record
     │
     ▼
Extraction Agent (LangChain + Claude Sonnet)
     │
     ├── Tool: parse_description()
     ├── Tool: parse_capability_field()
     ├── Tool: parse_procedure_list()
     └── Tool: parse_equipment_list()
     │
     ▼
Structured JSON Output (Pydantic-validated)
     │
     ▼
Gold Layer Write
```

**Key Prompt Design:**
The extraction prompt instructs the LLM to:
- Return ONLY capabilities it finds explicit evidence for
- Flag any capability claim it cannot verify in the text as `confidence: low`
- Never infer capabilities not mentioned
- Output valid JSON conforming to the Virtue Foundation Pydantic schema

**Batch Strategy:**
- Process records in batches of 50 using Spark UDFs or `applyInPandas()`
- MLflow 3 `mlflow.start_run()` wraps each batch — logs inputs, outputs, token cost
- Failed records are written to a `healthcare.gold.extraction_failures` table for retry

**Output:** Gold table with structured capability JSON per facility.

---

### Phase 3 — Trust Scoring Engine (Target: 3 hours)

**Goal:** Compute a 0–100 Trust Score per facility.

**Scoring Formula:**

| Dimension | Weight | Logic |
|---|---|---|
| Completeness | 25% | Ratio of non-null key fields (description, specialties, equipment, procedure, capability) |
| Consistency | 30% | Cross-field agreement (claims vs evidence); contradiction penalty = -15 pts each |
| Verifiability | 25% | Ratio of extracted claims backed by named specifics (equipment names, doctor names) |
| Recency | 10% | `recency_of_page_update` converted to a decay score (recent = higher) |
| Digital Presence | 10% | Social media activity as operational proxy |

**Contradiction Detection Rules (coded as Python functions):**
- `facilityTypeId == clinic` AND extracted_beds > 50 → flag
- Claim "Advanced Surgery" AND no anaesthesiologist found → flag
- Claim "24/7 Emergency" AND `numberDoctors == 1` → flag
- Claim "ICU" AND no ventilator/ICU-related equipment found → flag

**Output:** Trust score + breakdown JSON per facility, written to Gold table.

---

### Phase 4 — Vector Search & Query Agent (Target: 5 hours)

**Goal:** Enable natural language queries over all 10,000 records.

**Indexing:**
1. Concatenate key text fields into a single `searchable_text` string per record
2. Generate embeddings using Databricks `bge-large-en` or `text-embedding-3-small`
3. Index all 10,000 embeddings into Mosaic AI Vector Search
4. Store metadata (facility_id, state, specialties, trust_score, lat/long) alongside vectors

**Query Pipeline:**
```
User Query (natural language)
     │
     ▼
Query Parser (Claude Sonnet)
→ Extracts: [geography], [specialty], [procedure], [equipment], [min_trust_score]
     │
     ▼
Hybrid Retrieval
→ Vector Search (semantic similarity, top-50 candidates)
→ SQL Filter (apply geography, specialty, trust_score constraints)
     │
     ▼
Re-Ranker (Claude Sonnet)
→ Score each candidate against original query
→ Justify why each is included/excluded
     │
     ▼
Result Formatter
→ Top-N facilities with:
   - Name, address, contact
   - Trust Score + breakdown
   - Citation: exact sentence from source field
   - Distance from reference point
```

**MLflow Tracing:** Every step logged — parse → retrieve → rerank → format.

**Output:** FastAPI `/query` endpoint returning ranked JSON results.

---

### Phase 5 — Medical Desert Analysis (Target: 3 hours)

**Goal:** Identify geographic gaps in critical specialties.

**Method:**
1. Aggregate Gold table by `address_stateOrRegion` and `address_zipOrPostcode`
2. For each high-acuity specialty (Oncology, Dialysis, Neonatal ICU, Emergency Trauma, Mental Health):
   - Count facilities per PIN code with confirmed capability (`confidence != low`)
   - Compute nearest facility distance using Haversine formula on lat/long
3. Assign Desert Severity Score (0–100) per PIN code:
   - 0 facilities within 50km → severity 90–100
   - 1 facility within 50km, low trust → severity 60–89
   - 2+ facilities, high trust → severity 0–30
4. Aggregate to state level for heatmap overlay

**Output:** `healthcare.gold.desert_analysis` table + GeoJSON export for frontend.

---

### Phase 6 — Frontend & Dashboard (Target: 6 hours)

**Goal:** Build an intuitive interface for NGO planners and health administrators.

**Pages / Views:**

**1. Query Interface**
- Free-text search bar
- Optional filters: state, facility type, minimum trust score
- Results displayed as ranked cards with citations visible
- Each result card shows Trust Score ring, specialty tags, and distance

**2. Map View**
- All 10,000 facilities plotted as dots (colour = trust score)
- Cluster view for dense areas (e.g. Mumbai, Delhi)
- Toggle: Heatmap overlay for desert severity per specialty
- Click a facility → full profile modal with all extracted data and citations

**3. Desert Report View**
- Table of top-50 most critical medical deserts
- Filterable by state and specialty
- Export to PDF/CSV for NGO planning

**4. Trust Score Explorer**
- Distribution chart of trust scores across all facilities
- Breakdown by state and facility type
- Flag low-trust facilities claiming critical capabilities

**Frontend Tech Choice Decision:**

For hackathon speed, use **Streamlit** hosted on **Databricks Apps**:
- Single Python codebase — no React build pipeline
- Databricks Apps handles auth and hosting
- Folium for map (renders as HTML in Streamlit)
- Plotly for charts

For production quality (if time permits), use **React + FastAPI**:
- React-Leaflet for fully interactive map
- Recharts for dashboards
- Hosted on Databricks Apps

---

### Phase 7 — Validator Agent & Self-Correction (Target: 3 hours)

**Goal:** Reduce hallucinations and flag low-confidence extractions.

**Validator Agent Design:**
1. Sample 10% of extracted records (random + stratified by trust score band)
2. For each sampled record, re-run a **stripped-down extraction prompt** that asks the LLM only:
   - "Given this raw text, what capabilities are explicitly stated?"
   - "Does this match what was previously extracted?"
3. Compute agreement rate between primary extraction and validator
4. Disagreements → flag as `needs_review = True` in Gold table
5. Log all disagreements in MLflow for review

**Confidence Intervals:**
- For aggregate statistics (e.g. "% of Bihar facilities with ICU"), compute a bootstrap confidence interval that accounts for the 35.8% null rate in the `capability` field
- Report as range: "Between 12% and 28% of Bihar facilities claim ICU capability, but only 8% have high-confidence verification"

---

## 5. Data Flow Summary

```
Raw CSV
  │
  ├──► Bronze Delta Table (raw, 10,000 rows)
  │
  ├──► Silver Delta Table (cleaned, standardised, completeness scored)
  │
  ├──► NLP Extraction Agent (batch, MLflow traced)
  │         │
  │         └──► Gold Delta Table (enriched, trust scored, embeddings)
  │                   │
  │                   ├──► Vector Search Index (Mosaic AI)
  │                   ├──► Desert Analysis Table
  │                   └──► FastAPI / Streamlit Backend
  │                                   │
  │                                   └──► Frontend Dashboard
  │
  └──► Unity Catalog (governance, lineage, access control for all tables)
```

---

## 6. API Endpoints (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Natural language query → ranked facility results |
| `/facility/{id}` | GET | Full profile + trust score + citations for one facility |
| `/deserts` | GET | Medical desert report, filterable by state/specialty |
| `/map/data` | GET | GeoJSON of all facilities with metadata for map rendering |
| `/stats` | GET | Aggregate statistics (trust score distribution, specialty coverage) |

---

## 7. Team Roles (Suggested Split for Hackathon)

| Role | Responsibilities |
|---|---|
| Data Engineer | Phases 1 & 5 — Delta Lake setup, Silver transforms, desert analysis |
| AI/ML Engineer | Phases 2 & 7 — Extraction agent, validator, MLflow tracing |
| Backend Engineer | Phases 3 & 4 — Trust scorer, query agent, FastAPI endpoints |
| Frontend Engineer | Phase 6 — Dashboard, maps, query UI |

---

## 8. Hackathon Timeline

| Time Block | Phase | Deliverable |
|---|---|---|
| Hour 0–3 | Phase 1 | Bronze & Silver Delta tables live in Unity Catalog |
| Hour 3–9 | Phase 2 | Extraction agent processing all 10k records with MLflow tracing |
| Hour 9–12 | Phase 3 | Trust scores computed for all records |
| Hour 12–17 | Phase 4 | Query agent live, `/query` endpoint returning cited results |
| Hour 17–20 | Phase 5 | Desert analysis table + GeoJSON ready |
| Hour 20–26 | Phase 6 | Full dashboard deployed on Databricks Apps |
| Hour 26–29 | Phase 7 | Validator agent running, confidence intervals computed |
| Hour 29–30 | Final | Demo prep, README, MLflow trace screenshots |

---

## 9. Key Design Decisions & Rationale

**Why Claude Sonnet for extraction?**
The `description` and `capability` fields are written in diverse Indian English styles with medical jargon, abbreviations, and non-standard phrasing. Claude Sonnet's strong instruction-following and low hallucination rate make it the best fit for structured extraction with verifiable citations.

**Why Medallion Architecture?**
The raw data is too dirty to run agents on directly. The Bronze/Silver split ensures the extraction agent always works from cleaned, consistently-typed data — reducing prompt complexity and improving output reliability.

**Why Vector Search + SQL Hybrid?**
Pure SQL cannot handle fuzzy queries like "facility that can handle complicated childbirth." Pure vector search cannot enforce hard constraints like "must be in Rajasthan with trust score above 60." The hybrid approach handles both.

**Why Streamlit over React (for hackathon)?**
Streamlit allows a single Python developer to build a functional, deployed dashboard in 4–6 hours. React + FastAPI gives better UX but requires a minimum of 8–10 hours for a comparable result. The hackathon evaluation weights functionality over polish.

**Why not trust the structured fields directly?**
93.7% of `numberDoctors` and 99% of `capacity` are null. 84% of `equipment` is null. The structured fields are insufficient as primary intelligence sources — the system must derive its intelligence from text.

---

*Implementation Plan v1.0 — Aligned with FR/NFR Specification v1.0*