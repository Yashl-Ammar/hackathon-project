# Agentic Healthcare Intelligence System
## Requirements Specification Document
**Challenge:** Serving A Nation — Building Agentic Healthcare Maps for 1.4 Billion Lives
**Powered by:** Databricks Data Intelligence Platform

---

## 1. Background & Context

The system operates on a dataset of **10,000 Indian medical facilities** across all states. A preliminary audit of the dataset reveals severe data quality challenges that directly shape the requirements:

| Field | Null / Empty Rate | Implication |
|---|---|---|
| `capacity` | 99.0% | Cannot rely on structured capacity data |
| `numberDoctors` | 93.7% | Doctor counts must be inferred from free text |
| `equipment` | 84.0% | Equipment must be extracted from unstructured fields |
| `procedure` | 66.0% | Procedures must be parsed from descriptions |
| `capability` | 35.8% | Partial — must be cross-validated with other fields |
| `description` | 9.4% | Mostly present — primary NLP source |
| `specialties` | 0.0% | Fully populated — reliable anchor field |
| `latitude / longitude` | 0.0% | Fully populated — reliable for geo-mapping |

This means the system **cannot rely on structured fields alone**. The intelligence layer must extract, infer, and validate capabilities primarily from unstructured free-text fields.

---

## 2. Functional Requirements

### FR-1: Data Ingestion & Preprocessing

**FR-1.1 — Bulk Ingestion**
The system shall ingest all 10,000 facility records from the source CSV/Excel dataset into the Databricks Lakehouse (Delta Lake format) without data loss.

**FR-1.2 — Schema Standardisation**
The system shall map all raw fields to a standardised Pydantic-compatible schema (the Virtue Foundation Schema), normalising inconsistencies in casing, encoding, and list formatting (e.g., JSON-stringified arrays in `specialties`, `procedure`, `equipment`).

**FR-1.3 — Null Handling**
The system shall classify each record's completeness and flag fields that are missing, empty arrays, or contain placeholder values such as `"null"` or `[]`.

---

### FR-2: Unstructured Text Extraction (Intelligent Document Parsing)

**FR-2.1 — Capability Extraction**
The agent shall extract clinical capabilities from the `description`, `capability`, `procedure`, and `equipment` fields using a language model, producing structured outputs including:
- Available medical procedures
- Equipment present (e.g., ventilators, ICU beds, dialysis machines)
- Staffing details (e.g., number of doctors, specialist types)
- Operational status (e.g., 24/7 availability, emergency readiness)

**FR-2.2 — Specialty Parsing**
The agent shall parse and normalise the `specialties` field (which contains coded strings such as `"oncology"`, `"neonatalogy"`, `"emergencyMedicine"`) into human-readable labels and group them into clinical categories (e.g., High-Acuity, Primary Care, Surgical, Diagnostic).

**FR-2.3 — Claim Verification**
The agent shall cross-reference capability claims made in one field against evidence in other fields. For example:
- A facility claiming "Advanced Surgery" must have an anaesthesiologist mentioned somewhere.
- A facility claiming "ICU" must list related equipment or staffing evidence.

**FR-2.4 — Contradiction Flagging**
The agent shall flag records where extracted data contradicts structured metadata. Examples:
- `facilityTypeId = "clinic"` but description claims "full-service hospital with 200 beds."
- `numberDoctors = 1` but description claims "team of 50 specialists."

---

### FR-3: Trust Scoring Engine

**FR-3.1 — Per-Facility Trust Score**
The system shall compute a Trust Score (0–100) for every facility based on:
- **Completeness:** How many key fields are populated.
- **Consistency:** Whether claims across fields agree with each other.
- **Verifiability:** Whether claims are supported by specific evidence (named equipment, named staff) vs. generic statements.
- **Recency:** Whether the data has been recently updated (`recency_of_page_update`).
- **Digital Presence:** Social media activity as a proxy for operational status.

**FR-3.2 — Score Decomposition**
The Trust Score shall be decomposed into sub-scores per dimension so that users understand why a facility scored high or low.

**FR-3.3 — Contradiction Penalty**
Facilities where extracted data directly contradicts stated metadata shall receive an automatic penalty to their Trust Score and a human-readable explanation of the contradiction.

---

### FR-4: Agentic Query Interface

**FR-4.1 — Natural Language Querying**
The system shall accept free-text queries from users and resolve them against the 10,000-record knowledge base. Example queries:
- *"Find the nearest facility in rural Bihar that can perform an emergency appendectomy."*
- *"Which districts in Rajasthan have no dialysis centre?"*
- *"List hospitals in Kerala with a functional neonatal ICU and more than 5 specialists."*

**FR-4.2 — Multi-Attribute Reasoning**
The agent shall resolve queries involving multiple simultaneous constraints (geography + specialty + equipment + operational status) rather than keyword-matching on a single field.

**FR-4.3 — Ranked Results with Justification**
For every query, the system shall return a ranked list of matching facilities, each accompanied by:
- The Trust Score and its breakdown.
- The exact text snippet(s) from the facility record that justify why it was recommended.
- The distance from a reference location if one is provided.

**FR-4.4 — Agentic Traceability (Row-Level Citations)**
For every recommendation, the agent shall cite the specific field and sentence that justifies the conclusion. Example: *"Recommended because capability field states: 'Has a functional ICU with 6 ventilators and a full-time anaesthesiologist.'"*

---

### FR-5: Medical Desert Identification

**FR-5.1 — Gap Analysis by Region**
The system shall identify geographic regions (by state, district, and PIN code) where critical medical specialties are absent or severely under-represented, including:
- Emergency Trauma
- Oncology
- Dialysis / Nephrology
- Neonatal / Paediatric ICU
- Mental Health
- Obstetrics & Gynaecology

**FR-5.2 — Desert Severity Scoring**
Each identified gap shall receive a severity score based on:
- Population density of the area (if estimable from PIN code).
- Distance to the nearest facility with the required capability.
- Number of facilities with the required capability within a 50km and 100km radius.

**FR-5.3 — Actionable Reporting**
The system shall produce a summary report of the top medical deserts per state, formatted for NGO planners and healthcare policymakers.

---

### FR-6: Visual Dashboard & Map

**FR-6.1 — Interactive Facility Map**
The system shall render all 10,000 facilities on an interactive map of India, with colour coding by:
- Facility type (hospital, clinic, pharmacy, etc.)
- Trust Score (red = low trust, green = high trust)
- Specialty availability

**FR-6.2 — Desert Heatmap Overlay**
The map shall include an overlay layer that highlights medical deserts by PIN code or district, colour-coded by severity.

**FR-6.3 — Drill-Down**
Clicking on a facility on the map shall reveal its full profile including Trust Score breakdown, extracted capabilities, and raw citation snippets.

---

### FR-7: Self-Correction & Validation Agent

**FR-7.1 — Validator Agent**
A secondary Validator Agent shall independently re-evaluate a sample of the primary agent's extractions and flag cases where the primary agent may have hallucinated or over-inferred capabilities not supported by the source text.

**FR-7.2 — Confidence Intervals**
Where data is sparse or contradictory, the system shall report a confidence range rather than a point estimate, acknowledging uncertainty explicitly.

---

## 3. Non-Functional Requirements

### NFR-1: Performance

**NFR-1.1 — Batch Processing Throughput**
The system shall process all 10,000 facility records (extraction, scoring, indexing) within a reasonable batch window on Databricks serverless compute (target: under 60 minutes for full pipeline).

**NFR-1.2 — Query Latency**
Interactive natural language queries shall return ranked results within **10 seconds** for simple single-constraint queries and **30 seconds** for complex multi-constraint queries.

**NFR-1.3 — Vector Search Speed**
Semantic similarity search across the 10,000-record index (via Mosaic AI Vector Search) shall return candidates within **2 seconds**.

---

### NFR-2: Accuracy & Reliability

**NFR-2.1 — Extraction Fidelity**
The agent shall not fabricate capabilities that are not present in the source record. All extracted attributes must be traceable to at least one source sentence.

**NFR-2.2 — Hallucination Prevention**
The system shall implement guardrails (e.g., grounding checks, Validator Agent cross-reference) to reduce hallucinated extractions to a minimum. Any unverifiable claim shall be marked as `low_confidence`.

**NFR-2.3 — Consistency**
Running the same query twice shall produce the same ranked results and citations (deterministic or near-deterministic outputs).

---

### NFR-3: Scalability

**NFR-3.1 — Record Volume**
The architecture shall be designed to scale beyond 10,000 records to 100,000+ without requiring re-architecture — only additional compute.

**NFR-3.2 — Concurrent Users**
The query interface shall support at least 10 concurrent users without performance degradation.

---

### NFR-4: Observability & Traceability

**NFR-4.1 — MLflow 3 Tracing**
Every agent run shall be logged in MLflow 3, capturing:
- Input query or record ID
- Each reasoning step and tool call
- Token usage and cost per step
- Final output and confidence level

**NFR-4.2 — Audit Trail**
All Trust Score computations shall be fully auditable — a reviewer must be able to reproduce a score from the logged inputs and logic.

**NFR-4.3 — Step-Level Explainability**
The system shall expose its Chain of Thought at query time so that a human reviewer can follow the agent's reasoning without requiring access to logs.

---

### NFR-5: Data Quality & Governance

**NFR-5.1 — Unity Catalog Governance**
All datasets (raw, processed, extracted, scored) shall be registered in Databricks Unity Catalog with appropriate access controls, lineage tracking, and data quality tags.

**NFR-5.2 — Data Provenance**
Every derived field (e.g., extracted ICU status) shall be linked back to its source record and source field to maintain full provenance.

**NFR-5.3 — Missing Data Transparency**
The system shall never treat a missing field as evidence of absence. A null `equipment` field means *unknown*, not *no equipment*. This distinction must be reflected in Trust Scores and outputs.

---

### NFR-6: User Experience

**NFR-6.1 — Accessibility**
The interface shall be usable by non-technical users (NGO planners, health administrators) without requiring knowledge of SQL, Python, or data tools.

**NFR-6.2 — Transparency**
Every recommendation, score, and map annotation shall include a plain-English explanation. No black-box outputs.

**NFR-6.3 — Actionability**
Outputs shall be framed in terms of decisions. For example: *"District X has no oncology facility within 150km — consider targeting for NGO intervention."*

---

### NFR-7: Security & Privacy

**NFR-7.1 — Data Handling**
Facility contact details (phone, email) shall be treated as potentially sensitive and access-controlled within Unity Catalog.

**NFR-7.2 — No PII Leakage**
Patient data is not present in this dataset. The system shall ensure no external data sources containing patient PII are introduced during enrichment steps.

---

## 4. Constraints

| Constraint | Description |
|---|---|
| **Platform** | Must run on Databricks Free Edition using serverless compute |
| **Primary Tech Stack** | Agent Bricks, MLflow 3, Mosaic AI Vector Search, Unity Catalog |
| **No Ground Truth** | There is no labelled answer key — the system must validate its own outputs |
| **Data Messiness** | 84% of equipment fields are null; the agent must handle extreme sparsity gracefully |
| **Open Source Preferred** | Solutions should favour open standards (Delta Lake, MLflow, Apache Spark) |

---

## 5. Evaluation Mapping

| Evaluation Criterion | Weight | Requirements Addressed |
|---|---|---|
| Discovery & Verification | 35% | FR-2, FR-3, FR-7 |
| IDP Innovation | 30% | FR-2.1, FR-2.2, FR-2.3, FR-2.4 |
| Social Impact & Utility | 25% | FR-5, FR-6, NFR-6.3 |
| UX & Transparency | 10% | FR-4.3, FR-4.4, NFR-4.3, NFR-6 |

---

*Document Version 1.0 — Based on VF Hackathon Dataset (India, 10,000 records) and Challenge Brief.*
