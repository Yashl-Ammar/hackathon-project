"""
query_agent.py
CareMap — Healthcare Query Agent Module

All agent functions packaged as a standalone Python module
for import by the FastAPI backend (app.py).

Functions:
  parse_query()              — LLM call #1: parse natural language query
  hybrid_search()            — SQL search against Gold + Silver tables
  format_results()           — build citations from extracted data
  validate_top_results()     — LLM call #2: batch validate top 5 results
  check_medical_standards()  — rule-based standards check (no LLM)
  re_extract_with_feedback() — LLM call #3: self-correct disputed results
  query_healthcare()         — main orchestrator: runs full pipeline
"""

import json
import os
import mlflow
import mlflow.deployments

# ── Config ────────────────────────────────────────────────────
GOLD_TABLE   = "workspace.gold.facilities_enriched"
SILVER_TABLE = "workspace.silver.facilities_clean"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# ── Spark + MLflow setup ──────────────────────────────────────
spark_error = None
try:
    from pyspark.sql import SparkSession
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    spark = SparkSession.builder \
        .config("spark.databricks.service.client.enabled", "true") \
        .config("spark.databricks.sql.warehouse.id", warehouse_id) \
        .getOrCreate()
    print(f"Spark session created with warehouse: {warehouse_id}")
except Exception as e:
    spark_error = str(e)
    print(f"Spark error: {spark_error}")
    spark = None

try:
    import mlflow
    import mlflow.deployments
    mlflow.set_experiment("/Users/yashlammarr@gmail.com/query_agent_runs")
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    print("MLflow client created")
except Exception as e:
    print(f"MLflow error: {e}")
    deploy_client = None

# ── LLM helper ────────────────────────────────────────────────
def _call_llm(messages: list, max_tokens: int = 800) -> str:
    """Call LLM and return raw text response."""
    response = deploy_client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages":   messages,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
    )
    raw = response["choices"][0]["message"]["content"].strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _parse_json(raw: str, default):
    """Safely parse JSON, unwrapping properties key if present."""
    try:
        parsed = json.loads(raw)
        if "properties" in parsed and isinstance(parsed["properties"], dict):
            return parsed["properties"]
        return parsed
    except:
        return default


# ═══════════════════════════════════════════════════════════════
# STEP 1 — QUERY PARSER
# ═══════════════════════════════════════════════════════════════

PARSER_SYSTEM_PROMPT = """You are a medical query parser for an Indian healthcare intelligence system.
Extract structured search parameters from natural language queries.
Return ONLY a valid JSON object. No explanation, no markdown.

Indian states include: Maharashtra, Uttar Pradesh, Gujarat, Tamil Nadu, Kerala, Rajasthan,
West Bengal, Karnataka, Delhi, Telangana, Bihar, Haryana, Punjab, Madhya Pradesh,
Andhra Pradesh, Odisha, Jharkhand, Uttarakhand, Chhattisgarh, Assam, Jammu & Kashmir,
Himachal Pradesh, Goa, Tripura, Meghalaya, Manipur, Nagaland, Puducherry, Chandigarh.
"""

PARSER_SCHEMA = {
    "state":             "Indian state name or null if not specified",
    "city":              "City name or null",
    "facility_type":     "One of: hospital, clinic, dentist, doctor, pharmacy, or null",
    "specialties":       "List of medical specialties e.g. ['oncology', 'cardiology']",
    "requires_icu":      "true/false/null",
    "requires_emergency":"true/false/null",
    "requires_24_7":     "true/false/null",
    "requires_dialysis": "true/false/null",
    "min_trust_score":   "Minimum trust score 0-100 or null",
    "search_text":       "Core clinical search terms for semantic search"
}

def parse_query(query: str) -> dict:
    """Parse natural language query into structured filters. LLM call #1."""
    prompt = f"""Parse this healthcare facility search query for India:

Query: "{query}"

Return a JSON object with these fields:
{json.dumps(PARSER_SCHEMA, indent=2)}

Return ONLY the JSON object."""

    raw = _call_llm([
        {"role": "system", "content": PARSER_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt}
    ], max_tokens=500)

    return _parse_json(raw, {})


# ═══════════════════════════════════════════════════════════════
# STEP 2 — HYBRID SEARCH
# ═══════════════════════════════════════════════════════════════

def hybrid_search(parsed_query: dict, num_results: int = 20) -> list:
    """
    SQL search against Gold + Silver tables.
    Vector Search skipped — SQL-only mode for hackathon.
    """
    # Pre-build conditional clauses
    state         = parsed_query.get("state")
    city          = parsed_query.get("city")
    facility_type = parsed_query.get("facility_type")
    min_trust     = parsed_query.get("min_trust_score")
    req_icu       = parsed_query.get("requires_icu")
    req_emerg     = parsed_query.get("requires_emergency")
    req_dialysis  = parsed_query.get("requires_dialysis")

    state_clause    = f"AND s.address_stateOrRegion = '{state}'"         if state         else ""
    city_clause     = f"AND s.address_city LIKE '%{city}%'"              if city          else ""
    type_clause     = f"AND s.facilityTypeId = '{facility_type}'"        if facility_type else ""
    trust_clause    = f"AND g.trust_score >= {min_trust}"                if min_trust     else ""
    icu_clause      = "AND (g.extracted_availability LIKE '%\"has_icu\": true%' OR g.extracted_availability LIKE '%\"has_icu\":true%')"         if req_icu      else ""
    emerg_clause    = "AND (g.extracted_availability LIKE '%\"has_emergency\": true%' OR g.extracted_availability LIKE '%\"has_emergency\":true%')" if req_emerg else ""
    dialysis_clause = "AND (CAST(s.specialties AS STRING) LIKE '%nephrology%' OR CAST(s.specialties AS STRING) LIKE '%dialysis%')" if req_dialysis else ""

    query = f"""
        SELECT
            g.name,
            s.address_stateOrRegion,
            s.address_city,
            s.address_zipOrPostcode,
            s.facilityTypeId,
            s.latitude,
            s.longitude,
            s.specialties,
            s.officialPhone,
            s.officialWebsite,
            s.description,
            s.searchable_text,
            g.trust_score,
            g.trust_score_breakdown,
            g.confidence_level,
            g.extracted_capabilities,
            g.extracted_availability,
            g.extracted_staff,
            g.extracted_equipment,
            g.contradictions,
            g.extracted_bed_count
        FROM {GOLD_TABLE} g
        JOIN {SILVER_TABLE} s ON g.name = s.name
        WHERE g.trust_score IS NOT NULL
        {state_clause}
        {city_clause}
        {type_clause}
        {trust_clause}
        {icu_clause}
        {emerg_clause}
        {dialysis_clause}
        ORDER BY g.trust_score DESC
        LIMIT 100
    """

    df = spark.sql(query).toPandas()
    df["final_score"] = df["trust_score"]
    return df.head(num_results).to_dict("records")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — FORMAT RESULTS WITH CITATIONS
# ═══════════════════════════════════════════════════════════════

def format_results(candidates: list, original_query: str) -> list:
    """Format results with row-level citations from source text."""
    formatted = []
    for i, c in enumerate(candidates):
        avail     = _parse_json(c.get("extracted_availability"), {})
        caps      = _parse_json(c.get("extracted_capabilities"), [])
        staff     = _parse_json(c.get("extracted_staff"), {})
        breakdown = _parse_json(c.get("trust_score_breakdown"), {})

        # Build sentence citations
        sentence_citations = []
        if avail.get("has_icu"):
            sentence_citations.append("ICU confirmed (extracted from facility text)")
        if avail.get("has_emergency"):
            sentence_citations.append("Emergency services confirmed (extracted from facility text)")
        if avail.get("has_24_7"):
            sentence_citations.append("24/7 availability confirmed (extracted from facility text)")
        if caps:
            sentence_citations.append(f"Key capabilities extracted: {', '.join(caps[:3])}")
        if staff.get("doctor_count"):
            sentence_citations.append(f"{staff['doctor_count']} doctors identified in facility text")
        if not sentence_citations:
            sentence_citations.append("Basic facility information — insufficient text for detailed citation")

        # Raw source text snippet
        description   = str(c.get("description") or "")
        source_snippet = description[:400] if description else str(c.get("searchable_text") or "")[:400]

        # Trust justification
        trust_justification = []
        if breakdown:
            trust_justification = [
                f"Completeness: {breakdown.get('completeness', 0)}/100",
                f"Consistency: {breakdown.get('consistency', 0)}/100",
                f"Verifiability: {breakdown.get('verifiability', 0)}/100",
                f"Recency: {breakdown.get('recency', 0)}/100",
                f"Digital presence: {breakdown.get('digital_presence', 0)}/100",
            ]

        # Contradictions
        contradictions = []
        try:
            raw_contra = c.get("contradictions")
            if raw_contra is not None:
                if isinstance(raw_contra, str):
                    contradictions = json.loads(raw_contra)
                elif hasattr(raw_contra, 'size'):
                    contradictions = raw_contra.tolist() if raw_contra.size > 0 else []
                elif isinstance(raw_contra, list):
                    contradictions = raw_contra
        except:
            contradictions = []

        formatted.append({
            "rank":                i + 1,
            "name":                c.get("name"),
            "facility_type":       c.get("facilityTypeId"),
            "state":               c.get("address_stateOrRegion"),
            "city":                c.get("address_city"),
            "pin_code":            c.get("address_zipOrPostcode"),
            "phone":               c.get("officialPhone"),
            "website":             c.get("officialWebsite"),
            "latitude":            c.get("latitude"),
            "longitude":           c.get("longitude"),
            "trust_score":         c.get("trust_score"),
            "confidence":          c.get("confidence_level"),
            "has_icu":             avail.get("has_icu"),
            "has_emergency":       avail.get("has_emergency"),
            "has_24_7":            avail.get("has_24_7"),
            "key_capabilities":    caps[:5],
            "doctor_count":        staff.get("doctor_count"),
            "bed_count":           c.get("extracted_bed_count"),
            "citation":            " | ".join(sentence_citations),
            "sentence_citations":  sentence_citations,
            "source_text":         source_snippet,
            "source_field":        "description + capability statements",
            "trust_justification": trust_justification,
            "contradictions":      contradictions,
            "trust_breakdown":     breakdown,
        })

    return formatted


# ═══════════════════════════════════════════════════════════════
# STEP 4 — BATCH VALIDATOR
# ═══════════════════════════════════════════════════════════════

VALIDATOR_SYSTEM_PROMPT = """You are a medical claims validator for an Indian healthcare
intelligence system. Verify whether specific medical capability claims are actually
supported by the raw facility text.
Be strict — only confirm claims EXPLICITLY stated in the text.
Return ONLY a valid JSON object. No explanation, no markdown."""

def validate_top_results(results: list) -> list:
    """Batch validate top 5 results in one LLM call. LLM call #2."""
    if not results:
        return []

    facilities_text = ""
    for i, r in enumerate(results[:5]):
        facilities_text += f"""
FACILITY {i+1}: {r['name']}
Source text: "{r.get('source_text', 'No source text available')[:300]}"
Claims to verify:
  - has_icu: {r.get('has_icu')}
  - has_emergency: {r.get('has_emergency')}
  - has_24_7: {r.get('has_24_7')}
  - key_capabilities: {r.get('key_capabilities', [])}
---"""

    n = min(len(results), 5)
    prompt = f"""Validate these {n} medical facility claims against their source text.

{facilities_text}

For each facility return:
{{
  "facility_1": {{
    "has_icu_supported": true/false/null,
    "has_emergency_supported": true/false/null,
    "has_24_7_supported": true/false/null,
    "capabilities_supported": true/false,
    "validation_note": "one sentence",
    "overall_verdict": "VERIFIED" or "DISPUTED" or "INSUFFICIENT_TEXT"
  }},
  ...up to "facility_{n}"
}}

VERIFIED = all claims match source text
DISPUTED = at least one claim not supported
INSUFFICIENT_TEXT = text too vague to verify

Return ONLY the JSON object."""

    try:
        raw = _call_llm([
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ], max_tokens=1000)

        parsed = _parse_json(raw, {})

        validations = []
        for i in range(n):
            v = parsed.get(f"facility_{i+1}", {})
            validations.append({
                "facility_name":           results[i]["name"],
                "has_icu_supported":       v.get("has_icu_supported"),
                "has_emergency_supported": v.get("has_emergency_supported"),
                "has_24_7_supported":      v.get("has_24_7_supported"),
                "capabilities_supported":  v.get("capabilities_supported"),
                "validation_note":         v.get("validation_note", ""),
                "overall_verdict":         v.get("overall_verdict", "INSUFFICIENT_TEXT"),
            })
        return validations

    except Exception as e:
        return [
            {
                "facility_name":           r["name"],
                "has_icu_supported":       None,
                "has_emergency_supported": None,
                "has_24_7_supported":      None,
                "capabilities_supported":  None,
                "validation_note":         f"Validation error: {str(e)[:100]}",
                "overall_verdict":         "INSUFFICIENT_TEXT",
            }
            for r in results[:5]
        ]


# ═══════════════════════════════════════════════════════════════
# STEP 5 — SELF-CORRECTION LOOP
# ═══════════════════════════════════════════════════════════════

MEDICAL_STANDARDS = {
    "has_icu":       ["icu", "intensive care", "critical care", "ventilator", "intensivist"],
    "has_emergency": ["emergency", "a&e", "trauma", "urgent care", "casualty", "24/7", "round the clock"],
    "has_surgery":   ["operation theatre", "ot ", "surgeon", "surgical", "surgery"],
    "has_24_7":      ["24/7", "24 hours", "round the clock", "always open", "day and night"],
}

CORRECTION_SYSTEM_PROMPT = """You are a medical data extraction specialist.
A previous extraction was flagged as inaccurate. Re-extract MORE CAREFULLY,
applying strict evidence standards. Only include claims with EXPLICIT textual support.
Return ONLY a valid JSON object. No explanation, no markdown."""

def check_medical_standards(source_text: str, claims: dict) -> dict:
    """Rule-based medical standards check — no LLM needed."""
    text_lower = source_text.lower()
    results = {}
    for claim, keywords in MEDICAL_STANDARDS.items():
        claim_value = claims.get(claim)
        found = [kw for kw in keywords if kw in text_lower]
        if claim_value and not found:
            results[claim] = {"claimed": True, "supported": False, "evidence": None,
                              "standard_note": f"Expected one of: {keywords[:3]}"}
        elif claim_value and found:
            results[claim] = {"claimed": True, "supported": True, "evidence": found[0]}
        else:
            results[claim] = {"claimed": False, "supported": None, "evidence": None}
    return results


def re_extract_with_feedback(source_text: str, facility_name: str,
                             original_claims: dict, validator_note: str) -> dict:
    """Re-extract with validator feedback. LLM call #3 — only for DISPUTED results."""
    standards_check = check_medical_standards(source_text, original_claims)
    failed = [c for c, r in standards_check.items() if r.get("claimed") and not r.get("supported")]

    prompt = f"""Re-extract clinical information. Previous extraction was disputed.

Facility: {facility_name}
Source text: "{source_text[:600]}"

Validator feedback: {validator_note}
Medical standards check failed for: {', '.join(failed) if failed else 'none'}

Original claims:
- has_icu: {original_claims.get('has_icu')}
- has_emergency: {original_claims.get('has_emergency')}
- has_24_7: {original_claims.get('has_24_7')}
- key_capabilities: {original_claims.get('key_capabilities', [])}

Return JSON:
{{
  "has_icu": true/false/null,
  "has_emergency": true/false/null,
  "has_24_7": true/false/null,
  "key_capabilities": ["verified", "capabilities"],
  "doctor_count": number or null,
  "corrected_citation": "exact phrase from text",
  "correction_note": "what changed and why",
  "confidence": "high/medium/low"
}}

Only include claims with EXPLICIT evidence. Return ONLY the JSON."""

    try:
        raw = _call_llm([
            {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ], max_tokens=600)

        corrected = _parse_json(raw, {})
        corrected["standards_check"] = standards_check
        corrected["self_corrected"]  = True
        return corrected

    except Exception as e:
        return {
            "has_icu":            original_claims.get("has_icu"),
            "has_emergency":      original_claims.get("has_emergency"),
            "has_24_7":           original_claims.get("has_24_7"),
            "key_capabilities":   original_claims.get("key_capabilities", []),
            "corrected_citation": "Correction failed — original retained",
            "correction_note":    f"Error: {str(e)[:100]}",
            "confidence":         "low",
            "self_corrected":     False,
            "standards_check":    standards_check,
        }


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def query_healthcare(query: str, num_results: int = 10, verbose: bool = False) -> dict:
    """
    Full agentic pipeline:
    1. Parse query         (LLM #1)
    2. Hybrid SQL search
    3. Format with citations
    4. Batch validate      (LLM #2)
    5. Self-correct DISPUTED results (LLM #3, only if needed)
    6. Medical standards check (rule-based)

    Returns dict with results, chain_of_thought, validation, total_found.
    """
    chain_of_thought = []

    with mlflow.start_run(run_name=f"query_{query[:30]}"):
        mlflow.log_param("query", query)

        # Step 1 — Parse
        parsed = parse_query(query)
        mlflow.log_dict(parsed, "parsed_query.json")
        chain_of_thought.append({
            "step":   1,
            "action": "Query parsed",
            "detail": f"State: {parsed.get('state') or 'any'} | "
                      f"ICU: {parsed.get('requires_icu')} | "
                      f"Emergency: {parsed.get('requires_emergency')} | "
                      f"Search: '{parsed.get('search_text')}'"
        })

        # Step 2 — Search
        candidates = hybrid_search(parsed, num_results=num_results * 2)
        mlflow.log_metric("candidates_found", len(candidates))
        chain_of_thought.append({
            "step":   2,
            "action": "Hybrid search complete",
            "detail": f"SQL filters applied across 9,253 enriched records — "
                      f"{len(candidates)} facilities matched"
        })

        if not candidates:
            return {"results": [], "chain_of_thought": chain_of_thought,
                    "parsed_query": parsed, "total_found": 0, "validation": []}

        # Step 3 — Format
        results = format_results(candidates[:num_results], query)
        mlflow.log_metric("results_returned", len(results))
        if results:
            chain_of_thought.append({
                "step":   3,
                "action": "Results ranked by trust score",
                "detail": f"Top result: '{results[0]['name']}' "
                          f"(trust: {results[0]['trust_score']}, "
                          f"confidence: {results[0]['confidence']})"
            })

        # Step 4 — Validate
        validations    = validate_top_results(results[:5])
        verdicts       = [v["overall_verdict"] for v in validations]
        verified_count = verdicts.count("VERIFIED")
        disputed_count = verdicts.count("DISPUTED")
        mlflow.log_metric("results_validated", len(validations))
        chain_of_thought.append({
            "step":   4,
            "action": "Validator agent complete",
            "detail": f"Checked top {len(validations)} against source text — "
                      f"{verified_count} VERIFIED, {disputed_count} DISPUTED"
        })

        # Step 5 — Self-correction
        validation_map  = {v["facility_name"]: v for v in validations}
        corrected_count = 0

        for r in results:
            v       = validation_map.get(r["name"], {})
            verdict = v.get("overall_verdict", "NOT_VALIDATED")

            if verdict == "DISPUTED":
                corrected = re_extract_with_feedback(
                    source_text    = r.get("source_text", ""),
                    facility_name  = r["name"],
                    original_claims= {
                        "has_icu":          r.get("has_icu"),
                        "has_emergency":    r.get("has_emergency"),
                        "has_24_7":         r.get("has_24_7"),
                        "key_capabilities": r.get("key_capabilities", [])
                    },
                    validator_note = v.get("validation_note", "")
                )
                r["has_icu"]          = corrected.get("has_icu",          r["has_icu"])
                r["has_emergency"]    = corrected.get("has_emergency",    r["has_emergency"])
                r["has_24_7"]         = corrected.get("has_24_7",         r["has_24_7"])
                r["key_capabilities"] = corrected.get("key_capabilities", r["key_capabilities"])
                r["citation"]         = corrected.get("corrected_citation", r["citation"])
                r["self_corrected"]   = corrected.get("self_corrected", False)
                r["correction_note"]  = corrected.get("correction_note", "")
                r["standards_check"]  = corrected.get("standards_check", {})
                if corrected.get("confidence") == "low":
                    r["confidence"] = "low"
                corrected_count += 1
            else:
                r["standards_check"] = check_medical_standards(
                    r.get("source_text", ""),
                    {"has_icu": r.get("has_icu"), "has_emergency": r.get("has_emergency"),
                     "has_24_7": r.get("has_24_7")}
                )
                r["self_corrected"]  = False
                r["correction_note"] = ""

            r["validation_verdict"]     = verdict
            r["validation_note"]        = v.get("validation_note", "")
            r["icu_verified"]           = v.get("has_icu_supported")
            r["emergency_verified"]     = v.get("has_emergency_supported")
            r["availability_verified"]  = v.get("has_24_7_supported")

            if verdict == "DISPUTED" and not r.get("self_corrected"):
                r["trust_score_display"] = round(r["trust_score"] * 0.85, 1)
                r["trust_score_warning"] = "Trust score adjusted — disputed claims not correctable"
            elif r.get("self_corrected"):
                r["trust_score_display"] = round(r["trust_score"] * 0.90, 1)
                r["trust_score_warning"] = "Self-corrected — extraction revised after validation"
            else:
                r["trust_score_display"] = r["trust_score"]
                r["trust_score_warning"] = None

        chain_of_thought.append({
            "step":   5,
            "action": "Self-correction loop complete",
            "detail": f"{corrected_count} result(s) re-extracted with validator feedback. "
                      f"Medical standards check applied to all results."
        })

        mlflow.log_metric("self_corrected_count", corrected_count)
        mlflow.log_dict({"chain_of_thought": chain_of_thought}, "chain_of_thought.json")
        mlflow.log_dict({"validations": validations}, "validations.json")

    return {
        "results":          results,
        "chain_of_thought": chain_of_thought,
        "parsed_query":     parsed,
        "total_found":      len(candidates),
        "validation":       validations
    }
