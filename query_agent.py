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
import time
import mlflow
import mlflow.deployments

# ── Config ────────────────────────────────────────────────────
GOLD_TABLE   = "workspace.gold.facilities_enriched"
SILVER_TABLE = "workspace.silver.facilities_clean"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# ── Spark + MLflow setup ──────────────────────────────────────
spark_error = None
try:
    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.serverless(True).getOrCreate()
    print("Spark session created via DatabricksSession (serverless)")
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
@mlflow.trace(span_type="LLM", name="llm_call")
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
    "state":                   "Indian state name or null if not specified",
    "city":                    "City name or null",
    "facility_type":           "One of: hospital, clinic, dentist, doctor, pharmacy, or null",
    "specialties":             "List of medical specialties e.g. ['oncology', 'cardiology']",
    "requires_icu":            "true/false/null",
    "requires_emergency":      "true/false/null",
    "requires_24_7":           "true/false/null",
    "requires_dialysis":       "true/false/null",
    "requires_contradictions": "true if user wants facilities WITH data contradictions/disputes, false/null otherwise",
    "min_trust_score":         "Minimum trust score 0-100 or null",
    "search_text":             "Core clinical search terms — exclude meta-words like 'contradictions', 'disputes', 'claiming'"
}

@mlflow.trace(span_type="LLM", name="parse_query")
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

@mlflow.trace(name="hybrid_search")
def hybrid_search(parsed_query: dict, num_results: int = 20) -> list:
    """
    SQL search against Gold + Silver tables.
    Vector Search skipped — SQL-only mode for hackathon.
    """
    # Pre-build conditional clauses
    state                = parsed_query.get("state")
    city                 = parsed_query.get("city")
    facility_type        = parsed_query.get("facility_type")
    min_trust            = parsed_query.get("min_trust_score")
    req_icu              = parsed_query.get("requires_icu")
    req_emerg            = parsed_query.get("requires_emergency")
    req_dialysis         = parsed_query.get("requires_dialysis")
    req_contradictions   = parsed_query.get("requires_contradictions")
    search_text          = parsed_query.get("search_text") or ""
    specialties          = parsed_query.get("specialties") or []

    state_clause         = f"AND s.address_stateOrRegion LIKE '%{state}%'"   if state         else ""
    city_clause          = f"AND s.address_city LIKE '%{city}%'"             if city          else ""
    type_clause          = f"AND s.facilityTypeId = '{facility_type}'"       if facility_type else ""
    trust_clause         = f"AND g.trust_score >= {min_trust}"               if min_trust     else ""
    icu_clause           = "AND (g.extracted_availability LIKE '%\"has_icu\": true%' OR g.extracted_availability LIKE '%\"has_icu\":true%')"         if req_icu      else ""
    emerg_clause         = "AND (g.extracted_availability LIKE '%\"has_emergency\": true%' OR g.extracted_availability LIKE '%\"has_emergency\":true%')" if req_emerg else ""
    dialysis_clause      = "AND (CAST(s.specialties AS STRING) LIKE '%nephrology%' OR CAST(s.specialties AS STRING) LIKE '%dialysis%')" if req_dialysis else ""
    contradictions_clause = (
        "AND g.contradictions IS NOT NULL AND SIZE(g.contradictions) > 0"
    ) if req_contradictions else ""

    # Generic words that match everything or are meta-query words — exclude from keyword filter
    _STOP_WORDS = {
        "care", "clinic", "clinics", "hospital", "hospitals", "medical", "health",
        "centre", "center", "centers", "services", "service", "department", "unit",
        "find", "need", "want", "best", "good", "near", "india", "with", "that",
        "have", "their", "from", "this", "show", "give", "list", "looking",
        # meta-query words (intent words, not clinical terms)
        "claiming", "contradictions", "contradiction", "disputes", "disputed",
        "verified", "unverified", "flagged", "suspicious",
    }

    # Collect meaningful terms: min length 3 (catches "eye", "icu", "ent"), skip stop words
    text_terms = [w for w in search_text.lower().split() if len(w) >= 3 and w not in _STOP_WORDS]
    for sp in specialties[:4]:
        text_terms += [w for w in sp.lower().split() if len(w) >= 3 and w not in _STOP_WORDS]
    text_terms = list(dict.fromkeys(text_terms))[:8]  # dedupe, cap at 8 terms

    if text_terms:
        # Build per-term match flags for relevance scoring
        relevance_parts = []
        for t in text_terms:
            relevance_parts.append(
                f"(CASE WHEN s.searchable_text LIKE '%{t}%' OR g.extracted_capabilities LIKE '%{t}%' OR CAST(s.specialties AS STRING) LIKE '%{t}%' THEN 1 ELSE 0 END)"
            )
        relevance_expr = " + ".join(relevance_parts)
        # At least one term must match
        text_conditions = " OR ".join(
            f"s.searchable_text LIKE '%{t}%' OR g.extracted_capabilities LIKE '%{t}%' OR CAST(s.specialties AS STRING) LIKE '%{t}%'"
            for t in text_terms
        )
        text_clause    = f"AND ({text_conditions})"
        relevance_col  = f"({relevance_expr}) AS relevance_score"
        order_by       = "relevance_score DESC, g.trust_score DESC"
    else:
        text_clause   = ""
        relevance_col = "0 AS relevance_score"
        order_by      = "g.trust_score DESC"

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
            g.extracted_bed_count,
            {relevance_col}
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
        {contradictions_clause}
        {text_clause}
        ORDER BY {order_by}
        LIMIT 100
    """

    df = spark.sql(query).toPandas()
    df["final_score"] = df["trust_score"]
    return df.head(num_results).to_dict("records")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — FORMAT RESULTS WITH CITATIONS
# ═══════════════════════════════════════════════════════════════

@mlflow.trace(name="format_results")
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

@mlflow.trace(span_type="LLM", name="validate_top_results")
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


@mlflow.trace(span_type="LLM", name="re_extract_with_feedback")
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
# DESERT / GAP QUERY PATH
# ═══════════════════════════════════════════════════════════════

_DESERT_KEYWORDS = {
    "desert", "deserts", "gap", "gaps", "underserved", "no facility", "no facilities",
    "lack", "lacking", "missing", "absent", "unavailable", "shortage", "shortages",
    "without", "unserved", "deprived", "coverage", "uncovered",
}

# Map common clinical phrases → specialty keywords to search in desert_analysis table
_SPECIALTY_MAP = {
    "dialysis":    ["dialysis", "nephrology"],
    "kidney":      ["dialysis", "nephrology"],
    "renal":       ["dialysis", "nephrology"],
    "mental":      ["psychiatry", "mental health", "psychology"],
    "psychiatry":  ["psychiatry", "mental health"],
    "psychiatric": ["psychiatry", "mental health"],
    "trauma":      ["emergency", "trauma"],
    "emergency":   ["emergency", "trauma"],
    "cancer":      ["oncology"],
    "oncology":    ["oncology"],
    "heart":       ["cardiology"],
    "cardiac":     ["cardiology"],
    "cardiology":  ["cardiology"],
    "maternity":   ["obstetrics", "gynecology"],
    "obstetric":   ["obstetrics", "gynecology"],
    "gynecology":  ["gynecology", "obstetrics"],
    "eye":         ["ophthalmology"],
    "ophthal":     ["ophthalmology"],
    "bone":        ["orthopedics"],
    "joint":       ["orthopedics"],
    "ortho":       ["orthopedics"],
    "surgery":     ["surgery"],
    "surgical":    ["surgery"],
    "icu":         ["critical care", "icu"],
    "critical":    ["critical care"],
    "child":       ["pediatrics"],
    "pediatric":   ["pediatrics"],
}


def _is_desert_query(query: str) -> bool:
    words = set(query.lower().split())
    return bool(words & _DESERT_KEYWORDS)


def _extract_specialty_filters(query: str) -> list[str]:
    """Return a list of specialty keyword patterns to filter desert_analysis on."""
    words = query.lower().split()
    patterns = []
    for w in words:
        for key, specialties in _SPECIALTY_MAP.items():
            if w.startswith(key):
                patterns.extend(specialties)
    return list(dict.fromkeys(patterns))  # dedupe, preserve order


@mlflow.trace(name="query_desert")
def query_desert(query: str) -> dict:
    """
    Query workspace.gold.desert_analysis and state_summary to answer
    'which states lack / have no X' style questions.
    Returns a dict compatible with query_healthcare() output format.
    """
    chain_of_thought = []
    t0 = time.time()

    specialty_filters = _extract_specialty_filters(query)

    chain_of_thought.append({
        "step":      1,
        "action":    "Desert query detected",
        "detail":    f"Routing to medical desert analysis. Specialty focus: {specialty_filters or 'all'}",
        "timing_ms": round((time.time() - t0) * 1000),
        "metadata":  {"specialty_filters": specialty_filters, "query": query},
    })

    t0 = time.time()

    # Build specialty WHERE clause
    if specialty_filters:
        specialty_conditions = " OR ".join(
            f"LOWER(d.specialty) LIKE '%{sp}%'" for sp in specialty_filters
        )
        specialty_clause = f"AND ({specialty_conditions})"
    else:
        specialty_clause = ""

    desert_sql = f"""
        SELECT
            d.state,
            d.specialty,
            d.specialty_facility_count  AS facility_count,
            d.total_facilities,
            d.desert_severity,
            d.severity_label,
            d.action,
            d.centroid_lat AS lat,
            d.centroid_lng AS lng
        FROM workspace.gold.desert_analysis d
        WHERE 1=1
        {specialty_clause}
        ORDER BY d.desert_severity DESC
        LIMIT 50
    """

    state_sql = """
        SELECT
            state,
            max_desert_severity,
            overall_severity_label,
            critical_specialty_count,
            total_facilities,
            centroid_lat AS lat,
            centroid_lng AS lng
        FROM workspace.gold.state_summary
        ORDER BY max_desert_severity DESC
        LIMIT 30
    """

    desert_df = spark.sql(desert_sql).toPandas()
    state_df  = spark.sql(state_sql).toPandas()

    chain_of_thought.append({
        "step":      2,
        "action":    "Desert analysis queried",
        "detail":    f"Found {len(desert_df)} state-specialty gaps. "
                     f"{len(state_df)} states in summary.",
        "timing_ms": round((time.time() - t0) * 1000),
        "metadata":  {
            "rows_returned":    len(desert_df),
            "specialty_filter": specialty_filters or "all",
        },
    })

    # Format rows as pseudo-results so the frontend can render them as cards
    results = []
    for _, row in desert_df.iterrows():
        severity  = str(row.get("severity_label", "")).upper()
        state     = str(row.get("state", ""))
        specialty = str(row.get("specialty", ""))
        count     = int(row.get("facility_count", 0))
        total     = int(row.get("total_facilities", 0))
        score     = float(row.get("desert_severity", 0))
        action    = str(row.get("action", ""))

        # Trust score inverted — high desert severity = low coverage = low "trust"
        display_trust = round(max(0, 100 - score), 1)

        tags = [{"label": f"Desert: {severity}", "warn": severity in ("CRITICAL", "HIGH")}]
        if count == 0:
            tags.append({"label": "Zero facilities", "warn": True})
        tags.append({"label": specialty, "warn": False})

        results.append({
            "name":             f"{state} — {specialty}",
            "state":            state,
            "city":             "",
            "facility_type":    "Desert Zone",
            "trust_score":      display_trust,
            "trust_score_display": display_trust,
            "confidence":       "high",
            "citation":         action or f"Only {count} of {total} expected facilities present",
            "key_capabilities": [],
            "has_icu":          False,
            "has_emergency":    False,
            "has_24_7":         False,
            "tags":             tags,
            "validation_verdict": "NOT_VALIDATED",
            "self_corrected":   False,
            "desert_severity":  score,
            "is_desert_result": True,
            "latitude":         float(row.get("lat", 20.5)),
            "longitude":        float(row.get("lng", 78.9)),
        })

    chain_of_thought.append({
        "step":      3,
        "action":    "Results formatted",
        "detail":    f"Top gap: {results[0]['name']} (severity score {results[0]['desert_severity']:.0f}/100)" if results else "No gaps found",
        "timing_ms": 0,
        "metadata":  {"results_count": len(results)},
    })

    return {
        "results":          results,
        "chain_of_thought": chain_of_thought,
        "parsed_query":     {"search_text": query, "specialties": specialty_filters},
        "total_found":      len(results),
        "validation":       [],
        "is_desert_query":  True,
        "state_summary":    state_df.to_dict("records"),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

@mlflow.trace(name="query_healthcare")
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
    # Route desert / gap queries to the desert analysis path
    if spark is not None and _is_desert_query(query):
        return query_desert(query)

    chain_of_thought = []

    with mlflow.start_run(run_name=f"query_{query[:30]}"):
        mlflow.log_param("query", query)

        # Step 1 — Parse
        t0 = time.time()
        parsed = parse_query(query)
        mlflow.log_dict(parsed, "parsed_query.json")
        chain_of_thought.append({
            "step":      1,
            "action":    "Query parsed",
            "detail":    f"State: {parsed.get('state') or 'any'} | "
                         f"ICU: {parsed.get('requires_icu')} | "
                         f"Emergency: {parsed.get('requires_emergency')} | "
                         f"Search: '{parsed.get('search_text')}'",
            "timing_ms": round((time.time() - t0) * 1000),
            "metadata":  {
                "state":               parsed.get("state") or "any",
                "specialties":         parsed.get("specialties") or [],
                "requires_icu":        parsed.get("requires_icu"),
                "requires_emergency":  parsed.get("requires_emergency"),
                "requires_24_7":       parsed.get("requires_24_7"),
                "search_text":         parsed.get("search_text"),
            }
        })

        # Step 2 — Search
        t0 = time.time()
        candidates = hybrid_search(parsed, num_results=num_results * 2)
        mlflow.log_metric("candidates_found", len(candidates))
        chain_of_thought.append({
            "step":      2,
            "action":    "Hybrid search complete",
            "detail":    f"SQL filters applied across 9,253 enriched records — "
                         f"{len(candidates)} facilities matched",
            "timing_ms": round((time.time() - t0) * 1000),
            "metadata":  {
                "candidates_found": len(candidates),
                "filters": {k: parsed.get(k) for k in
                    ("state", "city", "facility_type", "requires_icu", "requires_emergency", "requires_dialysis")
                    if parsed.get(k)},
            }
        })

        if not candidates:
            return {"results": [], "chain_of_thought": chain_of_thought,
                    "parsed_query": parsed, "total_found": 0, "validation": []}

        # Step 3 — Format
        t0 = time.time()
        results = format_results(candidates[:num_results], query)
        mlflow.log_metric("results_returned", len(results))
        if results:
            chain_of_thought.append({
                "step":      3,
                "action":    "Results ranked by trust score",
                "detail":    f"Top result: '{results[0]['name']}' "
                             f"(trust: {results[0]['trust_score']}, "
                             f"confidence: {results[0]['confidence']})",
                "timing_ms": round((time.time() - t0) * 1000),
                "metadata":  {
                    "results_count": len(results),
                    "top_result":    results[0]["name"],
                    "top_trust":     results[0]["trust_score"],
                    "top_confidence":results[0]["confidence"],
                    "top_location":  f"{results[0].get('city', '')}, {results[0].get('state', '')}",
                }
            })

        # Step 4 — Validate
        t0 = time.time()
        validations    = validate_top_results(results[:5])
        verdicts       = [v["overall_verdict"] for v in validations]
        verified_count = verdicts.count("VERIFIED")
        disputed_count = verdicts.count("DISPUTED")
        mlflow.log_metric("results_validated", len(validations))
        chain_of_thought.append({
            "step":      4,
            "action":    "Validator agent complete",
            "detail":    f"Checked top {len(validations)} against source text — "
                         f"{verified_count} VERIFIED, {disputed_count} DISPUTED",
            "timing_ms": round((time.time() - t0) * 1000),
            "metadata":  {
                "verdicts": {v["facility_name"]: v["overall_verdict"] for v in validations},
                "verified": verified_count,
                "disputed": disputed_count,
                "insufficient": verdicts.count("INSUFFICIENT_TEXT"),
            }
        })

        # Step 5 — Self-correction
        t0 = time.time()
        validation_map  = {v["facility_name"]: v for v in validations}
        corrected_count = 0
        corrected_names = []

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
                if corrected.get("self_corrected"):
                    corrected_names.append(r["name"])
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
            "step":      5,
            "action":    "Self-correction loop complete",
            "detail":    f"{corrected_count} result(s) re-extracted with validator feedback. "
                         f"Medical standards check applied to all results.",
            "timing_ms": round((time.time() - t0) * 1000),
            "metadata":  {
                "corrected_count": corrected_count,
                "corrected":       corrected_names,
                "standards_checked": len(results),
            }
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
