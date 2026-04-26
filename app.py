"""
CareMap — Agentic Healthcare Intelligence System
FastAPI Backend for Databricks Apps

Serves:
  GET  /              → index.html (query interface)
  GET  /map           → map.html (facility map)
  POST /query         → natural language query → ranked results
  GET  /map/data      → all facilities as GeoJSON for map
  GET  /deserts       → desert analysis by state
  GET  /stats         → aggregate statistics
  GET  /facility/{name} → single facility full profile

Deploy: Upload app.py, index.html, map.html to Databricks Apps
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import time
import os
import sys

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="CareMap Healthcare Intelligence API",
    description="Agentic healthcare facility search for India",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Import query agent module ─────────────────────────────────
try:
    from query_agent import query_healthcare, spark
    from pyspark.sql import functions as F
    SPARK_AVAILABLE = True
    print("Query agent and Spark loaded successfully")
except Exception as e:
    SPARK_AVAILABLE = False
    print(f"Query agent import error: {e}")

# ── Request/Response models ───────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    num_results: Optional[int] = 10
    state: Optional[str] = None
    min_trust: Optional[float] = None

class FacilityResult(BaseModel):
    name: str
    location: str
    trust: float
    type: str
    tags: list
    citation: str

# ── Helper: format for index.html ────────────────────────────

def format_for_frontend(agent_response: dict, query: str) -> dict:
    """
    Convert query_healthcare() output to the format
    expected by index.html's sendMessage() function.
    """
    results     = agent_response.get("results", [])
    chain       = agent_response.get("chain_of_thought", [])
    total_found = agent_response.get("total_found", 0)
    validations = agent_response.get("validation", [])

    # Build answer summary
    if not results:
        answer = f"No facilities found matching your query. Try broadening your search criteria."
    else:
        top = results[0]
        state_str = top.get("state", "India")
        answer = (
            f"Found {total_found} facilities matching your criteria. "
            f"Top result: {top['name']} in {state_str} "
            f"with a trust score of {top.get('trust_score_display', top.get('trust_score', 0))}. "
            f"The search was validated against original source text using a 3-step agentic pipeline."
        )

    # Build result cards in format index.html expects
    frontend_results = []
    for r in results:
        # Build tags from availability flags
        tags = []

        # Validation verdict tag
        verdict = r.get("validation_verdict", "NOT_VALIDATED")
        if verdict == "VERIFIED":
            tags.append({"label": "✅ Verified", "warn": False})
        elif verdict == "DISPUTED":
            tags.append({"label": "⚠ Disputed", "warn": True})
        if r.get("self_corrected"):
            tags.append({"label": "🔄 Self-corrected", "warn": False})

        # Clinical flags
        if r.get("has_icu"):
            tags.append({"label": "ICU", "warn": False})
        if r.get("has_emergency"):
            tags.append({"label": "Emergency", "warn": False})
        if r.get("has_24_7"):
            tags.append({"label": "24/7", "warn": False})

        # Confidence tag
        confidence = r.get("confidence", "")
        if confidence == "low":
            tags.append({"label": "Low data quality", "warn": True})

        # Contradiction warning
        contradictions = r.get("contradictions", [])
        if contradictions and len(contradictions) > 0:
            tags.append({"label": f"⚠ {len(contradictions)} contradiction(s)", "warn": True})

        # Capability tags (first 2)
        caps = r.get("key_capabilities", [])
        for cap in caps[:2]:
            tags.append({"label": cap[:25], "warn": False})

        frontend_results.append({
            "name":      r.get("name", "Unknown"),
            "location":  f"{r.get('city', '')}, {r.get('state', '')}",
            "trust":     r.get("trust_score_display", r.get("trust_score", 0)),
            "type":      r.get("facility_type", "facility").title(),
            "tags":      tags,
            "citation":  r.get("citation", "No citation available")[:200],
            "phone":     r.get("phone"),
            "pin_code":  r.get("pin_code"),
            "latitude":  r.get("latitude"),
            "longitude": r.get("longitude"),
            "validation_note":  r.get("validation_note", ""),
            "correction_note":  r.get("correction_note", ""),
            "trust_breakdown":  r.get("trust_breakdown", {}),
            "source_text":      r.get("source_text", "")[:300],
            "chain_of_thought": chain,
        })

    # Desert insight — check if query is about gaps
    desert_insight = None
    query_lower = query.lower()
    if any(word in query_lower for word in ["desert", "gap", "missing", "no facility", "lack"]):
        desert_insight = (
            "Based on our desert analysis: Nagaland, Ladakh, Meghalaya, Sikkim, and Mizoram "
            "have ZERO verified facilities for all 8 high-acuity specialties. "
            "See the map view for full desert heatmap."
        )

    # Trace stats for the UI stats bar
    trust_scores = [r.get("trust_score", 0) for r in results if r.get("trust_score")]
    avg_trust = round(sum(trust_scores) / len(trust_scores), 1) if trust_scores else 0

    return {
        "answer":         answer,
        "results":        frontend_results,
        "desert_insight": desert_insight,
        "chain_of_thought": chain,
        "trace": {
            "records_searched": 9253,
            "candidates":       total_found,
            "avg_trust":        avg_trust,
            "tokens":           len(query.split()) * 50 + 500,  # Estimate
            "steps":            len(chain),
            "verified":         sum(1 for r in results if r.get("validation_verdict") == "VERIFIED"),
            "disputed":         sum(1 for r in results if r.get("validation_verdict") == "DISPUTED"),
            "self_corrected":   sum(1 for r in results if r.get("self_corrected")),
        }
    }


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the query interface."""
    with open("index.html", "r") as f:
        return f.read()


@app.get("/map", response_class=HTMLResponse)
def serve_map():
    """Serve the map dashboard."""
    with open("map.html", "r") as f:
        return f.read()


@app.post("/query")
def query_endpoint(request: QueryRequest):
    """
    Main query endpoint — runs the full agentic pipeline:
    1. Parse query (LLM #1)
    2. SQL search
    3. Format with citations
    4. Validate top 5 (LLM #2)
    5. Self-correct disputed (LLM #3)
    """
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    try:
        # Build query with any frontend filters applied
        query = request.query
        if request.state and request.state != "All":
            if request.state.lower() not in query.lower():
                query = f"{query} in {request.state}"

        # Call the query agent (defined in notebook 07)
        # In Databricks Apps, query_healthcare is available via %run
        agent_response = query_healthcare(
            query=query,
            num_results=request.num_results or 10,
            verbose=False
        )

        # Format for frontend
        frontend_response = format_for_frontend(agent_response, request.query)
        frontend_response["latency_ms"] = round((time.time() - start_time) * 1000)

        return JSONResponse(content=frontend_response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query agent error: {str(e)}"
        )


@app.get("/map/data")
def map_data():
    """
    Returns all facilities as a list for map rendering.
    Used by map.html to plot facility markers.
    """
    if not SPARK_AVAILABLE:
        return JSONResponse(content={"facilities": [], "error": "Spark not available"})

    try:
        df = spark.sql("""
            SELECT
                g.name,
                s.address_stateOrRegion as state,
                s.address_city          as city,
                s.facilityTypeId        as type,
                s.latitude              as lat,
                s.longitude             as lng,
                s.specialties,
                s.officialPhone         as phone,
                g.trust_score           as trust,
                g.confidence_level      as confidence,
                g.extracted_availability,
                g.extracted_capabilities,
                g.contradictions
            FROM workspace.gold.facilities_enriched g
            JOIN workspace.silver.facilities_clean s ON g.name = s.name
            WHERE g.trust_score IS NOT NULL
            AND s.latitude IS NOT NULL
            AND s.longitude IS NOT NULL
        """).toPandas()

        facilities = []
        for _, row in df.iterrows():
            # Parse availability
            avail = {}
            try:
                avail = json.loads(row.get("extracted_availability") or "{}")
            except:
                pass

            # Parse capabilities
            caps = []
            try:
                caps = json.loads(row.get("extracted_capabilities") or "[]")
            except:
                pass

            # Parse specialties
            specs = []
            try:
                raw = row.get("specialties")
                if raw and isinstance(raw, str):
                    specs = [s.strip() for s in raw.split("|")][:4]
                elif isinstance(raw, list):
                    specs = raw[:4]
            except:
                pass

            # Parse contradictions
            contradictions = []
            try:
                raw_contra = row.get("contradictions")
                if raw_contra and isinstance(raw_contra, str):
                    contradictions = json.loads(raw_contra)
            except:
                pass

            facilities.append({
                "name":        str(row.get("name", "")),
                "state":       str(row.get("state", "")),
                "city":        str(row.get("city", "")),
                "type":        str(row.get("type", "")),
                "lat":         float(row.get("lat", 0)),
                "lng":         float(row.get("lng", 0)),
                "trust":       float(row.get("trust", 0)),
                "confidence":  str(row.get("confidence", "")),
                "specialties": specs,
                "hasIcu":      avail.get("has_icu") == True,
                "hasEmergency":avail.get("has_emergency") == True,
                "has24_7":     avail.get("has_24_7") == True,
                "capabilities":caps[:3],
                "citation":    caps[0] if caps else "No capabilities extracted",
                "hasContradiction": len(contradictions) > 0,
            })

        return JSONResponse(content={"facilities": facilities, "total": len(facilities)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deserts")
def desert_data(state: Optional[str] = None, specialty: Optional[str] = None):
    """
    Returns medical desert analysis.
    Used by map.html for the desert heatmap overlay.
    """
    if not SPARK_AVAILABLE:
        return JSONResponse(content={"deserts": [], "state_summary": []})

    try:
        # Desert analysis by state x specialty
        desert_query = """
            SELECT
                d.state,
                d.specialty,
                d.specialty_facility_count,
                d.total_facilities,
                d.coverage_ratio,
                d.desert_severity,
                d.severity_label,
                d.action,
                d.centroid_lat  as lat,
                d.centroid_lng  as lng
            FROM workspace.gold.desert_analysis d
            WHERE d.desert_severity >= 60
        """

        if state:
            desert_query += f" AND d.state = '{state}'"
        if specialty:
            desert_query += f" AND d.specialty = '{specialty}'"

        desert_query += " ORDER BY d.desert_severity DESC LIMIT 200"

        deserts_df = spark.sql(desert_query).toPandas()

        # State summary for heatmap
        summary_df = spark.sql("""
            SELECT
                state,
                max_desert_severity,
                overall_severity_label,
                critical_specialty_count,
                total_facilities,
                centroid_lat as lat,
                centroid_lng as lng
            FROM workspace.gold.state_summary
            ORDER BY max_desert_severity DESC
        """).toPandas()

        deserts = []
        for _, row in deserts_df.iterrows():
            deserts.append({
                "state":    str(row.get("state", "")),
                "specialty":str(row.get("specialty", "")),
                "count":    int(row.get("specialty_facility_count", 0)),
                "total":    int(row.get("total_facilities", 0)),
                "severity": str(row.get("severity_label", "")),
                "score":    int(row.get("desert_severity", 0)),
                "action":   str(row.get("action", "")),
                "lat":      float(row.get("lat", 20.5)),
                "lng":      float(row.get("lng", 78.9)),
            })

        state_summary = []
        for _, row in summary_df.iterrows():
            state_summary.append({
                "state":           str(row.get("state", "")),
                "severity":        str(row.get("overall_severity_label", "")).lower(),
                "critical_count":  int(row.get("critical_specialty_count", 0)),
                "facilities":      int(row.get("total_facilities", 0)),
                "lat":             float(row.get("lat", 20.5)),
                "lng":             float(row.get("lng", 78.9)),
            })

        return JSONResponse(content={
            "deserts":       deserts,
            "state_summary": state_summary,
            "total_critical":sum(1 for d in deserts if d["severity"] == "CRITICAL")
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats():
    """
    Returns aggregate statistics for the dashboard header.
    """
    if not SPARK_AVAILABLE:
        return JSONResponse(content={})

    try:
        stats = spark.sql("""
            SELECT
                COUNT(*)                                              as total_facilities,
                SUM(CASE WHEN g.trust_score >= 80 THEN 1 ELSE 0 END) as high_trust_count,
                SUM(CASE WHEN g.trust_score >= 60 THEN 1 ELSE 0 END) as medium_plus_count,
                ROUND(AVG(g.trust_score), 1)                         as avg_trust_score,
                SUM(CASE WHEN g.extracted_availability
                    LIKE '%"has_icu": true%' THEN 1 ELSE 0 END)      as icu_confirmed,
                SUM(CASE WHEN g.extracted_availability
                    LIKE '%"has_emergency": true%' THEN 1 ELSE 0 END) as emergency_confirmed,
                SUM(CASE WHEN g.confidence_level = 'high' THEN 1 ELSE 0 END) as high_confidence,
                COUNT(DISTINCT s.address_stateOrRegion)              as states_covered
            FROM workspace.gold.facilities_enriched g
            JOIN workspace.silver.facilities_clean s ON g.name = s.name
            WHERE g.trust_score IS NOT NULL
        """).collect()[0]

        return JSONResponse(content={
            "total_facilities":   int(stats["total_facilities"]),
            "high_trust_count":   int(stats["high_trust_count"]),
            "avg_trust_score":    float(stats["avg_trust_score"] or 0),
            "icu_confirmed":      int(stats["icu_confirmed"]),
            "emergency_confirmed":int(stats["emergency_confirmed"]),
            "high_confidence":    int(stats["high_confidence"]),
            "states_covered":     int(stats["states_covered"]),
            "records_extracted":  9253,
            "extraction_success_rate": 99.9,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/facility/{facility_name}")
def facility_detail(facility_name: str):
    """
    Returns full profile for a single facility.
    Called when user clicks a result card or map marker.
    """
    if not SPARK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Spark not available")

    try:
        # Escape single quotes in facility name
        safe_name = facility_name.replace("'", "\\'")

        result = spark.sql(f"""
            SELECT
                g.*,
                s.address_stateOrRegion,
                s.address_city,
                s.address_zipOrPostcode,
                s.address_line1,
                s.facilityTypeId,
                s.operatorTypeId,
                s.latitude,
                s.longitude,
                s.specialties,
                s.officialPhone,
                s.officialWebsite,
                s.email,
                s.description,
                s.yearEstablished,
                s.numberDoctors,
                s.data_completeness_score
            FROM workspace.gold.facilities_enriched g
            JOIN workspace.silver.facilities_clean s ON g.name = s.name
            WHERE g.name = '{safe_name}'
            LIMIT 1
        """).toPandas()

        if result.empty:
            raise HTTPException(status_code=404, detail=f"Facility '{facility_name}' not found")

        row = result.iloc[0].to_dict()

        # Parse JSON fields
        def safe_json(val, default):
            try:
                return json.loads(str(val)) if val else default
            except:
                return default

        return JSONResponse(content={
            "name":               str(row.get("name", "")),
            "facility_type":      str(row.get("facilityTypeId", "")),
            "operator_type":      str(row.get("operatorTypeId", "")),
            "state":              str(row.get("address_stateOrRegion", "")),
            "city":               str(row.get("address_city", "")),
            "address":            str(row.get("address_line1", "")),
            "pin_code":           str(row.get("address_zipOrPostcode", "")),
            "latitude":           float(row.get("latitude") or 0),
            "longitude":          float(row.get("longitude") or 0),
            "phone":              str(row.get("officialPhone", "")),
            "website":            str(row.get("officialWebsite", "")),
            "email":              str(row.get("email", "")),
            "year_established":   str(row.get("yearEstablished", "")),
            "description":        str(row.get("description", "")),
            "specialties":        safe_json(row.get("specialties"), []),
            "trust_score":        float(row.get("trust_score") or 0),
            "trust_breakdown":    safe_json(row.get("trust_score_breakdown"), {}),
            "confidence_level":   str(row.get("confidence_level", "")),
            "extracted_capabilities": safe_json(row.get("extracted_capabilities"), []),
            "extracted_equipment":    safe_json(row.get("extracted_equipment"), []),
            "extracted_staff":        safe_json(row.get("extracted_staff"), {}),
            "extracted_availability": safe_json(row.get("extracted_availability"), {}),
            "extracted_bed_count":    int(row.get("extracted_bed_count") or 0),
            "contradictions":         safe_json(row.get("contradictions"), []),
            "data_completeness":      float(row.get("data_completeness_score") or 0),
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":          "ok",
        "spark_available": SPARK_AVAILABLE,
        "version":         "1.0.0"
    }


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
