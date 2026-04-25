"""
CareMap — Agentic Healthcare Intelligence System
Data Extraction Pipeline: Bronze → Silver → Gold

Run this in a Databricks notebook (Free Edition).
Processes VF_Hackathon_Dataset_India_Large.xlsx → Delta Lake → Gold enriched table.

Requirements:
    pip install openai pydantic pandas openpyxl mlflow
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────
import json, re, ast, math, time
from typing import Optional, List
from pathlib import Path

import pandas as pd
from openai import OpenAI
import mlflow
import mlflow.pyfunc
from pydantic import BaseModel, Field
# import tomllib

# with open("secrets.toml", "rb") as f:
#     secrets = tomllib.load(f)

# OPENAI_API_KEY = secrets["OPENAI_API_KEY"]

DATASET_PATH = "VF_Hackathon_Dataset_India_Large.csv"  # same folder
BATCH_SIZE   = 25     # records per LLM batch (keep low to avoid timeouts)
MAX_RECORDS  = 100   # set to e.g. 100 for testing; None = all 10k

client = OpenAI(api_key="")

# ─────────────────────────────────────────────────────────────
# 1. PYDANTIC SCHEMA  (Virtue Foundation Schema)
# ─────────────────────────────────────────────────────────────
class ExtractedCapability(BaseModel):
    has_icu:              Optional[bool]   = None
    icu_beds:             Optional[int]    = None
    has_emergency:        Optional[bool]   = None
    has_24_7:             Optional[bool]   = None
    has_surgery:          Optional[bool]   = None
    has_dialysis:         Optional[bool]   = None
    has_oncology:         Optional[bool]   = None
    has_neonatal_icu:     Optional[bool]   = None
    has_mental_health:    Optional[bool]   = None
    has_trauma_centre:    Optional[bool]   = None
    inferred_doctors:     Optional[int]    = None
    extracted_equipment:  List[str]        = Field(default_factory=list)
    extracted_procedures: List[str]        = Field(default_factory=list)
    operational_notes:    Optional[str]    = None
    confidence:           str              = "low"   # high / medium / low / insufficient_data

class TrustScore(BaseModel):
    total:                float   = 0.0          # 0–100
    completeness:         float   = 0.0          # 0–25
    consistency:          float   = 0.0          # 0–30
    verifiability:        float   = 0.0          # 0–25
    recency:              float   = 0.0          # 0–10
    digital_presence:     float   = 0.0          # 0–10
    contradictions:       List[str] = Field(default_factory=list)
    source_sentence:      Optional[str] = None   # sentence that most affected score

class EnrichedFacility(BaseModel):
    facility_id:          str
    name:                 Optional[str]
    address_city:         Optional[str]
    address_state:        Optional[str]
    address_zip:          Optional[str]
    latitude:             Optional[float]
    longitude:            Optional[float]
    facility_type:        Optional[str]
    specialties:          List[str]      = Field(default_factory=list)
    extracted:            ExtractedCapability
    trust:                TrustScore
    searchable_text:      str = ""       # concat for vector embedding


# ─────────────────────────────────────────────────────────────
# 2. BRONZE LAYER — load raw Excel
# ─────────────────────────────────────────────────────────────
def load_bronze(path: str, max_rows=None) -> pd.DataFrame:
    print(f"[Bronze] Loading {path} …")
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    df["facility_id"] = df.index.astype(str).str.zfill(5)
    print(f"[Bronze] Loaded {len(df):,} records, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────
# 3. SILVER LAYER — clean & standardise
# ─────────────────────────────────────────────────────────────
def _parse_json_list(val) -> List[str]:
    """Safely parse JSON-stringified arrays like '["oncology","cardiology"]'."""
    if pd.isna(val) or val in ("null", "[]", "", None):
        return []
    if isinstance(val, list):
        return [str(v) for v in val]
    try:
        parsed = json.loads(str(val))
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except Exception:
        try:
            parsed = ast.literal_eval(str(val))
            return [str(v) for v in parsed] if isinstance(parsed, list) else []
        except Exception:
            return [str(val)]

def _completeness_score(row) -> float:
    """0–1: fraction of key fields that are non-null."""
    key_fields = ["description", "specialties", "equipment", "procedure", "capability",
                  "numberDoctors", "capacity", "facilityTypeId"]
    filled = sum(1 for f in key_fields
                 if f in row and not pd.isna(row[f]) and str(row[f]) not in ("null","[]","","None"))
    return round(filled / len(key_fields), 3)

def clean_silver(df: pd.DataFrame) -> pd.DataFrame:
    print("[Silver] Cleaning and standardising …")
    df = df.copy()

    # parse JSON arrays
    for col in ["specialties", "procedure", "equipment", "capability"]:
        if col in df.columns:
            df[col + "_parsed"] = df[col].apply(_parse_json_list)
        else:
            df[col + "_parsed"] = [[] for _ in range(len(df))]

    # normalise casing
    for col in ["facilityTypeId", "operatorTypeId"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # numeric coercion
    for col in ["numberDoctors", "capacity", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # completeness score
    df["data_completeness"] = df.apply(_completeness_score, axis=1)
    df["has_text"]          = df["description"].notna() | (df["capability"].notna())

    print(f"[Silver] Done. Mean completeness: {df['data_completeness'].mean():.2f}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. EXTRACTION AGENT — LLM extracts structured capabilities
# ─────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a medical data extraction agent. Extract structured capabilities from this Indian hospital/clinic record.

Record:
  Name:        {name}
  Description: {description}
  Capability:  {capability}
  Procedures:  {procedures}
  Equipment:   {equipment}
  Specialties: {specialties}
  Facility type: {facility_type}
  Doctors listed: {num_doctors}

Extract ONLY what is explicitly stated. Never infer. If unsure, use null.

Return ONLY valid JSON with these exact keys:
{{
  "has_icu": true/false/null,
  "icu_beds": integer or null,
  "has_emergency": true/false/null,
  "has_24_7": true/false/null,
  "has_surgery": true/false/null,
  "has_dialysis": true/false/null,
  "has_oncology": true/false/null,
  "has_neonatal_icu": true/false/null,
  "has_mental_health": true/false/null,
  "has_trauma_centre": true/false/null,
  "inferred_doctors": integer or null,
  "extracted_equipment": ["list","of","equipment"],
  "extracted_procedures": ["list","of","procedures"],
  "operational_notes": "any notes about 24/7, part-time, closing hours",
  "confidence": "high" or "medium" or "low" or "insufficient_data",
  "source_sentence": "the exact sentence that best justified your extraction"
}}

confidence guide:
  high = multiple fields confirm the capability with specifics
  medium = one field mentions it with some detail
  low = mentioned in passing, no detail
  insufficient_data = description/capability fields are empty"""


def extract_one(row: pd.Series) -> dict:
    """Call Claude to extract structured capabilities from one facility record."""
    prompt = EXTRACTION_PROMPT.format(
        name         = row.get("name", "Unknown"),
        description  = row.get("description", "null") or "null",
        capability   = row.get("capability", "null") or "null",
        procedures   = row.get("procedure_parsed", []),
        equipment    = row.get("equipment_parsed", []),
        specialties  = row.get("specialties_parsed", []),
        facility_type= row.get("facilityTypeId", "unknown"),
        num_doctors  = row.get("numberDoctors", "null"),
    )
    try:
        response = client.chat.completions.create(
            model      = "gpt-4o-mini",   # cheap + fast; swap to gpt-4o for better accuracy
            max_tokens = 600,
            temperature= 0,               # deterministic outputs
            messages   = [
                {"role": "system", "content": "You are a medical data extraction agent. Return only valid JSON."},
                {"role": "user",   "content": prompt}
            ]
        )
        raw  = response.choices[0].message.content.replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        data["_llm_input_tokens"]  = response.usage.prompt_tokens
        data["_llm_output_tokens"] = response.usage.completion_tokens
        return data
    except Exception as e:
        return {"confidence": "insufficient_data", "_error": str(e)}


# ─────────────────────────────────────────────────────────────
# 5. TRUST SCORER
# ─────────────────────────────────────────────────────────────
def compute_trust(row: pd.Series, extracted: dict) -> dict:
    """
    Compute Trust Score (0–100) across 5 dimensions.
    """
    contradictions = []
    score_completeness  = 0.0
    score_consistency   = 0.0
    score_verifiability = 0.0
    score_recency       = 0.0
    score_digital       = 0.0

    # ── Completeness (25 pts) ──
    comp = row.get("data_completeness", 0)
    score_completeness = round(comp * 25, 1)

    # ── Recency (10 pts) ──
    recency = row.get("recency_of_page_update", None)
    if pd.notna(recency) and str(recency) not in ("null",""):
        score_recency = 8.0  # present = good
    else:
        score_recency = 2.0

    # ── Digital presence (10 pts) ──
    social = row.get("distinct_social_media_presence_count", 0) or 0
    score_digital = min(10.0, float(social) * 2.5)

    # ── Verifiability (25 pts) ──
    equip  = extracted.get("extracted_equipment",  [])
    procs  = extracted.get("extracted_procedures", [])
    specifics = len(equip) + len(procs)
    score_verifiability = min(25.0, specifics * 3.0)

    # ── Consistency (30 pts) — start full, deduct for contradictions ──
    score_consistency = 30.0

    ftype   = str(row.get("facilityTypeId", "")).lower()
    n_docs  = row.get("numberDoctors", None)
    descr   = str(row.get("description","") or "").lower()
    cap_raw = str(row.get("capability","") or "").lower()
    combined_text = descr + " " + cap_raw

    # Rule 1: clinic claims >50 beds
    beds = extracted.get("icu_beds") or 0
    bed_mentions = re.findall(r'(\d+)\s*bed', combined_text)
    max_beds = max([int(b) for b in bed_mentions], default=0)
    if ftype == "clinic" and max_beds > 50:
        score_consistency -= 15
        contradictions.append(f"Facility type=clinic but claims {max_beds} beds")

    # Rule 2: claims surgery but no anaesthesiologist
    has_surgery = extracted.get("has_surgery")
    if has_surgery:
        anaesth_mentioned = any(w in combined_text for w in
            ["anaesthesiologist","anesthesiologist","anaesthetist","anesthesia"])
        if not anaesth_mentioned:
            score_consistency -= 12
            contradictions.append("Claims surgery capability but no anaesthesiologist mentioned")

    # Rule 3: claims 24/7 but text hints at limited hours
    has_247 = extracted.get("has_24_7")
    close_words = ["closes", "closed at", "10pm","8pm","9pm","hours only","outpatient only"]
    if has_247 and any(w in combined_text for w in close_words):
        score_consistency -= 10
        contradictions.append("Claims 24/7 but text suggests limited operating hours")

    # Rule 4: claims ICU but no supporting equipment
    has_icu = extracted.get("has_icu")
    icu_equip_words = ["ventilator","icu","intensive care","oxygen","monitor"]
    if has_icu and not any(w in combined_text for w in icu_equip_words) and not equip:
        score_consistency -= 10
        contradictions.append("Claims ICU but no supporting equipment or text evidence found")

    # Rule 5: single doctor but claims team of specialists
    if pd.notna(n_docs) and n_docs == 1:
        if any(w in combined_text for w in ["team of","50 specialists","specialists available"]):
            score_consistency -= 15
            contradictions.append(f"numberDoctors=1 but claims team of specialists")

    score_consistency = max(0.0, score_consistency)

    total = score_completeness + score_consistency + score_verifiability + score_recency + score_digital

    return {
        "trust_total":        round(min(100.0, total), 1),
        "trust_completeness": score_completeness,
        "trust_consistency":  score_consistency,
        "trust_verifiability":score_verifiability,
        "trust_recency":      score_recency,
        "trust_digital":      score_digital,
        "contradictions":     contradictions,
        "source_sentence":    extracted.get("source_sentence",""),
    }


# ─────────────────────────────────────────────────────────────
# 6. SEARCHABLE TEXT — concat for vector embedding
# ─────────────────────────────────────────────────────────────
def build_searchable_text(row: pd.Series, extracted: dict) -> str:
    parts = [
        str(row.get("name","") or ""),
        str(row.get("description","") or ""),
        str(row.get("capability","") or ""),
        " ".join(row.get("specialties_parsed", []) or []),
        " ".join(row.get("procedure_parsed", []) or []),
        " ".join(row.get("equipment_parsed", []) or []),
        str(row.get("address_city","") or ""),
        str(row.get("address_stateOrRegion","") or ""),
        " ".join(extracted.get("extracted_equipment", []) or []),
        " ".join(extracted.get("extracted_procedures", []) or []),
    ]
    return " ".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────────────────────
# 7. GOLD PIPELINE — run extraction + scoring on all records
# ─────────────────────────────────────────────────────────────
def run_gold_pipeline(silver_df: pd.DataFrame) -> pd.DataFrame:
    print(f"[Gold] Starting extraction pipeline on {len(silver_df):,} records …")

    results   = []
    failures  = []
    total     = len(silver_df)
    start_all = time.time()

    # filter only records that have some text to extract from
    has_text_mask = silver_df["has_text"].fillna(False)
    df_text    = silver_df[has_text_mask].copy()
    df_no_text = silver_df[~has_text_mask].copy()
    print(f"[Gold] {len(df_text):,} records have text · {len(df_no_text):,} have no text (trust score only)")

    with mlflow.start_run(run_name="gold_extraction_pipeline"):
        mlflow.log_param("total_records", total)
        mlflow.log_param("records_with_text", len(df_text))
        mlflow.log_param("batch_size", BATCH_SIZE)

        total_tokens = 0

        # ── Process records WITH text ──
        for batch_start in range(0, len(df_text), BATCH_SIZE):
            batch = df_text.iloc[batch_start : batch_start + BATCH_SIZE]
            batch_tokens = 0

            with mlflow.start_span(name=f"batch_{batch_start//BATCH_SIZE}") as span:
                span.set_inputs({"batch_start": batch_start, "batch_size": len(batch)})
                batch_results = []

                for idx, (_, row) in enumerate(batch.iterrows()):
                    fid = row.get("facility_id", str(batch_start + idx))
                    try:
                        extracted = extract_one(row)
                        trust     = compute_trust(row, extracted)
                        s_text    = build_searchable_text(row, extracted)

                        batch_tokens += (
                            extracted.pop("_llm_input_tokens",  0) +
                            extracted.pop("_llm_output_tokens", 0)
                        )
                        err = extracted.pop("_error", None)

                        gold_row = {
                            "facility_id":            fid,
                            "name":                   row.get("name"),
                            "address_city":           row.get("address_city"),
                            "address_state":          row.get("address_stateOrRegion"),
                            "address_zip":            row.get("address_zipOrPostcode"),
                            "latitude":               row.get("latitude"),
                            "longitude":              row.get("longitude"),
                            "facility_type":          row.get("facilityTypeId"),
                            "specialties":            row.get("specialties_parsed", []),
                            "data_completeness":      row.get("data_completeness", 0),
                            "searchable_text":        s_text,
                            **{f"ext_{k}": v for k,v in extracted.items()},
                            **{f"trust_{k}": v for k,v in trust.items()
                               if k not in ("trust_total",)},
                            "trust_score":            trust["trust_total"],
                            "contradictions":         trust["contradictions"],
                            "source_sentence":        trust["source_sentence"],
                            "extraction_error":       err,
                        }
                        batch_results.append(gold_row)

                    except Exception as e:
                        failures.append({"facility_id": fid, "error": str(e)})

                results.extend(batch_results)
                total_tokens += batch_tokens
                span.set_outputs({"processed": len(batch_results), "tokens": batch_tokens})

            pct = min(100, int((batch_start + len(batch)) / len(df_text) * 100))
            elapsed = time.time() - start_all
            print(f"  [{pct:3d}%] Batch {batch_start//BATCH_SIZE+1} done · "
                  f"{len(results):,} enriched · {len(failures)} failed · "
                  f"{elapsed:.0f}s elapsed")

        # ── Process records WITHOUT text (trust score only) ──
        for _, row in df_no_text.iterrows():
            fid = row.get("facility_id", "?")
            trust = compute_trust(row, {})
            results.append({
                "facility_id":   fid,
                "name":          row.get("name"),
                "address_city":  row.get("address_city"),
                "address_state": row.get("address_stateOrRegion"),
                "address_zip":   row.get("address_zipOrPostcode"),
                "latitude":      row.get("latitude"),
                "longitude":     row.get("longitude"),
                "facility_type": row.get("facilityTypeId"),
                "specialties":   row.get("specialties_parsed", []),
                "trust_score":   trust["trust_total"],
                "contradictions":trust["contradictions"],
                "ext_confidence":"insufficient_data",
                "searchable_text": build_searchable_text(row, {}),
            })

        total_time = time.time() - start_all
        mlflow.log_metrics({
            "total_enriched":  len(results),
            "total_failures":  len(failures),
            "total_tokens":    total_tokens,
            "pipeline_seconds":total_time,
        })
        print(f"\n[Gold] ✅ Done in {total_time:.0f}s · "
              f"{len(results):,} enriched · {len(failures)} failures · "
              f"{total_tokens:,} tokens used")

    gold_df = pd.DataFrame(results)
    return gold_df, pd.DataFrame(failures) if failures else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 8. MEDICAL DESERT ANALYSIS
# ─────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in km between two lat/lon points."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

CRITICAL_SPECIALTIES = {
    "oncology":       ["oncology","cancer","chemotherapy","radiotherapy"],
    "dialysis":       ["dialysis","nephrology","kidney","renal"],
    "neonatal_icu":   ["neonatal","nicu","newborn icu","paediatric icu"],
    "emergency_trauma":["emergency","trauma","casualty","accident"],
    "mental_health":  ["psychiatry","mental health","psychology","behavioural"],
    "obstetrics":     ["obstetrics","gynaecology","maternity","delivery"],
}

def compute_desert_analysis(gold_df: pd.DataFrame) -> pd.DataFrame:
    print("[Desert] Computing medical desert analysis …")
    deserts = []

    for state, grp in gold_df.groupby("address_state"):
        if pd.isna(state): continue

        for specialty, keywords in CRITICAL_SPECIALTIES.items():
            # facilities in this state with this specialty (high/medium confidence)
            mask = grp["searchable_text"].str.lower().apply(
                lambda t: any(kw in str(t) for kw in keywords)
            ) & grp["ext_confidence"].isin(["high","medium"]) if "ext_confidence" in grp.columns else \
            grp["searchable_text"].str.lower().apply(
                lambda t: any(kw in str(t) for kw in keywords)
            )

            count = mask.sum()
            pop_state = len(grp)  # proxy: total facilities as denominator

            # severity: 0 = well served, 100 = critical desert
            if count == 0:
                severity = 95
            elif count / max(pop_state, 1) < 0.02:
                severity = 75
            elif count / max(pop_state, 1) < 0.05:
                severity = 45
            else:
                severity = 20

            deserts.append({
                "state":                  state,
                "specialty":              specialty,
                "facility_count":         int(count),
                "total_facilities_state": int(pop_state),
                "coverage_ratio":         round(count / max(pop_state,1), 4),
                "desert_severity":        severity,
                "action":                 f"{'CRITICAL: ' if severity>80 else ''}"
                                          f"{state} has only {count} verified {specialty} "
                                          f"facilities out of {pop_state} total."
            })

    desert_df = pd.DataFrame(deserts).sort_values("desert_severity", ascending=False)
    print(f"[Desert] Found {len(desert_df[desert_df['desert_severity']>80]):,} critical deserts")
    return desert_df


# ─────────────────────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
def save_outputs(gold_df: pd.DataFrame, desert_df: pd.DataFrame,
                 failures_df: pd.DataFrame, output_dir: str = "."):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gold_path    = Path(output_dir) / "gold_facilities_enriched.csv"
    desert_path  = Path(output_dir) / "desert_analysis.csv"
    failure_path = Path(output_dir) / "extraction_failures.csv"
    geojson_path = Path(output_dir) / "facilities_map.geojson"

    gold_df.to_csv(gold_path, index=False)
    desert_df.to_csv(desert_path, index=False)
    if not failures_df.empty:
        failures_df.to_csv(failure_path, index=False)

    # GeoJSON for map
    features = []
    for _, row in gold_df.dropna(subset=["latitude","longitude"]).iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
            "properties": {
                "id":           row.get("facility_id"),
                "name":         row.get("name"),
                "city":         row.get("address_city"),
                "state":        row.get("address_state"),
                "type":         row.get("facility_type"),
                "trust_score":  row.get("trust_score", 0),
                "specialties":  row.get("specialties", []),
                "has_icu":      row.get("ext_has_icu"),
                "has_emergency":row.get("ext_has_emergency"),
            }
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(geojson_path, "w") as f:
        json.dump(geojson, f)

    print(f"\n[Output] Files saved:")
    print(f"  Gold table   → {gold_path}")
    print(f"  Desert report→ {desert_path}")
    print(f"  GeoJSON map  → {geojson_path}")
    if not failures_df.empty:
        print(f"  Failures     → {failure_path}")


# ─────────────────────────────────────────────────────────────
# 10. MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CareMap — Healthcare Intelligence Extraction Pipeline")
    print("=" * 60)

    # ── Step 1: Bronze ──
    bronze_df = load_bronze(DATASET_PATH, max_rows=MAX_RECORDS)

    # ── Step 2: Silver ──
    silver_df = clean_silver(bronze_df)

    # ── Step 3+4+5: Gold (extraction + trust + searchable text) ──
    gold_df, failures_df = run_gold_pipeline(silver_df)

    # ── Step 6: Desert analysis ──
    desert_df = compute_desert_analysis(gold_df)

    # ── Step 7: Save ──
    save_outputs(gold_df, desert_df, failures_df, output_dir="caremap_output")

    # ── Quick summary ──
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total facilities enriched : {len(gold_df):,}")
    print(f"  Avg trust score           : {gold_df['trust_score'].mean():.1f}/100")
    print(f"  High trust (≥70)          : {(gold_df['trust_score']>=70).sum():,}")
    print(f"  Low trust  (<30)          : {(gold_df['trust_score']<30).sum():,}")
    print(f"  Critical deserts found    : {(desert_df['desert_severity']>80).sum():,}")
    print(f"  Extraction failures       : {len(failures_df):,}")
    print("=" * 60)
    print("  ✅ Ready to load into FastAPI / Mosaic AI Vector Search")