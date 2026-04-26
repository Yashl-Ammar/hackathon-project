# CareMap — Deployment Guide

## Files in this package

```
app.py           ← FastAPI backend (main entry point)
query_agent.py   ← All agent functions (query, validate, self-correct)
index.html       ← Query interface (wired to /query endpoint)
map.html         ← Map dashboard (loads real data from /map/data + /deserts)
requirements.txt ← Python dependencies
```

## Deploy on Databricks Apps

1. Go to Databricks workspace
2. Click Compute → Apps → Create App
3. Choose "Custom" app type
4. Upload all 5 files
5. Set entry point: `app.py`
6. Click Deploy

Databricks gives you a public URL automatically.

## Deploy on GitHub Pages (frontend only)

If Databricks Apps is not available on Free Edition:

1. Push index.html and map.html to your GitHub repo
2. Go to repo Settings → Pages → Deploy from branch
3. Add this line before </body> in both HTML files:
   <script>window.BACKEND_URL = 'YOUR_DATABRICKS_NOTEBOOK_URL'</script>

## API Endpoints

| Method | Endpoint        | Description                          |
|--------|----------------|--------------------------------------|
| GET    | /              | Query interface (index.html)         |
| GET    | /map           | Map dashboard (map.html)             |
| POST   | /query         | Natural language facility search     |
| GET    | /map/data      | All 9,253 facilities as JSON         |
| GET    | /deserts       | Medical desert analysis              |
| GET    | /stats         | Aggregate statistics                 |
| GET    | /facility/{name} | Single facility full profile       |
| GET    | /health        | Health check                         |

## Query API Example

POST /query
{
  "query": "Find hospitals in Kerala with ICU and emergency care",
  "num_results": 10
}

Response includes:
- results: ranked facilities with trust scores and citations
- chain_of_thought: 5-step reasoning trace
- validation: VERIFIED/DISPUTED/INSUFFICIENT_TEXT per result
- trace: stats for the UI bar

## Agent Pipeline (per query)

Step 1: Parse query          → LLM call #1
Step 2: SQL search           → Gold + Silver tables
Step 3: Format + cite        → Row-level citations from source text
Step 4: Batch validate       → LLM call #2 (top 5 results)
Step 5: Self-correct         → LLM call #3 (DISPUTED results only)
        + Medical standards check (rule-based, no LLM)
