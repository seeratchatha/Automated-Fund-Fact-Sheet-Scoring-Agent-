"""
PRCS Fact-Sheet Scoring Agent
=============================
Extends the PRCS Trust Score (returns-based) with a "Fact-Sheet Score"
(holdings/behaviour-based), using Gemini to read monthly AMC fact sheets.

This implements Track B1 (Holdings & behaviour) from the PRCS Research
Project Specification: style drift, concentration, sector positioning,
turnover, and manager commentary — direct measures of manager trustworthiness
rather than NAV-derived shadows of it.

SETUP
-----
pip install google-genai pdfplumber pandas

export GEMINI_API_KEY="your-key-here"

USAGE
-----
python fact_sheet_scoring_agent.py --pdf path/to/factsheet.pdf --category "Large Cap"

Or import score_fact_sheet() directly to run over a folder of AMC fact sheets.

IMPORTANT — point-in-time discipline
-------------------------------------
A fact sheet dated e.g. 2024-06-30 must ONLY ever be used to inform decisions
made AT OR AFTER that date. When you merge this into the PRCS pipeline, tag
every score with its `as_of_date` and join it the same way run_all.py joins
NAV data — never let a June fact sheet influence a scoring decision made in May.
"""

import os
import json
import glob
import argparse
from datetime import datetime

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise SystemExit("Run: pip install google-genai")

try:
    import pdfplumber
except ImportError:
    raise SystemExit("Run: pip install pdfplumber")


# ---------------------------------------------------------------------------
# 1. THE SCHEMA — this is what forces structured, checkable output.
#    Gemini's response_schema mode guarantees the JSON shape below.
# ---------------------------------------------------------------------------

FACT_SHEET_SCHEMA = {
    "type": "object",
    "properties": {
        "scheme_name": {"type": "string"},
        "amc_name": {"type": "string"},
        "as_of_date": {"type": "string", "description": "YYYY-MM-DD, the fact sheet's stated date"},
        "stated_category": {"type": "string"},

        "concentration_score": {"type": "integer", "description": "0-100, 100=well diversified"},
        "concentration_evidence": {"type": "string", "description": "Short quote or figure from the document, or 'NOT FOUND'"},

        "style_consistency_score": {"type": "integer", "description": "0-100, 100=fully matches stated category"},
        "style_consistency_evidence": {"type": "string"},

        "sector_alignment_score": {"type": "integer", "description": "0-100, 100=closely tracks benchmark sector weights"},
        "sector_alignment_evidence": {"type": "string"},

        "turnover_signal_score": {"type": "integer", "description": "0-100, 100=low/stable turnover"},
        "turnover_signal_evidence": {"type": "string"},

        "top_10_holdings_pct": {"type": "number", "description": "Sum of top 10 holdings weight, or -1 if not disclosed"},
        "cash_allocation_pct": {"type": "number", "description": "Cash/equivalents %, or -1 if not disclosed"},

        "missing_fields": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "scheme_name", "amc_name", "as_of_date", "stated_category",
        "concentration_score", "concentration_evidence",
        "style_consistency_score", "style_consistency_evidence",
        "sector_alignment_score", "sector_alignment_evidence",
        "turnover_signal_score", "turnover_signal_evidence",
        "top_10_holdings_pct", "cash_allocation_pct",
        "missing_fields", "confidence",
    ],
}


# ---------------------------------------------------------------------------
# 2. THE PROMPT
#    Two rules do almost all the work here:
#      (a) every score must be backed by a quoted/paraphrased evidence field
#      (b) anything not present in the document must be marked NOT FOUND,
#          never guessed — a fund's real risk is data quality itself.
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are a careful financial-document analyst scoring Indian
mutual fund monthly fact sheets. You are NOT a portfolio recommender — you are
producing a transparent, auditable, category-relative behavioural score for
a research pipeline. Follow these rules strictly:

1. Only use information that is actually present in the provided document.
   If a figure or statement is not in the document, set the score to 50
   (neutral/unknown — never guess high or low) and write "NOT FOUND" in the
   matching evidence field, and add the field name to missing_fields.
2. Every non-neutral score MUST be justified by a short (<25 word) quote or
   paraphrase from the document in the matching *_evidence field. A score
   without traceable evidence is not acceptable output.
3. Score each dimension independently. Do not let a low score on one
   dimension pull down your reasoning on another.
4. Style consistency: compare the fund's disclosed category (e.g. "Large Cap")
   against its actual disclosed holdings composition (e.g. % in large-cap vs
   mid/small-cap stocks, if stated). A fund holding significantly outside its
   stated mandate scores low here, regardless of returns.
5. Be conservative. This is a research signal, not a final verdict — when
   genuinely uncertain, score near 50 and say so via low confidence.
6. Output ONLY the JSON object matching the provided schema. No prose,
   no markdown fences, nothing else.
"""

USER_PROMPT_TEMPLATE = """Below is the extracted text of one monthly mutual fund
fact sheet. Read it and produce the structured score defined by the schema.

CATEGORY THIS FUND IS BEING EVALUATED AGAINST: {category}
CATEGORY BENCHMARK (if relevant for sector comparison): {benchmark}

--- FACT SHEET TEXT START ---
{fact_sheet_text}
--- FACT SHEET TEXT END ---
"""


# ---------------------------------------------------------------------------
# 3. THE AGENT CALL
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-3.5-flash"  # cheap/fast; use gemini-3.1-pro for harder fact sheets


def _init_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY as an environment variable first.")
    return genai.Client(api_key=api_key)


def extract_pdf_text(pdf_path: str) -> str:
    """Pull raw text out of a fact sheet PDF. AMC fact sheets vary wildly in
    layout, so this is intentionally simple — flag pages that fail to extract
    (likely scanned/image-only pages) rather than silently dropping them."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                text_parts.append(f"[PAGE {i+1}: NO EXTRACTABLE TEXT — possibly scanned/image]")
    return "\n".join(text_parts)


def extract_scheme_section(full_text: str, scheme_name: str, section_chars: int = 30_000) -> str:
    """Some AMCs (e.g. ABSL) publish ONE giant PDF covering every scheme in
    their fund family, with a table of contents up front. Blindly truncating
    to the first N characters of a multi-hundred-page document just grabs the
    cover page and index and never reaches the actual scheme data — which is
    silently wrong rather than loudly broken, so it's an easy bug to miss.

    Strategy: the scheme name typically appears once in the table of contents
    (early in the doc) and then again as the section header where its actual
    data starts. We skip the first occurrence and start extraction from the
    second one. Falls back to a plain truncation if the name only appears once
    (i.e. this looks like a genuine single-scheme fact sheet already)."""
    import re
    pattern = re.escape(scheme_name.upper())
    matches = [m.start() for m in re.finditer(pattern, full_text.upper())]

    if len(matches) < 2:
        # Single-scheme document, or name didn't match cleanly — don't guess.
        return full_text[:section_chars]

    start = matches[1]
    return full_text[start:start + section_chars]


def score_fact_sheet(fact_sheet_text: str, category: str, benchmark: str = "",
                      scheme_name: str = "") -> dict:
    """Send one fact sheet's text to Gemini and get back the structured score.
    If scheme_name is given, isolate that scheme's section first — important
    for multi-scheme "factsheet book" PDFs (see extract_scheme_section)."""
    client = _init_model()
    if scheme_name:
        relevant_text = extract_scheme_section(fact_sheet_text, scheme_name)
    else:
        relevant_text = fact_sheet_text[:100_000]  # guard against oversized docs
    prompt = USER_PROMPT_TEMPLATE.format(
        category=category,
        benchmark=benchmark or "not specified",
        fact_sheet_text=relevant_text,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=FACT_SHEET_SCHEMA,
            temperature=0.1,  # low temperature: consistency matters more than creativity here
        ),
    )
    result = json.loads(response.text)

    # Composite fact_sheet_score: simple equal-weighted average of 4 dimensions
    # (manager_commentary removed — structurally absent from per-fund fact sheets)
    component_keys = [
        "concentration_score", "style_consistency_score",
        "sector_alignment_score", "turnover_signal_score",
    ]
    result["fact_sheet_score"] = round(
        sum(result[k] for k in component_keys) / len(component_keys), 1
    )
    return result


# ---------------------------------------------------------------------------
# 4. BATCH RUNNER
#    Expects: data/factsheets/<AMC>/*.pdf
#    Each AMC folder can hold either:
#      - one PDF per scheme (plain single-scheme fact sheets), or
#      - one "factsheet book" PDF covering many schemes (e.g. ABSL) — or both.
#    extract_scheme_section() already handles either shape per-scheme, so the
#    batch runner just needs to know WHICH schemes to look for per AMC.
# ---------------------------------------------------------------------------

def run_batch(factsheet_dir: str, category_map: dict, out_csv: str, benchmark_map: dict = None) -> pd.DataFrame:
    """
    category_map: {amc_name: {scheme_name: category}}
        e.g. {"ABSL": {"ADITYA BIRLA SUN LIFE LARGE CAP FUND": "Large Cap",
                        "ADITYA BIRLA SUN LIFE SMALL CAP FUND": "Small Cap"}}
        amc_name must match the AMC's subfolder name under factsheet_dir.
    benchmark_map: optional {amc_name: {scheme_name: benchmark_name}}, same
        shape as category_map — only needed for schemes where you want
        sector_alignment_score scored against a named benchmark.
    Writes ONE combined CSV across all AMCs (amc_name is a column), not a
    separate CSV per AMC.
    """
    benchmark_map = benchmark_map or {}
    rows = []

    amc_dirs = [d for d in glob.glob(os.path.join(factsheet_dir, "*")) if os.path.isdir(d)]
    print(f"Found {len(amc_dirs)} AMC folder(s) under {factsheet_dir}")

    for amc_dir in amc_dirs:
        amc_name = os.path.basename(amc_dir)
        schemes = category_map.get(amc_name, {})
        if not schemes:
            print(f"  [skip AMC] {amc_name}: no scheme mapping in category_map")
            continue

        pdf_paths = glob.glob(os.path.join(amc_dir, "*.pdf"))
        if not pdf_paths:
            print(f"  [skip AMC] {amc_name}: no PDFs found")
            continue

        # Extract each PDF once and cache it — cheaper than re-parsing per scheme.
        pdf_texts = {}
        for p in pdf_paths:
            print(f"  [extracting] {os.path.basename(p)}")
            pdf_texts[p] = extract_pdf_text(p)

        for scheme_name, category in schemes.items():
            # Find which PDF actually contains this scheme. Prefer the PDF
            # where the name appears 2+ times — that's the "book" pattern
            # (once in the table of contents, once as the section header) —
            # since that's the strongest signal we've landed on real data,
            # not just a passing mention.
            best_path, best_count = None, 0
            for p, text in pdf_texts.items():
                count = text.upper().count(scheme_name.upper())
                if count > best_count:
                    best_path, best_count = p, count

            if best_path is None:
                print(f"    [skip] {scheme_name}: not found in any {amc_name} PDF")
                continue

            try:
                benchmark = benchmark_map.get(amc_name, {}).get(scheme_name, "")
                result = score_fact_sheet(
                    pdf_texts[best_path], category=category,
                    scheme_name=scheme_name, benchmark=benchmark,
                )
                result["amc_name"] = amc_name  # folder-derived, more reliable than model's guess
                result["source_file"] = best_path
                result["scored_at"] = datetime.utcnow().isoformat()
                rows.append(result)
                print(f"    [ok] {scheme_name}: fact_sheet_score={result['fact_sheet_score']} "
                      f"(confidence={result['confidence']})")
            except Exception as e:
                print(f"    [fail] {scheme_name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} scored fact sheets across {len(amc_dirs)} AMC(s) to {out_csv}")
    return df


# ---------------------------------------------------------------------------
# 5. UNIFORMITY CHECK — run this BEFORE trusting any score.
#    Different AMCs format fact sheets very differently. This tells you
#    whether the agent is actually extracting consistently across them,
#    or whether some AMCs are silently coming back mostly "NOT FOUND".
# ---------------------------------------------------------------------------

EVIDENCE_COLS = [
    "concentration_evidence", "style_consistency_evidence",
    "sector_alignment_evidence", "turnover_signal_evidence",
]


def check_uniformity(scores_csv: str) -> pd.DataFrame:
    """Per-AMC diagnostic: how often did fields come back NOT FOUND, and
    what's the confidence distribution? Low completeness for one AMC means
    its fact sheet layout needs a closer look (e.g. scanned pages, an
    unusual holdings table format) before you trust its scores."""
    df = pd.read_csv(scores_csv)

    def not_found_rate(row):
        hits = sum(1 for c in EVIDENCE_COLS if str(row.get(c, "")).strip().upper() == "NOT FOUND")
        return hits / len(EVIDENCE_COLS)

    df["not_found_rate"] = df.apply(not_found_rate, axis=1)

    summary = df.groupby("amc_name").agg(
        n_fact_sheets=("scheme_name", "count"),
        avg_not_found_rate=("not_found_rate", "mean"),
        pct_low_confidence=("confidence", lambda s: (s == "low").mean()),
        score_std=("fact_sheet_score", "std"),  # low variance across all funds = suspicious (lazy/uniform scoring)
    ).reset_index().sort_values("avg_not_found_rate", ascending=False)

    print(summary.to_string(index=False))
    flagged = summary[summary["avg_not_found_rate"] > 0.3]
    if len(flagged):
        print(f"\n⚠ {len(flagged)} AMC(s) have >30% missing fields on average — "
              f"review their fact sheet format before trusting these scores:")
        print(flagged["amc_name"].tolist())
    return summary


# ---------------------------------------------------------------------------
# 6. MERGE WITH THE EXISTING TRUST SCORE (rolling 3-year window)
#    NOT NEEDED YET — this is Phase 2, once the standalone score above has
#    proven itself. Left in place for when you get there; skip it for now.
# ---------------------------------------------------------------------------

def merge_with_trust_score(fact_sheet_scores_csv: str, pooled_panel_csv: str, out_csv: str):
    """
    Merges the fact-sheet score into the existing PRCS pooled panel.

    Steps (mirrors how the existing Trust Score components are combined):
      1. Percentile-rank fact_sheet_score WITHIN category, per quarter
         (same convention as scores.csv — never compare across categories).
      2. Roll up to a trailing-3-year (12-quarter) average per fund, since a
         single fact sheet is noisy but sustained behaviour over 3 years is
         a signal, matching the "consistency persists" design principle.
      3. Left-join onto the pooled panel on (scheme_code, quarter).
      4. Leave the ORIGINAL trust_score untouched as its own column — do not
         overwrite it. Per Track B1 rules, this new axis must be validated
         with its own out-of-sample information coefficient BEFORE being
         blended into the official Trust Score. Treat it as a candidate
         feature, not an automatic replacement.
    """
    fs = pd.read_csv(fact_sheet_scores_csv, parse_dates=["as_of_date"])
    fs["quarter"] = fs["as_of_date"].dt.to_period("Q").dt.to_timestamp()

    # 1. within-category, within-quarter percentile rank
    fs["fact_sheet_pctile"] = fs.groupby(["stated_category", "quarter"])["fact_sheet_score"] \
        .rank(pct=True) * 100

    # 2. trailing 3-year (12-quarter) rolling average per fund
    fs = fs.sort_values(["scheme_name", "quarter"])
    fs["fact_sheet_pctile_3y"] = (
        fs.groupby("scheme_name")["fact_sheet_pctile"]
        .transform(lambda s: s.rolling(window=12, min_periods=4).mean())
    )

    panel = pd.read_csv(pooled_panel_csv, parse_dates=["quarter"])
    merged = panel.merge(
        fs[["scheme_name", "quarter", "fact_sheet_pctile_3y", "confidence"]],
        left_on=["scheme_code", "quarter"],  # adjust join keys to match your actual panel schema
        right_on=["scheme_name", "quarter"],
        how="left",
    )
    merged.to_csv(out_csv, index=False)
    print(f"Merged panel saved to {out_csv}. "
          f"fact_sheet_pctile_3y coverage: "
          f"{merged['fact_sheet_pctile_3y'].notna().mean():.1%} of rows")
    return merged


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Score a single fact sheet PDF")
    parser.add_argument("--category", default="Large Cap")
    parser.add_argument("--scheme_name", default="",
                         help="Scheme name to isolate, needed when --pdf is a multi-scheme "
                              "'factsheet book' (e.g. 'ADITYA BIRLA SUN LIFE LARGE CAP FUND'). "
                              "Leave blank for a genuine single-scheme fact sheet PDF.")
    parser.add_argument("--batch_dir", help="Score a whole folder of fact sheets, organized as <batch_dir>/<AMC>/*.pdf")
    parser.add_argument("--config", default="scheme_config.json",
                         help="JSON file mapping AMC -> scheme -> category (and optionally benchmark). "
                              "See scheme_config.json.example for the expected shape. Only used with --batch_dir.")
    parser.add_argument("--out_csv", default="fact_sheet_scores.csv")
    parser.add_argument("--check_uniformity", action="store_true",
                         help="After batch scoring, check consistency across AMCs")
    args = parser.parse_args()

    if args.pdf:
        text = extract_pdf_text(args.pdf)
        result = score_fact_sheet(text, category=args.category, scheme_name=args.scheme_name)
        print(json.dumps(result, indent=2))
    elif args.batch_dir:
        if not os.path.exists(args.config):
            raise SystemExit(
                f"Config file not found: {args.config}\n"
                f"Batch mode needs to know which schemes to look for per AMC. Create a JSON file shaped like:\n"
                f'{{"category_map": {{"ABSL": {{"ADITYA BIRLA SUN LIFE LARGE CAP FUND": "Large Cap"}}}}, '
                f'"benchmark_map": {{"ABSL": {{"ADITYA BIRLA SUN LIFE LARGE CAP FUND": "NIFTY 100 TRI"}}}}}}\n'
                f"benchmark_map is optional. Pass its path with --config."
            )
        with open(args.config) as f:
            cfg = json.load(f)
        run_batch(
            args.batch_dir,
            category_map=cfg.get("category_map", {}),
            benchmark_map=cfg.get("benchmark_map", {}),
            out_csv=args.out_csv,
        )
        if args.check_uniformity:
            print("\n--- Uniformity check across AMCs ---")
            check_uniformity(args.out_csv)
    else:
        print("Pass --pdf <file> for a single test run, or --batch_dir <folder> for a batch run.")
