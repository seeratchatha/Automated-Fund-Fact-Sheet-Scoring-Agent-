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


FACT_SHEET_SCHEMA = {
    "type": "object",
    "properties": {
        "scheme_name": {"type": "string"},
        "amc_name": {"type": "string"},
        "as_of_date": {"type": "string"},
        "stated_category": {"type": "string"},
        "concentration_score": {"type": "integer"},
        "concentration_evidence": {"type": "string"},
        "style_consistency_score": {"type": "integer"},
        "style_consistency_evidence": {"type": "string"},
        "sector_alignment_score": {"type": "integer"},
        "sector_alignment_evidence": {"type": "string"},
        "turnover_signal_score": {"type": "integer"},
        "turnover_signal_evidence": {"type": "string"},
        "top_10_holdings_pct": {"type": "number"},
        "cash_allocation_pct": {"type": "number"},
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

SYSTEM_INSTRUCTION = """You are a financial-document analyst scoring Indian mutual fund 
monthly fact sheets. Produce transparent, auditable, category-relative behavioural scores.

Rules:
1. Only use information present in the document. If missing, set score to 50, write 
   "NOT FOUND" in evidence, and add field name to missing_fields.
2. Every non-neutral score MUST be justified by a quote or paraphrase (<25 words) in 
   the matching *_evidence field.
3. Score each dimension independently.
4. Style consistency: compare stated category against actual holdings composition.
5. Be conservative. When uncertain, score near 50 with low confidence.
6. Output ONLY the JSON object matching the schema. No prose, no markdown.
"""

USER_PROMPT_TEMPLATE = """Below is the extracted text of one monthly mutual fund fact sheet.
Read it and produce the structured score defined by the schema.

CATEGORY THIS FUND IS BEING EVALUATED AGAINST: {category}
CATEGORY BENCHMARK (if relevant for sector comparison): {benchmark}

--- FACT SHEET TEXT START ---
{fact_sheet_text}
--- FACT SHEET TEXT END ---
"""

MODEL_NAME = "gemini-3.5-flash"


def _init_model():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY as an environment variable first.")
    return genai.Client(api_key=api_key)


def extract_pdf_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                text_parts.append(f"[PAGE {i+1}: NO EXTRACTABLE TEXT]")
    return "\n".join(text_parts)


def extract_scheme_section(full_text: str, scheme_name: str, section_chars: int = 30_000) -> str:
    import re
    pattern = re.escape(scheme_name.upper())
    matches = [m.start() for m in re.finditer(pattern, full_text.upper())]

    if len(matches) < 2:
        return full_text[:section_chars]

    start = matches[1]
    return full_text[start:start + section_chars]


def score_fact_sheet(fact_sheet_text: str, category: str, benchmark: str = "",
                      scheme_name: str = "") -> dict:
    client = _init_model()
    if scheme_name:
        relevant_text = extract_scheme_section(fact_sheet_text, scheme_name)
    else:
        relevant_text = fact_sheet_text[:100_000]
    
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
            temperature=0.1,
        ),
    )
    
    result = json.loads(response.text)

    component_keys = [
        "concentration_score", "style_consistency_score",
        "sector_alignment_score", "turnover_signal_score",
    ]
    result["fact_sheet_score"] = round(
        sum(result[k] for k in component_keys) / len(component_keys), 1
    )
    return result


def run_batch(factsheet_dir: str, category_map: dict, out_csv: str, benchmark_map: dict = None) -> pd.DataFrame:
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

        pdf_texts = {}
        for p in pdf_paths:
            print(f"  [extracting] {os.path.basename(p)}")
            pdf_texts[p] = extract_pdf_text(p)

        for scheme_name, category in schemes.items():
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
                result["amc_name"] = amc_name
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


EVIDENCE_COLS = [
    "concentration_evidence", "style_consistency_evidence",
    "sector_alignment_evidence", "turnover_signal_evidence",
]


def check_uniformity(scores_csv: str) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)

    def not_found_rate(row):
        hits = sum(1 for c in EVIDENCE_COLS if str(row.get(c, "")).strip().upper() == "NOT FOUND")
        return hits / len(EVIDENCE_COLS)

    df["not_found_rate"] = df.apply(not_found_rate, axis=1)

    summary = df.groupby("amc_name").agg(
        n_fact_sheets=("scheme_name", "count"),
        avg_not_found_rate=("not_found_rate", "mean"),
        pct_low_confidence=("confidence", lambda s: (s == "low").mean()),
        score_std=("fact_sheet_score", "std"),
    ).reset_index().sort_values("avg_not_found_rate", ascending=False)

    print(summary.to_string(index=False))
    flagged = summary[summary["avg_not_found_rate"] > 0.3]
    if len(flagged):
        print(f"\n⚠ {len(flagged)} AMC(s) have >30% missing fields on average — "
              f"review their fact sheet format before trusting these scores:")
        print(flagged["amc_name"].tolist())
    return summary


def merge_with_trust_score(fact_sheet_scores_csv: str, pooled_panel_csv: str, out_csv: str):
    fs = pd.read_csv(fact_sheet_scores_csv, parse_dates=["as_of_date"])
    fs["quarter"] = fs["as_of_date"].dt.to_period("Q").dt.to_timestamp()

    fs["fact_sheet_pctile"] = fs.groupby(["stated_category", "quarter"])["fact_sheet_score"] \
        .rank(pct=True) * 100

    fs = fs.sort_values(["scheme_name", "quarter"])
    fs["fact_sheet_pctile_3y"] = (
        fs.groupby("scheme_name")["fact_sheet_pctile"]
        .transform(lambda s: s.rolling(window=12, min_periods=4).mean())
    )

    panel = pd.read_csv(pooled_panel_csv, parse_dates=["quarter"])
    merged = panel.merge(
        fs[["scheme_name", "quarter", "fact_sheet_pctile_3y", "confidence"]],
        left_on=["scheme_code", "quarter"],
        right_on=["scheme_name", "quarter"],
        how="left",
    )
    merged.to_csv(out_csv, index=False)
    print(f"Merged panel saved to {out_csv}. "
          f"fact_sheet_pctile_3y coverage: "
          f"{merged['fact_sheet_pctile_3y'].notna().mean():.1%} of rows")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Score a single fact sheet PDF")
    parser.add_argument("--category", default="Large Cap")
    parser.add_argument("--scheme_name", default="")
    parser.add_argument("--batch_dir", help="Score a whole folder of fact sheets")
    parser.add_argument("--config", default="scheme_config.json")
    parser.add_argument("--out_csv", default="fact_sheet_scores.csv")
    parser.add_argument("--check_uniformity", action="store_true")
    args = parser.parse_args()

    if args.pdf:
        text = extract_pdf_text(args.pdf)
        result = score_fact_sheet(text, category=args.category, scheme_name=args.scheme_name)
        print(json.dumps(result, indent=2))
    elif args.batch_dir:
        if not os.path.exists(args.config):
            raise SystemExit(f"Config file not found: {args.config}")
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
