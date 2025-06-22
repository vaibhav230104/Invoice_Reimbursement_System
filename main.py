print(">>> START main.py")
import os, zipfile, tempfile, time, asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from invoice_analysis import (
    extract_text_from_pdf,
    extract_text_from_pdf_ocr,
    rule_based_check,
    truncate,
)

# ---------- FastAPI app ----------
app = FastAPI(title="Invoice API (Stable)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic models ----------
class InvoiceResult(BaseModel):
    filename: str
    employee: str
    category: str
    status: str
    reason: str

class AnalyzeResponse(BaseModel):
    success: bool
    results: List[InvoiceResult]

def determine_category(name: str) -> str:
    n = name.lower()
    if "meal" in n: return "meal"
    if "cab"  in n or "ride" in n: return "cab"
    return "travel"

# ---------- Endpoint ----------
@app.post("/analyze-invoices/", response_model=AnalyzeResponse)
async def analyze_invoices(
    employee_name: str = Form(...),
    policy_pdf: UploadFile = File(...),
    invoices_zip: UploadFile = File(...),
):
    print(">>> Endpoint triggered", flush=True)
    ts = time.time()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save policy
            pol_path = os.path.join(tmpdir, "policy.pdf")
            with open(pol_path, "wb") as f:
                f.write(await policy_pdf.read())
            print(">>> policy saved")

            policy_text = await asyncio.to_thread(extract_text_from_pdf, pol_path)
            policy_short = truncate(policy_text, 100)
            print(">>> policy extracted & truncated")

            # Save ZIP
            zip_path = os.path.join(tmpdir, "invoices.zip")
            with open(zip_path, "wb") as f:
                f.write(await invoices_zip.read())
            print(">>> invoices ZIP saved")

            # Extract ZIP
            inv_dir = os.path.join(tmpdir, "invoices")
            os.makedirs(inv_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(inv_dir)
            print(">>> invoices extracted")

            # Iterate PDFs
            results = []
            for root, _, files in os.walk(inv_dir):
                for fname in files:
                    if not fname.lower().endswith(".pdf"):
                        print(">>> skipped non-pdf:", fname)
                        continue

                    pdf_path = os.path.join(root, fname)
                    print(">>> analyzing", fname)

                    try:
                        inv_text = await asyncio.to_thread(extract_text_from_pdf_ocr, pdf_path)
                        if not inv_text.strip():
                            print(">>> empty OCR output, skipped")
                            continue

                        category = determine_category(fname)
                        status, reason = rule_based_check(inv_text, category)
                        if status is None:
                            status = "UNKNOWN"
                            reason = "Rule-based check failed (LLM disabled)"

                        results.append(InvoiceResult(
                            filename=fname,
                            employee=employee_name,
                            category=category,
                            status=status,
                            reason=reason
                        ))
                        print(f">>> done: {fname} → {status}")
                    except Exception as file_err:
                        print(f">>> Error processing {fname}: {file_err}")

            print(">>> Completed in", round(time.time() - ts, 2), "s")
            return AnalyzeResponse(success=True, results=results)

    except Exception as e:
        print(">>> Global ERROR:", e)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
