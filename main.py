# main.py
print(">>> START main.py")

from fastapi import FastAPI, File, UploadFile, Form
print(">>> fastapi imports done")
from fastapi.responses import JSONResponse
print(">>> JSONResponse import done")
import zipfile, os, tempfile, re         
print(">>> os/zipfile/tempfile/re imports done")


from invoice_analysis import (
    extract_text_from_pdf,
    extract_text_from_pdf_ocr,
    rule_based_check,
    truncate,
    generate_prompt,
    call_groq,
    add_to_vector_store,
    analyze_single_invoice,                 
)

print(">>> invoice_analysis import done")
# ---------- helper functions ----------
def determine_category(filename: str) -> str:
    name = filename.lower()
    if "meal" in name:
        return "meal"
    if "cab" in name or "ride" in name:
        return "cab"
    return "travel"

def extract_status(response: str) -> str:
    for line in response.splitlines():
        if "status" in line.lower():
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"

def extract_reason(response: str) -> str:
    for line in response.splitlines():
        if "reason" in line.lower():
            return line.split(":", 1)[1].strip()
    return "No reason found"

app = FastAPI()

# -------------------------------------------------
# POST /analyze‑invoices
# -------------------------------------------------
@app.post("/analyze-invoices/")
async def analyze_invoices(
    employee_name: str = Form(...),
    policy_pdf:  UploadFile = File(...),
    invoices_zip: UploadFile = File(...),
):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            # ---------- save policy ----------
            policy_path = os.path.join(tmpdir, "policy.pdf")
            with open(policy_path, "wb") as f:
                f.write(await policy_pdf.read())
            print(">>> 1. Reading policy")
            policy_text  = extract_text_from_pdf(policy_path)
            
            print(">>> 2. Truncating policy")
            policy_short = truncate(policy_text, max_words=100)

            # ---------- save & unzip invoices ----------
            zip_path = os.path.join(tmpdir, "invoices.zip")
            with open(zip_path, "wb") as f:
                f.write(await invoices_zip.read())

            invoice_dir = os.path.join(tmpdir, "invoices")
            os.makedirs(invoice_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(invoice_dir)

            # ---------- analyse every PDF ----------
            results = []
            for file in os.listdir(invoice_dir):
                if not file.lower().endswith(".pdf"):
                    continue

                pdf_path = os.path.join(invoice_dir, file)
                print(f">>> Analyzing with analyze_single_invoice(): {file}")
                result = analyze_single_invoice(policy_short, pdf_path, determine_category(file))

                # Push to vector store
                add_to_vector_store(
                    doc_id=f"{employee_name}_{file}",
                    full_text=result["filename"],  # NOTE: use invoice text if needed
                    status=result["status"],
                    reason=result["reason"],
                    meta={
                        "employee": employee_name,
                        "filename": file,
                        "category": result["category"],
                        "status": result["status"],
                        "date": "",  # (you can re-add date extraction if needed)
                    },
                )

                result["employee"] = employee_name
                results.append(result)


        return JSONResponse(content={"success": True, "results": results})

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})


