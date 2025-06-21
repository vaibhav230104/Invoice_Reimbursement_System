# invoice_analysis.py
print(">>> START invoice_analysis.py")

import os, time, re
print(">>> os, time, re import done")

import pdfplumber
print(">>> pdfplumber import done")

import pytesseract
print(">>> pytesseract import done")

from pdf2image import convert_from_path
print(">>> pdf2image import done")

from openai import OpenAI
print(">>> openai import done")

from dotenv import load_dotenv
print(">>> dotenv import done")

from vectordb import embedder, collection
print(">>> vectordb import done")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\91902\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"                      

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")                  
)

def add_to_vector_store(
        doc_id: str,
        full_text: str,
        status: str,
        reason: str,
        meta: dict
):
    """Embeds the invoice text + conclusion and writes to Chroma."""
    # We embed both the raw invoice & the verdict so RAG can match either.
    embedding_input = full_text + "\n\nStatus: " + status + "\nReason: " + reason
    vec = embedder.encode(embedding_input).tolist()      # -> python list for Chroma

    collection.add(
        ids=[doc_id],                # must be unique
        embeddings=[vec],
        documents=[embedding_input], # for retrieval preview
        metadatas=[meta]
    )

# -----------------------------------------------------------
# 1. PDF TEXT EXTRACTION
# -----------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF using pdfplumber (works for true‑text PDFs)."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            if pg.extract_text():
                text += pg.extract_text() + "\n"
    return text.strip()

def extract_text_from_pdf_ocr(pdf_path: str) -> str:
    """
    OCR fallback for scanned PDFs.
    Requires:
        • Tesseract installed
        • Poppler (for pdf2image)   –> POPPLER_PATH
    """
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    all_text = ""
    for img in images:
        all_text += pytesseract.image_to_string(img) + "\n"
    print(">>> MOCK: OCR fallback triggered")
    return all_text.strip()

# -----------------------------------------------------------
# 2. SMALL HELPERS
# -----------------------------------------------------------

def truncate(text: str, max_words: int = 120) -> str:
    return " ".join(text.split()[:max_words])

def generate_prompt(policy, invoice_text, category) -> str:
    return f"""
You are a reimbursement assistant.

Policy:
\"\"\"
{policy}
\"\"\"

Invoice ({category.upper()}):
\"\"\"
{invoice_text}
\"\"\"

Only reply in this format:

Status: <Fully Reimbursed / Partially Reimbursed / Declined>
Reason: <1 sentence reason>
""".strip()

def call_groq(prompt, model: str = "llama3-8b-8192") -> str:
    """Calls Groq ChatCompletion (OpenAI compatible)."""
    print(">>> 5. Calling LLM")
    print(">>> Making real Groq API call")
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a reimbursement assistant. Reply ONLY in Status/Reason format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=150
    )
    print(">>> Groq API call done")
    return chat.choices[0].message.content.strip()




# -----------------------------------------------------------
# 3. RULE‑BASED CHECKS  (meal / travel / cab)
# -----------------------------------------------------------

def rule_based_check(text: str, category: str):
    """Returns (status, reason) OR (None, None) if uncertain."""
    text_lower = text.lower()
    lines = [ln.strip() for ln in text_lower.splitlines() if ln.strip()]

    # ---------- helpers ----------
    def extract_amount(text_block, keywords):
        """
        Return the largest numeric value (float) that appears on a line
        containing any of the given keywords.
        """
        max_val = 0
        for ln in text_block.lower().splitlines():
            if any(k in ln for k in keywords):
                clean = ln.replace(",", "")
                # merge split digits like '21 00' → '2100'
                clean = re.sub(r"(?<=\d)\s(?=\d)", "", clean)
                matches = re.findall(r"\d{1,6}(?:\.\d{1,2})?", clean)
                for m in matches:
                    try:
                        val = float(m)
                        if 1 <= val <= 50000:
                            max_val = max(max_val, val)
                    except ValueError:
                        pass
        return max_val
    # -----------------------------

    # =============== MEAL =================
    if category == "meal":
        alcohol_kw = ["whisky", "rum", "vodka", "wine", "beer",
                      "scotch", "gin", "tequila", "alcohol", "stag", "mc", "royal"]
        food_kw = ["biryani", "biriyani", "idli", "dosa", "chapati", "meal",
                   "meals", "tea", "coffee", "roti", "vada", "thali", "mini",
                   "paratha", "sandwich", "biriyvani"]

        alcohol_found = any(a in text_lower for a in alcohol_kw)
        food_found = any(f in text_lower for f in food_kw)

        total = extract_amount(text, ["total", "sub total", "subtotal"])
        if total == 0:
            return None, None            # unsure, fall back to LLM

        if not food_found and alcohol_found:
            return "Declined", "Only alcohol items found, not reimbursable."
        if not food_found:
            return "Declined", "No valid food items found."

        if total <= 200 and not alcohol_found:
            return "Fully Reimbursed", f"Meal within ₹200 limit (≈ ₹{int(total)}), no alcohol."
        if alcohol_found:
            return "Partially Reimbursed", "Alcohol excluded. ₹200 reimbursable."
        return "Partially Reimbursed", f"Meal amount ≈ ₹{int(total)} exceeds ₹200 limit."

    # =============== TRAVEL ===============
    if category == "travel":
        travel_kw = ["ticket", "train", "flight", "air", "fare"]
        if not any(k in text_lower for k in travel_kw):
            return None, None            # maybe not a travel invoice

        amount = extract_amount(text, ["total fare", "total", "fare", "amount"])
        if amount == 0:
            return None, None

        if amount <= 2000:
            return "Fully Reimbursed", f"Trip cost ≈ ₹{int(amount)} within ₹2000 limit."
        return "Partially Reimbursed", f"Trip cost ≈ ₹{int(amount)} exceeds ₹2000 per‑trip limit."

    # =============== CAB ==================
    if category == "cab":
        amount = extract_amount(text, ["total", "fare", "amount", "subtotal"])
        if amount == 0:
            return None, None

        if amount <= 150:
            return "Fully Reimbursed", f"Cab fare ≈ ₹{int(amount)} within ₹150 daily limit."
        return "Partially Reimbursed", f"Cab fare ≈ ₹{int(amount)} exceeds ₹150 daily limit."

    # If category unknown
    return None, None

# -----------------------------------------------------------
# 4. TOP‑LEVEL FUNCTION (used by FastAPI)
# -----------------------------------------------------------

def analyze_single_invoice(policy_short: str, pdf_path: str, category: str):
    print(f">>> 3. Reading invoice {os.path.basename(pdf_path)}")
    text = extract_text_from_pdf_ocr(pdf_path)
    print(f">>> 4. OCR done for {os.path.basename(pdf_path)}")

    status, reason = rule_based_check(text, category)
    print(f">>> Rule-based result: {status}, {reason}")  # NEW LINE

    if status is None or status == "Partially Reimbursed":
        print(">>> FALLBACK to LLM because rule_based_check was uncertain")
        prompt = generate_prompt(policy_short, truncate(text), category)
        print(">>> Prompt generated:\n", prompt)
        response = call_groq(prompt)
        print(">>> Groq response:", response)
        status_line = next((l for l in response.splitlines()
                            if "status" in l.lower()), "Status: UNKNOWN")
        reason_line = next((l for l in response.splitlines()
                            if "reason" in l.lower()), "Reason: Could not parse")

        status = status_line.split(":", 1)[1].strip()
        reason = reason_line.split(":", 1)[1].strip()

    print(">>> Returning final result")
    return {
        "filename": os.path.basename(pdf_path),
        "category": category,
        "status": status,
        "reason": reason
    }


