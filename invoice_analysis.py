print(">>> START invoice_analysis.py")
import os, re
import pdfplumber, pytesseract
from datetime import date 
from pdf2image import convert_from_path
from dotenv import load_dotenv
from vectordb import embedder, collection

# ---------- local paths ----------
pytesseract.pytesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
POPPLER_PATH = r"C:\\Users\\91902\\Downloads\\poppler-24.08.0\\Library\\bin"

# ---------- optional Groq ----------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    from openai import OpenAI
    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
else:
    groq_client = None

# ---------- PDF text helpers ----------
def extract_text_from_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()

def extract_text_from_pdf_ocr(path: str, dpi: int = 110) -> str:
    text = extract_text_from_pdf(path)
    if text.strip():
        return text
    imgs = convert_from_path(path, dpi=dpi, poppler_path=POPPLER_PATH)
    return "\n".join(pytesseract.image_to_string(img, lang="eng") for img in imgs).strip()

def truncate(txt: str, words: int = 120) -> str:
    return " ".join(txt.split()[:words])

def determine_category(name: str) -> str:
    n = name.lower()
    if "meal" in n: return "meal"
    if "cab"  in n: return "cab"
    if "travel" in n: return "travel"
    return "other"

# ---------- RULE-BASED FUNCTION ----------
def rule_based_check(text: str, category: str):
    """Returns (status, reason) OR (None, None) if uncertain."""
    text_lower = text.lower()
    lines = [ln.strip() for ln in text_lower.splitlines() if ln.strip()]

    # ============= helpers ==============
    def extract_amount(text, keywords):
        """
        Return the largest number (float) found on lines containing any of the keywords.
        Handles OCR errors like '21 00' by merging digits.
        """
        max_val = 0
        lines = text.lower().splitlines()

        for ln in lines:
            if any(k in ln for k in keywords):
                clean_line = ln.replace(",", "")
                # Match numbers: "1,000", "1000.00", "21 00", etc.
                matches = re.findall(r"(\d{1,3}(?:[\s,]\d{2,3})+|\d+\.\d+|\d+)", clean_line)

                for match in matches:
                    number = match.replace(" ", "")  # merge spaces: "21 00" → "2100"
                    try:
                        val = float(number)
                        if 10 <= val <= 50000:
                            max_val = max(max_val, val)
                    except:
                        continue
        return max_val

    # =============== MEAL =================
    if category == "meal":
        alcohol_kw = ["whisky", "rum", "vodka", "wine", "beer",
                      "scotch", "gin", "tequila", "alcohol", "stag", "mc", "royal"]
        food_kw = ["biryani", "biriyani", "idli", "dosa", "chapati", "meal",
                   "meals", "tea", "coffee", "roti", "vada", "thali", "mini",
                   "paratha", "sandwich", "biriyvani"]

        alcohol_found = any(a in text_lower for a in alcohol_kw)
        food_found    = any(f in text_lower for f in food_kw)

        total = extract_amount(text, ["total", "sub total", "subtotal"])
        if total == 0:
            return None, None

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
        travel_keywords = ["ticket", "air india", "train", "flight", "airasia", "indigo", "goair", "ixigo"]
        if not any(kw in text_lower for kw in travel_keywords):
            return "Declined", "No valid travel reference found."

        # Prioritize 'total fare'
        amount = extract_amount(text, ["total fare"])
        if not amount:
            amount = extract_amount(text, ["total", "fare", "amount", "price", "cost", "rs", "inr", "net amount"])

        if amount == 0:
            return "Declined", "No valid travel cost found."

        if amount <= 2000:
            return "Fully Reimbursed", f"Trip cost ≈ ₹{int(amount)} within ₹2000 limit."
        else:
            return "Partially Reimbursed", f"Trip cost ≈ ₹{int(amount)} exceeds ₹2000 per‑trip limit."



    # =============== CAB ==================
    if category == "cab":
        amount = extract_amount(text, ["total", "fare", "amount", "subtotal"])
        if amount == 0:
            return None, None
        if amount <= 150:
            return "Fully Reimbursed", f"Cab fare ≈ ₹{int(amount)} within ₹150 daily limit."
        return "Partially Reimbursed", f"Cab fare ≈ ₹{int(amount)} exceeds ₹150 daily limit."

    return None, None  # unknown category

# ---------- Remaining optional Groq/vector helpers ----------
def generate_prompt(policy, invoice, cat):
    return (
        "You are a reimbursement assistant.\n\n"
        "Policy:\n" + policy + "\n\n"
        f"Invoice ({cat.upper()}):\n" + invoice + "\n\n"
        "Reply ONLY:\n"
        "Status: Fully Reimbursed / Partially Reimbursed / Declined\n"
        "Reason: <one sentence>"
    )

def call_groq(prompt):
    if groq_client is None:
        return "Status: UNKNOWN\nReason: LLM disabled."
    resp = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "Return Status/Reason only."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120
    )
    return resp.choices[0].message.content.strip()

from datetime import date

def add_to_vector_store(doc_id, full_text, status, reason, meta):
    if embedder is None or collection is None:
        return
    try:
        snippet = " ".join(full_text.split()[:512])
        vec_text = f"{snippet}\n\nStatus:{status}\nReason:{reason}"
        vec = embedder.encode([vec_text])[0].tolist()

        meta["date"] = str(date.today())

        collection.add(
            ids=[doc_id],
            embeddings=[vec],
            documents=[vec_text],
            metadatas=[meta]
        )
    except Exception as e:
        print(">>> [VS] ERROR:", e)
