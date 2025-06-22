# 📄 Invoice Reimbursement System (Part 1)

This project provides an intelligent **Invoice Reimbursement Analysis API** built using FastAPI, rule-based logic, and optional LLM via Groq. It parses invoice PDFs and determines reimbursement eligibility based on HR policy documents.

---

## Features

- Upload HR Policy PDF + multiple Invoice PDFs (in ZIP)
- Rule-based + LLM (Groq) logic to evaluate reimbursement status
- Categories: **Meal**, **Cab**, **Travel**
- Status: `Fully Reimbursed`, `Partially Reimbursed`, `Declined`
- Embeds invoice + result using Sentence Transformers
- Stores embeddings in **ChromaDB** (vector store)

---

## Setup Instructions

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On Linux/macOS
```
---
### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
---

### 2. Install OCR Tools

# Tesseract OCR Setup


- [![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-blue)](https://github.com/tesseract-ocr/tesseract)
- [![Poppler](https://img.shields.io/badge/PDF-Poppler-brightgreen)](https://github.com/oschwartz10612/poppler-windows)
- Tesseract is a powerful OCR engine used to extract text from scanned PDFs.
  
---

### Installation

#### Tesseract OCR


- Download: [Tesseract OCR Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Install to: `C:\Program Files\Tesseract-OCR\`

#### Poppler for PDF Conversion


- Download: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- Extract to any location, e.g., `C:\tools\poppler-24.08.0`

---

### Configure Paths in Code

- Paste this into `invoice_analysis.py`:
- pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
- POPPLER_PATH = r"C:\\tools\\poppler-24.08.0\\Library\\bin"

---
### 3. Run the FastAPI Server

- By default, the app will be available at Swagger UI:
```bash
http://127.0.0.1:8000/docs 
```
---

## Vector Store

Each invoice is vectorized and stored using the ChromaDB vector database.
Stored metadata includes:

- Invoice filename (doc_id)

- Employee name

- Status & reason

- Invoice content (embedded)

- Submission date

---

## Project Structure
```bash
.
├── main.py                      # FastAPI app
├── invoice_analysis.py          # OCR + rule-based & Groq logic
├── vectordb.py                  # ChromaDB vector store setup
├── test_api.py                  # API testing script (manual/test)
├── invoice_jupyter_analysis.ipynb  # Notebook for step-by-step logic tests
├── requirements.txt             # Python dependencies
├── .env                         # Groq API Key
└── README.md                    # Project documentation
```
---

##  Results

```bash
{
  "success": true,
  "results": [
    {
      "filename": "Book 3.pdf",
      "employee": "Vaibhav",
      "category": "travel",
      "status": "Partially Reimbursed",
      "reason": "Trip cost ≈ ₹3688 exceeds ₹2000 per‑trip limit."
    },
    {
      "filename": "Book-cab-03.pdf",
      "employee": "Vaibhav",
      "category": "cab",
      "status": "Fully Reimbursed",
      "reason": "Cab fare ≈ ₹141 within ₹150 daily limit."
    },
    {
      "filename": "Meal Invoice 4.pdf",
      "employee": "Vaibhav",
      "category": "meal",
      "status": "Fully Reimbursed",
      "reason": "Meal within ₹200 limit (≈ ₹88), no alcohol."
    }
  ]
}
```
---
