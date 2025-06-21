# Invoice Reimbursement System 🧾

A lightweight and efficient tool to automate the process of uploading, analyzing, and verifying invoices for reimbursement purposes.

---

## 📌 Features

- Upload and parse invoice PDFs/images
- Extract key details (vendor, date, amount, GST, etc.)
- Auto-validation of invoice entries
- Admin panel to review and approve reimbursements
- Database storage for all invoice entries

---

## 🚀 Tech Stack

- **Frontend**: HTML/CSS (or React, if used)
- **Backend**: Python (Flask / Django)
- **Database**: SQLite / PostgreSQL / MySQL
- **OCR**: Tesseract / EasyOCR (for image-to-text)
- **Deployment**: (mention if hosted on Heroku, Render, etc.)

---


## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/vaibhav230104/Invoice_Reimbursement_System.git
cd Invoice_Reimbursement_System

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py            # or flask run / streamlit run main.py

---

