# Intelligent CV Analyzer

FastAPI-based CV screening tool that loads batches of resumes (PDF/DOCX/DOC), extracts text, and compares skills against predefined job descriptions using three classic string-matching algorithms (Brute Force, Rabin-Karp, KMP). The project includes a Tailwind/Chart.js dashboard for visualizing match quality and algorithm performance.

## Key Features
- Batch CV ingestion with filename validation and duplicate/latest-submission filtering
- Text extraction for PDF and Word documents with basic error handling
- Keyword matching via Brute Force, Rabin-Karp, and KMP with per-algorithm metrics
- Ranked candidate results plus matched/missing keyword breakdowns
- Dataset loader endpoint and interactive web UI for analysis and charts

## Tech Stack
- Backend: Python 3.8+, FastAPI, Uvicorn
- Frontend: HTML, TailwindCSS CDN, vanilla JS, Chart.js
- Parsing: python-docx, PyPDF2

## Repository Layout
- 23i-2523_Algo_Asst-2_DS-A/Application/main.py — FastAPI application and string-matching logic
- 23i-2523_Algo_Asst-2_DS-A/Application/index.html — Interactive dashboard
- 23i-2523_Algo_Asst-2_DS-A/Application/requirements.txt — Python dependencies
- 23i-2523_Algo_Asst-2_DS-A/Job_Descriptions/ — Three role templates with keywords
- DataSet/ — Place CV files here (ignored by git)

## Setup
1) (Optional) Create and activate a virtual environment
```
python -m venv .venv
./.venv/Scripts/activate
```
2) Install dependencies
```
pip install -r 23i-2523_Algo_Asst-2_DS-A/Application/requirements.txt
```
3) Add input data
- Drop CV files into DataSet/ (PDF/DOCX/DOC). Filenames must follow the pattern `##x####` (e.g., `23i2523.pdf`).
- Job description keyword lists live in 23i-2523_Algo_Asst-2_DS-A/Job_Descriptions/.

## Run
From the Application directory, start the API server:
```
cd 23i-2523_Algo_Asst-2_DS-A/Application
python main.py
```
Open the UI at http://127.0.0.1:8000 and use **Load DataSet** or upload CVs manually, then select a job description and analyze.

## Notes
- DataSet/ is git-ignored to avoid committing large or sensitive CV files.
- If you prefer auto-reload during development, you can run `uvicorn main:app --reload` from the Application directory.
