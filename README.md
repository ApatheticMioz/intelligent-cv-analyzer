# Intelligent CV Analyzer

An intelligent CV screening system that implements three classical string-matching algorithms (Brute Force, Rabin-Karp, KMP) to automatically extract skills from resumes and evaluate job fit for recruitment automation.

> **Status:** Archived / Refactored

## Description

This FastAPI-based application analyzes batches of resumes (PDF/DOCX/DOC), extracts text content, and compares candidate skills against predefined job descriptions using string-matching algorithms. It provides a modern web dashboard for visualizing match quality, algorithm performance comparison, and candidate rankings.

**Key Features:**
- Three string-matching algorithms with real-time performance tracking
- Batch CV analysis with candidate ranking by relevance score
- Interactive web dashboard with TailwindCSS and Chart.js visualizations
- Support for PDF and DOCX document formats
- Algorithm performance comparison (execution time, character comparisons)
- Matched/missing keyword breakdowns per candidate

## Project Structure

```
intelligent-cv-analyzer/
├── main.py                 # FastAPI application and algorithm implementations
├── index.html              # Interactive web dashboard
├── requirements.txt        # Python dependencies
├── job_descriptions/       # Predefined job role templates
│   ├── AI_ML_Engineer_JD.txt
│   ├── Data_Scientist_JD.txt
│   └── Full_Stack_Web_Developer_JD.txt
├── images/                 # Documentation images
├── LICENSE                 # MIT License
├── CONTRIBUTING.md         # Contribution guidelines
├── CHANGELOG.md            # Version history
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ApatheticMioz/intelligent-cv-analyzer.git
   cd intelligent-cv-analyzer
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare CV data:
   - Create a `DataSet/` folder in the project root
   - Add CV files (PDF/DOCX/DOC format)
   - Filenames should follow pattern: `##x####` (e.g., `23i2523.pdf`)

## Usage

1. Start the server:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

3. Use the web interface to:
   - Click **Load DataSet** to import CVs from the DataSet folder
   - Select a job description (AI/ML Engineer, Data Scientist, or Full-Stack Developer)
   - Click **Analyze All CVs** to run the analysis
   - View ranked results and algorithm performance metrics

### Development Mode

For auto-reload during development:
```bash
uvicorn main:app --reload
```

## Tech Stack

- **Backend:** Python 3.8+, FastAPI, Uvicorn
- **Frontend:** HTML5, TailwindCSS (CDN), Vanilla JavaScript, Chart.js
- **Document Parsing:** PyPDF2, python-docx

## Algorithms Implemented

| Algorithm | Time Complexity | Space Complexity | Best Use Case |
|-----------|-----------------|------------------|---------------|
| Brute Force | O(n×m) | O(1) | Short patterns, simple implementation |
| Rabin-Karp | O(n+m) avg | O(1) | Multiple pattern searches |
| KMP | O(n+m) | O(m) | Guaranteed linear performance |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
