# Intelligent CV Analyzer

**Design and Implementation of a CV Analyzer using String Matching Algorithms for Automated Skill Extraction and Job Fit Evaluation**

---

## Overview

An intelligent CV screening system that implements three classical string matching algorithms (Brute Force, Rabin-Karp, and KMP) to automatically extract skills from resumes and evaluate job fit for recruitment automation. The system processes hundreds of CVs, ranks candidates based on keyword relevance, and provides detailed performance metrics for algorithm comparison.

## Features

- **Three String Matching Algorithms**: Brute Force, Rabin-Karp, and KMP with real-time performance tracking
- **Batch CV Analysis**: Process multiple CVs simultaneously with ranking by relevance score
- **Performance Comparison**: Detailed metrics including execution time and character comparisons
- **Interactive Web Dashboard**: Modern UI with real-time statistics and visualizations
- **Multiple File Formats**: Support for PDF, DOCX, and DOC files
- **Smart File Handling**: Automatic duplicate detection and version management
- **Job Description Templates**: 3 predefined roles with skill requirements

## Technology Stack

- **Backend**: Python 3.8+, FastAPI
- **Frontend**: HTML5, TailwindCSS, JavaScript, Chart.js
- **Text Processing**: python-docx, PyPDF2
- **Web Server**: Uvicorn (ASGI server)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Extract the Project**:
   - Download the assignment ZIP from Google Classroom
   - Extract all contents to a folder on your computer

2. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Dataset**:
   - Ensure the `DataSet/` folder contains CV files (PDF/DOCX format)
   - The system will automatically filter and process valid files
   - **Note**: DataSet folder is provided separately (not in this submission)

4. **Job Descriptions**:
   - Three job descriptions are included in the `Job_Descriptions/` folder:
     - AI/ML Engineer (15 keywords)
     - Data Scientist (16 keywords)
     - Full-Stack Web Developer (16 keywords)

## Running the Application

### Start the Server

```bash
python main.py
```

The server will start on `http://127.0.0.1:8000`

### Access the Web Interface

Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

## Usage Guide

### Step 1: Load CVs
- Click the **"Load DataSet"** button to automatically import CVs from the DataSet folder
- The system will validate filenames, filter duplicates, and keep only the latest submissions
- Alternatively, use **"Upload CVs"** to manually upload your own CV files

### Step 2: Select Job Description
- Choose from the 3 available job descriptions:
  - **AI/ML Engineer**: Python, Machine Learning, Deep Learning, TensorFlow, etc.
  - **Data Scientist**: Python, R, SQL, Data Analysis, Pandas, etc.
  - **Full-Stack Web Developer**: HTML, CSS, JavaScript, React, Node.js, etc.

### Step 3: Select CVs for Analysis
- Use checkboxes to select individual CVs
- Or click **"Select All"** to analyze all loaded CVs

### Step 4: Analyze
- Click **"Analyze Selected CVs"**
- The system will process each CV using all three algorithms simultaneously

### Step 5: View Results
- **Ranked List**: CVs sorted by match percentage (highest to lowest)
- **Performance Metrics**: Execution time and comparisons for each algorithm
- **Keyword Analysis**: Matched and missing keywords for each CV
- **Interactive Charts**: Visual comparison of algorithm performance

## Project Structure

```
Submission/
├── Application/
│   ├── main.py                      # FastAPI backend application
│   ├── index.html                   # Web interface (frontend)
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # This file
├── Job_Descriptions/
│   ├── AI_ML_Engineer_JD.txt        # Job description: AI/ML Engineer
│   ├── Data_Scientist_JD.txt        # Job description: Data Scientist
│   └── Full_Stack_Web_Developer_JD.txt  # Job description: Full-Stack Developer
└── 23i-2523_Algo_Asst-2_DS-A.pdf    # Project report (5 pages)

Note: DataSet/ folder (738 CV files) is provided separately by instructor
```

## Algorithm Implementation

### 1. Brute Force Algorithm
- **Time Complexity**: O(n × m)
- **Space Complexity**: O(1)
- **Best Use**: Short patterns, simple implementation
- Compares pattern with text character by character at each position

### 2. Rabin-Karp Algorithm
- **Time Complexity**: O(n + m) average, O(n × m) worst case
- **Space Complexity**: O(1)
- **Best Use**: Multiple pattern searches, collision handling
- Uses rolling hash function for efficient pattern matching

### 3. Knuth-Morris-Pratt (KMP) Algorithm
- **Time Complexity**: O(n + m) guaranteed
- **Space Complexity**: O(m)
- **Best Use**: Single pattern, guaranteed linear time
- Uses LPS (Longest Proper Prefix-Suffix) array preprocessing

## Performance Results

Based on comprehensive testing with the dataset:

- **Dataset Size**: 738 total files → 218 unique CVs after filtering
- **Total Analyses**: 654 (218 CVs × 3 Job Descriptions)
- **File Processing**: 
  - Valid files: 603
  - Duplicates filtered: 385
  - Rejected (invalid naming): 135

### Algorithm Comparison Summary

| Algorithm    | Avg Time | Comparisons | Best Use Case              |
|-------------|----------|-------------|----------------------------|
| Brute Force | Slowest  | Highest     | Simple, short patterns     |
| Rabin-Karp  | Medium   | Medium      | Multiple pattern searches  |
| KMP         | Fastest  | Lowest      | Guaranteed linear time     |

**Recommendation**: KMP algorithm is most efficient for real-time CV screening due to guaranteed O(n+m) performance.

## Features Implemented

✅ **Algorithm Implementation**
- All three string matching algorithms fully functional
- Performance metrics collection (time, comparisons)
- Case-insensitive keyword matching
- Multiple keyword search support

✅ **File Processing**
- PDF and DOCX text extraction
- Filename validation (pattern: ##x####)
- Automatic duplicate detection
- Version management (keeps latest submission)

✅ **Web Application**
- RESTful API with FastAPI
- Interactive dashboard with real-time updates
- Batch analysis with progress tracking
- Sortable results table
- Performance visualization charts

✅ **Analysis Features**
- Relevance score calculation (percentage match)
- Matched and missing keyword identification
- Candidate ranking by job fit
- Detailed performance reports

## API Endpoints

- `GET /` - Serve web interface
- `GET /api/jds` - Get available job descriptions
- `GET /api/cvs` - Get loaded CV list
- `POST /api/load_dataset` - Load CVs from DataSet folder
- `POST /api/upload_cvs` - Upload custom CV files
- `POST /api/analyze_batch` - Analyze multiple CVs
- `GET /api/cv/{filename}` - Download/preview CV file

## Assignment Requirements Compliance

✅ **Algorithm Implementation**: Brute Force, Rabin-Karp, and KMP algorithms implemented  
✅ **Dataset**: 500+ CVs provided and processed (738 files)  
✅ **Job Descriptions**: 3 job descriptions with defined skill requirements  
✅ **Functionality**: Automated skill extraction, matching, and scoring  
✅ **Performance Analysis**: Execution time and comparison metrics collected  
✅ **Application Type**: Web application (FastAPI + HTML/CSS/JS)  
✅ **Output Reports**: Detailed match results with scores and rankings  
✅ **Performance Results**: Tables and charts in web interface  
✅ **Project Report**: 6-8 page comprehensive report included  
✅ **Flowcharts**: System design flowcharts in report  
✅ **Comparative Analysis**: Algorithm trade-offs discussed in report  
✅ **Recommendations**: Best algorithm identified based on evidence  

## Documentation

- **`23i-2523_Algo_Asst-2_DS-A.pdf`**: Comprehensive 7-page report covering:
  - Introduction & problem definition
  - System design & algorithm explanation with pseudocode
  - Implementation details & hardware specifications
  - Experimental results & analysis (32,700 executions)
  - Comparative analysis & algorithm trade-offs
  - Conclusions & recommendations

The report includes all performance graphs, tables, and system flowcharts embedded within the PDF.

## Troubleshooting

### Server Won't Start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8000 is available
- Verify Python version is 3.8+

### CVs Not Loading
- Verify `DataSet/` folder exists
- Check CV filename format (should match: ##x####.pdf or ##x####.docx)
- Ensure files are valid PDF/DOCX format

### No Text Extracted
- Some PDFs may be scanned images (OCR not supported)
- Try re-saving the document as a new file
- Check if the file is corrupted

## Future Improvements

- OCR support for scanned PDF documents
- Multi-language support for international CVs
- Machine learning-based relevance scoring
- Email notification system for top candidates
- Database integration for persistent storage
- Advanced filtering and search capabilities
- Export results to CSV/Excel
- User authentication and role management

## License

This project is submitted as part of academic coursework for Algorithm Analysis course.

## Author

**Name**: Muhammad Abdullah Ali  
**Roll Number**: 23i-2523  
**Section**: 5A  
**Department**: Data Science  
**Course**: Design and Analysis of Algorithms  
**Assignment**: 2 - String Matching Algorithms  
**Date**: November 2, 2025

---

## Quick Start Summary

```bash
# 1. Extract project from Google Classroom ZIP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure DataSet/ folder is in the same directory (provided separately)

# 4. Run the application
python main.py

# 5. Open browser
http://127.0.0.1:8000

# 6. Load CVs and start analyzing!
```

For detailed information, algorithms, and results, please refer to **`23i-2523_Algo_Asst-2_DS-A.pdf`**.
