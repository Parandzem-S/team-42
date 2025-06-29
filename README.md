# 🏦 Team 42 — Cambridge BoE Project

A multi-bank AI-powered financial analysis platform developed for the Bank of England Student Project by Team 42 (University of Cambridge).  
This repository features tools and applications for extracting, processing, and analyzing financial communication data from major banks using advanced language models and data pipelines.

---

## 🚀 Features

- **AI-Driven Analysis**:  
  Analyze banking earnings transcripts, Q&A sessions, and presentations using Large Language Models (LLMs) such as Microsoft's Phi-4 and OpenAI's GPT-4.
- **Multi-Bank Support**:  
  Unified data processing and extraction pipelines for major banks (e.g., JPMorgan, HSBC, Deutsche Bank).
- **Live Document Processing**:  
  Real-time Q&A and semantic analysis of PDF transcripts; no pre-computed results or persistent data storage.
- **Streamlit Application**:  
  User-friendly interface for uploading documents, running live analysis, and exploring semantic insights.
- **Privacy & Security**:  
  All inference runs locally (for most methods); no data sent to external APIs unless explicitly configured.

---

## 🏗️ Project Structure

```
AI Streamlit App on HuggingFace/
├── AI Streamlit App on HuggingFace/    # Streamlit-based front-end app & model integration
├── Deutsche_Bank/                      # Deutsche Bank processing scripts & data
├── Final Codes/                        # Consolidated/cleaned Jupyter notebooks for data processing
├── HSBC/                               # HSBC-specific scripts and presentations
├── JPM/                                # JPMorgan-specific scripts and data
├── Json Conversion/                    # Utilities for prompt/keyword analysis and JSON workflows
├── Testing/                            # Experimental scripts, notebooks, and test routines
├── output/                             # Output files (generated reports, CSVs, processed data)
├── .github/                            # GitHub configuration and workflows
├── README.md                           # This file
└── ...                                 # Other files and folders as needed
```

### Folder Descriptions

- **AI Streamlit App on HuggingFace/** — Main Streamlit application for live document analysis, semantic exploration, and LLM integration.
- **Deutsche_Bank/** — Notebooks, scripts, and data pipelines for Deutsche Bank earnings and Q&A analysis.
- **Final Codes/** — Clean, unified Jupyter notebooks for processing and extracting insights across all banks.
- **HSBC/** — Scripts, data, and presentation files specific to HSBC.
- **JPM/** — JPMorgan scripts, extracted data, and processing pipelines.
- **Json Conversion/** — Utilities for prompt management, JSON transformation, and keyword checking.
- **Testing/** — Experimental notebooks and scripts for prototyping and validation.
- **output/** — Folder for generated outputs (reports, CSVs, etc.).
- **.github/** — Repository configuration, workflows, and issue templates.

---

## ⚙️ Technologies Used

- **Python** (core language)
- **Jupyter Notebook** (data processing workflows)
- **Streamlit** (interactive dashboard)
- **Transformers (HuggingFace)**, **PyTorch** (LLM integration)
- **Pandas, NumPy** (data manipulation)
- **NLTK, FinBERT** (natural language & sentiment analysis)
- **PyPDF2, openpyxl, pdfplumber** (document parsing)

---

## 🧑‍💻 Setup & Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/Parandzem-S/team-42.git
   cd team-42
   ```

2. **Install dependencies (for Streamlit App):**
   ```bash
   cd "AI Streamlit App on HuggingFace"
   pip install -r requirements.txt
   ```

3. **Run Streamlit App:**
   ```bash
   streamlit run app.py
   ```

4. **For Data Processing Notebooks:**
   - Open any notebook in `Final Codes/`, `Deutsche_Bank/`, or other relevant subdirectories using Jupyter Lab/Notebook or Google Colab.
   - Install required libraries as prompted in each notebook (e.g., `pip install pandas pdfplumber openpyxl`).

5. **API Keys (if using GPT-4):**
   - Set your OpenAI API key as an environment variable:  
     `export OPENAI_API_KEY=your_openai_key_here`

---

## 📊 Example Use Cases

- **Upload a PDF transcript.**  
  Get instant semantic summaries, Q&A extraction, and risk communication analysis.
- **Compare communication patterns across banks**  
  using unified data outputs and semantic metrics.
- **Perform sentiment and coverage analysis**  
  on Q&A sections using FinBERT and custom LLM prompts.

---

## 👥 Team Members

- Daniela Miccoli
- Garen Arevian
- Geoffrey Payne
- Adrian Mircea Guinea
- Parandzem Sargsyan

---

## 🔒 Privacy & Security

- All AI inference (Methods 1 & 2) runs locally.
- No data leaves your environment unless you use Method 3 (fetches public research data from GitHub).
- Temporary files are automatically cleaned after processing.

---

## 📄 License

_This project is for academic research and demonstration purposes. Please contact the maintainers for usage or collaboration inquiries._

---

## 🌐 More Info

See detailed app documentation in  
[AI Streamlit App on HuggingFace/README.md](AI%20Streamlit%20App%20on%20HuggingFace/README.md)

---
