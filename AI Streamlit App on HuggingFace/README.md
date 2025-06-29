---
title: Multi-Bank AI Financial Analysis Platform
emoji: 🏦
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.46.0
app_file: app.py
pinned: false
suggested_hardware: t4-small 
license: mit
---

# 🏦 Team 42 - Cambridge - Bank of England Student Project

## Multi-Bank AI Financial Analysis Platform

A powerful Streamlit application that analyzes banking documents using Microsoft's Phi-4 AI model with **live-only analysis** and semantic communication pattern insights. Extract insights from PDF transcripts, perform real-time Q&A analysis, and explore semantic patterns in banking communication.

## Team members

- **Daniela Miccoli**
- Garen Arevian
- **Geoffrey Payne**
- **Adrian Mircea Guinea**
- **Parandzem Sargsyan**

## ✨ Key Features

- 📄 **PDF Processing**: Extract Q&A data from HSBC transcript PDFs
- 🧠 **Live AI Analysis**: Real-time Phi-4 powered insight generation (no pre-computed results)
- 📊 **Semantic Analysis**: Pre-computed communication pattern analysis
- 💬 **Interactive Chat**: Ask questions about your banking data
- 📊 **Data Visualization**: Interactive charts and analysis dashboards
- 💾 **Export Options**: Download processed data in CSV, JSON, or Excel formats
- 🌐 **Multi-Bank Analysis**: Compare across 7 major banks with optimized performance

## 🔥 Analysis Methods

**This platform offers three distinct analysis approaches:**
- ✅ **Methods 1 & 2**: Live AI processing using Phi-4 models
- ✅ **Method 3**: Pre-computed semantic analysis insights
- ✅ Fresh insights generated for each analysis run (Methods 1 & 2)
- ✅ Transparent, live model processing
- ✅ Results automatically saved after each analysis

## 🚀 Quick Start

### Method 1 (HSBC PDF Analysis):
1. Upload HSBC transcript PDFs directly
2. Extract Q&A data using PDF processing  
3. **Run live AI analysis** focused on risk detection
4. Real-time insight generation and answer coverage
5. Detailed transparency analysis

### Method 2 (Multi-Bank Analysis):
1. Pre-processed multi-bank transcript data access
2. Filter by bank, year, quarter (multi-select)
3. **Run live AI analysis** for financial metrics extraction
4. Real-time cross-bank comparative analysis

### Method 3 (Semantic Analysis):
1. **NEW**: Load semantic and risk analysis based on answer completeness and risk detection from method 1
2. Explore communication patterns in banking discourse
3. Risk & distress communication analysis
4. Answer coverage and completeness patterns
5. Statistical insights into rhetorical strategies

## ⚡ Performance Optimizations

- **Smart Caching**: Data is cached locally for faster subsequent loads
- **Parallel Downloads**: Multiple bank files downloaded simultaneously  
- **Efficient Processing**: Vectorized operations and optimized algorithms
- **Live Processing**: Real-time AI inference with progress tracking (Methods 1 & 2)
- **Instant Access**: Pre-computed semantic insights (Method 3)
- **Memory Management**: Optimized batch processing for large datasets

## 🔧 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Supported Banks (Method 2)

- Barclays (BARC)
- Credit Suisse/UBS (CS)
- Deutsche Bank (DB) 
- HSBC (HSBC)
- JP Morgan (JPM)
- Santander (SAN)
- UniCredit (UC)

## 🎯 Analysis Features

**Live AI Processing (Methods 1 & 2):**
- **Real-time Insights**: Generated fresh for each analysis run
- **Risk Assessment**: Live Yes/No/Unclear risk detection
- **Answer Coverage**: Real-time Complete/Partial/None evaluation
- **Financial Metrics**: Live extraction of metrics, trends, and status
- **Interactive Chat**: Natural language queries with live AI responses

**Semantic Analysis (Method 3):**
- **Communication Patterns**: Pre-computed analysis of rhetorical strategies
- **Risk Communication**: How banks communicate uncertainty and distress
- **Answer Quality**: Linguistic markers of complete vs. incomplete responses
- **Hedging Analysis**: Use of uncertainty and conditional language
- **Statistical Insights**: Quantitative analysis of communication trends

## 🔒 Privacy & Security

- **Local Processing**: All AI inference happens locally (Methods 1 & 2)
- **No External APIs**: Data never leaves your environment (when run locally)
- **Live Analysis**: No pre-computed results stored or shared (Methods 1 & 2)
- **GitHub Data**: Method 3 fetches public research data from GitHub
- **Temporary Storage**: Files automatically cleaned after processing

## 📁 Project Structure

```
AI Streamlit App on HuggingFace/
├── app.py                     # Main Streamlit application (live analysis + semantic insights)
├── phi4_agent.py              # Phi-4 Method 1 AI model integration
├── phi4_method2_agent.py      # Phi-4 Method 2 AI model integration
├── pdf_processor.py           # PDF extraction logic
├── multi_bank_extractor.py    # Multi-bank data processing
├── data_extractor.py          # Data processing utilities
├── config.py                  # Configuration settings (live analysis + Method 3)
├── requirements.txt           # Python dependencies
├── Semantic_Analysis_by_Risk_Distress.csv      # Method 3 risk communication data
└── Semantic_Analysis_by_Answer_Coverage.csv    # Method 3 answer quality data
```

## 💡 Usage Tips

- **First Time**: Model download may take 5-15 minutes (Methods 1 & 2)
- **Live Analysis**: Each analysis runs fresh - no cached results (Methods 1 & 2)
- **Method 3**: Instant data loading from GitHub - no model required
- **Performance**: Analysis time varies by data size and complexity
- **Method 2**: First load downloads data, subsequent loads are cached
- **Cache**: Clear cache if you encounter data issues

## 🧠 About the Technology

- **AI Model**: Microsoft Phi-4-mini-instruct (Methods 1 & 2)
- **Processing**: Live analysis + pre-computed semantic insights
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Data Sources**: PDF extraction, multi-bank datasets, GitHub CSV files
- **Architecture**: Optimized for real-time AI processing and semantic analysis

## 🔄 Analysis Workflows

**Live Analysis Workflow (Methods 1 & 2):**
1. **Data Input**: Upload PDFs or select pre-processed data
2. **Processing**: Extract and prepare data for analysis
3. **Live AI Analysis**: Real-time Phi-4 model processing
4. **Results Display**: Fresh insights displayed immediately
5. **Auto-Save**: Results automatically saved to files
6. **Export**: Download processed data and live analysis results

**Semantic Analysis Workflow (Method 3):**
1. **Data Loading**: Semantic and risk analysis based on answer completeness and risk detection from method 1
2. **Pattern Exploration**: Browse communication insights
3. **Statistical Review**: Examine quantitative findings
4. **Export**: Download semantic analysis datasets

## 🎓 Academic Context

**Team 42 - Cambridge University & Bank of England Collaboration**

This project demonstrates:
- Real-time AI analysis capabilities in financial contexts
- Transparent, live model processing without pre-computation
- Advanced natural language processing for banking documents
- **NEW**: Semantic analysis of financial communication patterns
- Cross-institutional collaborative research methodologies
- **Communication Strategy Analysis**: How banks use language to manage stakeholder expectations

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the application**: `streamlit run app.py`
4. **Select your method** in the sidebar:
   - Method 1: HSBC PDF analysis
   - Method 2: Multi-bank financial metrics
   - **Method 3: Semantic and risk analysis dashboard**
5. **Process your data** with the method-specific controls
6. **Run live AI analysis** for fresh insights (Methods 1 & 2)
7. **Explore semantic patterns** and statistical findings (Method 3)
8. **Export results** and chat with the AI

---

**Note**: This platform offers both live AI processing (Methods 1 & 2) and semantic analysis (Method 3). All live analysis results are generated in real-time using Phi-4 models, while Method 3 provides semantic and risk analysis based on answer completeness and risk detection patterns for banking communication research.
