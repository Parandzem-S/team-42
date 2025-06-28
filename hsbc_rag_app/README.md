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

A powerful Streamlit application that analyzes banking documents using Microsoft's Phi-4 AI model with **live-only analysis**. Extract insights from PDF transcripts, perform real-time Q&A analysis, and interact with your banking data through an intelligent chat interface.

## Team members

- Garen Arevian
- Geoffrey Payne
- Daniela M
- Mircea Adrian Guinea
- Parandzem Sargsyan

## ✨ Key Features

- 📄 **PDF Processing**: Extract Q&A data from HSBC transcript PDFs
- 🧠 **Live AI Analysis**: Real-time Phi-4 powered insight generation (no pre-computed results)
- 💬 **Interactive Chat**: Ask questions about your banking data
- 📊 **Data Visualization**: Interactive charts and analysis dashboards
- 💾 **Export Options**: Download processed data in CSV, JSON, or Excel formats
- 🌐 **Multi-Bank Analysis**: Compare across 7 major banks with optimized performance

## 🔥 Live Analysis Only

**This platform exclusively uses live AI processing:**
- ✅ All analysis runs in real-time using Phi-4 models
- ✅ No pre-computed or cached analysis results
- ✅ Fresh insights generated for each analysis run
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

## ⚡ Performance Optimizations

- **Smart Caching**: Data is cached locally for faster subsequent loads
- **Parallel Downloads**: Multiple bank files downloaded simultaneously  
- **Efficient Processing**: Vectorized operations and optimized algorithms
- **Live Processing**: Real-time AI inference with progress tracking
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

## 🎯 Live AI Analysis Features

- **Real-time Insights**: Generated fresh for each analysis run
- **Risk Assessment**: Live Yes/No/Unclear risk detection
- **Answer Coverage**: Real-time Complete/Partial/None evaluation
- **Financial Metrics**: Live extraction of metrics, trends, and status
- **Interactive Chat**: Natural language queries with live AI responses

## 🔒 Privacy & Security

- **Local Processing**: All AI inference happens locally
- **No External APIs**: Data never leaves your environment
- **Live Analysis**: No pre-computed results stored or shared
- **Temporary Storage**: Files automatically cleaned after processing

## 📁 Project Structure

```
hsbc_rag_app/
├── app.py                     # Main Streamlit application (live analysis only)
├── phi4_agent.py              # Phi-4 Method 1 AI model integration
├── phi4_method2_agent.py      # Phi-4 Method 2 AI model integration
├── pdf_processor.py           # PDF extraction logic
├── multi_bank_extractor.py    # Multi-bank data processing
├── data_extractor.py          # Data processing utilities
├── config.py                  # Configuration settings (live analysis only)
└── requirements.txt           # Python dependencies
```

## 💡 Usage Tips

- **First Time**: Model download may take 5-15 minutes
- **Live Analysis**: Each analysis runs fresh - no cached results
- **Performance**: Analysis time varies by data size and complexity
- **Method 2**: First load downloads data, subsequent loads are cached
- **Cache**: Clear cache if you encounter data issues

## 🧠 About the Technology

- **AI Model**: Microsoft Phi-4-mini-instruct
- **Processing**: Live analysis only - no pre-computed results
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Architecture**: Optimized for real-time AI processing

## 🔄 Live Analysis Workflow

1. **Data Input**: Upload PDFs or select pre-processed data
2. **Processing**: Extract and prepare data for analysis
3. **Live AI Analysis**: Real-time Phi-4 model processing
4. **Results Display**: Fresh insights displayed immediately
5. **Auto-Save**: Results automatically saved to files
6. **Export**: Download processed data and live analysis results

## 🎓 Academic Context

**Team 42 - Cambridge University & Bank of England Collaboration**

This project demonstrates:
- Real-time AI analysis capabilities in financial contexts
- Transparent, live model processing without pre-computation
- Advanced natural language processing for banking documents
- Cross-institutional collaborative research methodologies

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the application**: `streamlit run app.py`
4. **Select your method** in the sidebar
5. **Process your data** with the method-specific controls
6. **Run live AI analysis** for fresh insights
7. **Explore results** and chat with the AI

---

**Note**: This platform exclusively uses live AI processing. All analysis results are generated in real-time using Phi-4 models, ensuring transparent and fresh insights for every analysis run.