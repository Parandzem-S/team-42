"""
Configuration file for Team 42 - Cambridge - Bank of England Student Project
Multi-Bank AI Financial Analysis Platform
Updated to include Method 3 semantic analysis functionality
"""

# ===== STREAMLIT APP CONFIGURATION =====
STREAMLIT_CONFIG = {
    "page_title": "Team 42 - Cambridge - Bank of England Student Project",
    "page_icon": "🏦", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ===== UI COLORS (Original Theme without Pink) =====
UI_CONFIG = {
    "primary_color": "#DC143C",  # HSBC Red (original)
    "secondary_color": "#B71C1C",  # Darker Red (original)
    "accent_color": "#FF6B35",  # Orange accent (was pink, now orange)
    "background_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",  # Original blue-purple
    "method_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",  # Blue-purple instead of pink
    "button_gradient": "linear-gradient(135deg, #DC143C 0%, #B71C1C 100%)",  # Red gradient
    "text_color": "#333333",  # Original dark gray
    "sidebar_bg": "#F5F5F5"  # Light gray
}

# ===== BASIC FILE LIMITS =====
PDF_CONFIG = {
    "max_file_size_mb": 50,
    "max_files_per_upload": 10
}

# ===== METHOD 2 BANK NAMES =====
BANK_NAMES = {
    'barc': 'Barclays',
    'cs': 'Credit Suisse/UBS', 
    'db': 'Deutsche Bank',
    'hsbc': 'HSBC',
    'jpm': 'JP Morgan',
    'san': 'Santander',
    'uc': 'UniCredit'
}

# ===== METHOD 3 CONFIGURATION ===== 
# 🔄 CHANGE #1: UPDATED GitHub URLs to use new path
METHOD3_CONFIG = {
    "risk_distress_url": "https://raw.githubusercontent.com/Parandzem-S/team-42/main/AI%20Streamlit%20App%20on%20HuggingFace/Semantic_Analysis_by_Risk_Distress.csv",
    "answer_coverage_url": "https://raw.githubusercontent.com/Parandzem-S/team-42/main/AI%20Streamlit%20App%20on%20HuggingFace/Semantic_Analysis_by_Answer_Coverage.csv",
    "description": "Semantic Analysis Dashboard",
    "features": [
        "Semantic and risk analysis based on answer completeness and risk detection from method 1",
        "Risk & distress communication patterns",
        "Answer coverage and completeness analysis",
        "Modal verbs, conditional phrases, and confidence terms",
        "Superficial analysis of specific expressions in answer texts",
        "Hedging terms and uncertainty indicators",
        "Statistical insights into communication strategies"
    ]
}

# ===== ANALYSIS CONFIGURATION =====
ANALYSIS_CONFIG = {
    "live_analysis_only": True,  # Force live analysis only
    "enable_result_caching": True,  # Cache results for performance
    "save_results_automatically": True,  # Auto-save results to files
    "max_qa_pairs_display": 50,  # Maximum Q&A pairs to display in UI
    "default_qa_pairs_display": 10,  # Default number of Q&A pairs to show
    "method3_data_cache_duration": 3600  # Cache Method 3 data for 1 hour (seconds)
}

# ===== METHOD DESCRIPTIONS =====
METHOD_DESCRIPTIONS = {
    "method1": {
        "name": "Phi-4 Method 1: HSBC PDF Processor",
        "icon": "📄",
        "features": [
            "Upload HSBC transcript PDFs directly",
            "Extract Q&A data using PDF processing",
            "Select number of Q&A pairs for analysis",
            "AI analysis focused on risk detection",
            "Insight generation and answer coverage",
            "Detailed transparency analysis"
        ]
    },
    "method2": {
        "name": "Phi-4 Method 2: Multi-Bank Financial Metrics Extractor",
        "icon": "🏦",
        "features": [
            "Pre-processed multi-bank transcript data access",
            "Filter by bank, year, quarter (multi-select)",
            "Select number of Q&A pairs for analysis",
            "Detects and extracts financial metrics discussed",
            "Captures metric values, trends (increase/decrease/stable)",
            "Determines if questions were answered or avoided",
            "Cross-bank comparative financial analysis"
        ]
    },
    "method3": {
        "name": "Method 3: Semantic Analysis Dashboard",
        "icon": "📊",
        "features": METHOD3_CONFIG["features"]
    }
}

# That's it! All analysis is now live-only with no pre-computed result loading, plus Method 3 semantic analysis