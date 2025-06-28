"""
Configuration file for Team 42 - Cambridge - Bank of England Student Project
Multi-Bank AI Financial Analysis Platform
Updated to remove pre-computed analysis loading functionality
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

# ===== ANALYSIS CONFIGURATION =====
ANALYSIS_CONFIG = {
    "live_analysis_only": True,  # Force live analysis only
    "enable_result_caching": True,  # Cache results for performance
    "save_results_automatically": True,  # Auto-save results to files
    "max_qa_pairs_display": 50,  # Maximum Q&A pairs to display in UI
    "default_qa_pairs_display": 10  # Default number of Q&A pairs to show
}

# That's it! All analysis is now live-only with no pre-computed result loading