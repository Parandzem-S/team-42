"""
Simple Configuration file for HSBC Banking AI Agent
Contains only basic settings needed for Streamlit - your core code remains unchanged
"""

# ===== STREAMLIT APP CONFIGURATION =====
STREAMLIT_CONFIG = {
    "page_title": "HSBC Banking AI Agent",
    "page_icon": "🏦", 
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# ===== UI COLORS (HSBC Brand) =====
UI_CONFIG = {
    "primary_color": "#DC143C",  # HSBC Red
    "secondary_color": "#B71C1C",
    "background_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "text_color": "#333333"
}

# ===== BASIC FILE LIMITS =====
PDF_CONFIG = {
    "max_file_size_mb": 50,
    "max_files_per_upload": 10
}

# That's it! Your original code in phi4_agent.py and pdf_processor.py remains exactly the same
# No changes to your model loading, generation args, or processing logic