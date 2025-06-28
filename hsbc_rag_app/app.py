import streamlit as st
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re

# Import your custom modules
from pdf_processor import extract_qa_from_pdf, is_internal_speaker, is_external_speaker
from phi4_agent import Phi4BankingAgent
from phi4_method2_agent import Phi4Method2Agent
from multi_bank_extractor import MultiBankDataExtractor
from data_extractor import DataExtractor
from config import STREAMLIT_CONFIG, UI_CONFIG, BANK_NAMES

# Configure Streamlit page
STREAMLIT_CONFIG['page_title'] = "Team 42 - Cambridge - Bank of England Student Project"
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS for better styling with larger method selection buttons
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        color: {UI_CONFIG['primary_color']};
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        line-height: 1.2;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {UI_CONFIG['text_color']};
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }}
    .metric-card {{
        background: {UI_CONFIG['background_gradient']};
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }}
    .method-card {{
        background: {UI_CONFIG['method_gradient']};
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        border: 2px solid transparent;
    }}
    .method-card.selected {{
        border: 2px solid {UI_CONFIG['accent_color']};
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
    }}
    .chat-message {{
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {UI_CONFIG['primary_color']};
    }}
    .stButton > button {{
        background: {UI_CONFIG['button_gradient']};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .sidebar .stRadio > div {{
        background-color: {UI_CONFIG['sidebar_bg']};
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }}
    .sidebar .stRadio > div > label {{
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.5rem;
    }}
    .bank-info {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }}
    .metric-bullet {{
        background: #f0f8ff;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-left: 3px solid #4CAF50;
        border-radius: 3px;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data_method1' not in st.session_state:
    st.session_state.processed_data_method1 = None
if 'processed_data_method2' not in st.session_state:
    st.session_state.processed_data_method2 = None
if 'phi4_agent_method1' not in st.session_state:
    st.session_state.phi4_agent_method1 = None
if 'phi4_agent_method2' not in st.session_state:
    st.session_state.phi4_agent_method2 = None
if 'phi4_method2_model_cache' not in st.session_state:
    st.session_state.phi4_method2_model_cache = None
if 'multi_bank_extractor' not in st.session_state:
    st.session_state.multi_bank_extractor = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results_method1' not in st.session_state:
    st.session_state.analysis_results_method1 = None
if 'analysis_results_method2' not in st.session_state:
    st.session_state.analysis_results_method2 = None
if 'selected_method' not in st.session_state:
    st.session_state.selected_method = "method1"
if 'selected_banks_display' not in st.session_state:
    st.session_state.selected_banks_display = []
if 'selected_years_display' not in st.session_state:
    st.session_state.selected_years_display = []
if 'selected_quarters_display' not in st.session_state:
    st.session_state.selected_quarters_display = []

def initialize_phi4_agent_method1():
    """Initialize Phi-4 agent Method 1"""
    if st.session_state.phi4_agent_method1 is None:
        with st.spinner("🤖 Loading Phi-4 Method 1 model... This may take a few minutes on first load."):
            try:
                st.session_state.phi4_agent_method1 = Phi4BankingAgent()
                st.success("✅ Phi-4 Method 1 model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load Phi-4 Method 1 model: {str(e)}")
                return False
    return True

def initialize_phi4_agent_method2():
    """Initialize Phi-4 agent Method 2 with proper session state caching"""
    if st.session_state.phi4_agent_method2 is None:
        # Create the agent instance (without loading the model yet)
        st.session_state.phi4_agent_method2 = Phi4Method2Agent()
        
        # The model will be loaded lazily when first needed
        st.info("🤖 Phi-4 Method 2 agent initialized. Model will load when first analysis is run.")
    return True

def initialize_multi_bank_extractor():
    """Initialize multi-bank data extractor lazily"""
    if st.session_state.multi_bank_extractor is None:
        st.session_state.multi_bank_extractor = MultiBankDataExtractor()

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files using Method 1 extraction logic"""
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            # Use Method 1 extraction function
            qa_rows = extract_qa_from_pdf(tmp_path, file_label=uploaded_file.name)
            all_results.extend(qa_rows)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    progress_bar.empty()
    status_text.empty()
    
    if all_results:
        df = pd.DataFrame(all_results)
        df = df[df['Speaker name'].str.strip() != ""]
        return df
    return None

def method_selection_sidebar():
    """Method selection in sidebar with larger, more prominent buttons"""
    st.markdown("### 🔄 Select Processing Method")
    
    # Method selection radio buttons with custom styling
    method_choice = st.radio(
        "Choose your analysis method:",
        options=["method1", "method2"],
        format_func=lambda x: "📄 Phi-4 Method 1: HSBC PDF Processor" if x == "method1" else "🏦 Phi-4 Method 2: Multi-Bank Financial Metrics Extractor",
        key="method_selection"
    )
    
    # Only update if method actually changed (prevents unnecessary rerun)
    if method_choice != st.session_state.selected_method:
        st.session_state.selected_method = method_choice
        # Clear processed data when switching methods
        if method_choice == "method1":
            st.session_state.processed_data_method2 = None
            st.session_state.analysis_results_method2 = None
        else:
            st.session_state.processed_data_method1 = None
            st.session_state.analysis_results_method1 = None
    
    # Method descriptions with bullet points
    if st.session_state.selected_method == "method1":
        st.info("""📄 **Phi-4 Method 1: HSBC PDF Upload & Analysis**

• Upload HSBC transcript PDFs directly
• Extract Q&A data using PDF processing  
• AI analysis focused on risk detection
• Insight generation and answer coverage
• Detailed transparency analysis""")
    else:
        st.info("""🏦 **Phi-4 Method 2: Multi-Bank Financial Metrics Extractor**

• Pre-processed multi-bank transcript data access
• Filter by bank, year, quarter (multi-select)
• Detects and extracts financial metrics discussed
• Captures metric values, trends (increase/decrease/stable)
• Determines if questions were answered or avoided
• Cross-bank comparative financial analysis""")
    
    st.markdown("---")

def get_bank_display_info():
    """Get formatted display information for selected banks"""
    if not st.session_state.selected_banks_display:
        return "All Banks"
    
    bank_names = st.session_state.multi_bank_extractor.bank_names if st.session_state.multi_bank_extractor else BANK_NAMES
    formatted_banks = [bank_names.get(bank, bank) for bank in st.session_state.selected_banks_display]
    
    if len(formatted_banks) <= 3:
        return ", ".join(formatted_banks)
    else:
        return f"{', '.join(formatted_banks[:2])} + {len(formatted_banks)-2} more"

def display_current_selection():
    """Display current Method 2 selection in all tabs"""
    if st.session_state.selected_method == "method2" and st.session_state.processed_data_method2 is not None:
        banks_display = get_bank_display_info()
        years_display = ", ".join(map(str, st.session_state.selected_years_display)) if st.session_state.selected_years_display else "All Years"
        quarters_display = ", ".join(map(str, st.session_state.selected_quarters_display)) if st.session_state.selected_quarters_display else "All Quarters"
        
        st.markdown(f"""
        <div class="bank-info">
            🏦 <strong>Current Selection:</strong> {banks_display} | 📅 {years_display} | 📊 {quarters_display}
        </div>
        """, unsafe_allow_html=True)

def display_data_overview_method1(df):
    """Display overview of Phi-4 Method 1 processed data"""
    st.markdown("### 📊 Phi-4 Method 1 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_speakers = df['Speaker name'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_speakers}</h3>
            <p>Unique Speakers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        qa_records = len(df[df['Question No'].notna()])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{qa_records}</h3>
            <p>Q&A Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_files = df['File'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_files}</h3>
            <p>Files Processed</p>
        </div>
        """, unsafe_allow_html=True)

def display_data_overview_method2(df):
    """Display overview of Phi-4 Method 2 processed data"""
    st.markdown("### 📊 Phi-4 Method 2 Data Overview")
    
    # Display current selection
    display_current_selection()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_speakers = df['speaker'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_speakers}</h3>
            <p>Unique Speakers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        qa_records = len(df[df['type'] == 'query']) if 'type' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{qa_records}</h3>
            <p>Q&A Queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_banks = df['bank'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_banks}</h3>
            <p>Banks Covered</p>
        </div>
        """, unsafe_allow_html=True)

def create_visualizations_method1(df):
    """Create visualizations for Phi-4 Method 1 data"""
    st.markdown("### 📈 Phi-4 Method 1 Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speaker distribution
        speaker_counts = df['Speaker name'].value_counts().head(10)
        fig_speakers = px.bar(
            x=speaker_counts.values, 
            y=speaker_counts.index,
            orientation='h',
            title="Top 10 Speakers by Frequency",
            labels={'x': 'Number of Statements', 'y': 'Speaker'},
            color_discrete_sequence=[UI_CONFIG['primary_color']]
        )
        fig_speakers.update_layout(height=400)
        st.plotly_chart(fig_speakers, use_container_width=True)
    
    with col2:
        # Institution distribution
        institution_counts = df['Institution'].value_counts().head(10)
        fig_institutions = px.pie(
            values=institution_counts.values,
            names=institution_counts.index,
            title="Distribution by Institution",
            color_discrete_sequence=px.colors.sequential.Sunset
        )
        fig_institutions.update_layout(height=400)
        st.plotly_chart(fig_institutions, use_container_width=True)
    
    # Timeline analysis if year/quarter data available
    if 'Year' in df.columns and 'Quarter' in df.columns:
        df_timeline = df.groupby(['Year', 'Quarter']).size().reset_index(name='Count')
        df_timeline['Period'] = df_timeline['Year'].astype(str) + ' Q' + df_timeline['Quarter'].astype(str)
        
        fig_timeline = px.line(
            df_timeline, 
            x='Period', 
            y='Count',
            title="Activity Over Time",
            markers=True,
            color_discrete_sequence=[UI_CONFIG['accent_color']]
        )
        fig_timeline.update_layout(height=300)
        st.plotly_chart(fig_timeline, use_container_width=True)

def create_visualizations_method2(df):
    """Create visualizations for Phi-4 Method 2 data"""
    st.markdown("### 📈 Phi-4 Method 2 Data Visualizations")
    
    # Display current selection
    display_current_selection()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bank distribution
        bank_counts = df['bank'].value_counts()
        bank_names_map = st.session_state.multi_bank_extractor.bank_names if st.session_state.multi_bank_extractor else BANK_NAMES
        bank_labels = [bank_names_map.get(bank, bank) for bank in bank_counts.index]
        
        fig_banks = px.pie(
            values=bank_counts.values,
            names=bank_labels,
            title="Distribution by Bank",
            color_discrete_sequence=px.colors.sequential.Sunset
        )
        fig_banks.update_layout(height=400)
        st.plotly_chart(fig_banks, use_container_width=True)
    
    with col2:
        # Speaker role distribution
        if 'title' in df.columns:
            role_counts = df['title'].value_counts().head(10)
            fig_roles = px.bar(
                x=role_counts.values,
                y=role_counts.index,
                orientation='h',
                title="Top 10 Speaker Roles",
                labels={'x': 'Count', 'y': 'Role'},
                color_discrete_sequence=[UI_CONFIG['primary_color']]
            )
            fig_roles.update_layout(height=400)
            st.plotly_chart(fig_roles, use_container_width=True)
    
    # Bank-Year-Quarter analysis
    if all(col in df.columns for col in ['bank', 'year', 'quarter']):
        df_timeline = df.groupby(['bank', 'year', 'quarter']).size().reset_index(name='Count')
        df_timeline['Period'] = df_timeline['year'].astype(str) + ' ' + df_timeline['quarter'].astype(str)
        
        # Map bank codes to names
        df_timeline['Bank_Name'] = df_timeline['bank'].map(bank_names_map)
        
        fig_timeline = px.line(
            df_timeline,
            x='Period',
            y='Count',
            color='Bank_Name',
            title="Activity Over Time by Bank",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)

def save_analysis_results(results, method_name):
    """Save analysis results to file"""
    filename = f"analysis_results_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return filename
    except Exception as e:
        st.error(f"Failed to save results: {e}")
        return None

def run_phi4_analysis_method1(df):
    """Run Phi-4 Method 1 analysis on the data - LIVE ONLY"""
    if not initialize_phi4_agent_method1():
        return None
    
    st.markdown("### 🧠 AI Analysis with Phi-4 Method 1")
    
    with st.spinner("🔍 Running live Phi-4 Method 1 AI analysis on Q&A data..."):
        try:
            analysis_results = st.session_state.phi4_agent_method1.analyze_qa_data(df)
            # Save results
            filename = save_analysis_results(analysis_results, "method1")
            if filename:
                st.info(f"💾 Results saved to {filename}")
            return analysis_results
        except Exception as e:
            st.error(f"Phi-4 Method 1 analysis failed: {str(e)}")
            return None

def run_phi4_analysis_method2(df):
    """Run Phi-4 Method 2 analysis on the data - LIVE ONLY"""
    if not initialize_phi4_agent_method2():
        return None
    
    st.markdown("### 🧠 AI Analysis with Phi-4 Method 2")
    
    with st.spinner("🔍 Running live Phi-4 Method 2 AI analysis on multi-bank data..."):
        try:
            analysis_results = st.session_state.phi4_agent_method2.analyze_multibank_data(df)
            # Save results
            filename = save_analysis_results(analysis_results, "method2")
            if filename:
                st.info(f"💾 Results saved to {filename}")
            return analysis_results
        except Exception as e:
            st.error(f"Phi-4 Method 2 analysis failed: {str(e)}")
            return None

def parse_method2_metrics(output_text):
    """Parse Method 2 AI output into structured bullet points - handles various output formats"""
    if not output_text or output_text == "Processing failed":
        return []
    
    metrics = []
    
    try:
        # Method 1: Look for the pattern "Financial metric: X · Level: Y · Question: Z"
        import re
        pattern = r'Financial metric:\s*([^·]+)(?:·\s*Level:\s*([^·]+))?(?:·\s*Question:\s*([^·]+))?'
        matches = re.findall(pattern, output_text, re.IGNORECASE)
        
        for match in matches:
            metric_name = match[0].strip() if match[0] else "Unknown"
            level = match[1].strip() if match[1] else "NO LEVEL"
            status = match[2].strip() if match[2] else "UNCLEAR"
            
            # Try to extract trend from level text
            trend = "NO TRAJECTORY"
            level_lower = level.lower()
            if any(word in level_lower for word in ['above', 'up', 'increase', 'grow', 'higher']):
                trend = "INCREASE"
            elif any(word in level_lower for word in ['below', 'down', 'decrease', 'lower', 'decline']):
                trend = "DECREASE"
            elif any(word in level_lower for word in ['stable', 'flat', 'maintain']):
                trend = "STABLE"
            
            metrics.append({
                'metric': metric_name,
                'level': level,
                'trend': trend,
                'status': status
            })
        
        # Method 2: Look for standard list format ['X', 'Y', 'Z', 'W']
        if not metrics and '[' in output_text and ']' in output_text:
            start = output_text.find('[')
            end = output_text.rfind(']') + 1
            list_str = output_text[start:end]
            
            import ast
            try:
                metric_list = ast.literal_eval(list_str)
                if isinstance(metric_list, list) and len(metric_list) >= 4:
                    for i in range(0, len(metric_list), 4):
                        if i + 3 < len(metric_list):
                            metrics.append({
                                'metric': str(metric_list[i]).strip(),
                                'level': str(metric_list[i + 1]).strip(),
                                'trend': str(metric_list[i + 2]).strip(),
                                'status': str(metric_list[i + 3]).strip()
                            })
            except (ValueError, SyntaxError):
                pass
        
        # Method 3: Look for pipe-separated format
        if not metrics:
            lines = output_text.split('\n')
            for line in lines:
                line = line.strip()
                if '|' in line and line.count('|') >= 3:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        metrics.append({
                            'metric': parts[0].strip(),
                            'level': parts[1].strip(),
                            'trend': parts[2].strip(),
                            'status': parts[3].strip()
                        })
        
        # Method 4: Extract any mentioned financial terms as fallback
        if not metrics:
            financial_terms = re.findall(r'\b(RoTE|ROE|ROA|NIM|CET1|Tier\s*1|revenue|profit|earnings|capital|dividend|FTSE|S&P|NASDAQ|buyback|generation)\b', output_text, re.IGNORECASE)
            for term in set(financial_terms):  # Remove duplicates
                metrics.append({
                    'metric': term,
                    'level': 'See raw output',
                    'trend': 'See raw output', 
                    'status': 'ANSWERED' if 'ANSWERED' in output_text.upper() else 'UNCLEAR'
                })
    
    except Exception as e:
        # Error fallback
        metrics.append({
            'metric': 'Parsing Error',
            'level': f'Error: {str(e)}',
            'trend': 'See raw output',
            'status': 'ERROR'
        })
    
    return metrics

def display_analysis_results_method1(analysis_results):
    """Display Phi-4 Method 1 analysis results - Show ALL pairs"""
    if not analysis_results:
        return
    
    st.markdown("### 🎯 Phi-4 Method 1 Analysis Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_count = analysis_results.get('risk_detected_pairs', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{risk_count}</h3>
            <p>Risk Indicators Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        answered_count = analysis_results.get('fully_answered_pairs', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{answered_count}</h3>
            <p>Fully Answered Questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        insights_count = analysis_results.get('valid_insight_pairs', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{insights_count}</h3>
            <p>Valid Insights Generated</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results - Show MORE pairs, not just 5
    if 'qa_analyses' in analysis_results:
        total_pairs = len(analysis_results['qa_analyses'])
        st.markdown(f"#### 💡 Key Insights from {total_pairs} Q&A Pairs")
        
        # Allow user to choose how many to display
        max_display = st.slider("Number of Q&A pairs to display:", 5, min(50, total_pairs), 10)
        
        for i, qa in enumerate(analysis_results['qa_analyses'][:max_display]):
            if qa.get('processing_status') == 'success':
                with st.expander(f"Q&A Pair {i+1} - {qa['grouping_info']['year']} Q{qa['grouping_info']['quarter']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Question:**")
                        st.text_area("", qa['question_text'][:300] + ("..." if len(qa['question_text']) > 300 else ""), height=100, key=f"q1_{i}")
                        
                        st.markdown("**Key Finding:**")
                        st.info(qa['extracted_results']['key_findings'])
                    
                    with col2:
                        st.markdown("**Answer:**")
                        st.text_area("", qa['answer_text'][:300] + ("..." if len(qa['answer_text']) > 300 else ""), height=100, key=f"a1_{i}")
                        
                        col2a, col2b = st.columns(2)
                        with col2a:
                            risk_color = "🔴" if qa['extracted_results']['risk_or_distress'] == "Yes" else "🟢"
                            st.markdown(f"**Risk:** {risk_color} {qa['extracted_results']['risk_or_distress']}")
                        
                        with col2b:
                            st.markdown(f"**Coverage:** {qa['extracted_results']['answer_coverage']}")

def display_analysis_results_method2(analysis_results):
    """Display Phi-4 Method 2 analysis results with formatted bullet points"""
    if not analysis_results:
        return
    
    st.markdown("### 🎯 Phi-4 Method 2 Analysis Results")
    
    # Display current selection
    display_current_selection()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metrics_count = analysis_results.get('extracted_metrics', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{metrics_count}</h3>
            <p>Financial Metrics Extracted</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        answered_count = analysis_results.get('answered_questions', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{answered_count}</h3>
            <p>Questions Answered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avoided_count = analysis_results.get('avoided_questions', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avoided_count}</h3>
            <p>Questions Avoided</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results with formatted metrics
    if 'qa_analyses' in analysis_results:
        total_pairs = len(analysis_results['qa_analyses'])
        st.markdown(f"#### 💡 Financial Metrics Analysis from {total_pairs} Q&A Pairs")
        
        # Allow user to choose how many to display
        max_display = st.slider("Number of Q&A pairs to display:", 5, min(50, total_pairs), 10, key="method2_slider")
        
        for i, qa in enumerate(analysis_results['qa_analyses'][:max_display]):
            if qa.get('processing_status') == 'success':
                # Get bank info if available
                bank_info = ""
                if 'bank' in qa:
                    bank_names_map = st.session_state.multi_bank_extractor.bank_names if st.session_state.multi_bank_extractor else BANK_NAMES
                    bank_name = bank_names_map.get(qa['bank'], qa['bank'])
                    bank_info = f" - {bank_name}"
                
                with st.expander(f"Q&A Pair {i+1}{bank_info} - Metrics: {qa.get('metrics_extracted', 0)}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Question:**")
                        question_text = qa.get('query_text', qa.get('question', 'No question text'))
                        st.text_area("", question_text[:300] + ("..." if len(question_text) > 300 else ""), height=100, key=f"q2_{i}")
                        
                        st.markdown("**Financial Metrics Found:**")
                        # Parse and display metrics with dropdown selection
                        raw_output = qa.get('raw_llm_output', '')
                        metrics = parse_method2_metrics(raw_output)
                        
                        if metrics:
                            # Create dropdown for metric selection
                            metric_options = [f"📊 {idx + 1}. {metric['metric']}" for idx, metric in enumerate(metrics)]
                            
                            # Add summary option
                            metric_options.insert(0, "📋 Overview - All Metrics")
                            
                            selected_option = st.selectbox(
                                f"Select metric to view details ({len(metrics)} total):",
                                metric_options,
                                key=f"metric_select_{i}",
                                help="Choose a metric to see its detailed analysis"
                            )
                            
                            if selected_option == "📋 Overview - All Metrics":
                                # Show compact overview of all metrics
                                st.markdown("**📋 Metrics Summary:**")
                                for idx, metric in enumerate(metrics, 1):
                                    status_icon = "✅" if metric.get('status', '').upper() == 'ANSWERED' else "❌" if metric.get('status', '').upper() == 'AVOIDED' else "❓"
                                    st.markdown(f"**{idx}.** {metric['metric']} {status_icon}")
                                
                                # Summary stats
                                answered_count = sum(1 for m in metrics if m.get('status', '').upper() == 'ANSWERED')
                                avoided_count = sum(1 for m in metrics if m.get('status', '').upper() == 'AVOIDED')
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Answered", answered_count, delta=None)
                                with col_b:
                                    st.metric("Avoided", avoided_count, delta=None)
                            
                            else:
                                # Show details for selected metric
                                selected_idx = int(selected_option.split('.')[0].replace('📊', '').strip()) - 1
                                selected_metric = metrics[selected_idx]
                                
                                # Create color coding for trends and status
                                trend = selected_metric.get('trend', 'NO TRAJECTORY')
                                if trend.upper() in ['INCREASE', 'UP', 'GROW', 'ABOVE']:
                                    trend_color = "🟢"
                                    trend_delta = "positive"
                                elif trend.upper() in ['DECREASE', 'DOWN', 'DECLINE', 'BELOW']:
                                    trend_color = "🔴"
                                    trend_delta = "negative"
                                elif trend.upper() in ['STABLE', 'FLAT', 'MAINTAIN']:
                                    trend_color = "🔵"
                                    trend_delta = "off"
                                else:
                                    trend_color = "⚪"
                                    trend_delta = "off"
                                
                                status = selected_metric.get('status', 'UNCLEAR')
                                if status.upper() == 'ANSWERED':
                                    status_color = "✅"
                                    status_bg = "success"
                                elif status.upper() in ['AVOIDED', 'NOT ANSWERED']:
                                    status_color = "❌"
                                    status_bg = "error"
                                else:
                                    status_color = "❓"
                                    status_bg = "warning"
                                
                                # Display selected metric details in a nice format
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                                    padding: 1.5rem;
                                    border-radius: 15px;
                                    border-left: 5px solid #4CAF50;
                                    margin: 1rem 0;
                                ">
                                    <h4 style="margin-top: 0; color: #1f4e79;">📊 {selected_metric['metric']}</h4>
                                    
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;">
                                        <div>
                                            <strong>💰 Level:</strong><br>
                                            <span style="background: #fff; padding: 0.3rem 0.8rem; border-radius: 8px; border: 1px solid #ddd;">
                                                {selected_metric['level']}
                                            </span>
                                        </div>
                                        <div>
                                            <strong>{trend_color} Trend:</strong><br>
                                            <span style="background: #fff; padding: 0.3rem 0.8rem; border-radius: 8px; border: 1px solid #ddd;">
                                                {trend}
                                            </span>
                                        </div>
                                    </div>
                                    
                                    <div style="margin-top: 1rem;">
                                        <strong>{status_color} Answer Status:</strong><br>
                                        <span style="
                                            background: {'#d4edda' if status_bg == 'success' else '#f8d7da' if status_bg == 'error' else '#fff3cd'};
                                            color: {'#155724' if status_bg == 'success' else '#721c24' if status_bg == 'error' else '#856404'};
                                            padding: 0.3rem 0.8rem;
                                            border-radius: 8px;
                                            border: 1px solid {'#c3e6cb' if status_bg == 'success' else '#f5c6cb' if status_bg == 'error' else '#ffeaa7'};
                                            font-weight: bold;
                                        ">
                                            {status}
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Always show raw output in collapsible section for reference
                            with st.expander(f"🔍 Raw AI Output ({len(metrics)} metrics parsed)"):
                                st.code(raw_output, language="text")
                        
                        else:
                            # Fallback: show count or raw output preview
                            metrics_count = qa.get('metrics_extracted', 0)
                            if metrics_count > 0:
                                st.info(f"📊 {metrics_count} financial metrics detected (parsing failed)")
                            else:
                                st.warning("⚠️ No financial metrics found")
                            
                            if raw_output and raw_output != "Processing failed":
                                with st.expander("📄 View Raw AI Output"):
                                    st.code(raw_output, language="text")
                    
                    with col2:
                        st.markdown("**Answers:**")
                        if 'answer_texts' in qa:
                            combined_answers = " ".join(qa['answer_texts'])
                        else:
                            combined_answers = qa.get('answer', 'No answer text')
                        st.text_area("", combined_answers[:300] + ("..." if len(combined_answers) > 300 else ""), height=100, key=f"a2_{i}")
                        
                        col2a, col2b = st.columns(2)
                        with col2a:
                            answer_color = "🟢" if qa.get('question_answered', False) else "🔴"
                            st.markdown(f"**Answered:** {answer_color} {'Yes' if qa.get('question_answered', False) else 'No'}")
                        
                        with col2b:
                            avoid_color = "🔴" if qa.get('question_avoided', False) else "🟢"
                            st.markdown(f"**Avoided:** {avoid_color} {'Yes' if qa.get('question_avoided', False) else 'No'}")
                        
                        if qa.get('raw_llm_output') and qa['raw_llm_output'] != "Processing failed":
                            with st.expander("Raw LLM Output"):
                                st.code(qa['raw_llm_output'])

def chat_interface():
    """Interactive chat interface with Phi-4"""
    current_data = None
    current_agent = None
    method_name = ""
    
    if st.session_state.selected_method == "method1" and st.session_state.processed_data_method1 is not None:
        current_data = st.session_state.processed_data_method1
        current_agent = st.session_state.phi4_agent_method1
        method_name = "Phi-4 Method 1"
    elif st.session_state.selected_method == "method2" and st.session_state.processed_data_method2 is not None:
        current_data = st.session_state.processed_data_method2
        current_agent = st.session_state.phi4_agent_method2
        method_name = "Phi-4 Method 2"
    
    st.markdown(f"### 💬 Chat with {method_name} AI Agent")
    
    # Display current selection for Method 2
    if st.session_state.selected_method == "method2":
        display_current_selection()
    
    if current_data is None:
        st.warning(f"⚠️ Please process data using {st.session_state.selected_method.upper()} first to enable chat functionality.")
        return
    
    # Chat input
    user_question = st.text_input(f"Ask a question about the {method_name} data:", key="chat_input")
    
    if st.button("Send", key="send_chat") and user_question:
        if current_agent or (st.session_state.selected_method == "method1" and initialize_phi4_agent_method1()) or (st.session_state.selected_method == "method2" and initialize_phi4_agent_method2()):
            with st.spinner("🤖 Generating response..."):
                try:
                    if st.session_state.selected_method == "method1":
                        agent = st.session_state.phi4_agent_method1
                    else:
                        agent = st.session_state.phi4_agent_method2
                    
                    response = agent.answer_question(user_question, current_data)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response,
                        "method": method_name,
                        "timestamp": datetime.now()
                    })
                    
                except Exception as e:
                    st.error(f"Failed to generate response: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("#### 📝 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.container():
                st.markdown(f"""
                <div class="chat-message">
                    <strong>Q ({chat['method']}):</strong> {chat['question']}<br>
                    <strong>A:</strong> {chat['answer']}<br>
                    <small>🕐 {chat['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)

def method1_sidebar():
    """Sidebar controls for Phi-4 Method 1"""
    st.markdown("### 📄 Phi-4 Method 1 Controls")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload HSBC PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload transcript PDFs for analysis"
    )
    
    if uploaded_files and st.button("🔄 Process Phi-4 Method 1 Files", key="process_method1"):
        with st.spinner("📄 Processing PDF files..."):
            df = process_uploaded_files(uploaded_files)
            if df is not None:
                st.session_state.processed_data_method1 = df
                st.success(f"✅ Phi-4 Method 1: Processed {len(df):,} records from {len(uploaded_files)} files")
            else:
                st.error("❌ No data could be extracted from the files")
    
    # Analysis controls
    if st.session_state.processed_data_method1 is not None:
        st.markdown("---")
        if st.button("🧠 Run Phi-4 Method 1 AI Analysis", key="run_method1_analysis"):
            analysis_results = run_phi4_analysis_method1(st.session_state.processed_data_method1)
            if analysis_results:
                st.session_state.analysis_results_method1 = analysis_results

def get_filtered_options(selected_banks, selected_years, df_sample):
    """Get filtered year and quarter options based on selected banks and years (cascading)"""
    all_banks = sorted(list(df_sample['bank'].unique()))
    
    if selected_banks:
        # Filter data based on selected banks
        filtered_df = df_sample[df_sample['bank'].isin(selected_banks)]
        available_years = sorted(list(filtered_df['year'].unique()))
        
        # Further filter by years if provided
        if selected_years:
            year_filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
            available_quarters = sorted(list(year_filtered_df['quarter'].unique()))
        else:
            available_quarters = sorted(list(filtered_df['quarter'].unique()))
    else:
        # Return all options if no banks selected
        available_years = sorted(list(df_sample['year'].unique()))
        if selected_years:
            year_filtered_df = df_sample[df_sample['year'].isin(selected_years)]
            available_quarters = sorted(list(year_filtered_df['quarter'].unique()))
        else:
            available_quarters = sorted(list(df_sample['quarter'].unique()))
    
    return {
        'banks': all_banks,
        'years': available_years,
        'quarters': available_quarters
    }

def method2_sidebar():
    """Optimized sidebar controls for Phi-4 Method 2 with proper cascading filtering"""
    st.markdown("### 🏦 Phi-4 Method 2 Controls")
    
    # Initialize extractor lazily only when needed
    if st.session_state.selected_method == "method2" and st.session_state.multi_bank_extractor is None:
        initialize_multi_bank_extractor()
    
    if st.session_state.multi_bank_extractor is None:
        st.warning("⚠️ Please wait while initializing multi-bank extractor...")
        return
    
    # Get available filters
    try:
        filters = st.session_state.multi_bank_extractor.get_available_filters()
        df_sample = st.session_state.multi_bank_extractor.load_real_data()
    except Exception as e:
        st.error(f"Failed to load filters: {e}")
        return
    
    # Multi-select for banks
    st.markdown("**Select Banks:**")
    selected_banks = st.multiselect(
        "Choose one or more banks:",
        options=filters['banks'],
        format_func=lambda x: st.session_state.multi_bank_extractor.bank_names.get(x, x),
        help="• Select multiple banks for comparative analysis\n• Leave empty to include all banks",
        key="selected_banks_multiselect"
    )
    
    # Get filtered options based on selected banks (cascading)
    filter_options = get_filtered_options(selected_banks, [], df_sample)
    available_years = filter_options['years']
    
    # Multi-select for years (filtered based on banks)
    st.markdown("**Select Years:**")
    selected_years = st.multiselect(
        "Choose one or more years:",
        options=available_years,
        help="• Select multiple years for temporal analysis\n• Options filtered based on selected banks",
        key="selected_years_multiselect"
    )
    
    # Get quarters filtered by both banks and years (cascading)
    filter_options_with_years = get_filtered_options(selected_banks, selected_years, df_sample)
    available_quarters = filter_options_with_years['quarters']
    
    # Multi-select for quarters (filtered based on banks and years)
    st.markdown("**Select Quarters:**")
    selected_quarters = st.multiselect(
        "Choose one or more quarters:",
        options=available_quarters,
        help="• Select multiple quarters for detailed analysis\n• Options filtered based on selected banks and years",
        key="selected_quarters_multiselect"
    )
    
    # Show current filter summary
    if selected_banks or selected_years or selected_quarters:
        st.markdown("---")
        st.markdown("**📋 Current Filter Summary:**")
        
        bank_names = st.session_state.multi_bank_extractor.bank_names
        banks_display = [bank_names.get(b, b) for b in selected_banks] if selected_banks else ["All Banks"]
        years_display = [str(y) for y in selected_years] if selected_years else ["All Years"]
        quarters_display = [str(q) for q in selected_quarters] if selected_quarters else ["All Quarters"]
        
        st.info(f"""
        🏦 **Banks:** {', '.join(banks_display[:3])}{'...' if len(banks_display) > 3 else ''}
        📅 **Years:** {', '.join(years_display)}
        📊 **Quarters:** {', '.join(quarters_display)}
        """)
    
    # Process button
    process_button = st.button("🔄 Process Phi-4 Method 2 Data", key="process_method2")
    
    # Clear cache section with explanation
    st.markdown("---")
    st.markdown("**Troubleshooting:**")
    st.info("💡 Click 'Clear Cache' if you experience:\n• Stale data issues\n• Filter dropdown problems\n• Data loading errors\n• Want to force fresh download")
    
    if st.button("🗑️ Clear Cache & Reset Data", key="clear_cache"):
        try:
            if st.session_state.multi_bank_extractor and hasattr(st.session_state.multi_bank_extractor, 'clear_cache'):
                st.session_state.multi_bank_extractor.clear_cache()
            else:
                # Manual cache clearing if method doesn't exist
                st.session_state.multi_bank_extractor = None
                if os.path.exists("all_banks_transcript_split.xlsx"):
                    os.remove("all_banks_transcript_split.xlsx")
                st.success("✅ Cache cleared manually!")
            
            st.session_state.processed_data_method2 = None
            st.rerun()
        except Exception as e:
            st.error(f"Cache clear failed: {e}")
            # Force manual reset
            st.session_state.multi_bank_extractor = None
            st.session_state.processed_data_method2 = None
            st.success("✅ Forced cache reset completed!")
    
    # Process data with progress tracking
    if process_button:
        if not selected_banks and not selected_years and not selected_quarters:
            st.warning("⚠️ Please select at least one filter (bank, year, or quarter)")
        else:
            with st.spinner("🏦 Processing multi-bank data..."):
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading data...")
                    progress_bar.progress(25)
                    
                    # Validate combinations exist
                    test_df = df_sample
                    if selected_banks:
                        test_df = test_df[test_df['bank'].isin(selected_banks)]
                    if selected_years:
                        test_df = test_df[test_df['year'].isin(selected_years)]
                    if selected_quarters:
                        test_df = test_df[test_df['quarter'].isin(selected_quarters)]
                    
                    if len(test_df) == 0:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ No data found with the selected combination of filters. Please adjust your selection.")
                        return
                    
                    # Process data with selected filters
                    df = st.session_state.multi_bank_extractor.process_data_multiselect(
                        selected_banks, selected_years, selected_quarters
                    )
                    
                    progress_bar.progress(75)
                    status_text.text("Finalizing...")
                    
                    if df is not None and len(df) > 0:
                        st.session_state.processed_data_method2 = df
                        
                        # Store selection for display in other tabs
                        st.session_state.selected_banks_display = selected_banks
                        st.session_state.selected_years_display = selected_years
                        st.session_state.selected_quarters_display = selected_quarters
                        
                        # Show selection summary
                        bank_names = st.session_state.multi_bank_extractor.bank_names
                        banks_str = ", ".join([bank_names.get(b, b) for b in selected_banks]) if selected_banks else "All Banks"
                        years_str = ", ".join(map(str, selected_years)) if selected_years else "All Years"
                        quarters_str = ", ".join(map(str, selected_quarters)) if selected_quarters else "All Quarters"
                        
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        
                        # Count query/answer pairs for validation
                        query_count = len(df[df['type'] == 'query']) if 'type' in df.columns else 0
                        answer_count = len(df[df['type'] == 'answer']) if 'type' in df.columns else 0
                        
                        st.success(f"✅ Phi-4 Method 2: Processed {len(df):,} records\n\n**Selection:**\n• Banks: {banks_str}\n• Years: {years_str}\n• Quarters: {quarters_str}\n\n**Q&A Data:**\n• {query_count} Questions\n• {answer_count} Answers")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ No data found with selected filters")
                        
                except Exception as e:
                    st.error(f"❌ Error processing Phi-4 Method 2 data: {str(e)}")
                    if "progress_bar" in locals():
                        progress_bar.empty()
                    if "status_text" in locals():
                        status_text.empty()
    
    # Analysis controls
    if st.session_state.processed_data_method2 is not None:
        st.markdown("---")
        
        # Show model status
        if st.session_state.phi4_method2_model_cache is not None:
            st.success("🤖 Phi-4 Method 2 model is cached and ready!")
        elif st.session_state.phi4_agent_method2 is not None and st.session_state.phi4_agent_method2.model_loaded:
            st.info("🤖 Phi-4 Method 2 model is loaded and ready!")
        else:
            st.info("🤖 Phi-4 Method 2 model will load when analysis starts (first time only)")
        
        if st.button("🧠 Run Phi-4 Method 2 AI Analysis", key="run_method2_analysis"):
            analysis_results = run_phi4_analysis_method2(st.session_state.processed_data_method2)
            if analysis_results:
                st.session_state.analysis_results_method2 = analysis_results

def export_section():
    """Export options for current method"""
    st.markdown("---")
    st.markdown("### 💾 Export Options")
    
    if st.session_state.selected_method == "method1" and st.session_state.processed_data_method1 is not None:
        if st.button("📊 Download Phi-4 Method 1 Data", key="download_method1_data"):
            csv = st.session_state.processed_data_method1.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"phi4_method1_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if st.session_state.analysis_results_method1 and st.button("📋 Download Phi-4 Method 1 Analysis", key="download_method1_analysis"):
            json_str = json.dumps(st.session_state.analysis_results_method1, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"phi4_method1_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    elif st.session_state.selected_method == "method2" and st.session_state.processed_data_method2 is not None:
        if st.button("📊 Download Phi-4 Method 2 Data", key="download_method2_data"):
            csv = st.session_state.processed_data_method2.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"phi4_method2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if st.session_state.analysis_results_method2 and st.button("📋 Download Phi-4 Method 2 Analysis", key="download_method2_analysis"):
            json_str = json.dumps(st.session_state.analysis_results_method2, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"phi4_method2_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    """Main Streamlit app with updated heading"""
    
    # Updated header with Team 42 info
    st.markdown('<h1 class="main-header">🏦 Team 42 - Cambridge - Bank of England Student Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-Bank AI Financial Analysis Platform | Powered by Phi-4 for intelligent transcript analysis and insights</p>', unsafe_allow_html=True)
    
    # Sidebar with method selection and controls
    with st.sidebar:
        st.markdown("### 🛠️ Controls")
        
        # Method selection at the top of sidebar
        method_selection_sidebar()
        
        # Method-specific controls
        if st.session_state.selected_method == "method1":
            method1_sidebar()
        else:
            method2_sidebar()
        
        export_section()
    
    # Main content based on selected method and available data
    current_data = None
    current_analysis = None
    
    if st.session_state.selected_method == "method1":
        current_data = st.session_state.processed_data_method1
        current_analysis = st.session_state.analysis_results_method1
        method_display = "Phi-4 Method 1"
    else:
        current_data = st.session_state.processed_data_method2
        current_analysis = st.session_state.analysis_results_method2
        method_display = "Phi-4 Method 2"
    
    if current_data is not None:
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🧠 AI Analysis", "💬 Chat"])
        
        with tab1:
            if st.session_state.selected_method == "method1":
                display_data_overview_method1(current_data)
            else:
                display_data_overview_method2(current_data)
            
            # Data preview
            st.markdown(f"### 📋 {method_display} Data Preview")
            if st.session_state.selected_method == "method2":
                display_current_selection()
            
            st.dataframe(
                current_data.head(20),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            if st.session_state.selected_method == "method1":
                create_visualizations_method1(current_data)
            else:
                create_visualizations_method2(current_data)
        
        with tab3:
            if current_analysis:
                if st.session_state.selected_method == "method1":
                    display_analysis_results_method1(current_analysis)
                else:
                    display_analysis_results_method2(current_analysis)
            else:
                st.info(f"👆 Click 'Run {method_display} AI Analysis' in the sidebar to generate live insights")
        
        with tab4:
            chat_interface()
    
    else:
        # Welcome screen
        st.markdown(f"""
        ### 🎯 Welcome to the Team 42 Multi-Bank AI Financial Analysis Platform
        
        **Current Method: {method_display}**
        
        This Cambridge - Bank of England student project uses advanced AI (Phi-4) to analyze banking documents with two powerful methods:
        
        **📄 Phi-4 Method 1 - HSBC PDF Processor:**
        • Upload HSBC transcript PDFs directly
        • Extract Q&A data using advanced PDF processing
        • AI analysis focused on risk detection and insights
        • Detailed answer coverage evaluation
        • Comprehensive transparency analysis
        
        **🏦 Phi-4 Method 2 - Multi-Bank Financial Metrics Extractor:**
        • Access pre-processed multi-bank transcript data
        • Filter by specific banks, years, and quarters (multi-select supported)
        • Detects and extracts financial metrics discussed in transcripts
        • Captures metric values, trends (increase/decrease/stable), and status
        • Determines if questions were answered or avoided by management
        • Cross-bank comparative financial analysis and benchmarking
        
        **🔥 Live Analysis Features:**
        • 🧠 **Real-time AI Processing**: All analysis runs live using Phi-4 models
        • 💬 **Interactive Chat**: Ask questions about your data using natural language
        • 📊 **Rich Visualizations**: Method-specific charts and graphs with bank context
        • 💾 **Export Options**: Download processed data and analysis results
        • 🔄 **Flexible Processing**: Switch between methods seamlessly in the sidebar
        • ⚡ **Performance Optimized**: Fast data loading and instant method switching
        
        **Get Started:**
        1. Select your preferred method in the sidebar (larger buttons for easy selection)
        2. Use the method-specific controls to upload files or select filters
        3. Process your data with progress tracking
        4. Run live AI analysis for comprehensive insights
        5. Explore your data and chat with the AI
        
        ---
        **About the Technology:**
        • Built with Streamlit for easy deployment and user experience
        • Uses Microsoft Phi-4 with dual processing approaches
        • Supports both PDF extraction and pre-processed data analysis
        • Optimized for performance with caching and lazy loading
        • All AI processing happens live - no pre-computed results
        • No data leaves your environment (when run locally)
        
        **Team 42 - Cambridge University & Bank of England Collaboration**
        """)

if __name__ == "__main__":
    main()