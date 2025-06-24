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
from data_extractor import DataExtractor
from config import STREAMLIT_CONFIG, UI_CONFIG

# Configure Streamlit page
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS for better styling
st.markdown(f"""
<style>
    .main-header {{
        font-size: 3rem;
        color: {UI_CONFIG['primary_color']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }}
    .sub-header {{
        font-size: 1.5rem;
        color: {UI_CONFIG['text_color']};
        margin: 1rem 0;
    }}
    .metric-card {{
        background: {UI_CONFIG['background_gradient']};
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }}
    .chat-message {{
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {UI_CONFIG['primary_color']};
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {UI_CONFIG['primary_color']} 0%, {UI_CONFIG['secondary_color']} 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'phi4_agent' not in st.session_state:
    st.session_state.phi4_agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def initialize_phi4_agent():
    """Initialize Phi-4 agent with your configuration"""
    if st.session_state.phi4_agent is None:
        with st.spinner("🤖 Loading Phi-4 model... This may take a few minutes on first load."):
            try:
                st.session_state.phi4_agent = Phi4BankingAgent()
                st.success("✅ Phi-4 model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load Phi-4 model: {str(e)}")
                return False
    return True

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files using your extraction logic"""
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
            # Use your extraction function
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

def display_data_overview(df):
    """Display overview of processed data"""
    st.markdown("### 📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
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

def create_visualizations(df):
    """Create interactive visualizations"""
    st.markdown("### 📈 Data Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speaker distribution
        speaker_counts = df['Speaker name'].value_counts().head(10)
        fig_speakers = px.bar(
            x=speaker_counts.values, 
            y=speaker_counts.index,
            orientation='h',
            title="Top 10 Speakers by Frequency",
            labels={'x': 'Number of Statements', 'y': 'Speaker'}
        )
        fig_speakers.update_layout(height=400)
        st.plotly_chart(fig_speakers, use_container_width=True)
    
    with col2:
        # Institution distribution
        institution_counts = df['Institution'].value_counts().head(10)
        fig_institutions = px.pie(
            values=institution_counts.values,
            names=institution_counts.index,
            title="Distribution by Institution"
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
            markers=True
        )
        fig_timeline.update_layout(height=300)
        st.plotly_chart(fig_timeline, use_container_width=True)

def run_phi4_analysis(df):
    """Run Phi-4 analysis on the data"""
    if not initialize_phi4_agent():
        return None
    
    st.markdown("### 🧠 AI Analysis with Phi-4")
    
    with st.spinner("🔍 Running AI analysis on Q&A data..."):
        try:
            # Use your Phi-4 analysis logic
            analysis_results = st.session_state.phi4_agent.analyze_qa_data(df)
            return analysis_results
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

def display_analysis_results(analysis_results):
    """Display Phi-4 analysis results"""
    if not analysis_results:
        return
    
    st.markdown("### 🎯 Analysis Results")
    
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
    
    # Detailed results
    if 'qa_analyses' in analysis_results:
        st.markdown("#### 💡 Key Insights")
        for i, qa in enumerate(analysis_results['qa_analyses'][:5]):  # Show top 5
            if qa.get('processing_status') == 'success':
                with st.expander(f"Q&A Pair {i+1} - {qa['grouping_info']['year']} Q{qa['grouping_info']['quarter']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Question:**")
                        st.text_area("", qa['question_text'][:300] + "...", height=100, key=f"q_{i}")
                        
                        st.markdown("**Key Finding:**")
                        st.info(qa['extracted_results']['key_findings'])
                    
                    with col2:
                        st.markdown("**Answer:**")
                        st.text_area("", qa['answer_text'][:300] + "...", height=100, key=f"a_{i}")
                        
                        col2a, col2b = st.columns(2)
                        with col2a:
                            risk_color = "🔴" if qa['extracted_results']['risk_or_distress'] == "Yes" else "🟢"
                            st.markdown(f"**Risk:** {risk_color} {qa['extracted_results']['risk_or_distress']}")
                        
                        with col2b:
                            st.markdown(f"**Coverage:** {qa['extracted_results']['answer_coverage']}")

def chat_interface():
    """Interactive chat interface with Phi-4"""
    st.markdown("### 💬 Chat with AI Agent")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process PDF files first to enable chat functionality.")
        return
    
    # Chat input
    user_question = st.text_input("Ask a question about the banking data:", key="chat_input")
    
    if st.button("Send", key="send_chat") and user_question:
        if initialize_phi4_agent():
            with st.spinner("🤖 Generating response..."):
                try:
                    response = st.session_state.phi4_agent.answer_question(
                        user_question, 
                        st.session_state.processed_data
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response,
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
                    <strong>Q:</strong> {chat['question']}<br>
                    <strong>A:</strong> {chat['answer']}<br>
                    <small>🕐 {chat['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 HSBC Banking AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("#### Powered by Phi-4 for intelligent Q&A analysis and insights")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Controls")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload HSBC PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload transcript PDFs for analysis"
        )
        
        if uploaded_files and st.button("🔄 Process Files"):
            with st.spinner("📄 Processing PDF files..."):
                df = process_uploaded_files(uploaded_files)
                if df is not None:
                    st.session_state.processed_data = df
                    st.success(f"✅ Processed {len(df)} records from {len(uploaded_files)} files")
                else:
                    st.error("❌ No data could be extracted from the files")
        
        # Analysis controls
        if st.session_state.processed_data is not None:
            st.markdown("---")
            if st.button("🧠 Run AI Analysis"):
                analysis_results = run_phi4_analysis(st.session_state.processed_data)
                if analysis_results:
                    st.session_state.analysis_results = analysis_results
            
            # Export options
            st.markdown("---")
            st.markdown("### 💾 Export Options")
            
            if st.button("📊 Download Processed Data"):
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"hsbc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            if st.session_state.analysis_results and st.button("📋 Download Analysis Results"):
                json_str = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"hsbc_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Main content
    if st.session_state.processed_data is not None:
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "🧠 AI Analysis", "💬 Chat"])
        
        with tab1:
            display_data_overview(st.session_state.processed_data)
            
            # Data preview
            st.markdown("### 📋 Data Preview")
            st.dataframe(
                st.session_state.processed_data.head(20),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            create_visualizations(st.session_state.processed_data)
        
        with tab3:
            if st.session_state.analysis_results:
                display_analysis_results(st.session_state.analysis_results)
            else:
                st.info("👆 Click 'Run AI Analysis' in the sidebar to generate insights")
        
        with tab4:
            chat_interface()
    
    else:
        # Welcome screen
        st.markdown("""
        ### 🎯 Welcome to the HSBC Banking AI Agent
        
        This application uses advanced AI (Phi-4) to analyze HSBC banking documents and provide intelligent insights.
        
        **Features:**
        - 📄 **PDF Processing**: Extract Q&A data from HSBC transcripts
        - 🧠 **AI Analysis**: Phi-4 powered insight generation
        - 💬 **Interactive Chat**: Ask questions about your data
        - 📊 **Visualizations**: Interactive charts and graphs
        - 💾 **Export**: Download processed data and analysis results
        
        **Get Started:**
        1. Upload PDF files using the sidebar
        2. Click "Process Files" to extract data
        3. Run AI analysis for insights
        4. Explore your data and chat with the AI
        
        ---
        **About the Technology:**
        - Built with Streamlit for easy deployment
        - Uses Microsoft Phi-4 for local AI processing
        - No data leaves your environment
        """)

if __name__ == "__main__":
    main()
    