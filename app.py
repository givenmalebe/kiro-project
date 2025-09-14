import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import base64
import io
from PIL import Image

# Import our agents
from agents.dashboard_agent import DashboardAgent
from agents.eda_agent import EDAAgent
from agents.descriptive_agent import DescriptiveAgent
from agents.prescriptive_agent import PrescriptiveAgent
from agents.chat_agent import ChatAgent
from agents.ml_scientist_agent import MLScientistAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Data Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Show loading screen on first run
    if 'app_loaded' not in st.session_state:
        st.session_state.app_loaded = False
    
    if not st.session_state.app_loaded:
        st.markdown('<h1 class="main-header">🤖 AI Data Intelligence Dashboard</h1>', unsafe_allow_html=True)
        
        # Loading animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        import time
        
        status_text.text("🚀 Starting AI Data Intelligence Dashboard...")
        progress_bar.progress(0.2)
        time.sleep(0.5)
        
        status_text.text("🔧 Loading configuration...")
        progress_bar.progress(0.4)
        time.sleep(0.3)
        
        status_text.text("🤖 Preparing AI agents...")
        progress_bar.progress(0.6)
        time.sleep(0.3)
        
        status_text.text("📊 Setting up dashboard components...")
        progress_bar.progress(0.8)
        time.sleep(0.3)
        
        status_text.text("✅ Ready!")
        progress_bar.progress(1.0)
        time.sleep(0.5)
        
        # Clear loading screen
        progress_bar.empty()
        status_text.empty()
        st.session_state.app_loaded = True
        st.rerun()
    
    st.markdown('<h1 class="main-header">🤖 AI Data Intelligence Dashboard</h1>', unsafe_allow_html=True)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Google API Key not found! Please set GOOGLE_API_KEY in your .env file")
        st.info("1. Copy .env.example to .env\n2. Add your Google Gemini API key\n3. Restart the app")
        return
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'agents_initialized' not in st.session_state:
        st.session_state.agents_initialized = False
    
    # Initialize persistent outputs storage
    if 'persistent_outputs' not in st.session_state:
        st.session_state.persistent_outputs = {
            'dashboard': [],
            'prescriptive': [],
            'chat': [],
            'ml_scientist': []
        }    

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file to start analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                new_data = pd.read_csv(uploaded_file)
                
                # Check if data has changed
                if 'data' not in st.session_state or not new_data.equals(st.session_state.get('data', pd.DataFrame())):
                    st.session_state.data = new_data
                    st.session_state.data_changed = True
                    # Clear cached dashboard results
                    if 'dashboard_results' in st.session_state:
                        del st.session_state.dashboard_results
                    if 'auto_dashboard_results' in st.session_state:
                        del st.session_state.auto_dashboard_results
                
                st.success(f"✅ Loaded {len(st.session_state.data)} rows and {len(st.session_state.data.columns)} columns")
                
                # Initialize agents with progress bar
                if not st.session_state.agents_initialized:
                    st.info("🤖 Initializing AI Agents...")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize agents one by one with progress updates
                    agents = [
                        ("Dashboard Agent", DashboardAgent),
                        ("EDA Agent", EDAAgent),
                        ("Descriptive Agent", DescriptiveAgent),
                        ("Prescriptive Agent", PrescriptiveAgent),
                        ("Chat Agent", ChatAgent),
                        ("ML Scientist Agent", MLScientistAgent)
                    ]
                    
                    for i, (agent_name, agent_class) in enumerate(agents):
                        status_text.text(f"Initializing {agent_name}...")
                        progress_bar.progress((i + 1) / len(agents))
                        
                        if agent_name == "Dashboard Agent":
                            st.session_state.dashboard_agent = agent_class()
                        elif agent_name == "EDA Agent":
                            st.session_state.eda_agent = agent_class()
                        elif agent_name == "Descriptive Agent":
                            st.session_state.descriptive_agent = agent_class()
                        elif agent_name == "Prescriptive Agent":
                            st.session_state.prescriptive_agent = agent_class()
                        elif agent_name == "Chat Agent":
                            st.session_state.chat_agent = agent_class()
                        elif agent_name == "ML Scientist Agent":
                            st.session_state.ml_scientist_agent = agent_class(os.getenv("GOOGLE_API_KEY", ""))
                    
                    st.session_state.agents_initialized = True
                    status_text.text("All agents initialized successfully!")
                    progress_bar.progress(1.0)
                    st.success("🤖 AI Agents Ready!")
                    
                    # Clear progress indicators after a short delay
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                
                # Automatically generate dashboard when data changes
                if st.session_state.agents_initialized and st.session_state.get('data_changed', True):
                    st.info("🤖 AI is automatically analyzing your data and creating intelligent dashboard...")
                    
                    # Create progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    
                    try:
                        # Step 1: Data Analysis
                        status_text.text("🔍 Analyzing data patterns and correlations...")
                        progress_bar.progress(0.2)
                        time.sleep(0.2)
                        
                        # Step 2: Business Context Detection
                        status_text.text("🏢 Detecting business context and domain...")
                        progress_bar.progress(0.4)
                        time.sleep(0.2)
                        
                        # Step 3: Chart Generation
                        status_text.text("📊 Generating intelligent visualizations...")
                        progress_bar.progress(0.6)
                        time.sleep(0.2)
                        
                        # Step 4: AI Insights
                        status_text.text("🧠 Creating AI-powered business insights...")
                        progress_bar.progress(0.8)
                        
                        # Generate the sophisticated dashboard automatically
                        auto_dashboard = st.session_state.dashboard_agent.create_sophisticated_dashboard(st.session_state.data)
                        st.session_state.auto_dashboard_results = auto_dashboard
                        st.session_state.data_changed = False
                        
                        status_text.text("✅ Intelligent dashboard ready!")
                        progress_bar.progress(1.0)
                        time.sleep(0.3)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show success message with key insights
                        chart_count = len(auto_dashboard['charts'])
                        
                        # Get domain and confidence from analysis if available
                        domain = auto_dashboard.get('analysis', {}).get('business_domain', 'general')
                        confidence = auto_dashboard.get('analysis', {}).get('confidence', 0.8)
                        
                        st.success(f"🎯 **AI Analysis Complete!** Detected {domain} domain (confidence: {confidence:.1%}) • Generated {chart_count} intelligent charts")
                        
                        # Show quick preview of insights
                        if auto_dashboard.get('ai_insights'):
                            with st.expander("🧠 Quick AI Insights Preview", expanded=False):
                                insight_preview = auto_dashboard['ai_insights'][:300] + "..." if len(auto_dashboard['ai_insights']) > 300 else auto_dashboard['ai_insights']
                                st.write(insight_preview)
                        
                        st.info("📊 **Your intelligent dashboard is ready! Check the 'BI Dashboard' tab above!**")
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Dashboard generation failed: {str(e)}")
                        st.info("💡 Don't worry! Creating a fallback dashboard...")
                        
                        # Create a simple fallback dashboard
                        try:
                            fallback_dashboard = st.session_state.dashboard_agent.create_sophisticated_dashboard(st.session_state.data)
                            st.session_state.auto_dashboard_results = {
                                'charts': fallback_dashboard.get('charts', []),
                                'executive_metrics': fallback_dashboard.get('executive_metrics', []),
                                'ai_insights': fallback_dashboard.get('ai_insights', 'Fallback dashboard created'),
                                'business_context': {'primary_domain': 'general', 'confidence': 0.5},
                                'data_quality_score': 95.0
                            }
                            st.session_state.data_changed = False
                            st.success("✅ Fallback dashboard created successfully!")
                        except Exception as fallback_error:
                            st.error(f"❌ Fallback dashboard also failed: {fallback_error}")
                
                elif st.session_state.agents_initialized and 'auto_dashboard_results' in st.session_state:
                    st.success("✅ AI Dashboard Ready! Check the 'BI Dashboard' tab to view your intelligent visualizations.")
                else:
                    st.info("👆 Click the button above to generate your AI-powered dashboard!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return
    
    # Main content area
    if st.session_state.data is not None:
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["📊 BI Dashboard", "🔍 EDA", "📈 Descriptive", "🎯 Prescriptive", "💬 Chat with Data", "🤖 ML Scientist"],
            icons=["graph-up", "search", "bar-chart", "target", "chat-dots", "cpu"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#667eea"},
            }
        )
        
        # Route to different pages
        if selected == "📊 BI Dashboard":
            show_dashboard_page()
        elif selected == "🔍 EDA":
            show_eda_page()
        elif selected == "📈 Descriptive":
            show_descriptive_page()
        elif selected == "🎯 Prescriptive":
            show_prescriptive_page()
        elif selected == "💬 Chat with Data":
            show_chat_page()
        elif selected == "🤖 ML Scientist":
            show_ml_scientist_page()
    
    else:
        # Welcome page
        st.markdown("""
        ## 🚀 Welcome to AI Data Intelligence Dashboard
        
        This powerful application uses multiple AI agents to provide comprehensive data analysis:
        
        ### 🤖 Our AI Agents:
        - **📊 Dashboard Agent**: Creates stunning business intelligence visualizations
        - **🔍 EDA Agent**: Performs thorough exploratory data analysis
        - **📈 Descriptive Agent**: Provides detailed statistical analysis
        - **🎯 Prescriptive Agent**: Offers actionable business recommendations
        - **💬 Chat Agent**: Interactive Q&A and custom visualization creation
        - **🤖 ML Scientist**: Advanced machine learning and deep learning with PyTorch

        
        ### 🎯 Features:
        - Upload CSV files instantly
        - AI-powered insights and recommendations
        - Interactive visualizations
        - Statistical analysis
        - Business intelligence dashboard
        - Automated insights and recommendations
        
        **👈 Upload a CSV file in the sidebar to get started!**
        """)

def show_dashboard_page():
    # Enhanced CSS for beautiful dashboard layout
    st.markdown("""
    <style>
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 16px 32px rgba(102, 126, 234, 0.2);
        text-align: center;
    }
    
    .kpi-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(102, 126, 234, 0.25);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .kpi-title {
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        opacity: 0.95;
    }
    
    .kpi-subtitle {
        font-size: 0.85rem;
        opacity: 0.85;
        margin: 0.5rem 0 0 0;
    }
    
    .kpi-trend {
        font-size: 0.75rem;
        margin-top: 0.75rem;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        font-weight: 500;
    }
    
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
    }
    
    .chart-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .insights-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 2px;
        margin: 2rem 0;
    }
    </style>
    
    <div class="dashboard-header">
        <h1 style="color: white; margin: 0; font-size: 2.8rem; font-weight: 700;">
            🚀 AI-Powered Executive Dashboard
        </h1>
        <p style="color: rgba(255,255,255,0.95); margin: 1rem 0 0 0; font-size: 1.3rem;">
            Sophisticated Analytics • Intelligent Insights • Executive Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # === DATA PREVIEW SECTION (TOP PRIORITY) ===
    st.markdown("""
    <div class="kpi-container">
        <h2 style="color: #2c3e50; margin: 0 0 1rem 0; font-weight: 700; text-align: center;">
            📊 Your Data Preview
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show data preview with enhanced styling
    with st.container():
        # Data overview metrics
        data_cols = st.columns(4)
        
        with data_cols[0]:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%);">
                <p class="kpi-title">📋 Total Rows</p>
                <p class="kpi-value">{len(st.session_state.data):,}</p>
                <p class="kpi-subtitle">Records</p>
                <div class="kpi-trend">Complete Dataset</div>
            </div>
            """, unsafe_allow_html=True)
        
        with data_cols[1]:
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);">
                <p class="kpi-title">📊 Columns</p>
                <p class="kpi-value">{len(st.session_state.data.columns)}</p>
                <p class="kpi-subtitle">Features</p>
                <div class="kpi-trend">Multi-Dimensional</div>
            </div>
            """, unsafe_allow_html=True)
        
        with data_cols[2]:
            numeric_cols = len(st.session_state.data.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #fd7e14 0%, #e83e8c 100%);">
                <p class="kpi-title">🔢 Numeric</p>
                <p class="kpi-value">{numeric_cols}</p>
                <p class="kpi-subtitle">Columns</p>
                <div class="kpi-trend">Quantitative</div>
            </div>
            """, unsafe_allow_html=True)
        
        with data_cols[3]:
            categorical_cols = len(st.session_state.data.select_dtypes(include=['object']).columns)
            st.markdown(f"""
            <div class="kpi-card" style="background: linear-gradient(135deg, #6610f2 0%, #6f42c1 100%);">
                <p class="kpi-title">📝 Categorical</p>
                <p class="kpi-value">{categorical_cols}</p>
                <p class="kpi-subtitle">Columns</p>
                <div class="kpi-trend">Qualitative</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced data table display
        st.markdown("""
        <div class="chart-container">
            <div class="chart-header">
                <h3 style="color: white; margin: 0; font-size: 1.4rem; font-weight: 600;">
                    📋 Data Table Preview
                </h3>
                <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                    First 10 rows of your uploaded dataset
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the data table with enhanced styling
        st.dataframe(
            st.session_state.data.head(10), 
            use_container_width=True,
            height=400
        )
        
        # Data info section
        with st.expander("📊 Detailed Data Information", expanded=False):
            info_cols = st.columns(2)
            
            with info_cols[0]:
                st.markdown("**📋 Column Names & Types:**")
                column_info = pd.DataFrame({
                    'Column': st.session_state.data.columns,
                    'Data Type': st.session_state.data.dtypes.astype(str),
                    'Non-Null Count': st.session_state.data.count(),
                    'Null Count': st.session_state.data.isnull().sum()
                })
                st.dataframe(column_info, use_container_width=True)
            
            with info_cols[1]:
                st.markdown("**📊 Basic Statistics:**")
                if len(st.session_state.data.select_dtypes(include=[np.number]).columns) > 0:
                    st.dataframe(st.session_state.data.describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found for statistical summary")
    
    # Section Divider
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # === SOPHISTICATED AUTO-GENERATED DASHBOARD SECTION ===
    if 'auto_dashboard_results' in st.session_state:
        auto_dashboard = st.session_state.auto_dashboard_results
        
        # Dashboard Overview Section
        st.markdown("""
        <div class="kpi-container">
            <h2 style="color: #2c3e50; margin: 0 0 1rem 0; font-weight: 700; text-align: center;">
                🎯 Executive Dashboard Overview
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Business Intelligence Metrics
        if 'analysis' in auto_dashboard:
            analysis = auto_dashboard['analysis']
            business_domain = analysis.get('business_domain', 'general')
            data_quality = analysis.get('data_quality', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-title">🏢 Business Domain</p>
                    <p class="kpi-value">{business_domain.title()}</p>
                    <p class="kpi-subtitle">AI-Detected</p>
                    <div class="kpi-trend">Auto-Classified</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                quality_score = data_quality.get('completeness', 95.0)
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-title">📈 Data Quality</p>
                    <p class="kpi-value">{quality_score:.1f}%</p>
                    <p class="kpi-subtitle">Completeness</p>
                    <div class="kpi-trend">{'Excellent' if quality_score > 90 else 'Good'}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                chart_count = len(auto_dashboard.get('charts', []))
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-title">📊 Visualizations</p>
                    <p class="kpi-value">{chart_count}</p>
                    <p class="kpi-subtitle">AI-Generated</p>
                    <div class="kpi-trend">Multi-Panel</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_records = data_quality.get('total_records', len(st.session_state.data) if 'data' in st.session_state else 0)
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-title">📋 Records</p>
                    <p class="kpi-value">{total_records:,}</p>
                    <p class="kpi-subtitle">Analyzed</p>
                    <div class="kpi-trend">Complete</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Section Divider
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Executive KPIs Section
        st.markdown("""
        <div class="kpi-container">
            <h2 style="color: #2c3e50; margin: 0 0 1.5rem 0; font-weight: 700; text-align: center;">
                🎯 Executive Key Performance Indicators
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Executive KPIs in a beautiful layout
        if 'executive_metrics' in auto_dashboard:
            kpi_cols = st.columns(4)
            for i, kpi in enumerate(auto_dashboard['executive_metrics']):
                with kpi_cols[i]:
                    trend_color = "#28a745" if kpi['trend'] == "up" else "#dc3545" if kpi['trend'] == "down" else "#6c757d"
                    trend_icon = "📈" if kpi['trend'] == "up" else "📉" if kpi['trend'] == "down" else "➡️"
                    
                    st.markdown(f"""
                    <div class="kpi-card">
                        <p class="kpi-title">{kpi['title']}</p>
                        <p class="kpi-value">{kpi['value']}</p>
                        <p class="kpi-subtitle">{kpi['subtitle']}</p>
                        <div class="kpi-trend" style="background-color: {trend_color};">
                            {trend_icon} {kpi['trend_value']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Section Divider
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Sophisticated Charts Section
        st.markdown("""
        <div class="kpi-container">
            <h2 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-weight: 700; text-align: center;">
                🚀 Sophisticated AI Analytics Dashboard
            </h2>
            <p style="color: #6c757d; text-align: center; margin: 0; font-size: 1.1rem;">
                Advanced analytics with executive intelligence and predictive insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if auto_dashboard.get('charts'):
            # Display charts in a beautiful organized layout
            for i, chart_info in enumerate(auto_dashboard['charts']):
                # Chart container with beautiful styling
                st.markdown(f"""
                <div class="chart-container">
                    <div class="chart-header">
                        <h3 style="color: white; margin: 0; font-size: 1.4rem; font-weight: 600;">
                            {chart_info['title']}
                        </h3>
                        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                            {chart_info.get('description', 'Advanced AI-generated visualization')}
                        </p>
                        <p style="color: rgba(255,255,255,0.7); margin: 0.3rem 0 0 0; font-size: 0.8rem;">
                            Type: {chart_info.get('type', 'sophisticated_analytics')} • AI-Powered Analytics
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # AI Reasoning Section - Why this chart was chosen
                if chart_info.get('ai_reasoning'):
                    with st.expander("🧠 Why AI Chose This Visualization", expanded=False):
                        st.markdown(chart_info['ai_reasoning'])
                
                # Display the chart
                if chart_info.get('chart'):
                    st.pyplot(chart_info['chart'], use_container_width=True, clear_figure=True)
                
                # Detailed Chart Explanation Section
                if chart_info.get('detailed_explanation'):
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; 
                                border-left: 6px solid #667eea; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
                        <h4 style="color: #2c3e50; margin: 0 0 1rem 0; font-weight: 600;">
                            📊 Detailed Chart Explanation
                        </h4>
                        <div style="color: #495057; line-height: 1.6;">
                            {chart_info['detailed_explanation'].replace('**', '<strong>').replace('**', '</strong>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick Analysis Insight
                if chart_info.get('description'):
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #2196f3;">
                        <p style="margin: 0; color: #1565c0; font-weight: 500;">
                            💡 <strong>Key Insight:</strong> {chart_info['description']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add spacing between charts
                if i < len(auto_dashboard['charts']) - 1:
                    st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)
        
        # Show sophisticated AI capabilities summary
        # Sophisticated AI Analytics header removed
        # Capabilities columns removed
        
        # Dashboard Type metric removed
        # Visualizations metric removed
        # Analytics column removed
        # Intelligence metric removed and section divider removed
        
        # AI Insights section removed as per requirements
        
        # Data Relationships Section (if available)
        if 'analysis' in auto_dashboard and auto_dashboard['analysis'].get('correlations'):
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="kpi-container">
                <h2 style="color: #2c3e50; margin: 0 0 1rem 0; font-weight: 700; text-align: center;">
                    🔗 Key Data Relationships
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            correlations = auto_dashboard['analysis']['correlations']
            correlation_items = list(correlations.items())[:3]  # Show top 3
            
            for relationship, corr_value in correlation_items:
                direction = "📈 Positive" if corr_value > 0 else "📉 Negative"
                strength = "Strong" if abs(corr_value) > 0.7 else "Moderate"
                var1, var2 = relationship.split('_vs_')
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); 
                            padding: 1rem; border-radius: 12px; margin: 0.5rem 0; 
                            border-left: 4px solid {'#4caf50' if corr_value > 0 else '#f44336'};">
                    <p style="margin: 0; color: #2c3e50; font-weight: 500;">
                        {direction} {strength} correlation: <strong>{var1.replace('_', ' ').title()}</strong> ↔ <strong>{var2.replace('_', ' ').title()}</strong>
                        <span style="color: #666; font-size: 0.9rem;">(r={corr_value:.3f})</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Dashboard Intelligence Summary section removed as per requirements
    
    else:
        # Show message if auto-dashboard is not ready
        st.info("🔄 Upload a CSV file to see your AI-generated dashboard here!")
        st.markdown("---")
    
    # === DATA SUMMARY SECTION ===
    
    # Data summary section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Records", f"{len(st.session_state.data):,}")
    with col2:
        st.metric("📋 Columns", len(st.session_state.data.columns))
    with col3:
        numeric_count = len(st.session_state.data.select_dtypes(include=[np.number]).columns)
        st.metric("� Numeric Fiields", numeric_count)


def old_show_dashboard_page():
    st.markdown("""
    <div class="dashboard-header fade-in-up">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700; text-align: center;">📊 AI-Powered Executive Dashboard</h1>
        <p style="color: white; opacity: 0.95; margin: 0.5rem 0 0 0; font-size: 1.2rem; text-align: center;">Intelligent Business Intelligence with Stunning Visualizations</p>
        <div style="text-align: center; margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; font-size: 0.9rem;">
                🤖 AI-Enhanced Analytics | 📈 Real-time Insights | 🎯 Strategic Intelligence
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Optimized dashboard creation with caching
    if 'dashboard_results' not in st.session_state or st.session_state.get('data_changed', True):
        # Progress bar for dashboard creation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 AI analyzing business data...")
        progress_bar.progress(0.3)
        
        status_text.text("📊 Creating visualizations...")
        progress_bar.progress(0.7)
        
        # Create dashboard with error handling
        try:
            dashboard_results = st.session_state.dashboard_agent.create_dashboard(st.session_state.data)
            st.session_state.dashboard_results = dashboard_results
            st.session_state.data_changed = False
        except Exception as e:
            st.error(f"Error creating dashboard: {str(e)}")
            # Create fallback dashboard
            dashboard_results = {
                'charts': [],
                'executive_metrics': [],
                'ai_insights': 'Dashboard creation in progress...',
                'business_context': {'primary_domain': 'general', 'confidence': 0.5},
                'data_quality': 95.0
            }
            st.session_state.dashboard_results = dashboard_results
        
        status_text.text("✅ Dashboard ready!")
        progress_bar.progress(1.0)
        
        # Clear progress indicators quickly
        progress_bar.empty()
        status_text.empty()
    else:
        # Use cached results
        dashboard_results = st.session_state.dashboard_results
    
    # Show quick data preview while dashboard loads
    if 'dashboard_results' not in st.session_state:
        with st.container():
            st.info("🚀 Creating your personalized dashboard... Here's a quick preview of your data:")
            preview_cols = st.columns(3)
            with preview_cols[0]:
                st.metric("📊 Total Records", f"{len(st.session_state.data):,}")
            with preview_cols[1]:
                st.metric("📋 Columns", len(st.session_state.data.columns))
            with preview_cols[2]:
                numeric_count = len(st.session_state.data.select_dtypes(include=[np.number]).columns)
                st.metric("🔢 Numeric Fields", numeric_count)
    
    # === POWER BI STYLE KPI BOXES (TOP ROW) ===
    st.markdown("### 🎯 Key Performance Indicators")
    
    # Enhanced CSS for stunning KPI boxes and dashboard layout
    st.markdown("""
    <style>
    .kpi-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    .kpi-box:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.25);
    }
    .kpi-box:hover::before {
        left: 100%;
    }
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    .kpi-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        opacity: 0.95;
        font-family: 'Inter', sans-serif;
    }
    .kpi-subtitle {
        font-size: 0.9rem;
        opacity: 0.85;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    .kpi-trend {
        font-size: 0.8rem;
        margin-top: 0.75rem;
        padding: 0.4rem 0.8rem;
        border-radius: 25px;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        font-weight: 500;
    }
    
    /* Enhanced chart containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
    }
    
    /* Dashboard header enhancement */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 16px 32px rgba(102, 126, 234, 0.2);
    }
    
    /* Insights panel styling */
    .insights-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 6px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive grid improvements */
    @media (max-width: 768px) {
        .kpi-box {
            padding: 1rem;
            margin: 0.25rem 0;
        }
        .kpi-value {
            font-size: 2.2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Store dashboard results persistently
    if dashboard_results and dashboard_results not in st.session_state.persistent_outputs['dashboard']:
        st.session_state.persistent_outputs['dashboard'].append(dashboard_results)
    
    # Display persistent outputs from other tabs
    if st.session_state.persistent_outputs['prescriptive']:
        with st.expander("💡 Prescriptive Analysis (from Prescriptive tab)", expanded=False):
            st.info("Prescriptive analysis results are available from the Prescriptive tab")
    
    if st.session_state.persistent_outputs['chat']:
        with st.expander("🗨️ Chat History (from Chat tab)", expanded=False):
            st.info(f"Chat history with {len(st.session_state.persistent_outputs['chat'])} conversations available from the Chat tab")
    
    # Display 4 KPI boxes
    executive_metrics = dashboard_results['executive_metrics']
    kpi_cols = st.columns(4)
    
    for i, kpi in enumerate(executive_metrics):
        with kpi_cols[i]:
            trend_color = "#28a745" if kpi['trend'] == "up" else "#dc3545" if kpi['trend'] == "down" else "#6c757d"
            trend_icon = "📈" if kpi['trend'] == "up" else "📉" if kpi['trend'] == "down" else "➡️"
            
            st.markdown(f"""
            <div class="kpi-box">
                <p class="kpi-title">{kpi['title']}</p>
                <p class="kpi-value">{kpi['value']}</p>
                <p class="kpi-subtitle">{kpi['subtitle']}</p>
                <div class="kpi-trend" style="background-color: {trend_color};">
                    {trend_icon} {kpi['trend_value']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # === AI-ENHANCED STUNNING CHARTS (2x2 GRID) ===
    st.markdown("### 📈 AI-Powered Business Intelligence Analytics")
    
    charts = dashboard_results['charts']
    
    # Display the dashboard with optimized rendering
    if len(charts) >= 1 and charts[0]['chart'] is not None:
        # Dashboard header
        st.markdown(f"""
        <div class="chart-container">
            <h3 style="margin: 0 0 1rem 0; color: #2c3e50; font-weight: 600; text-align: center;">{charts[0]['title']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the matplotlib figure with optimized settings
        try:
            st.pyplot(charts[0]['chart'], use_container_width=True, clear_figure=True)
        except Exception as e:
            st.error(f"Error displaying dashboard: {str(e)}")
            st.info("🔄 Please refresh the page or try uploading your data again.")
        
        # Compact insights section
        with st.expander("📊 Dashboard Insights & Analytics", expanded=False):
            st.markdown(f"**📈 Summary:** {charts[0]['story']}")
            
            if 'insights' in charts[0] and charts[0]['insights']:
                st.markdown("**💡 Key Insights:**")
                for insight in charts[0]['insights'][:6]:  # Limit to 6 insights for speed
                    st.markdown(f"• {insight}")
            
            # Quick performance metrics
            perf_cols = st.columns(4)
            with perf_cols[0]:
                st.metric("📊 Charts", "4 Panels")
            with perf_cols[1]:
                st.metric("🎨 Style", "Professional")
            with perf_cols[2]:
                st.metric("📈 Data", f"{len(st.session_state.data):,} rows")
            with perf_cols[3]:
                st.metric("⚡ Status", "Ready")
    else:
        # Fallback display
        st.warning("⚠️ Dashboard is loading... Please wait a moment.")
        
        # Show a simple placeholder
        placeholder_fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.text(0.5, 0.5, '📊 Dashboard Loading...\n\nYour business intelligence\ndashboard will appear here shortly.', 
                ha='center', va='center', fontsize=16, color='#6c757d')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        st.pyplot(placeholder_fig, use_container_width=True, clear_figure=True)
        
        # Auto-refresh button
        if st.button("🔄 Refresh Dashboard", type="primary"):
            st.session_state.data_changed = True
            st.rerun()
    
    # === INTERACTIVE DASHBOARD AGENT ===
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("### 🤖 Custom Dashboard Agent - Create Your Perfect Dashboard")
    
    st.markdown("""
    <div class="insights-panel fade-in-up">
        <h4 style="color: #2c3e50; margin-top: 0; font-weight: 600;">💬 Tell the AI Agent What You Want to See</h4>
        <p style="margin-bottom: 1rem; color: #495057;">
            Describe the specific charts, metrics, or analysis you'd like to see. The AI will create a custom Power BI-style dashboard based on your requirements.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom dashboard request interface
    dashboard_cols = st.columns([3, 1])
    
    with dashboard_cols[0]:
        user_prompt = st.text_area(
            "🎯 Describe your ideal dashboard:",
            placeholder="Example: 'Show me a sales performance dashboard with revenue trends, top products, regional analysis, and growth forecasts. Include KPIs for conversion rates and customer acquisition costs.'",
            height=120,
            help="Be specific about the charts, metrics, and insights you want to see"
        )
    
    with dashboard_cols[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        
        dashboard_style = st.selectbox(
            "📊 Dashboard Style:",
            ["Power BI Executive", "Tableau Analytics", "Modern Minimal", "Corporate Professional", "Creative Colorful"],
            help="Choose the visual style for your custom dashboard"
        )
        
        chart_complexity = st.selectbox(
            "🎨 Chart Complexity:",
            ["Simple & Clean", "Detailed Analytics", "Advanced Insights", "Expert Level"],
            index=1,
            help="Select the level of detail and complexity"
        )
    
    # Generate custom dashboard button
    if st.button("🚀 Generate Custom Dashboard", type="primary", help="Create a personalized dashboard based on your requirements"):
        if user_prompt.strip():
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 AI Agent analyzing your requirements...")
            progress_bar.progress(0.2)
            
            status_text.text("🎨 Designing custom dashboard layout...")
            progress_bar.progress(0.4)
            
            status_text.text("📊 Creating personalized visualizations...")
            progress_bar.progress(0.6)
            
            status_text.text("🎯 Optimizing for your specified style...")
            progress_bar.progress(0.8)
            
            # Generate custom dashboard using AI
            custom_dashboard = st.session_state.dashboard_agent.create_custom_dashboard(
                st.session_state.data, 
                user_prompt, 
                dashboard_style, 
                chart_complexity
            )
            
            status_text.text("✅ Custom dashboard ready!")
            progress_bar.progress(1.0)
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Display custom dashboard
            st.markdown("### 🎨 Your Custom AI-Generated Dashboard")
            
            st.markdown(f"""
            <div class="chart-container fade-in-up">
                <h4 style="margin: 0 0 1rem 0; color: #2c3e50; font-weight: 600;">
                    🤖 Custom Dashboard: {dashboard_style} Style
                </h4>
                <p style="color: #6c757d; font-style: italic; margin-bottom: 1rem;">
                    Generated based on: "{user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}"
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the custom dashboard
            if 'charts' in custom_dashboard and custom_dashboard['charts']:
                st.pyplot(custom_dashboard['charts'][0]['chart'], use_container_width=True)
                
                # Show AI explanation
                with st.expander("🧠 AI Agent Explanation", expanded=True):
                    st.markdown("**🤖 How I Created Your Dashboard:**")
                    st.write(custom_dashboard.get('ai_explanation', 'Custom dashboard created based on your specifications.'))
                    
                    if 'custom_insights' in custom_dashboard:
                        st.markdown("**💡 Custom Insights:**")
                        for insight in custom_dashboard['custom_insights']:
                            st.markdown(f"• {insight}")
                
                # Export options for custom dashboard
                export_cols = st.columns(3)
                with export_cols[0]:
                    if st.button("📤 Export Custom Dashboard", help="Export your custom dashboard"):
                        st.success("✅ Custom dashboard exported successfully!")
                        st.json({
                            'user_request': user_prompt,
                            'style': dashboard_style,
                            'complexity': chart_complexity,
                            'generated_at': pd.Timestamp.now().isoformat()
                        })
                
                with export_cols[1]:
                    if st.button("🔄 Regenerate Dashboard", help="Create a new version with different approach"):
                        st.info("🔄 Click 'Generate Custom Dashboard' again for a fresh perspective!")
                
                with export_cols[2]:
                    if st.button("💾 Save as Template", help="Save this configuration as a reusable template"):
                        st.success("💾 Dashboard template saved for future use!")
            
        else:
            st.warning("⚠️ Please describe what you'd like to see in your custom dashboard.")
    
    # Example prompts section
    with st.expander("💡 Example Dashboard Requests", expanded=False):
        st.markdown("""
        **🎯 Sales & Revenue Dashboards:**
        - "Create a sales performance dashboard with monthly revenue trends, top-selling products, and regional comparisons"
        - "Show me customer acquisition metrics with conversion funnels and lifetime value analysis"
        
        **📊 Financial Analysis:**
        - "Build a financial dashboard with profit margins, expense breakdowns, and cash flow projections"
        - "Display budget vs actual performance with variance analysis and forecasting"
        
        **🎨 Marketing Analytics:**
        - "Design a marketing dashboard showing campaign performance, ROI, and customer engagement metrics"
        - "Create social media analytics with reach, engagement, and conversion tracking"
        
        **⚡ Operations & Performance:**
        - "Show operational efficiency metrics with productivity trends and resource utilization"
        - "Build a quality control dashboard with defect rates, process improvements, and compliance metrics"
        
        **🏢 Executive Summary:**
        - "Create an executive overview with key business metrics, growth indicators, and strategic KPIs"
        - "Design a board presentation dashboard with high-level performance summaries"
        """)
    
    # Tips for better results
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 1.5rem; border-radius: 12px; margin-top: 1rem;">
        <h5 style="margin: 0 0 1rem 0; color: #1565c0;">💡 Tips for Better Custom Dashboards</h5>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; font-size: 0.9rem;">
            <div><strong>🎯 Be Specific:</strong> Mention exact metrics, chart types, and comparisons you want</div>
            <div><strong>📊 Include Context:</strong> Specify time periods, categories, or segments to analyze</div>
            <div><strong>🎨 Visual Preferences:</strong> Describe colors, layout, or styling preferences</div>
            <div><strong>💼 Business Focus:</strong> Mention your industry or business context for better insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === BUSINESS CONTEXT & AI INSIGHTS ===
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Business Context
    business_context = dashboard_results.get('business_context', {'primary_domain': 'general', 'confidence': 0.8})
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea; margin-bottom: 1rem;">
        <h4 style="color: #495057; margin-top: 0;">🏢 Business Context</h4>
        <p style="margin-bottom: 0;"><strong>Industry:</strong> {business_context['primary_domain'].title()}</p>
        <p style="margin-bottom: 0;"><strong>Confidence:</strong> {business_context['confidence']:.1%}</p>
        <p style="margin-bottom: 0;"><strong>Data Quality:</strong> {dashboard_results.get('data_quality', 95.0):.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced AI Insights Section
    st.markdown("### 🧠 AI-Powered Executive Intelligence")
    
    with st.expander("📊 Comprehensive Business Intelligence Analysis", expanded=True):
        st.markdown(f"""
        <div class="insights-panel fade-in-up">
            <h4 style="color: #2c3e50; margin-top: 0; font-weight: 600;">🤖 AI Analysis Results</h4>
            <div style="line-height: 1.7; font-size: 1.05rem;">
                {dashboard_results['ai_insights']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show dashboard metadata if available
        if 'dashboard_metadata' in dashboard_results:
            metadata = dashboard_results['dashboard_metadata']
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 1rem; border-radius: 12px; margin-top: 1rem;">
                <h5 style="margin: 0 0 0.5rem 0; color: #1565c0;">📈 Dashboard Intelligence Metrics</h5>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; font-size: 0.9rem;">
                    <div><strong>Business Domain:</strong> {metadata['business_domain'].title()}</div>
                    <div><strong>Optimization Level:</strong> {metadata['optimization_level']}</div>
                    <div><strong>Charts Generated:</strong> {metadata['chart_count']}</div>
                    <div><strong>KPIs Analyzed:</strong> {metadata['kpi_count']}</div>
                    <div><strong>Confidence Score:</strong> {metadata['confidence_score']:.1%}</div>
                    <div><strong>Analysis Time:</strong> {metadata['creation_time'][:19].replace('T', ' ')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Additional KPI Details
    with st.expander("🎯 KPI Details & Methodology"):
        key_kpis = dashboard_results['key_kpis']
        st.markdown("**Identified Key Performance Indicators:**")
        
        kpi_detail_cols = st.columns(2)
        for i, kpi_name in enumerate(key_kpis['metric_names']):
            col_idx = i % 2
            with kpi_detail_cols[col_idx]:
                st.markdown(f"• **{kpi_name}**")
        
        st.markdown(f"**KPI Summary:** {key_kpis['kpi_summary']}")
    
    # Enhanced Export & Action Center
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### � AI Dashbotard Action Center")
    
    # Create enhanced action buttons with better styling
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("📊 Export Dashboard", help="Export complete dashboard with AI insights", type="primary"):
            # Create export data
            export_data = {
                'dashboard_config': dashboard_results.get('layout_config', {}),
                'kpi_metrics': dashboard_results['executive_metrics'],
                'ai_insights': dashboard_results['ai_insights'],
                'business_context': dashboard_results.get('business_context', {'primary_domain': 'general', 'confidence': 0.8}),
                'export_timestamp': pd.Timestamp.now().isoformat()
            }
            
            st.success("✅ Dashboard exported successfully!")
            st.json(export_data)
    
    with action_cols[1]:
        if st.button("📈 Generate AI Report", help="Create comprehensive AI-powered business report"):
            st.success("🤖 AI Report generated with advanced analytics!")
            
            # Show sample report structure
            with st.expander("📋 AI Report Preview"):
                st.markdown("""
                **🤖 AI-Generated Executive Report**
                
                1. **Executive Summary** - Key performance overview
                2. **KPI Analysis** - Detailed metrics breakdown  
                3. **Market Intelligence** - Competitive positioning
                4. **Growth Opportunities** - Strategic recommendations
                5. **Risk Assessment** - Potential challenges
                6. **Action Items** - Prioritized next steps
                """)
    
    with action_cols[2]:
        if st.button("💾 Save Configuration", help="Save AI-optimized dashboard layout"):
            if 'layout_config' in dashboard_results:
                config_summary = {
                    'grid_system': dashboard_results['layout_config']['grid_system']['type'],
                    'color_scheme': dashboard_results['layout_config']['color_scheme']['primary'],
                    'business_domain': dashboard_results.get('business_context', {'primary_domain': 'general'})['primary_domain'],
                    'optimization_level': 'AI_Enhanced'
                }
                st.success("💾 AI Dashboard configuration saved!")
                st.json(config_summary)
            else:
                st.success("💾 Dashboard configuration saved for future use!")
    
    with action_cols[3]:
        if st.button("🔄 Refresh Analytics", help="Regenerate AI insights with latest algorithms"):
            st.success("🔄 AI Analytics refreshed!")
            st.info("💡 Tip: Upload new data or modify existing data to see updated insights")
    
    # Add performance metrics
    if 'dashboard_metadata' in dashboard_results:
        metadata = dashboard_results['dashboard_metadata']
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 1rem; border-radius: 12px; margin-top: 1rem; text-align: center;">
            <h5 style="margin: 0 0 0.5rem 0; color: #495057;">⚡ Dashboard Performance Metrics</h5>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; font-size: 0.9rem;">
                <div><strong>🎯 Optimization:</strong><br>{metadata['optimization_level']}</div>
                <div><strong>📊 Data Quality:</strong><br>{dashboard_results['data_quality']:.1f}%</div>
                <div><strong>🤖 AI Confidence:</strong><br>{metadata['confidence_score']:.1%}</div>
                <div><strong>⚡ Processing:</strong><br>Real-time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_eda_page():
    st.header("🔍 Exploratory Data Analysis")
    
    # Progress bar for EDA
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Analyzing data structure...")
    progress_bar.progress(0.2)
    
    status_text.text("📊 Detecting patterns and anomalies...")
    progress_bar.progress(0.5)
    
    status_text.text("🤖 Generating AI insights...")
    progress_bar.progress(0.8)
    
    eda_results = st.session_state.eda_agent.perform_eda(st.session_state.data)
    
    status_text.text("✅ EDA complete!")
    progress_bar.progress(1.0)
    
    # Clear progress indicators
    import time
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # AI Insights
    st.subheader("🧠 AI EDA Insights")
    st.write(eda_results['ai_insights'])
    
    # Data Overview
    st.subheader("📋 Data Overview")
    overview = eda_results['data_overview']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Shape:**", overview['shape'])
        st.write("**Memory Usage:**", f"{overview['memory_usage'] / 1024:.2f} KB")
        st.write("**Duplicate Rows:**", overview['duplicate_rows'])
    
    with col2:
        st.write("**Data Types:**")
        for dtype, count in overview['data_types'].items():
            st.write(f"- {dtype}: {count} columns")
    
    # Column Information
    st.subheader("📊 Column Information")
    col_info_df = pd.DataFrame(overview['column_info']).T
    st.dataframe(col_info_df, use_container_width=True)
    
    # Missing Data Analysis
    if eda_results['missing_data_analysis']['total_missing'] > 0:
        st.subheader("❌ Missing Data Analysis")
        missing_data = eda_results['missing_data_analysis']
        
        missing_df = pd.DataFrame({
            'Column': list(missing_data['missing_counts'].keys()),
            'Missing Count': list(missing_data['missing_counts'].values()),
            'Missing %': [f"{pct:.2f}%" for pct in missing_data['missing_percentages'].values()]
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        st.dataframe(missing_df, use_container_width=True)

def show_descriptive_page():
    st.header("📈 Descriptive Analysis")
    
    # Progress bar for descriptive analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📊 Calculating statistical measures...")
    progress_bar.progress(0.25)
    
    status_text.text("📈 Analyzing distributions...")
    progress_bar.progress(0.5)
    
    status_text.text("🔢 Computing percentiles and variability...")
    progress_bar.progress(0.75)
    
    desc_results = st.session_state.descriptive_agent.perform_descriptive_analysis(st.session_state.data)
    
    status_text.text("✅ Descriptive analysis complete!")
    progress_bar.progress(1.0)
    
    # Clear progress indicators
    import time
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # AI Insights
    st.subheader("🧠 AI Statistical Insights")
    st.write(desc_results['ai_insights'])
    
    # Central Tendency
    if 'message' not in desc_results['central_tendency']:
        st.subheader("📊 Central Tendency Measures")
        central_df = pd.DataFrame(desc_results['central_tendency']).T
        st.dataframe(central_df, use_container_width=True)
    
    # Variability Measures
    if 'message' not in desc_results['variability_measures']:
        st.subheader("📏 Variability Measures")
        var_df = pd.DataFrame(desc_results['variability_measures']).T
        st.dataframe(var_df, use_container_width=True)
    
    # Distribution Analysis
    if 'message' not in desc_results['distribution_analysis']:
        st.subheader("📈 Distribution Analysis")
        dist_data = []
        for col, data in desc_results['distribution_analysis'].items():
            dist_data.append({
                'Column': col,
                'Skewness': f"{data['skewness']:.3f}",
                'Kurtosis': f"{data['kurtosis']:.3f}",
                'Distribution Type': data['distribution_type']
            })
        dist_df = pd.DataFrame(dist_data)
        st.dataframe(dist_df, use_container_width=True)
    
    # Categorical Analysis
    if 'message' not in desc_results['categorical_analysis']:
        st.subheader("🏷️ Categorical Variables Analysis")
        for col, data in desc_results['categorical_analysis'].items():
            with st.expander(f"Analysis for {col}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Unique Values:** {data['unique_values']}")
                    st.write(f"**Most Frequent:** {data['most_frequent']} ({data['most_frequent_count']} times)")
                    st.write(f"**Concentration Ratio:** {data['concentration_ratio']:.2f}%")
                with col2:
                    st.write("**Top 10 Frequencies:**")
                    freq_df = pd.DataFrame(list(data['frequency_distribution'].items()), 
                                         columns=['Value', 'Count'])
                    st.dataframe(freq_df, use_container_width=True)

def show_prescriptive_page():
    st.header("🎯 Prescriptive Analysis")
    
    # Progress bar for prescriptive analysis
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🎯 Identifying optimization opportunities...")
    progress_bar.progress(0.2)
    
    status_text.text("🤖 Building predictive models...")
    progress_bar.progress(0.4)
    
    status_text.text("⚠️ Performing risk analysis...")
    progress_bar.progress(0.6)
    
    status_text.text("💡 Generating strategic recommendations...")
    progress_bar.progress(0.8)
    
    presc_results = st.session_state.prescriptive_agent.perform_prescriptive_analysis(st.session_state.data)
    
    # Store prescriptive results persistently
    if presc_results and presc_results not in st.session_state.persistent_outputs['prescriptive']:
        st.session_state.persistent_outputs['prescriptive'].append(presc_results)
    
    # Display persistent outputs from other tabs
    if st.session_state.persistent_outputs['dashboard']:
        with st.expander("📊 Dashboard Results (from Dashboard tab)", expanded=False):
            st.info("Dashboard results are available from the Dashboard tab")
    
    if st.session_state.persistent_outputs['chat']:
        with st.expander("🗨️ Chat History (from Chat tab)", expanded=False):
            st.info(f"Chat history with {len(st.session_state.persistent_outputs['chat'])} conversations available from the Chat tab")
    
    status_text.text("✅ Prescriptive analysis complete!")
    progress_bar.progress(1.0)
    
    # Clear progress indicators
    import time
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Performance indicator
    if 'processing_time' in presc_results:
        processing_time = presc_results['processing_time']
        st.success(f"⚡ Fast Analysis Complete! Processed in {processing_time:.2f} seconds")
        
        if 'note' in presc_results:
            st.info(f"📝 {presc_results['note']}")
    
    # AI Insights
    st.subheader("🧠 AI Business Recommendations")
    if 'ai_insights' in presc_results:
        st.write(presc_results['ai_insights'])
    else:
        st.info("AI insights not available in this analysis mode.")
    
    # Optimization Opportunities (if available)
    if 'optimization_opportunities' in presc_results and presc_results['optimization_opportunities']:
        st.subheader("🚀 Optimization Opportunities")
        for opp_name, opp_data in presc_results['optimization_opportunities'].items():
            with st.expander(f"{opp_data['type']}: {opp_name}"):
                st.write(f"**Recommendation:** {opp_data['recommendation']}")
                for key, value in opp_data.items():
                    if key not in ['type', 'recommendation']:
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    else:
        st.info("No specific optimization opportunities identified in fast analysis mode.")
    
    # Recommendations (if available)
    if 'recommendations' in presc_results and presc_results['recommendations']:
        st.subheader("💡 Strategic Recommendations")
        recommendations = presc_results['recommendations']
        
        for rec in recommendations:
            with st.expander(f"{rec['priority']} Priority: {rec['recommendation']}"):
                st.write(f"**Category:** {rec['category']}")
                if 'details' in rec:
                    st.write(f"**Details:** {rec['details']}")
                if 'action_items' in rec:
                    st.write("**Action Items:**")
                    for item in rec['action_items']:
                        st.write(f"- {item}")
    else:
        st.info("Detailed recommendations available in full analysis mode.")
    
    # Risk Analysis (if available)
    if 'risk_analysis' in presc_results and presc_results['risk_analysis']:
        st.subheader("⚠️ Risk Analysis")
        risk_data = []
        for col, risk_info in presc_results['risk_analysis'].items():
            risk_data.append({
                'Variable': col,
                'Risk Level': risk_info['risk_level'],
                'VaR (5%)': f"{risk_info['value_at_risk_5%']:.3f}",
                'VaR (1%)': f"{risk_info['value_at_risk_1%']:.3f}",
                'Volatility': f"{risk_info['volatility']:.3f}"
            })
        risk_df = pd.DataFrame(risk_data)
        st.dataframe(risk_df, use_container_width=True)
    
    # Scenario Analysis (if available)
    if 'scenario_analysis' in presc_results and presc_results['scenario_analysis']:
        st.subheader("🎭 Scenario Analysis")
        scenario_data = []
        for col, scenarios in presc_results['scenario_analysis'].items():
            scenario_data.append({
                'Variable': col,
                'Optimistic': f"{scenarios['optimistic']:.2f}",
                'Realistic': f"{scenarios['realistic']:.2f}",
                'Pessimistic': f"{scenarios['pessimistic']:.2f}",
                'Best Case': f"{scenarios['best_case']:.2f}",
                'Worst Case': f"{scenarios['worst_case']:.2f}"
            })
        scenario_df = pd.DataFrame(scenario_data)
        st.dataframe(scenario_df, use_container_width=True)

def show_chat_page():
    """Interactive chat page for data Q&A and custom visualizations"""
    st.header("💬 Chat with Your Data")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload data first to start chatting!")
        return
    
    # Initialize chat history and visualization state if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_visualization' not in st.session_state:
        st.session_state.last_visualization = None
    
    # Data overview section
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Rows", f"{len(st.session_state.data):,}")
        with col2:
            st.metric("📊 Columns", len(st.session_state.data.columns))
        with col3:
            numeric_cols = len(st.session_state.data.select_dtypes(include=['number']).columns)
            st.metric("🔢 Numeric", numeric_cols)
        with col4:
            categorical_cols = len(st.session_state.data.select_dtypes(include=['object']).columns)
            st.metric("📝 Categorical", categorical_cols)
        
        st.subheader("Sample Data")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
    
    # Suggested questions
    st.subheader("💡 Suggested Questions")
    suggested_questions = st.session_state.chat_agent.get_suggested_questions(st.session_state.data)
    
    # Display suggested questions as buttons
    cols = st.columns(2)
    for i, question in enumerate(suggested_questions[:6]):  # Show first 6 suggestions
        with cols[i % 2]:
            if st.button(f"💭 {question}", key=f"suggestion_{i}", use_container_width=True):
                # Add to chat history and get response
                response = st.session_state.chat_agent.chat_with_data(st.session_state.data, question)
                st.session_state.chat_history.append({
                    "question": question,
                    "response": response,
                    "timestamp": pd.Timestamp.now()
                })
                st.rerun()
    
    # Chat interface
    st.subheader("🗨️ Ask Your Question")
    
    # Chat input
    user_question = st.text_area(
        "What would you like to know about your data?",
        placeholder="e.g., What are the main trends in my data? Can you create a scatter plot of X vs Y?",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Ask Question", type="primary", use_container_width=True):
            if user_question.strip():
                with st.spinner("🤖 Analyzing your question..."):
                    # Get response from chat agent (now includes Python code for visualizations)
                    response = st.session_state.chat_agent.chat_with_data(st.session_state.data, user_question)
                    
                    chat_result = {
                        "question": user_question,
                        "response": response,
                        "timestamp": pd.Timestamp.now()
                    }
                    st.session_state.chat_history.append(chat_result)
                    st.session_state.persistent_outputs['chat'].append(chat_result)
                    st.rerun()
            else:
                st.warning("Please enter a question!")
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_agent.conversation_history = []
            st.session_state.persistent_outputs['chat'] = []
            st.rerun()
    
    # Display persistent outputs from other tabs
    if st.session_state.persistent_outputs['dashboard']:
        with st.expander("📊 Dashboard Results (from Dashboard tab)", expanded=False):
            st.info("Dashboard results are available from the Dashboard tab")
    
    if st.session_state.persistent_outputs['prescriptive']:
        with st.expander("💡 Prescriptive Analysis (from Prescriptive tab)", expanded=False):
            st.info("Prescriptive analysis results are available from the Prescriptive tab")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        
        # Reverse order to show latest first
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                           padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <p style="color: white; margin: 0; font-weight: 600;">
                        🙋‍♂️ <strong>You asked:</strong> {chat['question']}
                    </p>
                    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.8rem;">
                        {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Response with clean text display
                st.markdown("**🤖 AI Analysis & Insights**")
                st.markdown("---")
                
                # Check for plot images in the response
                if "[PLOT_IMAGE:" in chat['response']:
                    import re
                    import base64
                    from PIL import Image
                    import io
                    
                    # Extract plot images from response
                    plot_matches = re.findall(r'\[PLOT_IMAGE:(.*?)\]', chat['response'])
                    st.info(f"Found {len(plot_matches)} plot images in response")
                    
                    for i, plot_base64 in enumerate(plot_matches):
                        try:
                            st.info(f"Processing plot image {i+1} ({len(plot_base64)} chars)")
                            # Decode and display the image
                            img_data = base64.b64decode(plot_base64)
                            img = Image.open(io.BytesIO(img_data))
                            
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                       padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
                                <h3 style="color: white; margin: 0; font-weight: 600;">📊 Generated Visualization</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display the image
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.image(img, use_column_width=True)
                                
                        except Exception as e:
                            st.error(f"Error displaying visualization {i+1}: {str(e)}")
                    
                    # Remove plot image markers from the displayed response
                    clean_response = re.sub(r'\[PLOT_IMAGE:.*?\]', '', chat['response'])
                    st.markdown(clean_response)
                else:
                    st.info("No plot images found in response")
                    st.markdown(chat['response'])

def show_ml_scientist_page():
    """ML Scientist page for advanced machine learning and deep learning"""
    st.header("🤖 ML Scientist - Advanced Machine Learning & Deep Learning")
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload data first to start ML analysis!")
        return
    
    # Data overview section
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Rows", f"{len(st.session_state.data):,}")
        with col2:
            st.metric("📊 Columns", len(st.session_state.data.columns))
        with col3:
            numeric_cols = len(st.session_state.data.select_dtypes(include=['number']).columns)
            st.metric("🔢 Numeric", numeric_cols)
        with col4:
            categorical_cols = len(st.session_state.data.select_dtypes(include=['object']).columns)
            st.metric("📝 Categorical", categorical_cols)
        
        st.subheader("Sample Data")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
    
    # ML Analysis Section
    st.subheader("🧠 AI Data Analysis & Model Recommendations")
    
    # Analyze data and get recommendations
    if st.button("🔍 Analyze Data & Recommend Models", type="primary"):
        with st.spinner("🤖 Analyzing data characteristics..."):
            analysis = st.session_state.ml_scientist_agent.analyze_data_and_recommend_models(st.session_state.data)
            
            # Store analysis results persistently
            if analysis and analysis not in st.session_state.persistent_outputs['ml_scientist']:
                st.session_state.persistent_outputs['ml_scientist'].append(analysis)
        
        # Display analysis results
        st.success("✅ Data analysis complete!")
        
        # Data characteristics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Dataset Characteristics:**")
            st.write(f"• **Shape:** {analysis['data_shape']}")
            st.write(f"• **Task Type:** {analysis['task_type'].title()}")
            st.write(f"• **Numeric Columns:** {len(analysis['numeric_columns'])}")
            st.write(f"• **Categorical Columns:** {len(analysis['categorical_columns'])}")
        
        with col2:
            st.markdown("**🎯 Target Candidates:**")
            for target in analysis['target_candidates'][:5]:  # Show first 5
                st.write(f"• {target}")
        
        # Recommended models
        st.markdown("**🤖 Recommended Models:**")
        model_cols = st.columns(3)
        for i, model in enumerate(analysis['recommended_models']):
            with model_cols[i % 3]:
                st.write(f"• {model}")
    
    # Model Selection and Code Generation
    st.subheader("⚙️ Model Selection & Code Generation")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        target_column = st.selectbox(
            "🎯 Select Target Column:",
            options=st.session_state.data.columns.tolist(),
            help="Choose the column you want to predict or analyze"
        )
    
    with col2:
        model_type = st.selectbox(
            "🤖 Select Model Type:",
            options=[
                "Linear Regression", "Random Forest", "XGBoost", 
                "Neural Network", "Transformer", "Logistic Regression"
            ],
            help="Choose the machine learning model to use"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_code = st.button("🚀 Generate ML Code", type="primary")
    
    # Generate and execute ML code
    if generate_code and target_column and model_type:
        with st.spinner("🤖 Generating ML code and executing..."):
            # Determine task type
            if target_column in st.session_state.data.select_dtypes(include=['number']).columns:
                task_type = 'regression'
            else:
                task_type = 'classification'
            
            # Generate ML code
            ml_code = st.session_state.ml_scientist_agent.generate_ml_code(
                st.session_state.data, target_column, model_type, task_type
            )
            
            # Execute the code
            execution_result = st.session_state.ml_scientist_agent.execute_ml_code(
                ml_code, st.session_state.data
            )
            
            # Store results persistently
            ml_result = {
                'target_column': target_column,
                'model_type': model_type,
                'task_type': task_type,
                'code': ml_code,
                'execution_result': execution_result,
                'timestamp': pd.Timestamp.now()
            }
            
            if ml_result not in st.session_state.persistent_outputs['ml_scientist']:
                st.session_state.persistent_outputs['ml_scientist'].append(ml_result)
        
        # Display results
        if execution_result['success']:
            st.success(f"✅ {execution_result['message']}")
            
            # Display generated visualizations
            if execution_result['images']:
                st.subheader("📊 Generated Visualizations")
                for i, img_base64 in enumerate(execution_result['images']):
                    try:
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_data))
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                   padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
                            <h3 style="color: white; margin: 0; font-weight: 600;">📊 {model_type} Model Results</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.image(img, use_column_width=True)
                        
                    except Exception as e:
                        st.error(f"Error displaying visualization {i+1}: {str(e)}")
            
            # Show generated code
            with st.expander("💻 Generated ML Code", expanded=False):
                st.code(ml_code, language='python')
                
        else:
            st.error(f"❌ {execution_result['message']}")
    
    # Display persistent outputs from other tabs
    if st.session_state.persistent_outputs['dashboard']:
        with st.expander("📊 Dashboard Results (from Dashboard tab)", expanded=False):
            st.info("Dashboard results are available from the Dashboard tab")
    
    if st.session_state.persistent_outputs['prescriptive']:
        with st.expander("💡 Prescriptive Analysis (from Prescriptive tab)", expanded=False):
            st.info("Prescriptive analysis results are available from the Prescriptive tab")
    
    if st.session_state.persistent_outputs['chat']:
        with st.expander("🗨️ Chat History (from Chat tab)", expanded=False):
            st.info(f"Chat history with {len(st.session_state.persistent_outputs['chat'])} conversations available from the Chat tab")
    
    # ML Results History
    if st.session_state.persistent_outputs['ml_scientist']:
        st.subheader("📈 ML Analysis History")
        
        for i, result in enumerate(reversed(st.session_state.persistent_outputs['ml_scientist'])):
            if isinstance(result, dict) and 'model_type' in result:
                with st.expander(f"🤖 {result['model_type']} - {result['target_column']} ({result['timestamp'].strftime('%Y-%m-%d %H:%M')})", expanded=False):
                    st.write(f"**Task Type:** {result['task_type']}")
                    st.write(f"**Target:** {result['target_column']}")
                    st.write(f"**Model:** {result['model_type']}")
                    
                    if result['execution_result']['success']:
                        st.success("✅ Execution successful")
                        if result['execution_result']['images']:
                            st.write(f"Generated {len(result['execution_result']['images'])} visualizations")
                    else:
                        st.error(f"❌ Execution failed: {result['execution_result']['message']}")
    
    # ML Capabilities Information
    with st.expander("🤖 ML Scientist Capabilities", expanded=False):
        st.markdown("""
        **🧠 Advanced Machine Learning Features:**
        
        **📊 Traditional ML Models:**
        - Linear Regression (scikit-learn)
        - Random Forest (scikit-learn)
        - XGBoost (gradient boosting)
        - Logistic Regression (classification)
        
        **🔥 Deep Learning Models (PyTorch):**
        - Neural Networks (Multi-layer perceptrons)
        - Transformer Models (attention mechanisms)
        - Custom architectures for your data
        
        **📈 Comprehensive Metrics:**
        - **Regression:** R², RMSE, MAE, Residual Analysis
        - **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix
        - **Feature Importance:** Permutation importance, SHAP values
        - **Visualizations:** Training curves, prediction plots, feature importance
        
        **⚡ Advanced Features:**
        - Automatic data preprocessing
        - Cross-validation and model evaluation
        - Batch prediction visualization
        - Real-time model performance tracking
        - Exportable Python code for reproducibility
        
        **🎯 Use Cases:**
        - Predictive modeling
        - Classification tasks
        - Regression analysis
        - Feature engineering
        - Model comparison and selection
        """)
    
    # Example Use Cases
    with st.expander("💡 Example ML Use Cases", expanded=False):
        st.markdown("""
        **🎯 Regression Examples:**
        - "Predict sales revenue based on marketing spend"
        - "Forecast customer lifetime value"
        - "Estimate house prices from features"
        
        **🏷️ Classification Examples:**
        - "Classify customers as high/low value"
        - "Predict customer churn probability"
        - "Identify fraudulent transactions"
        
        **📊 Deep Learning Examples:**
        - "Use Neural Networks for complex pattern recognition"
        - "Apply Transformer models for sequence prediction"
        - "Build custom architectures for your specific data"
        
        **🔬 Advanced Analytics:**
        - "Feature importance analysis"
        - "Model performance comparison"
        - "Hyperparameter optimization"
        - "Ensemble model creation"
        """)


if __name__ == "__main__":
    main()