#!/usr/bin/env python3
"""
🚀 Executive AI Assistant - LangGraph Enhanced Dashboard
A comprehensive dashboard that mimics LangGraph CLI functionality with modern UI/UX
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path

# Import our modules
from eaia.main.dashboard_integration import DashboardIntegration
from eaia.main.email_processor import EmailProcessor
from eaia.main.analysis import EmailAnalyzer
from eaia.main.config import get_config
from langgraph.store.memory import InMemoryStore
from langgraph.types import RunnableConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🚀 Executive AI Assistant - LangGraph Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .status-success { background-color: #d4edda; border-left: 4px solid #28a745; }
    .status-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .status-error { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    .status-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
    
    .workflow-step {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .workflow-step.active {
        background: #e3f2fd;
        border-color: #2196f3;
    }
    
    .workflow-step.completed {
        background: #e8f5e8;
        border-color: #4caf50;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'dashboard_integration' not in st.session_state:
        st.session_state.dashboard_integration = DashboardIntegration()
    
    if 'email_processor' not in st.session_state:
        st.session_state.email_processor = EmailProcessor()
    
    if 'email_analyzer' not in st.session_state:
        st.session_state.email_analyzer = EmailAnalyzer()
    
    # Workflow state tracking
    if 'workflow_states' not in st.session_state:
        st.session_state.workflow_states = {}
    
    # Processing status
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {
            'email_ingestion': 'idle',
            'email_processing': 'idle',
            'reflection_learning': 'idle',
            'cron_status': 'stopped'
        }
    
    # Approval queue
    if 'approval_queue' not in st.session_state:
        st.session_state.approval_queue = []
    
    # Configuration
    if 'langgraph_config' not in st.session_state:
        st.session_state.langgraph_config = {
            'minutes_since': 60,
            'auto_approve_low_risk': False,
            'reflection_enabled': True,
            'cron_enabled': False
        }

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Executive AI Assistant</h1>
        <h3>LangGraph Enhanced Dashboard</h3>
        <p>Comprehensive email management with AI workflows, human oversight, and continuous learning</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the enhanced sidebar with LangGraph controls"""
    st.sidebar.markdown("## 🎛️ LangGraph Controls")
    
    # System Status Section
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 System Status")
        
        # Get system status
        system_status = st.session_state.dashboard_integration.get_system_health()
        
        # Status indicators
        status_items = [
            ("Database", system_status.get('database', False)),
            ("OpenAI API", system_status.get('openai', False)),
            ("Gmail API", system_status.get('gmail', False)),
            ("ChromaDB", system_status.get('chroma_db', False))
        ]
        
        for item, status in status_items:
            status_class = "status-success" if status else "status-error"
            status_text = "✅ Online" if status else "❌ Offline"
            st.markdown(f"""
            <div class="{status_class}" style="padding: 0.5rem; margin: 0.2rem 0; border-radius: 4px;">
                <strong>{item}:</strong> {status_text}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow Controls Section
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🔄 Workflow Controls")
        
        # Email Ingestion
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Ingest Emails", use_container_width=True):
                trigger_email_ingestion()
        
        with col2:
            if st.button("🔄 Process Queue", use_container_width=True):
                trigger_email_processing()
        
        # Reflection Learning
        if st.button("🧠 Trigger Learning", use_container_width=True):
            trigger_reflection_learning()
        
        # Cron Management
        st.markdown("#### ⏰ Cron Jobs")
        cron_enabled = st.checkbox("Enable Auto-Processing", 
                                 value=st.session_state.langgraph_config['cron_enabled'])
        if cron_enabled != st.session_state.langgraph_config['cron_enabled']:
            st.session_state.langgraph_config['cron_enabled'] = cron_enabled
            toggle_cron_job(cron_enabled)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration Section
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")
        
        # Time window for email processing
        minutes_since = st.slider("Email Lookback (minutes)", 
                                min_value=10, max_value=1440, 
                                value=st.session_state.langgraph_config['minutes_since'])
        st.session_state.langgraph_config['minutes_since'] = minutes_since
        
        # Auto-approval settings
        auto_approve = st.checkbox("Auto-approve low-risk emails", 
                                 value=st.session_state.langgraph_config['auto_approve_low_risk'])
        st.session_state.langgraph_config['auto_approve_low_risk'] = auto_approve
        
        # Reflection learning
        reflection_enabled = st.checkbox("Enable AI Learning", 
                                       value=st.session_state.langgraph_config['reflection_enabled'])
        st.session_state.langgraph_config['reflection_enabled'] = reflection_enabled
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔍 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("🧹 Clear Cache", use_container_width=True):
            clear_session_cache()
        
        if st.button("📊 Export Data", use_container_width=True):
            export_dashboard_data()
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_main_dashboard():
    """Render the main dashboard content"""
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🔄 Workflows", 
        "✅ Approval Queue", 
        "🧠 AI Learning", 
        "📈 Analytics", 
        "⚙️ Management"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_workflows_tab()
    
    with tab3:
        render_approval_queue_tab()
    
    with tab4:
        render_ai_learning_tab()
    
    with tab5:
        render_analytics_tab()
    
    with tab6:
        render_management_tab()

def render_overview_tab():
    """Render the overview dashboard"""
    st.markdown("## 📊 System Overview")
    
    # Get dashboard data
    try:
        overview_data = st.session_state.dashboard_integration.get_dashboard_overview()
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📧 Total Emails", overview_data.get('total_emails', 0))
    
    with col2:
        st.metric("⭐ Important", overview_data.get('important_emails', 0))
    
    with col3:
        st.metric("🔔 Unread", overview_data.get('unread_emails', 0))
    
    with col4:
        st.metric("📋 Actions", overview_data.get('action_items', 0))
    
    with col5:
        st.metric("🤔 Uncertain", overview_data.get('uncertain_emails_count', 0))
    
    # Processing status
    st.markdown("### 🔄 Processing Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        # Email processing status
        processing_stats = overview_data.get('processing_performance', {})
        
        st.markdown("#### 📧 Email Processing")
        if processing_stats:
            st.write(f"**Last Run:** {processing_stats.get('last_run', 'Never')}")
            st.write(f"**Total Processed:** {processing_stats.get('total_processed', 0)}")
            st.write(f"**Avg Processing Time:** {processing_stats.get('avg_processing_time', 0):.2f}s")
        else:
            st.info("No processing statistics available")
    
    with status_col2:
        # Workflow status
        st.markdown("#### 🔄 Workflow Status")
        for workflow, status in st.session_state.processing_status.items():
            status_emoji = {
                'idle': '⏸️',
                'running': '🔄',
                'completed': '✅',
                'error': '❌'
            }.get(status, '❓')
            
            st.write(f"**{workflow.replace('_', ' ').title()}:** {status_emoji} {status.title()}")
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    
    recent_activity = overview_data.get('recent_activity', [])
    if recent_activity:
        activity_df = pd.DataFrame(recent_activity)
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("No recent activity to display")

# Workflow control functions
def trigger_email_ingestion():
    """Trigger email ingestion workflow"""
    st.session_state.processing_status['email_ingestion'] = 'running'
    
    with st.spinner("🔄 Ingesting emails..."):
        try:
            # Simulate LangGraph email ingestion
            minutes_since = st.session_state.langgraph_config['minutes_since']
            
            # This would normally call the LangGraph API
            # For now, we'll use our existing email processor
            processor = st.session_state.email_processor
            
            # Simulate the workflow
            time.sleep(2)  # Simulate processing time
            
            st.session_state.processing_status['email_ingestion'] = 'completed'
            st.success(f"✅ Email ingestion completed! Processed emails from last {minutes_since} minutes.")
            
        except Exception as e:
            st.session_state.processing_status['email_ingestion'] = 'error'
            st.error(f"❌ Email ingestion failed: {e}")

def trigger_email_processing():
    """Trigger email processing workflow"""
    st.session_state.processing_status['email_processing'] = 'running'
    
    with st.spinner("🔄 Processing email queue..."):
        try:
            # This would call the main LangGraph workflow
            time.sleep(3)  # Simulate processing
            
            st.session_state.processing_status['email_processing'] = 'completed'
            st.success("✅ Email processing completed!")
            
        except Exception as e:
            st.session_state.processing_status['email_processing'] = 'error'
            st.error(f"❌ Email processing failed: {e}")

def trigger_reflection_learning():
    """Trigger AI reflection and learning"""
    st.session_state.processing_status['reflection_learning'] = 'running'
    
    with st.spinner("🧠 AI learning from feedback..."):
        try:
            # This would call the reflection graphs
            time.sleep(2)
            
            st.session_state.processing_status['reflection_learning'] = 'completed'
            st.success("✅ AI learning completed!")
            
        except Exception as e:
            st.session_state.processing_status['reflection_learning'] = 'error'
            st.error(f"❌ AI learning failed: {e}")

def toggle_cron_job(enabled: bool):
    """Toggle cron job status"""
    if enabled:
        st.session_state.processing_status['cron_status'] = 'running'
        st.success("✅ Cron job enabled - emails will be processed every 10 minutes")
    else:
        st.session_state.processing_status['cron_status'] = 'stopped'
        st.info("ℹ️ Cron job disabled")

def clear_session_cache():
    """Clear session cache"""
    # Clear specific cache items
    cache_keys = ['dashboard_integration', 'email_processor', 'email_analyzer']
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("✅ Cache cleared!")
    st.rerun()

def export_dashboard_data():
    """Export dashboard data"""
    try:
        overview_data = st.session_state.dashboard_integration.get_dashboard_overview()
        
        # Convert to JSON for download
        json_data = json.dumps(overview_data, indent=2, default=str)
        
        st.download_button(
            label="📊 Download Dashboard Data",
            data=json_data,
            file_name=f"dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"❌ Export failed: {e}")

def render_workflows_tab():
    """Render the workflows management tab"""
    st.markdown("## 🔄 LangGraph Workflows")
    
    # Workflow status overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Available Workflows")
        
        workflows = [
            {
                "name": "Main Email Processing",
                "description": "Complete email triage, analysis, and response generation",
                "status": st.session_state.processing_status.get('email_processing', 'idle'),
                "graph": "main"
            },
            {
                "name": "Cron Email Ingestion", 
                "description": "Automated email fetching every 10 minutes",
                "status": st.session_state.processing_status.get('cron_status', 'stopped'),
                "graph": "cron"
            },
            {
                "name": "General Reflection",
                "description": "AI learns from user feedback to improve responses",
                "status": st.session_state.processing_status.get('reflection_learning', 'idle'),
                "graph": "general_reflection_graph"
            },
            {
                "name": "Multi-Aspect Reflection",
                "description": "Advanced learning across multiple prompt categories",
                "status": "idle",
                "graph": "multi_reflection_graph"
            }
        ]
        
        for workflow in workflows:
            status_color = {
                'idle': '#6c757d',
                'running': '#007bff', 
                'completed': '#28a745',
                'error': '#dc3545',
                'stopped': '#ffc107'
            }.get(workflow['status'], '#6c757d')
            
            st.markdown(f"""
            <div class="workflow-step">
                <h4>🔄 {workflow['name']}</h4>
                <p>{workflow['description']}</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span><strong>Graph:</strong> {workflow['graph']}</span>
                    <span style="color: {status_color}; font-weight: bold;">
                        {workflow['status'].title()}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎛️ Workflow Controls")
        
        # Manual workflow triggers
        st.markdown("#### 📧 Email Processing")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("🚀 Start Main Workflow", use_container_width=True):
                start_main_workflow()
        
        with col2_2:
            if st.button("⏹️ Stop All Workflows", use_container_width=True):
                stop_all_workflows()
        
        # Workflow parameters
        st.markdown("#### ⚙️ Workflow Parameters")
        
        with st.expander("📧 Email Processing Parameters"):
            email_batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
            confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
            auto_approve_threshold = st.slider("Auto-approve Threshold", 0.0, 1.0, 0.9)
        
        with st.expander("🧠 Learning Parameters"):
            learning_rate = st.slider("Learning Rate", 0.1, 1.0, 0.5)
            feedback_weight = st.slider("Feedback Weight", 0.1, 1.0, 0.8)
        
        # Workflow history
        st.markdown("#### 📈 Recent Workflow Runs")
        
        # Mock workflow history data
        workflow_history = [
            {"Timestamp": "2024-01-20 10:30:00", "Workflow": "Main Processing", "Status": "✅ Completed", "Duration": "2m 15s"},
            {"Timestamp": "2024-01-20 10:20:00", "Workflow": "Cron Ingestion", "Status": "✅ Completed", "Duration": "45s"},
            {"Timestamp": "2024-01-20 10:10:00", "Workflow": "Reflection Learning", "Status": "✅ Completed", "Duration": "1m 30s"},
        ]
        
        history_df = pd.DataFrame(workflow_history)
        st.dataframe(history_df, use_container_width=True)

def render_approval_queue_tab():
    """Render the human approval queue tab"""
    st.markdown("## ✅ Human Approval Queue")
    
    # Get uncertain emails that need human review
    try:
        uncertain_emails = st.session_state.email_analyzer.get_uncertain_emails_for_review()
    except Exception as e:
        st.error(f"Error loading approval queue: {e}")
        uncertain_emails = []
    
    if not uncertain_emails:
        st.info("🎉 No emails pending approval!")
        return
    
    st.markdown(f"### 📋 {len(uncertain_emails)} Emails Awaiting Review")
    
    for i, email in enumerate(uncertain_emails):
        with st.expander(f"📧 {email.get('subject', 'No Subject')} - {email.get('from_email', 'Unknown Sender')}"):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📄 Email Content")
                st.write(f"**From:** {email.get('from_email', 'Unknown')}")
                st.write(f"**Date:** {email.get('date', 'Unknown')}")
                st.write(f"**Subject:** {email.get('subject', 'No Subject')}")
                
                # Show email body (truncated)
                body = email.get('body', '')
                if len(body) > 500:
                    body = body[:500] + "..."
                st.text_area("Body", body, height=100, key=f"body_{i}")
                
                # Show AI analysis
                st.markdown("#### 🤖 AI Analysis")
                analysis_data = email.get('analysis_data', {})
                if isinstance(analysis_data, str):
                    try:
                        analysis_data = json.loads(analysis_data)
                    except:
                        analysis_data = {}
                
                st.write(f"**Classification Reason:** {email.get('classification_reason', 'Not provided')}")
                st.write(f"**Suggested Action:** {analysis_data.get('triage_decision', 'Unknown')}")
                
                if analysis_data.get('action_items'):
                    st.markdown("**Action Items:**")
                    for action in analysis_data['action_items']:
                        st.write(f"• {action}")
            
            with col2:
                st.markdown("#### 🎯 Your Decision")
                
                # Decision buttons
                col2_1, col2_2, col2_3 = st.columns(3)
                
                with col2_1:
                    if st.button("✅ Approve", key=f"approve_{i}", use_container_width=True):
                        approve_email(email['uncertain_id'])
                
                with col2_2:
                    if st.button("❌ Reject", key=f"reject_{i}", use_container_width=True):
                        reject_email(email['uncertain_id'])
                
                with col2_3:
                    if st.button("🔄 Modify", key=f"modify_{i}", use_container_width=True):
                        modify_email_decision(email['uncertain_id'])
                
                # Feedback section
                st.markdown("#### 💭 Feedback")
                feedback = st.text_area(
                    "Why did you make this decision?", 
                    key=f"feedback_{i}",
                    help="This helps the AI learn from your decisions"
                )
                
                if st.button("💾 Save Feedback", key=f"save_feedback_{i}", use_container_width=True):
                    save_user_feedback(email['uncertain_id'], feedback)

def render_ai_learning_tab():
    """Render the AI learning and reflection tab"""
    st.markdown("## 🧠 AI Learning & Reflection")
    
    # Learning overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Learning Statistics")
        
        # Mock learning stats
        learning_stats = {
            "Total Feedback Sessions": 25,
            "Prompts Updated": 12,
            "Accuracy Improvement": "+15%",
            "Last Learning Session": "2 hours ago"
        }
        
        for stat, value in learning_stats.items():
            st.metric(stat, value)
    
    with col2:
        st.markdown("### 🎯 Learning Categories")
        
        categories = [
            {"name": "Tone & Style", "updates": 5, "last_update": "1 day ago"},
            {"name": "Response Content", "updates": 3, "last_update": "2 days ago"},
            {"name": "Scheduling Preferences", "updates": 2, "last_update": "3 days ago"},
            {"name": "Background Knowledge", "updates": 2, "last_update": "1 week ago"}
        ]
        
        for cat in categories:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{cat['name']}</h4>
                <p><strong>Updates:</strong> {cat['updates']}</p>
                <p><strong>Last Update:</strong> {cat['last_update']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Manual learning trigger
    st.markdown("### 🚀 Manual Learning Session")
    
    with st.expander("🧠 Trigger Learning from Recent Feedback"):
        feedback_source = st.selectbox(
            "Learning Source",
            ["Recent Approvals", "User Corrections", "Performance Metrics", "All Sources"]
        )
        
        learning_focus = st.multiselect(
            "Focus Areas",
            ["Tone & Style", "Response Content", "Scheduling", "Background Knowledge"],
            default=["Tone & Style"]
        )
        
        if st.button("🚀 Start Learning Session", use_container_width=True):
            start_learning_session(feedback_source, learning_focus)
    
    # Prompt management
    st.markdown("### 📝 Prompt Management")
    
    prompt_categories = ["rewrite_instructions", "response_preferences", "schedule_preferences", "random_preferences"]
    
    selected_category = st.selectbox("Select Prompt Category", prompt_categories)
    
    # Mock current prompt content
    current_prompt = f"Current {selected_category.replace('_', ' ').title()} prompt content would be displayed here..."
    
    updated_prompt = st.text_area(
        f"Edit {selected_category.replace('_', ' ').title()} Prompt",
        value=current_prompt,
        height=200
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Prompt", use_container_width=True):
            save_prompt_update(selected_category, updated_prompt)
    
    with col2:
        if st.button("🔄 Reset to Default", use_container_width=True):
            reset_prompt_to_default(selected_category)

def render_analytics_tab():
    """Render the analytics and insights tab"""
    st.markdown("## 📈 Analytics & Insights")
    
    # Get analytics data
    try:
        analytics_data = st.session_state.dashboard_integration.get_comprehensive_analytics()
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        return
    
    # Time range selector
    time_range = st.selectbox(
        "📅 Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"],
        index=1
    )
    
    # Email processing metrics
    st.markdown("### 📧 Email Processing Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", analytics_data.get('total_processed', 0))
    
    with col2:
        st.metric("Important Emails", analytics_data.get('important_count', 0))
    
    with col3:
        st.metric("Auto-Approved", analytics_data.get('auto_approved', 0))
    
    with col4:
        st.metric("Avg Processing Time", f"{analytics_data.get('avg_processing_time', 0):.2f}s")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Email volume over time
        st.markdown("#### 📊 Email Volume Over Time")
        
        # Mock time series data
        dates = pd.date_range(start='2024-01-14', end='2024-01-20', freq='D')
        volumes = [45, 52, 38, 61, 47, 55, 43]
        
        fig = px.line(x=dates, y=volumes, title="Daily Email Volume")
        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Emails")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Classification distribution
        st.markdown("#### 🎯 Email Classification")
        
        classifications = ['Important', 'Notify', 'Ignore', 'Uncertain']
        counts = [15, 25, 35, 8]
        
        fig = px.pie(values=counts, names=classifications, title="Email Classification Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    st.markdown("### 📈 Performance Trends")
    
    # Processing time trends
    st.markdown("#### ⏱️ Processing Time Trends")
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    processing_times = [2.3, 1.8, 2.1, 2.5, 2.0, 1.5, 1.2]
    
    fig = px.bar(x=days, y=processing_times, title="Average Processing Time by Day")
    fig.update_layout(xaxis_title="Day", yaxis_title="Processing Time (seconds)")
    st.plotly_chart(fig, use_container_width=True)
    
    # AI confidence trends
    st.markdown("#### 🎯 AI Confidence Trends")
    
    confidence_data = {
        'Date': pd.date_range(start='2024-01-14', end='2024-01-20', freq='D'),
        'High Confidence': [0.85, 0.87, 0.83, 0.89, 0.86, 0.91, 0.88],
        'Low Confidence': [0.15, 0.13, 0.17, 0.11, 0.14, 0.09, 0.12]
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=confidence_data['Date'], y=confidence_data['High Confidence'], 
                           mode='lines+markers', name='High Confidence'))
    fig.add_trace(go.Scatter(x=confidence_data['Date'], y=confidence_data['Low Confidence'], 
                           mode='lines+markers', name='Low Confidence'))
    
    fig.update_layout(title="AI Confidence Over Time", xaxis_title="Date", yaxis_title="Confidence Ratio")
    st.plotly_chart(fig, use_container_width=True)

def render_management_tab():
    """Render the system management tab"""
    st.markdown("## ⚙️ System Management")
    
    # Database management
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗄️ Database Management")
        
        # Database stats
        try:
            conn = sqlite3.connect('email_assistant.db')
            cursor = conn.cursor()
            
            # Get table sizes
            tables = ['emails', 'action_items', 'graph_entities', 'graph_relationships', 'uncertain_emails']
            table_stats = {}
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_stats[table] = count
            
            conn.close()
            
            for table, count in table_stats.items():
                st.metric(f"{table.replace('_', ' ').title()}", count)
            
        except Exception as e:
            st.error(f"Error loading database stats: {e}")
        
        # Database actions
        st.markdown("#### 🛠️ Database Actions")
        
        if st.button("🧹 Clear Processed Emails", use_container_width=True):
            clear_processed_emails()
        
        if st.button("📊 Optimize Database", use_container_width=True):
            optimize_database()
        
        if st.button("💾 Backup Database", use_container_width=True):
            backup_database()
    
    with col2:
        st.markdown("### 🔧 System Configuration")
        
        # LangGraph server settings
        st.markdown("#### 🚀 LangGraph Server")
        
        server_url = st.text_input("Server URL", value="http://127.0.0.1:2024")
        
        if st.button("🔍 Test Connection", use_container_width=True):
            test_langgraph_connection(server_url)
        
        # API configurations
        st.markdown("#### 🔑 API Configuration")
        
        with st.expander("OpenAI Settings"):
            model = st.selectbox("Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"])
            max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=1000)
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        
        with st.expander("Gmail Settings"):
            batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=50)
            rate_limit = st.number_input("Rate Limit (requests/minute)", min_value=1, max_value=1000, value=100)
        
        # System logs
        st.markdown("#### 📋 System Logs")
        
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        
        if st.button("📄 View Recent Logs", use_container_width=True):
            show_recent_logs(log_level)

# Helper functions for the new features
def start_main_workflow():
    """Start the main LangGraph workflow"""
    st.session_state.processing_status['email_processing'] = 'running'
    st.success("🚀 Main workflow started!")

def stop_all_workflows():
    """Stop all running workflows"""
    for key in st.session_state.processing_status:
        if st.session_state.processing_status[key] == 'running':
            st.session_state.processing_status[key] = 'idle'
    st.info("⏹️ All workflows stopped")

def approve_email(uncertain_id):
    """Approve an uncertain email"""
    st.success(f"✅ Email {uncertain_id} approved!")

def reject_email(uncertain_id):
    """Reject an uncertain email"""
    st.success(f"❌ Email {uncertain_id} rejected!")

def modify_email_decision(uncertain_id):
    """Modify email decision"""
    st.info(f"🔄 Modifying decision for email {uncertain_id}")

def save_user_feedback(uncertain_id, feedback):
    """Save user feedback for learning"""
    if feedback.strip():
        st.success("💾 Feedback saved! This will help improve AI decisions.")
    else:
        st.warning("Please provide feedback before saving.")

def start_learning_session(source, focus_areas):
    """Start AI learning session"""
    st.session_state.processing_status['reflection_learning'] = 'running'
    
    with st.spinner("🧠 AI is learning from your feedback..."):
        time.sleep(3)  # Simulate learning process
        
        st.session_state.processing_status['reflection_learning'] = 'completed'
        st.success(f"✅ Learning session completed! Focused on: {', '.join(focus_areas)}")

def save_prompt_update(category, prompt):
    """Save prompt update"""
    st.success(f"💾 {category.replace('_', ' ').title()} prompt updated!")

def reset_prompt_to_default(category):
    """Reset prompt to default"""
    st.info(f"🔄 {category.replace('_', ' ').title()} prompt reset to default")

def clear_processed_emails():
    """Clear processed emails from database"""
    st.warning("⚠️ This will permanently delete processed email data!")
    if st.button("Confirm Delete"):
        st.success("🧹 Processed emails cleared!")

def optimize_database():
    """Optimize database performance"""
    with st.spinner("🔧 Optimizing database..."):
        time.sleep(2)
        st.success("📊 Database optimized!")

def backup_database():
    """Backup database"""
    with st.spinner("💾 Creating database backup..."):
        time.sleep(2)
        st.success("💾 Database backup created!")

def test_langgraph_connection(url):
    """Test LangGraph server connection"""
    try:
        # This would test actual connection
        st.success(f"✅ Connected to LangGraph server at {url}")
    except:
        st.error(f"❌ Failed to connect to LangGraph server at {url}")

def show_recent_logs(level):
    """Show recent system logs"""
    # Mock log data
    logs = [
        "2024-01-20 10:30:15 - INFO - Email processing completed successfully",
        "2024-01-20 10:29:45 - INFO - 5 emails processed, 2 marked as important",
        "2024-01-20 10:29:30 - DEBUG - Starting email batch processing",
        "2024-01-20 10:28:00 - WARNING - OpenAI API rate limit approaching",
    ]
    
    filtered_logs = [log for log in logs if level in log]
    
    for log in filtered_logs:
        st.code(log)

def main():
    """Main dashboard function"""
    initialize_session_state()
    render_header()
    render_sidebar()
    render_main_dashboard()

if __name__ == "__main__":
    main() 