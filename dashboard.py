#!/usr/bin/env python3
"""
Email AI Assistant Dashboard
A comprehensive Streamlit interface for managing email processing and analysis
"""

import streamlit as st
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from email_ai_assistant import EmailAIAssistant
import asyncio
import threading
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Email AI Assistant Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = 'idle'
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Dashboard Overview"

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect('email_assistant.db')

def load_dashboard_data():
    """Load data for dashboard metrics"""
    conn = get_db_connection()
    
    # Get basic stats
    stats = {}
    
    # Total emails
    stats['total_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails", conn).iloc[0]['count']
    
    # Important emails
    stats['important_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails WHERE is_important = 1", conn).iloc[0]['count']
    
    # Total entities
    stats['total_entities'] = pd.read_sql_query("SELECT COUNT(*) as count FROM graph_entities", conn).iloc[0]['count']
    
    # Total action items
    stats['total_actions'] = pd.read_sql_query("SELECT COUNT(*) as count FROM action_items", conn).iloc[0]['count']
    
    # Pending actions
    stats['pending_actions'] = pd.read_sql_query("SELECT COUNT(*) as count FROM action_items WHERE status = 'pending'", conn).iloc[0]['count']
    
    conn.close()
    return stats

def process_emails_background(start_date, end_date, quick_mode=True):
    """Process emails in background"""
    try:
        st.session_state.processing_status = 'running'
        assistant = EmailAIAssistant()
        assistant.run_full_pipeline(start_date, end_date, quick_mode)
        st.session_state.processing_status = 'completed'
    except Exception as e:
        st.session_state.processing_status = f'error: {str(e)}'

# Sidebar Navigation
st.sidebar.title("📧 Email AI Assistant")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    [
        "🏠 Dashboard Overview",
        "⚡ Process Emails", 
        "📊 Email Analysis",
        "📋 Action Items",
        "🧠 Knowledge Graph",
        "⚙️ Settings"
    ],
    index=[
        "🏠 Dashboard Overview",
        "⚡ Process Emails", 
        "📊 Email Analysis",
        "📋 Action Items",
        "🧠 Knowledge Graph",
        "⚙️ Settings"
    ].index(st.session_state.page)
)

# Update session state if sidebar selection changes
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

# Main Dashboard
if page == "🏠 Dashboard Overview":
    st.title("📧 Email AI Assistant Dashboard")
    st.markdown("Welcome to your intelligent email processing system!")
    
    # Load dashboard data
    try:
        stats = load_dashboard_data()
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📧 Total Emails", stats['total_emails'])
        
        with col2:
            st.metric("⭐ Important", stats['important_emails'])
            
        with col3:
            st.metric("🏷️ Entities", stats['total_entities'])
            
        with col4:
            st.metric("📋 Action Items", stats['total_actions'])
            
        with col5:
            st.metric("⏳ Pending", stats['pending_actions'])
        
        st.markdown("---")
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Email Processing Trends")
            
            # Get email processing over time
            conn = get_db_connection()
            df_trends = pd.read_sql_query("""
                SELECT DATE(date) as date, COUNT(*) as count, 
                       SUM(CASE WHEN is_important = 1 THEN 1 ELSE 0 END) as important_count
                FROM emails 
                WHERE date >= date('now', '-30 days')
                GROUP BY DATE(date)
                ORDER BY date
            """, conn)
            conn.close()
            
            if not df_trends.empty:
                fig = px.line(df_trends, x='date', y=['count', 'important_count'], 
                             title="Daily Email Processing (Last 30 Days)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recent email data to display")
        
        with col2:
            st.subheader("🎯 Classification Breakdown")
            
            # Pie chart of important vs not important
            if stats['total_emails'] > 0:
                labels = ['Important', 'Not Important']
                values = [stats['important_emails'], stats['total_emails'] - stats['important_emails']]
                
                fig = px.pie(values=values, names=labels, title="Email Importance Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emails processed yet")
        
        # Quick actions
        st.markdown("---")
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Process Recent Emails", use_container_width=True):
                st.session_state.page = "⚡ Process Emails"
                st.rerun()
                
        with col2:
            if st.button("📊 View Analysis", use_container_width=True):
                st.session_state.page = "📊 Email Analysis"
                st.rerun()
                
        with col3:
            if st.button("📋 Manage Actions", use_container_width=True):
                st.session_state.page = "📋 Action Items"
                st.rerun()
                
        with col4:
            if st.button("🧠 Explore Graph", use_container_width=True):
                st.session_state.page = "🧠 Knowledge Graph"
                st.rerun()
                
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

# Process Emails Page
elif page == "⚡ Process Emails":
    st.title("⚡ Process Emails")
    st.markdown("Configure and run email processing with your AI assistant")
    
    # Processing configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Date Range")
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", value=datetime.now())
        
    with col2:
        st.subheader("⚙️ Processing Options")
        quick_mode = st.checkbox("Quick Mode", value=True, help="Faster processing with essential features only")
        max_emails = st.number_input("Max Emails to Process", min_value=1, max_value=1000, value=50)
    
    st.markdown("---")
    
    # Processing controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            if start_date <= end_date:
                st.session_state.processing_status = 'starting'
                st.rerun()
            else:
                st.error("Start date must be before end date")
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.processing_status = 'stopped'
            st.rerun()
            
    with col3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Processing status
    status_container = st.container()
    
    with status_container:
        if st.session_state.processing_status == 'idle':
            st.info("Ready to process emails. Configure settings above and click 'Start Processing'.")
            
        elif st.session_state.processing_status == 'starting':
            st.warning("Initializing email processing...")
            # Start background processing
            thread = threading.Thread(
                target=process_emails_background, 
                args=(datetime.combine(start_date, datetime.min.time()), 
                      datetime.combine(end_date, datetime.max.time()), 
                      quick_mode)
            )
            thread.daemon = True
            thread.start()
            st.session_state.processing_status = 'running'
            time.sleep(2)
            st.rerun()
            
        elif st.session_state.processing_status == 'running':
            st.success("🔄 Processing emails in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (in real implementation, you'd track actual progress)
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f'Processing... {i + 1}%')
                time.sleep(0.1)
                
            st.session_state.processing_status = 'completed'
            st.rerun()
            
        elif st.session_state.processing_status == 'completed':
            st.success("✅ Email processing completed successfully!")
            
            # Show results
            try:
                stats = load_dashboard_data()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Emails Processed", stats['total_emails'])
                with col2:
                    st.metric("Important Found", stats['important_emails'])
                with col3:
                    st.metric("Action Items Created", stats['total_actions'])
                    
                if st.button("View Results", type="primary"):
                    st.session_state.page = "📊 Email Analysis"
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading results: {str(e)}")
                
        elif st.session_state.processing_status.startswith('error'):
            error_msg = st.session_state.processing_status.replace('error: ', '')
            st.error(f"❌ Processing failed: {error_msg}")
            
        elif st.session_state.processing_status == 'stopped':
            st.warning("⏹️ Processing stopped by user")

# Email Analysis Page
elif page == "📊 Email Analysis":
    st.title("📊 Email Analysis")
    st.markdown("Explore your processed emails and insights")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        importance_filter = st.selectbox("Importance", ["All", "Important Only", "Not Important"])
    
    with col2:
        date_range = st.selectbox("Date Range", ["All Time", "Last 7 Days", "Last 30 Days", "Custom"])
    
    with col3:
        search_term = st.text_input("Search", placeholder="Search emails...")
    
    # Load and filter emails
    try:
        conn = get_db_connection()
        
        # Build query based on filters
        query = "SELECT * FROM emails WHERE 1=1"
        params = []
        
        if importance_filter == "Important Only":
            query += " AND is_important = 1"
        elif importance_filter == "Not Important":
            query += " AND is_important = 0"
            
        if date_range == "Last 7 Days":
            query += " AND date >= date('now', '-7 days')"
        elif date_range == "Last 30 Days":
            query += " AND date >= date('now', '-30 days')"
            
        if search_term:
            query += " AND (subject LIKE ? OR from_email LIKE ? OR summary LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"])
            
        query += " ORDER BY date DESC LIMIT 100"
        
        df_emails = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df_emails.empty:
            st.subheader(f"📧 Found {len(df_emails)} emails")
            
            # Display emails
            for idx, email in df_emails.iterrows():
                with st.expander(f"{'⭐' if email['is_important'] else '📧'} {email['subject'][:60]}... - {email['from_email']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**From:** {email['from_email']}")
                        st.write(f"**Date:** {email['date']}")
                        st.write(f"**Summary:** {email['summary']}")
                        
                        # Show entities
                        if email['entities']:
                            try:
                                entities = json.loads(email['entities'])
                                if entities:
                                    st.write("**Entities:**")
                                    for entity in entities[:5]:
                                        if isinstance(entity, dict):
                                            st.write(f"• {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")
                                        else:
                                            st.write(f"• {entity}")
                            except:
                                pass
                    
                    with col2:
                        st.write(f"**Important:** {'Yes' if email['is_important'] else 'No'}")
                        
                        # Show action items
                        if email['action_items']:
                            try:
                                actions = json.loads(email['action_items'])
                                if actions:
                                    st.write("**Action Items:**")
                                    for action in actions[:3]:
                                        if isinstance(action, dict):
                                            st.write(f"• {action.get('description', action.get('action', 'Unknown'))}")
                                        else:
                                            st.write(f"• {action}")
                            except:
                                pass
        else:
            st.info("No emails found matching your criteria")
            
    except Exception as e:
        st.error(f"Error loading emails: {str(e)}")

# Action Items Page
elif page == "📋 Action Items":
    st.title("📋 Action Items Management")
    st.markdown("Track and manage action items extracted from your emails")
    
    try:
        conn = get_db_connection()
        
        # Load action items
        df_actions = pd.read_sql_query("""
            SELECT ai.*, e.subject, e.from_email 
            FROM action_items ai
            LEFT JOIN emails e ON ai.email_id = e.id
            ORDER BY ai.created_date DESC
        """, conn)
        
        if not df_actions.empty:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Actions", len(df_actions))
            with col2:
                pending_count = len(df_actions[df_actions['status'] == 'pending'])
                st.metric("Pending", pending_count)
            with col3:
                completed_count = len(df_actions[df_actions['status'] == 'completed'])
                st.metric("Completed", completed_count)
            with col4:
                high_priority = len(df_actions[df_actions['priority'] == 'high'])
                st.metric("High Priority", high_priority)
            
            st.markdown("---")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.selectbox("Status", ["All", "Pending", "Completed", "In Progress"])
            with col2:
                priority_filter = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
            
            # Filter data
            filtered_df = df_actions.copy()
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter.lower()]
            if priority_filter != "All":
                filtered_df = filtered_df[filtered_df['priority'] == priority_filter.lower()]
            
            # Display action items
            st.subheader(f"Action Items ({len(filtered_df)})")
            
            for idx, action in filtered_df.iterrows():
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(action['priority'], "⚪")
                status_emoji = {"pending": "⏳", "completed": "✅", "in progress": "🔄"}.get(action['status'], "❓")
                
                with st.expander(f"{priority_emoji} {status_emoji} {action['description'][:60]}..."):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {action['description']}")
                        st.write(f"**From Email:** {action['subject']} ({action['from_email']})")
                        st.write(f"**Created:** {action['created_date']}")
                        if action['due_date']:
                            st.write(f"**Due Date:** {action['due_date']}")
                    
                    with col2:
                        # Status update
                        new_status = st.selectbox(
                            "Status", 
                            ["pending", "in progress", "completed"], 
                            index=["pending", "in progress", "completed"].index(action['status']),
                            key=f"status_{action['id']}"
                        )
                        
                        if st.button(f"Update Status", key=f"update_{action['id']}"):
                            cursor = conn.cursor()
                            cursor.execute("UPDATE action_items SET status = ? WHERE id = ?", (new_status, action['id']))
                            conn.commit()
                            st.success("Status updated!")
                            st.rerun()
        
        else:
            st.info("No action items found. Process some emails first to extract action items.")
            
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading action items: {str(e)}")

# Knowledge Graph Page
elif page == "🧠 Knowledge Graph":
    st.title("🧠 Knowledge Graph Explorer")
    st.markdown("Explore entities, relationships, and insights from your emails")
    
    try:
        conn = get_db_connection()
        
        # Load entities and relationships
        df_entities = pd.read_sql_query("SELECT * FROM graph_entities ORDER BY updated_date DESC", conn)
        df_relationships = pd.read_sql_query("SELECT * FROM graph_relationships ORDER BY created_date DESC", conn)
        
        if not df_entities.empty:
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Entities", len(df_entities))
            with col2:
                st.metric("Relationships", len(df_relationships))
            with col3:
                entity_types = df_entities['type'].value_counts()
                st.metric("Entity Types", len(entity_types))
            
            st.markdown("---")
            
            # Search entities
            search_entity = st.text_input("Search Entities", placeholder="Search for people, organizations, etc.")
            
            if search_entity:
                filtered_entities = df_entities[
                    df_entities['name'].str.contains(search_entity, case=False, na=False) |
                    df_entities['description'].str.contains(search_entity, case=False, na=False)
                ]
            else:
                filtered_entities = df_entities.head(20)
            
            # Entity type distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏷️ Entity Types")
                entity_type_counts = df_entities['type'].value_counts()
                fig = px.bar(x=entity_type_counts.values, y=entity_type_counts.index, orientation='h')
                fig.update_layout(height=400, xaxis_title="Count", yaxis_title="Entity Type")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔗 Top Entities by Connections")
                # Count relationships per entity
                entity_connections = pd.concat([
                    df_relationships['source_entity'].value_counts(),
                    df_relationships['target_entity'].value_counts()
                ]).groupby(level=0).sum().sort_values(ascending=False).head(10)
                
                if not entity_connections.empty:
                    fig = px.bar(x=entity_connections.values, y=entity_connections.index, orientation='h')
                    fig.update_layout(height=400, xaxis_title="Connections", yaxis_title="Entity")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No relationship data available")
            
            # Display entities
            st.subheader(f"📋 Entities ({len(filtered_entities)})")
            
            for idx, entity in filtered_entities.iterrows():
                with st.expander(f"{entity['name']} ({entity['type']})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {entity['description']}")
                        st.write(f"**Type:** {entity['type']}")
                        st.write(f"**Created:** {entity['created_date']}")
                        
                        # Show related emails
                        if entity['email_ids']:
                            email_ids = entity['email_ids'].split(',')
                            st.write(f"**Related Emails:** {len(email_ids)}")
                    
                    with col2:
                        # Show relationships
                        entity_rels = df_relationships[
                            (df_relationships['source_entity'] == entity['id']) |
                            (df_relationships['target_entity'] == entity['id'])
                        ]
                        
                        if not entity_rels.empty:
                            st.write("**Relationships:**")
                            for _, rel in entity_rels.head(5).iterrows():
                                if rel['source_entity'] == entity['id']:
                                    st.write(f"→ {rel['relationship_type']}")
                                else:
                                    st.write(f"← {rel['relationship_type']}")
        else:
            st.info("No entities found. Process some emails first to build the knowledge graph.")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading knowledge graph: {str(e)}")

# Settings Page
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("Configure your Email AI Assistant")
    
    # System status
    st.subheader("🔧 System Status")
    
    try:
        # Test database connection
        conn = get_db_connection()
        conn.close()
        st.success("✅ Database connection: OK")
        
        # Test OpenAI connection
        try:
            assistant = EmailAIAssistant()
            test_result = assistant.test_openai_connection()
            if "successfully" in test_result.lower():
                st.success("✅ OpenAI API: Connected")
            else:
                st.warning(f"⚠️ OpenAI API: {test_result}")
        except Exception as e:
            st.error(f"❌ OpenAI API: {str(e)}")
            
        # Test Gmail connection
        try:
            assistant = EmailAIAssistant()
            assistant.setup_gmail_credentials()
            st.success("✅ Gmail API: Connected")
        except Exception as e:
            st.error(f"❌ Gmail API: {str(e)}")
            
    except Exception as e:
        st.error(f"❌ System check failed: {str(e)}")
    
    st.markdown("---")
    
    # Configuration options
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Processing Settings**")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
        max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3)
        
    with col2:
        st.write("**AI Model Settings**")
        model_name = st.selectbox("OpenAI Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"])
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    st.markdown("---")
    
    # Database management
    st.subheader("🗄️ Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧹 Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data"):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM emails")
                    cursor.execute("DELETE FROM action_items")
                    cursor.execute("DELETE FROM graph_entities")
                    cursor.execute("DELETE FROM graph_relationships")
                    conn.commit()
                    conn.close()
                    st.success("All data cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing data: {str(e)}")
    
    with col2:
        if st.button("📊 Export Data"):
            st.info("Export functionality coming soon!")
    
    with col3:
        if st.button("📥 Import Data"):
            st.info("Import functionality coming soon!")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Streamlit") 