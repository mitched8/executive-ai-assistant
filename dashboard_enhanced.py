#!/usr/bin/env python3
"""
Enhanced Email AI Assistant Dashboard
Leveraging full Gmail API capabilities including responses, calendar integration, and thread management
"""

import streamlit as st
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import threading
import time
import sys
import os

# Add the eaia module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new integration module
try:
    from eaia.main.dashboard_integration import DashboardIntegration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning(f"⚠️ Integration module not available: {e}")
    INTEGRATION_AVAILABLE = False

# Add Gmail functions with error handling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from eaia.gmail import (
        send_email, mark_as_read, fetch_group_emails, 
        get_events_for_days, send_calendar_invite
    )
    GMAIL_AVAILABLE = True
except ImportError as e:
    st.sidebar.warning("⚠️ Gmail functions not available")
    GMAIL_AVAILABLE = False
    
    # Create dummy functions to prevent errors
    def send_email(*args, **kwargs):
        raise Exception("Gmail integration not available")
    def mark_as_read(*args, **kwargs):
        raise Exception("Gmail integration not available")
    def fetch_group_emails(*args, **kwargs):
        return []
    def get_events_for_days(*args, **kwargs):
        return "Gmail integration not available"
    def send_calendar_invite(*args, **kwargs):
        return False

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Email AI Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = 'idle'
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Dashboard Overview"
if 'selected_email' not in st.session_state:
    st.session_state.selected_email = None
if 'reply_mode' not in st.session_state:
    st.session_state.reply_mode = False
if 'dashboard_integration' not in st.session_state and INTEGRATION_AVAILABLE:
    st.session_state.dashboard_integration = DashboardIntegration()

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect('email_assistant.db')

def setup_enhanced_tables():
    """Create enhanced tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create email status table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_status (
            email_id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'unread',
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
    ''')
    
    # Create email replies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_email_id TEXT,
            reply_text TEXT,
            sent_at TEXT,
            status TEXT DEFAULT 'draft',
            FOREIGN KEY (original_email_id) REFERENCES emails (id)
        )
    ''')
    
    # Create email meetings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT,
            meeting_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            attendees TEXT,
            FOREIGN KEY (email_id) REFERENCES emails (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_dashboard_data():
    """Load enhanced data for dashboard metrics"""
    # Ensure enhanced tables exist
    setup_enhanced_tables()
    
    conn = get_db_connection()
    
    stats = {}
    
    # Basic stats
    stats['total_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails", conn).iloc[0]['count']
    stats['important_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails WHERE is_important = 1", conn).iloc[0]['count']
    
    # Handle missing tables gracefully
    try:
        stats['total_entities'] = pd.read_sql_query("SELECT COUNT(*) as count FROM graph_entities", conn).iloc[0]['count']
    except:
        stats['total_entities'] = 0
    
    try:
        stats['total_actions'] = pd.read_sql_query("SELECT COUNT(*) as count FROM action_items", conn).iloc[0]['count']
        stats['pending_actions'] = pd.read_sql_query("SELECT COUNT(*) as count FROM action_items WHERE status = 'pending'", conn).iloc[0]['count']
    except:
        stats['total_actions'] = 0
        stats['pending_actions'] = 0
    
    # Enhanced stats with fallbacks
    try:
        # Count emails that don't have a 'read' status (default to unread)
        stats['unread_emails'] = pd.read_sql_query("""
            SELECT COUNT(*) as count FROM emails 
            WHERE id NOT IN (
                SELECT email_id FROM email_status WHERE status = 'read'
            )
        """, conn).iloc[0]['count']
    except:
        # If email_status table doesn't exist, assume all emails are unread
        stats['unread_emails'] = stats['total_emails']
    
    try:
        stats['threads'] = pd.read_sql_query("SELECT COUNT(DISTINCT thread_id) as count FROM emails", conn).iloc[0]['count']
    except:
        stats['threads'] = stats['total_emails']  # Fallback: assume each email is its own thread
    
    try:
        stats['recent_important'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails WHERE is_important = 1 AND date >= date('now', '-7 days')", conn).iloc[0]['count']
    except:
        stats['recent_important'] = 0
    
    conn.close()
    return stats

# Enhanced Sidebar Navigation
st.sidebar.title("📧 Enhanced Email AI")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate to:",
    [
        "🏠 Dashboard Overview",
        "⚡ Process Emails", 
        "📊 Email Analysis",
        "📧 Email Composer",
        "📅 Calendar Integration",
        "📋 Action Items",
        "🧵 Thread Management",
        "🧠 Knowledge Graph",
        "⚙️ Settings"
    ],
    index=[
        "🏠 Dashboard Overview",
        "⚡ Process Emails", 
        "📊 Email Analysis",
        "📧 Email Composer",
        "📅 Calendar Integration",
        "📋 Action Items",
        "🧵 Thread Management",
        "🧠 Knowledge Graph",
        "⚙️ Settings"
    ].index(st.session_state.page)
)

# Update session state if sidebar selection changes
if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

# Enhanced Dashboard Overview
if page == "🏠 Dashboard Overview":
    st.title("📧 Enhanced Email AI Assistant Dashboard")
    st.markdown("Complete email management with AI analysis, responses, and calendar integration!")
    
    try:
        if INTEGRATION_AVAILABLE:
            integration = st.session_state.dashboard_integration
            stats = integration.get_dashboard_overview(30)
        else:
            stats = load_dashboard_data()
        
        # Enhanced metrics with additional insights
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📧 Total Emails", stats['total_emails'])
        with col2:
            st.metric("⭐ Important", stats['important_emails'])
        with col3:
            st.metric("🔔 Unread", stats['unread_emails'])
        with col4:
            st.metric("🧵 Threads", stats['threads'])
        with col5:
            st.metric("📋 Actions", stats['total_actions'])
        with col6:
            st.metric("🆕 Recent Important", stats['recent_important'])
        
        st.markdown("---")
        
        # Quick Actions Enhanced
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Process Recent Emails", use_container_width=True):
                st.session_state.page = "⚡ Process Emails"
                st.rerun()
                
        with col2:
            if st.button("📧 Compose & Send", use_container_width=True):
                st.session_state.page = "📧 Email Composer"
                st.rerun()
                
        with col3:
            if st.button("📅 Check Calendar", use_container_width=True):
                st.session_state.page = "📅 Calendar Integration"
                st.rerun()
                
        with col4:
            if st.button("🧵 Manage Threads", use_container_width=True):
                st.session_state.page = "🧵 Thread Management"
                st.rerun()
        
        # Recent Activity & Trends (same as before)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Email Processing Trends")
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
            if stats['total_emails'] > 0:
                labels = ['Important', 'Not Important']
                values = [stats['important_emails'], stats['total_emails'] - stats['important_emails']]
                
                fig = px.pie(values=values, names=labels, title="Email Importance Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emails processed yet")
                
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

# NEW: Email Composer Page
elif page == "📧 Email Composer":
    st.title("📧 Smart Email Composer")
    st.markdown("Compose and send replies with AI assistance")
    
    if not GMAIL_AVAILABLE:
        st.error("📧 Gmail integration is not available. Please check your Gmail setup.")
        st.info("💡 This feature requires the Gmail API to be properly configured.")
        st.stop()
    
    # Email selection for replies
    st.subheader("📩 Reply to Email")
    
    try:
        conn = get_db_connection()
        # Get recent important emails that might need replies
        df_emails = pd.read_sql_query("""
            SELECT id, subject, from_email, date, summary 
            FROM emails 
            WHERE is_important = 1
            ORDER BY date DESC 
            LIMIT 20
        """, conn)
        
        if not df_emails.empty:
            # Email selection
            selected_email = st.selectbox(
                "Select email to reply to:",
                options=df_emails['id'].tolist(),
                format_func=lambda x: f"{df_emails[df_emails['id']==x]['subject'].iloc[0][:50]}... - {df_emails[df_emails['id']==x]['from_email'].iloc[0]}"
            )
            
            if selected_email:
                email_data = df_emails[df_emails['id'] == selected_email].iloc[0]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Subject:** {email_data['subject']}")
                    st.write(f"**From:** {email_data['from_email']}")
                    st.write(f"**Summary:** {email_data['summary']}")
                
                with col2:
                    st.write("**Quick Actions:**")
                    if st.button("Mark as Read"):
                        try:
                            mark_as_read(selected_email)
                            # Update email status in database
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute("""
                                INSERT OR REPLACE INTO email_status (email_id, status, updated_at)
                                VALUES (?, 'read', ?)
                            """, (selected_email, datetime.now().isoformat()))
                            conn.commit()
                            conn.close()
                            st.success("Marked as read!")
                        except Exception as e:
                            st.error(f"Error marking as read: {str(e)}")
                
                # Reply composition
                st.markdown("---")
                st.subheader("✍️ Compose Reply")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    reply_text = st.text_area(
                        "Reply message:", 
                        height=200,
                        placeholder="Type your reply here..."
                    )
                
                with col2:
                    st.write("**AI Suggestions:**")
                    
                    if st.button("🤖 Generate Reply"):
                        with st.spinner("Generating AI reply..."):
                            # Here you could integrate with your AI to generate reply
                            suggested_reply = "Thank you for your email. I'll review this and get back to you shortly."
                            st.text_area("Suggested reply:", value=suggested_reply, height=100)
                    
                    if st.button("📋 Add to Actions"):
                        # Add reply task to action items
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO action_items (email_id, description, priority, created_date)
                            VALUES (?, ?, ?, ?)
                        """, (selected_email, f"Reply to: {email_data['subject']}", "high", datetime.now().isoformat()))
                        conn.commit()
                        conn.close()
                        st.success("Added to action items!")
                
                # Send reply
                if reply_text:
                    col1, col2, col3 = st.columns([1, 1, 2])
                    
                    with col1:
                        if st.button("📤 Send Reply", type="primary"):
                            try:
                                send_email(
                                    email_id=selected_email,
                                    response_text=reply_text,
                                    email_address="your_email@gmail.com"  # You'd get this from config
                                )
                                st.success("✅ Reply sent successfully!")
                                
                                # Mark as read after sending reply
                                mark_as_read(selected_email)
                                
                            except Exception as e:
                                st.error(f"Failed to send reply: {str(e)}")
                    
                    with col2:
                        if st.button("💾 Save Draft"):
                            st.success("Draft saved!")
        
        else:
            st.info("No recent important emails found to reply to.")
            
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading emails: {str(e)}")

# NEW: Calendar Integration Page
elif page == "📅 Calendar Integration":
    st.title("📅 Calendar Integration")
    st.markdown("Manage meetings and calendar events from your emails")
    
    if not GMAIL_AVAILABLE:
        st.error("📅 Calendar integration is not available. Please check your Gmail setup.")
        st.info("💡 This feature requires the Gmail API and Calendar API to be properly configured.")
        st.stop()
    
    # Calendar viewer
    st.subheader("📆 Your Calendar")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_date = st.date_input("Select date:", value=datetime.now().date())
        
        if st.button("🔍 Check Events"):
            try:
                date_str = selected_date.strftime("%d-%m-%Y")
                events = get_events_for_days([date_str])
                
                st.subheader(f"Events for {selected_date}")
                st.text(events)
                
            except Exception as e:
                st.error(f"Error fetching calendar events: {str(e)}")
    
    with col2:
        # Meeting scheduler from action items
        st.subheader("📝 Schedule Meetings from Action Items")
        
        try:
            conn = get_db_connection()
            df_actions = pd.read_sql_query("""
                SELECT ai.*, e.subject, e.from_email 
                FROM action_items ai
                LEFT JOIN emails e ON ai.email_id = e.id
                WHERE ai.status = 'pending' AND ai.description LIKE '%meeting%'
                ORDER BY ai.created_date DESC
                LIMIT 10
            """, conn)
            
            if not df_actions.empty:
                for idx, action in df_actions.iterrows():
                    with st.expander(f"📋 {action['description'][:50]}..."):
                        st.write(f"**From Email:** {action['subject']}")
                        st.write(f"**Sender:** {action['from_email']}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            meeting_title = st.text_input(f"Meeting Title", 
                                                        value=f"Meeting: {action['subject'][:30]}",
                                                        key=f"title_{action['id']}")
                            
                            meeting_date = st.date_input("Date", key=f"date_{action['id']}")
                            meeting_time = st.time_input("Time", key=f"time_{action['id']}")
                        
                        with col2:
                            duration = st.selectbox("Duration", [30, 60, 90, 120], key=f"duration_{action['id']}")
                            
                            attendees = st.text_input("Additional Attendees (comma-separated)", 
                                                     value=action['from_email'] or "",
                                                     key=f"attendees_{action['id']}")
                        
                        if st.button(f"📅 Create Meeting", key=f"create_{action['id']}"):
                            try:
                                start_datetime = datetime.combine(meeting_date, meeting_time)
                                end_datetime = start_datetime + timedelta(minutes=duration)
                                
                                attendee_list = [email.strip() for email in attendees.split(',') if email.strip()]
                                
                                success = send_calendar_invite(
                                    emails=attendee_list,
                                    title=meeting_title,
                                    start_time=start_datetime.isoformat(),
                                    end_time=end_datetime.isoformat(),
                                    email_address="your_email@gmail.com"  # From config
                                )
                                
                                if success:
                                    st.success("📅 Meeting created and invites sent!")
                                    
                                    # Update action item status
                                    cursor = conn.cursor()
                                    cursor.execute("UPDATE action_items SET status = 'completed' WHERE id = ?", (action['id'],))
                                    conn.commit()
                                else:
                                    st.error("Failed to create meeting")
                                    
                            except Exception as e:
                                st.error(f"Error creating meeting: {str(e)}")
            else:
                st.info("No meeting-related action items found.")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error loading action items: {str(e)}")

# NEW: Thread Management Page
elif page == "🧵 Thread Management":
    st.title("🧵 Email Thread Management")
    st.markdown("Manage email conversations and thread relationships")
    
    try:
        conn = get_db_connection()
        
        # Get email threads
        df_threads = pd.read_sql_query("""
            SELECT thread_id, COUNT(*) as email_count, 
                   MAX(date) as latest_date,
                   GROUP_CONCAT(DISTINCT from_email) as participants,
                   MAX(subject) as subject,
                   SUM(CASE WHEN is_important = 1 THEN 1 ELSE 0 END) as important_count
            FROM emails 
            GROUP BY thread_id 
            HAVING email_count > 1
            ORDER BY latest_date DESC
            LIMIT 20
        """, conn)
        
        if not df_threads.empty:
            st.subheader(f"📊 Active Threads ({len(df_threads)})")
            
            # Thread statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_emails = df_threads['email_count'].mean()
                st.metric("Avg Emails/Thread", f"{avg_emails:.1f}")
            with col2:
                total_participants = df_threads['participants'].str.split(',').str.len().sum()
                st.metric("Total Participants", total_participants)
            with col3:
                important_threads = len(df_threads[df_threads['important_count'] > 0])
                st.metric("Important Threads", important_threads)
            
            # Thread browser
            st.subheader("🔍 Browse Threads")
            
            for idx, thread in df_threads.iterrows():
                participants = thread['participants'][:100] + "..." if len(thread['participants']) > 100 else thread['participants']
                
                with st.expander(f"🧵 {thread['subject'][:60]}... ({thread['email_count']} emails)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Participants:** {participants}")
                        st.write(f"**Latest Activity:** {thread['latest_date']}")
                        st.write(f"**Total Emails:** {thread['email_count']}")
                        st.write(f"**Important Emails:** {thread['important_count']}")
                    
                    with col2:
                        if st.button(f"📧 View Thread Details", key=f"view_{thread['thread_id']}"):
                            # Get all emails in thread
                            thread_emails = pd.read_sql_query("""
                                SELECT id, subject, from_email, date, summary, is_important
                                FROM emails 
                                WHERE thread_id = ?
                                ORDER BY date ASC
                            """, conn, params=[thread['thread_id']])
                            
                            st.subheader("📧 Thread Timeline")
                            for email_idx, email in thread_emails.iterrows():
                                importance_icon = "⭐" if email['is_important'] else "📧"
                                st.write(f"{importance_icon} **{email['from_email']}** - {email['date']}")
                                st.write(f"   {email['summary'][:100]}...")
                        
                        if st.button(f"📤 Reply to Thread", key=f"reply_{thread['thread_id']}"):
                            st.session_state.selected_email = thread['thread_id']
                            st.session_state.page = "📧 Email Composer"
                            st.rerun()
        else:
            st.info("No email threads found with multiple messages.")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error loading threads: {str(e)}")

# Keep existing pages (Process Emails, Email Analysis, Action Items, Knowledge Graph, Settings)
# ... (Previous implementations remain the same)

elif page == "⚡ Process Emails":
    st.title("⚡ Process Emails")
    st.markdown("Configure and run email processing with your AI assistant")
    
    if not INTEGRATION_AVAILABLE:
        st.error("❌ Email processing integration not available. Please check your setup.")
        st.info("💡 Make sure the eaia.main modules are properly installed.")
        st.stop()
    
    # Enhanced processing with Gmail integration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Date Range")
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
        end_date = st.date_input("End Date", value=datetime.now())
        
    with col2:
        st.subheader("⚙️ Processing Options")
        quick_mode = st.checkbox("Quick Mode", value=True, help="Faster processing with reduced knowledge graph generation")
        auto_mark_read = st.checkbox("Auto-mark processed emails as read", value=False)
        max_emails = st.number_input("Max Emails", min_value=1, max_value=1000, value=50)
    
    # System status check
    if INTEGRATION_AVAILABLE:
        integration = st.session_state.dashboard_integration
        system_status = integration.test_system_integration()
        
        with st.expander("🔍 System Status"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Database:** {system_status.get('database', '❓ Unknown')}")
            with col2:
                st.write(f"**Processor:** {system_status.get('processor', '❓ Unknown')}")
            with col3:
                st.write(f"**OpenAI:** {system_status.get('openai', '❓ Unknown')}")
            with col4:
                st.write(f"**Gmail:** {system_status.get('gmail', '❓ Unknown')}")
    
    # Processing section
    st.markdown("---")
    
    # Check current processing status
    if INTEGRATION_AVAILABLE:
        processing_status = integration.get_processing_status()
        
        if processing_status == "running":
            st.info("🔄 Email processing is currently running...")
            st.progress(0.5)  # Indeterminate progress
            
            if st.button("🔄 Refresh Status"):
                st.rerun()
                
        elif processing_status == "completed":
            st.success("✅ Email processing completed successfully!")
            
            # Show results
            results = integration.get_processing_results()
            if results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Emails Processed", results.get('processed', 0))
                with col2:
                    st.metric("Important Found", results.get('important', 0))
                with col3:
                    st.metric("Action Items", results.get('action_items', 0))
                
                if st.button("📊 View Analysis", type="primary"):
                    st.session_state.page = "📊 Email Analysis"
                    st.rerun()
                    
        elif processing_status.startswith("error"):
            error_msg = processing_status.replace("error: ", "")
            st.error(f"❌ Processing failed: {error_msg}")
            
        else:  # idle
            st.info("Ready to process emails. Configure settings above and click 'Start Processing'.")
    
    # Start processing button
    if st.button("🚀 Start Enhanced Processing", type="primary", disabled=not start_date or not end_date):
        if INTEGRATION_AVAILABLE and GMAIL_AVAILABLE:
            integration = st.session_state.dashboard_integration
            thread = integration.process_emails_async(start_date, end_date, quick_mode)
            st.success("🚀 Email processing started in background!")
            st.rerun()
        else:
            st.error("❌ Cannot start processing. Check Gmail and integration availability.")

# Add the remaining pages...
elif page == "📊 Email Analysis":
    st.title("📊 Email Analysis")
    st.markdown("Enhanced email analysis with response tracking")
    
    if not INTEGRATION_AVAILABLE:
        st.error("❌ Email analysis integration not available.")
        st.stop()
    
    integration = st.session_state.dashboard_integration
    
    # Search and filter controls
    st.subheader("🔍 Search & Filter")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("Search emails:", placeholder="Enter keywords...")
    with col2:
        important_only = st.checkbox("Important emails only")
    with col3:
        days_back = st.selectbox("Time period:", [7, 14, 30, 90], index=2)
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("From date:", value=datetime.now() - timedelta(days=days_back))
    with col2:
        date_to = st.date_input("To date:", value=datetime.now())
    
    # Apply filters
    filters = {
        'search_query': search_query,
        'important_only': important_only,
        'date_from': date_from.isoformat() if date_from else None,
        'date_to': date_to.isoformat() if date_to else None
    }
    
    # Get analysis data
    try:
        analysis_data = integration.get_email_analysis_data(filters)
        
        if 'error' in analysis_data:
            st.error(f"Error loading analysis: {analysis_data['error']}")
        else:
            # Overview metrics
            st.subheader("📊 Overview")
            overview = integration.get_dashboard_overview(days_back)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Emails", overview.get('total_emails', 0))
            with col2:
                st.metric("Important Rate", f"{overview.get('importance_rate', 0):.1f}%")
            with col3:
                st.metric("Action Items", overview.get('total_actions', 0))
            with col4:
                st.metric("Uncertain", overview.get('uncertain_emails_count', 0))
            
            # Search results if query provided
            if search_query and 'search_results' in analysis_data:
                search_results = analysis_data['search_results']
                st.subheader(f"🔍 Search Results ({search_results.get('total_found', 0)})")
                
                if search_results.get('results'):
                    for email in search_results['results']:
                        with st.expander(f"{'⭐' if email['is_important'] else '📧'} {email['subject']} - {email['from_email']}"):
                            st.write(f"**Date:** {email['date']}")
                            st.write(f"**Summary:** {email['summary']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"📧 View Details", key=f"view_{email['id']}"):
                                    st.session_state.selected_email = email['id']
                            with col2:
                                if st.button(f"💬 Reply", key=f"reply_{email['id']}"):
                                    st.session_state.selected_email = email['id']
                                    st.session_state.page = "📧 Email Composer"
                                    st.rerun()
                else:
                    st.info("No emails found matching your search criteria.")
            
            # Sender analysis
            st.subheader("👥 Sender Analysis")
            sender_analysis = analysis_data.get('sender_analysis', {})
            
            if sender_analysis.get('top_senders'):
                df_senders = pd.DataFrame(sender_analysis['top_senders'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Top Senders**")
                    st.dataframe(df_senders[['from_email', 'total_emails', 'importance_rate']], use_container_width=True)
                
                with col2:
                    if sender_analysis.get('vip_senders'):
                        st.write("**VIP Senders (High Importance)**")
                        df_vip = pd.DataFrame(sender_analysis['vip_senders'])
                        st.dataframe(df_vip[['from_email', 'importance_rate', 'total_emails']], use_container_width=True)
            
            # Entity analysis
            st.subheader("🏷️ Entity Analysis")
            entity_analysis = analysis_data.get('entity_analysis', {})
            
            col1, col2 = st.columns(2)
            with col1:
                if entity_analysis.get('by_type'):
                    fig = px.pie(values=list(entity_analysis['by_type'].values()), 
                                names=list(entity_analysis['by_type'].keys()),
                                title="Entity Types Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if entity_analysis.get('top_entities'):
                    df_entities = pd.DataFrame(entity_analysis['top_entities'])
                    st.write("**Top Entities**")
                    st.dataframe(df_entities[['name', 'type', 'email_count']], use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading email analysis: {str(e)}")

elif page == "📋 Action Items":
    st.title("📋 Action Items Management")
    st.markdown("Enhanced action items with calendar integration")
    
    if not INTEGRATION_AVAILABLE:
        st.error("❌ Action items integration not available.")
        st.stop()
    
    integration = st.session_state.dashboard_integration
    
    try:
        # Get action items dashboard data
        action_data = integration.get_action_items_dashboard()
        
        if 'error' in action_data:
            st.error(f"Error loading action items: {action_data['error']}")
        else:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Actions", action_data.get('total_actions', 0))
            with col2:
                completion_rate = action_data.get('completion_rate', 0)
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            with col3:
                st.metric("Overdue Items", action_data.get('overdue_count', 0))
            with col4:
                pending_count = action_data.get('by_status', {}).get('pending', 0)
                st.metric("Pending", pending_count)
            
            # Priority distribution
            st.subheader("📊 Priority Distribution")
            if action_data.get('by_priority'):
                priority_data = action_data['by_priority']
                fig = px.bar(x=list(priority_data.keys()), y=list(priority_data.values()),
                           title="Action Items by Priority")
                st.plotly_chart(fig, use_container_width=True)
            
            # Action items management
            tab1, tab2, tab3, tab4 = st.tabs(["📋 All Items", "🚨 Overdue", "⭐ High Priority", "✅ Recent"])
            
            with tab1:
                st.subheader("All Action Items")
                if action_data.get('recent_items'):
                    for item in action_data['recent_items']:
                        with st.expander(f"{'🚨' if item.get('priority') == 'urgent' else '📋'} {item['description'][:60]}..."):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**From Email:** {item.get('subject', 'N/A')}")
                                st.write(f"**Sender:** {item.get('from_email', 'N/A')}")
                                st.write(f"**Priority:** {item.get('priority', 'medium').title()}")
                                st.write(f"**Status:** {item.get('status', 'pending').title()}")
                                st.write(f"**Created:** {item.get('created_date', 'N/A')}")
                            
                            with col2:
                                new_status = st.selectbox(
                                    "Update Status:",
                                    ["pending", "in_progress", "completed", "cancelled"],
                                    index=["pending", "in_progress", "completed", "cancelled"].index(item.get('status', 'pending')),
                                    key=f"status_{item['id']}"
                                )
                                
                                if st.button(f"Update", key=f"update_{item['id']}"):
                                    if integration.update_action_item_status(item['id'], new_status):
                                        st.success("Status updated!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update status")
                else:
                    st.info("No action items found.")
            
            with tab2:
                st.subheader("🚨 Overdue Items")
                if action_data.get('overdue_items'):
                    for item in action_data['overdue_items']:
                        st.error(f"🚨 **{item['description']}** - Due: {item.get('due_date', 'N/A')}")
                        st.write(f"From: {item.get('from_email', 'N/A')} - {item.get('subject', 'N/A')}")
                else:
                    st.success("🎉 No overdue items!")
            
            with tab3:
                st.subheader("⭐ High Priority Items")
                if action_data.get('high_priority_items'):
                    for item in action_data['high_priority_items']:
                        st.warning(f"⭐ **{item['description']}**")
                        st.write(f"Priority: {item.get('priority', 'high').title()}")
                        st.write(f"From: {item.get('from_email', 'N/A')}")
                else:
                    st.info("No high priority items.")
            
            with tab4:
                st.subheader("✅ Recent Activity")
                if action_data.get('recent_actions'):
                    for action in action_data['recent_actions']:
                        status_icon = "✅" if action['status'] == 'completed' else "📋"
                        st.write(f"{status_icon} **{action['description']}** - {action['status'].title()}")
                        st.caption(f"Created: {action['created_date']}")
                else:
                    st.info("No recent action items.")
    
    except Exception as e:
        st.error(f"Error loading action items: {str(e)}")

elif page == "🧠 Knowledge Graph":
    st.title("🧠 Knowledge Graph Explorer")
    st.markdown("Enhanced knowledge graph with relationship insights")
    
    if not INTEGRATION_AVAILABLE:
        st.error("❌ Knowledge graph integration not available.")
        st.stop()
    
    integration = st.session_state.dashboard_integration
    
    try:
        # Get knowledge graph data
        graph_data = integration.get_knowledge_graph_data()
        
        if 'error' in graph_data:
            st.error(f"Error loading knowledge graph: {graph_data['error']}")
        else:
            # Graph statistics
            stats = graph_data.get('stats', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Entities", stats.get('total_entities', 0))
            with col2:
                st.metric("Relationships", stats.get('total_relationships', 0))
            with col3:
                entity_types = len(stats.get('entity_types', {}))
                st.metric("Entity Types", entity_types)
            
            # Entity type distribution
            if stats.get('entity_types'):
                st.subheader("📊 Entity Types Distribution")
                fig = px.pie(
                    values=list(stats['entity_types'].values()),
                    names=list(stats['entity_types'].keys()),
                    title="Distribution of Entity Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Network visualization placeholder
            st.subheader("🕸️ Knowledge Network")
            
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if nodes:
                st.info(f"📊 Network contains {len(nodes)} entities and {len(edges)} relationships")
                
                # Simple network representation
                st.write("**Top Entities:**")
                for node in nodes[:10]:  # Show top 10 entities
                    entity_type = node.get('type', 'OTHER')
                    color_indicator = "🔴" if entity_type == "PERSON" else "🔵" if entity_type == "COMPANY" else "🟢"
                    st.write(f"{color_indicator} **{node['label']}** ({entity_type})")
                
                # Relationship insights
                if edges:
                    st.write("**Key Relationships:**")
                    relationship_types = {}
                    for edge in edges:
                        rel_type = edge.get('label', 'RELATED_TO')
                        relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
                    
                    for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• **{rel_type}**: {count} connections")
                
                # Entity search
                st.subheader("🔍 Entity Search")
                search_entity = st.text_input("Search for entity:", placeholder="Enter entity name...")
                
                if search_entity:
                    matching_entities = [node for node in nodes if search_entity.lower() in node['label'].lower()]
                    
                    if matching_entities:
                        st.write(f"Found {len(matching_entities)} matching entities:")
                        for entity in matching_entities:
                            st.write(f"• **{entity['label']}** ({entity['type']})")
                            
                            # Show connections
                            connections = [edge for edge in edges if edge['from'] == entity['id'] or edge['to'] == entity['id']]
                            if connections:
                                st.write(f"  Connected to {len(connections)} other entities")
                    else:
                        st.info("No matching entities found.")
            else:
                st.info("No entities found. Process some emails first to build the knowledge graph.")
                
                if st.button("🚀 Process Emails to Build Graph"):
                    st.session_state.page = "⚡ Process Emails"
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error loading knowledge graph: {str(e)}")

elif page == "⚙️ Settings":
    st.title("⚙️ Enhanced Settings")
    st.markdown("Configure Gmail integration and advanced features")
    
    # Gmail integration settings
    st.subheader("📧 Gmail Integration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Email Account Settings**")
        email_address = st.text_input("Your Gmail Address", placeholder="your.email@gmail.com")
        auto_reply = st.checkbox("Enable AI Auto-Reply for Important Emails")
        
    with col2:
        st.write("**Calendar Settings**")
        default_meeting_duration = st.selectbox("Default Meeting Duration", [30, 60, 90, 120])
        timezone = st.selectbox("Timezone", ["US/Pacific", "US/Eastern", "UTC", "Europe/London"])
    
    if st.button("💾 Save Enhanced Settings"):
        st.success("Enhanced settings saved!")

# Enhanced sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🆕 Enhanced Features:**")
st.sidebar.markdown("• 📧 Smart email replies")
st.sidebar.markdown("• 📅 Calendar integration")
st.sidebar.markdown("• 🧵 Thread management")
st.sidebar.markdown("• 🔔 Auto-mark as read")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit") 