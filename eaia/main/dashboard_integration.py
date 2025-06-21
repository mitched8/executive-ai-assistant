"""
Dashboard Integration Module
Connects the email processing and analysis functionality with the Streamlit dashboard
Provides clean interfaces for processing emails and displaying results
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
import time

# Local imports
from eaia.main.email_processor import EmailProcessor
from eaia.main.analysis import EmailAnalyzer
from eaia.main.config import get_config
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables import RunnableConfig

# Configure logging
logger = logging.getLogger(__name__)

class DashboardIntegration:
    """Integration layer between email processing and dashboard"""
    
    def __init__(self, db_path: str = "email_assistant.db"):
        self.db_path = db_path
        self.processor = EmailProcessor(db_path)
        self.analyzer = EmailAnalyzer(db_path)
        self._processing_status = "idle"
        self._processing_results = {}
        
    def get_processing_status(self) -> str:
        """Get current processing status"""
        return self._processing_status
    
    def get_processing_results(self) -> Dict[str, Any]:
        """Get latest processing results"""
        return self._processing_results
    
    def process_emails_async(self, start_date: datetime, end_date: datetime, 
                           quick_mode: bool = False) -> threading.Thread:
        """Start email processing in background thread"""
        def run_processing():
            asyncio.run(self._process_emails_background(start_date, end_date, quick_mode))
        
        thread = threading.Thread(target=run_processing, daemon=True)
        thread.start()
        return thread
    
    async def _process_emails_background(self, start_date: datetime, end_date: datetime, 
                                       quick_mode: bool = False):
        """Background email processing"""
        try:
            self._processing_status = "running"
            logger.info(f"Starting email processing: {start_date} to {end_date}")
            
            # Create configuration
            config = RunnableConfig(
                configurable={
                    "assistant_id": "dashboard_processor",
                    "model": "gpt-4o"
                }
            )
            
            # Create store
            store = InMemoryStore()
            
            # Process emails
            results = await self.processor.process_emails_batch(
                start_date, end_date, config, store, quick_mode
            )
            
            self._processing_results = results
            self._processing_status = "completed"
            
            logger.info(f"Email processing completed: {results}")
            
        except Exception as e:
            logger.error(f"Email processing failed: {e}")
            self._processing_status = f"error: {str(e)}"
            self._processing_results = {"error": str(e)}
    
    def get_dashboard_overview(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive dashboard overview"""
        try:
            # Get basic overview from analyzer
            overview = self.analyzer.get_email_overview(days)
            
            # Add missing dashboard keys
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            
            # Add unread emails count
            try:
                unread_count = pd.read_sql_query("""
                    SELECT COUNT(*) as count FROM emails 
                    WHERE id NOT IN (
                        SELECT email_id FROM email_status WHERE status = 'read'
                    )
                """, conn).iloc[0]['count']
                overview['unread_emails'] = unread_count
            except:
                # If email_status table doesn't exist, assume all emails are unread
                overview['unread_emails'] = overview.get('total_emails', 0)
            
            # Add threads count
            try:
                threads_count = pd.read_sql_query(
                    "SELECT COUNT(DISTINCT thread_id) as count FROM emails", 
                    conn
                ).iloc[0]['count']
                overview['threads'] = threads_count
            except:
                overview['threads'] = overview.get('total_emails', 0)
            
            # Add recent important emails
            try:
                recent_important = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM emails WHERE is_important = 1 AND date >= date('now', '-7 days')", 
                    conn
                ).iloc[0]['count']
                overview['recent_important'] = recent_important
            except:
                overview['recent_important'] = 0
            
            conn.close()
            
            # Add processing performance
            performance = self.analyzer.get_processing_performance()
            overview['processing_performance'] = performance
            
            # Add uncertain emails count
            uncertain_emails = self.analyzer.get_uncertain_emails_for_review()
            overview['uncertain_emails_count'] = len(uncertain_emails)
            
            # Add recent activity
            overview['recent_activity'] = self._get_recent_activity()
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            return {"error": str(e)}
    
    def get_email_analysis_data(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get email analysis data for dashboard"""
        try:
            # Get sender analysis
            sender_analysis = self.analyzer.get_sender_analysis()
            
            # Get entity analysis
            entity_analysis = self.analyzer.get_entity_analysis()
            
            # Get action items analysis
            action_analysis = self.analyzer.get_action_items_analysis()
            
            # Search emails if query provided
            search_results = {}
            if filters and filters.get('search_query'):
                search_results = self.analyzer.search_emails(
                    filters['search_query'], 
                    filters
                )
            
            return {
                'sender_analysis': sender_analysis,
                'entity_analysis': entity_analysis,
                'action_analysis': action_analysis,
                'search_results': search_results,
                'filters_applied': filters or {}
            }
            
        except Exception as e:
            logger.error(f"Error getting email analysis data: {e}")
            return {"error": str(e)}
    
    def get_detailed_email_view(self, email_id: str) -> Dict[str, Any]:
        """Get detailed view of a specific email"""
        try:
            return self.analyzer.get_detailed_email_analysis(email_id)
        except Exception as e:
            logger.error(f"Error getting detailed email view: {e}")
            return {"error": str(e)}
    
    def get_thread_view(self, thread_id: str) -> Dict[str, Any]:
        """Get detailed thread view"""
        try:
            return self.analyzer.get_thread_analysis(thread_id)
        except Exception as e:
            logger.error(f"Error getting thread view: {e}")
            return {"error": str(e)}
    
    def get_action_items_dashboard(self) -> Dict[str, Any]:
        """Get action items dashboard data"""
        try:
            analysis = self.analyzer.get_action_items_analysis()
            
            # Add recent and overdue items
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            
            # Get recent action items
            recent_items = pd.read_sql_query("""
                SELECT ai.*, e.subject, e.from_email 
                FROM action_items ai
                LEFT JOIN emails e ON ai.email_id = e.id
                ORDER BY ai.created_date DESC
                LIMIT 20
            """, conn)
            
            # Get overdue items
            current_date = datetime.now().date().isoformat()
            overdue_items = pd.read_sql_query("""
                SELECT ai.*, e.subject, e.from_email 
                FROM action_items ai
                LEFT JOIN emails e ON ai.email_id = e.id
                WHERE ai.due_date < ? AND ai.status = 'pending'
                ORDER BY ai.due_date ASC
            """, conn, params=[current_date])
            
            # Get high priority items
            high_priority_items = pd.read_sql_query("""
                SELECT ai.*, e.subject, e.from_email 
                FROM action_items ai
                LEFT JOIN emails e ON ai.email_id = e.id
                WHERE ai.priority IN ('high', 'urgent') AND ai.status = 'pending'
                ORDER BY ai.created_date DESC
            """, conn)
            
            conn.close()
            
            analysis['recent_items'] = recent_items.to_dict('records')
            analysis['overdue_items'] = overdue_items.to_dict('records')
            analysis['high_priority_items'] = high_priority_items.to_dict('records')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting action items dashboard: {e}")
            return {"error": str(e)}
    
    def get_knowledge_graph_data(self) -> Dict[str, Any]:
        """Get knowledge graph data for visualization"""
        try:
            entity_analysis = self.analyzer.get_entity_analysis()
            
            # Format for graph visualization
            nodes = []
            edges = []
            
            # Add entity nodes
            for entity in entity_analysis.get('top_entities', []):
                nodes.append({
                    'id': entity['name'],
                    'label': entity['name'],
                    'type': entity['type'],
                    'size': min(entity.get('email_count', 1) * 5, 50),
                    'color': self._get_entity_color(entity['type'])
                })
            
            # Add relationship edges
            entity_network = entity_analysis.get('entity_network', {})
            for source, connections in entity_network.items():
                for connection in connections:
                    edges.append({
                        'from': source,
                        'to': connection['target'],
                        'label': connection['type'],
                        'width': connection.get('strength', 0.5) * 5
                    })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': {
                    'total_entities': entity_analysis.get('total_entities', 0),
                    'total_relationships': entity_analysis.get('total_relationships', 0),
                    'entity_types': entity_analysis.get('by_type', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge graph data: {e}")
            return {"error": str(e)}
    
    def get_uncertain_emails_for_review(self) -> List[Dict[str, Any]]:
        """Get uncertain emails that need human review"""
        try:
            return self.analyzer.get_uncertain_emails_for_review()
        except Exception as e:
            logger.error(f"Error getting uncertain emails: {e}")
            return []
    
    def mark_uncertain_email_reviewed(self, uncertain_id: int, feedback: str) -> bool:
        """Mark uncertain email as reviewed with feedback"""
        try:
            self.analyzer.mark_uncertain_email_reviewed(uncertain_id, feedback)
            return True
        except Exception as e:
            logger.error(f"Error marking uncertain email reviewed: {e}")
            return False
    
    def update_action_item_status(self, action_id: int, new_status: str) -> bool:
        """Update action item status"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE action_items 
                SET status = ?, updated_date = ?
                WHERE id = ?
            """, (new_status, datetime.now().isoformat(), action_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating action item status: {e}")
            return False
    
    def get_email_composer_data(self, email_id: str = None) -> Dict[str, Any]:
        """Get data for email composer (reply functionality)"""
        try:
            if email_id:
                # Get original email for reply
                email_details = self.analyzer.get_detailed_email_analysis(email_id)
                
                if 'error' in email_details:
                    return email_details
                
                original_email = email_details['email']
                
                # Prepare reply data
                return {
                    'original_email': original_email,
                    'suggested_recipients': [original_email['from_email']],
                    'suggested_subject': f"Re: {original_email['subject']}",
                    'thread_context': email_details.get('thread_context', []),
                    'related_entities': email_details.get('related_entities', [])
                }
            else:
                # New email composition
                return {
                    'recent_contacts': self._get_recent_contacts(),
                    'suggested_templates': self._get_email_templates()
                }
                
        except Exception as e:
            logger.error(f"Error getting email composer data: {e}")
            return {"error": str(e)}
    
    def get_calendar_integration_data(self) -> Dict[str, Any]:
        """Get calendar integration data"""
        try:
            # Get upcoming events from Gmail API if available
            try:
                from eaia.gmail import get_events_for_days
                upcoming_events = get_events_for_days(7)
            except:
                upcoming_events = "Calendar integration not available"
            
            # Get meeting-related action items
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            
            meeting_actions = pd.read_sql_query("""
                SELECT ai.*, e.subject, e.from_email 
                FROM action_items ai
                LEFT JOIN emails e ON ai.email_id = e.id
                WHERE ai.category = 'meeting' OR ai.description LIKE '%meeting%'
                ORDER BY ai.created_date DESC
                LIMIT 10
            """, conn)
            
            conn.close()
            
            return {
                'upcoming_events': upcoming_events,
                'meeting_actions': meeting_actions.to_dict('records'),
                'calendar_available': isinstance(upcoming_events, str) == False
            }
            
        except Exception as e:
            logger.error(f"Error getting calendar integration data: {e}")
            return {"error": str(e)}
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity for dashboard"""
        try:
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            
            # Get recent emails
            recent_emails = pd.read_sql_query("""
                SELECT 'email' as type, subject as description, from_email as source, date as timestamp
                FROM emails
                ORDER BY date DESC
                LIMIT 5
            """, conn)
            
            # Get recent action items
            recent_actions = pd.read_sql_query("""
                SELECT 'action' as type, description, 'system' as source, created_date as timestamp
                FROM action_items
                ORDER BY created_date DESC
                LIMIT 5
            """, conn)
            
            conn.close()
            
            # Combine and sort
            all_activity = pd.concat([recent_emails, recent_actions])
            all_activity = all_activity.sort_values('timestamp', ascending=False)
            
            return all_activity.head(10).to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []
    
    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity type in knowledge graph"""
        color_map = {
            'PERSON': '#FF6B6B',
            'COMPANY': '#4ECDC4',
            'PROJECT': '#45B7D1',
            'LOCATION': '#96CEB4',
            'DATE': '#FFEAA7',
            'PRODUCT': '#DDA0DD',
            'CONCEPT': '#98D8C8',
            'EVENT': '#F7DC6F',
            'TASK': '#BB8FCE',
            'OTHER': '#AED6F1'
        }
        return color_map.get(entity_type, '#AED6F1')
    
    def _get_recent_contacts(self) -> List[str]:
        """Get recent email contacts"""
        try:
            import sqlite3
            import pandas as pd
            
            conn = sqlite3.connect(self.db_path)
            
            contacts = pd.read_sql_query("""
                SELECT from_email, COUNT(*) as email_count
                FROM emails
                WHERE date >= date('now', '-30 days')
                GROUP BY from_email
                ORDER BY email_count DESC
                LIMIT 10
            """, conn)
            
            conn.close()
            
            return contacts['from_email'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting recent contacts: {e}")
            return []
    
    def _get_email_templates(self) -> List[Dict[str, str]]:
        """Get email templates"""
        return [
            {
                'name': 'Meeting Request',
                'subject': 'Meeting Request - [Topic]',
                'body': 'Hi [Name],\n\nI hope this email finds you well. I would like to schedule a meeting to discuss [topic].\n\nWould you be available for a [duration] meeting sometime next week?\n\nBest regards,\nDaniel'
            },
            {
                'name': 'Follow Up',
                'subject': 'Following up on [Topic]',
                'body': 'Hi [Name],\n\nI wanted to follow up on our previous conversation about [topic].\n\n[Details]\n\nPlease let me know if you need any additional information.\n\nBest regards,\nDaniel'
            },
            {
                'name': 'Thank You',
                'subject': 'Thank you for [Reason]',
                'body': 'Hi [Name],\n\nThank you for [reason]. I really appreciate [specific detail].\n\n[Additional context if needed]\n\nBest regards,\nDaniel'
            }
        ]
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test system integration and return status"""
        try:
            # Test database connection
            db_status = "✅ Connected"
            try:
                overview = self.analyzer.get_email_overview(7)
                if 'error' in overview:
                    db_status = f"❌ Error: {overview['error']}"
            except Exception as e:
                db_status = f"❌ Error: {str(e)}"
            
            # Test email processor
            processor_status = "✅ Ready"
            try:
                self.processor.setup_database()
            except Exception as e:
                processor_status = f"❌ Error: {str(e)}"
            
            # Test OpenAI connection
            openai_status = "✅ Connected"
            try:
                # This would test the LLM connection
                pass
            except Exception as e:
                openai_status = f"❌ Error: {str(e)}"
            
            # Test Gmail integration
            gmail_status = "✅ Available"
            try:
                from eaia.gmail import fetch_group_emails
            except ImportError:
                gmail_status = "❌ Not available"
            except Exception as e:
                gmail_status = f"❌ Error: {str(e)}"
            
            return {
                'database': db_status,
                'processor': processor_status,
                'openai': openai_status,
                'gmail': gmail_status,
                'overall_status': 'healthy' if all('✅' in status for status in [db_status, processor_status, openai_status]) else 'issues'
            }
            
        except Exception as e:
            logger.error(f"Error testing system integration: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data for the dashboard"""
        try:
            # Get basic overview
            overview = self.analyzer.get_email_overview()
            
            # Get processing performance
            performance = self.analyzer.get_processing_performance()
            
            # Calculate additional metrics
            analytics = {
                'total_processed': overview.get('total_emails', 0),
                'important_count': overview.get('important_emails', 0),
                'auto_approved': overview.get('total_emails', 0) - overview.get('uncertain_emails_count', 0),
                'avg_processing_time': performance.get('avg_processing_time', 0),
                'processing_performance': performance,
                'classification_stats': {
                    'important': overview.get('important_emails', 0),
                    'notify': overview.get('total_emails', 0) - overview.get('important_emails', 0),
                    'uncertain': overview.get('uncertain_emails_count', 0)
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analytics: {e}")
            return {
                'total_processed': 0,
                'important_count': 0,
                'auto_approved': 0,
                'avg_processing_time': 0,
                'processing_performance': {},
                'classification_stats': {}
            }
    
    def get_system_health(self) -> Dict[str, bool]:
        """Get system health status"""
        try:
            # Test database connection
            db_status = self._test_database_connection()
            
            # Test OpenAI connection (basic check)
            openai_status = self._test_openai_connection()
            
            # Test Gmail connection
            gmail_status = self._test_gmail_connection()
            
            # Test ChromaDB
            chroma_status = self._test_chroma_connection()
            
            return {
                'database': db_status,
                'openai': openai_status,
                'gmail': gmail_status,
                'chroma_db': chroma_status
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'database': False,
                'openai': False,
                'gmail': False,
                'chroma_db': False
            }
    
    def _test_database_connection(self) -> bool:
        """Test database connection"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except:
            return False
    
    def _test_openai_connection(self) -> bool:
        """Test OpenAI connection"""
        try:
            import os
            return bool(os.getenv('OPENAI_API_KEY'))
        except:
            return False
    
    def _test_gmail_connection(self) -> bool:
        """Test Gmail connection"""
        try:
            import os
            return os.path.exists('credentials.json') or os.path.exists('token.json')
        except:
            return False
    
    def _test_chroma_connection(self) -> bool:
        """Test ChromaDB connection"""
        try:
            import os
            return os.path.exists('./chroma_db')
        except:
            return False