"""
Advanced Email Analysis Module
Provides comprehensive analytics and insights for processed emails
Integrates with the dashboard for visualization and reporting
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict, Counter
import re

# Configure logging
logger = logging.getLogger(__name__)

class EmailAnalyzer:
    """Advanced email analysis and insights generator"""
    
    def __init__(self, db_path: str = "email_assistant.db"):
        self.db_path = db_path
        self.ensure_tables_exist()
    
    def ensure_tables_exist(self):
        """Ensure all required tables exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist and create if missing
        tables = [
            'emails', 'action_items', 'graph_entities', 
            'graph_relationships', 'uncertain_emails', 
            'email_status', 'processing_stats'
        ]
        
        existing_tables = cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ({})
        """.format(','.join('?' * len(tables))), tables).fetchall()
        
        existing_table_names = [table[0] for table in existing_tables]
        
        # Create missing tables with basic structure
        if 'emails' not in existing_table_names:
            cursor.execute('''
                CREATE TABLE emails (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    subject TEXT,
                    from_email TEXT,
                    to_email TEXT,
                    date TEXT,
                    body TEXT,
                    labels TEXT,
                    is_important BOOLEAN DEFAULT 0,
                    summary TEXT,
                    entities TEXT,
                    action_items TEXT
                )
            ''')
        
        if 'action_items' not in existing_table_names:
            cursor.execute('''
                CREATE TABLE action_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email_id TEXT,
                    description TEXT,
                    priority TEXT DEFAULT 'medium',
                    status TEXT DEFAULT 'pending',
                    created_date TEXT
                )
            ''')
        
        if 'graph_entities' not in existing_table_names:
            cursor.execute('''
                CREATE TABLE graph_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    description TEXT,
                    email_ids TEXT,
                    created_date TEXT
                )
            ''')
        
        if 'graph_relationships' not in existing_table_names:
            cursor.execute('''
                CREATE TABLE graph_relationships (
                    id TEXT PRIMARY KEY,
                    source_entity TEXT,
                    target_entity TEXT,
                    relationship_type TEXT,
                    description TEXT,
                    strength REAL DEFAULT 0.5,
                    email_ids TEXT,
                    created_date TEXT
                )
            ''')
        
        conn.commit()
        conn.close()
    
    def get_email_overview(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive email overview statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Date filter
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            # Basic email stats
            total_emails = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM emails WHERE date >= ?", 
                conn, params=[cutoff_date]
            ).iloc[0]['count']
            
            important_emails = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM emails WHERE is_important = 1 AND date >= ?", 
                conn, params=[cutoff_date]
            ).iloc[0]['count']
            
            # Action items stats
            try:
                total_actions = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM action_items WHERE created_date >= ?", 
                    conn, params=[cutoff_date]
                ).iloc[0]['count']
                
                pending_actions = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM action_items WHERE status = 'pending' AND created_date >= ?", 
                    conn, params=[cutoff_date]
                ).iloc[0]['count']
            except:
                total_actions = 0
                pending_actions = 0
            
            # Entity stats
            try:
                total_entities = pd.read_sql_query(
                    "SELECT COUNT(*) as count FROM graph_entities WHERE created_date >= ?", 
                    conn, params=[cutoff_date]
                ).iloc[0]['count']
            except:
                total_entities = 0
            
            # Thread analysis
            threads_df = pd.read_sql_query(
                "SELECT thread_id, COUNT(*) as email_count FROM emails WHERE date >= ? GROUP BY thread_id", 
                conn, params=[cutoff_date]
            )
            
            total_threads = len(threads_df)
            avg_emails_per_thread = threads_df['email_count'].mean() if not threads_df.empty else 0
            
            # Sender analysis
            senders_df = pd.read_sql_query(
                "SELECT from_email, COUNT(*) as email_count FROM emails WHERE date >= ? GROUP BY from_email ORDER BY email_count DESC LIMIT 10", 
                conn, params=[cutoff_date]
            )
            
            # Daily email trend
            daily_emails = pd.read_sql_query(
                "SELECT DATE(date) as email_date, COUNT(*) as count FROM emails WHERE date >= ? GROUP BY DATE(date) ORDER BY email_date", 
                conn, params=[cutoff_date]
            )
            
            # Importance trend
            importance_trend = pd.read_sql_query(
                "SELECT DATE(date) as email_date, SUM(CASE WHEN is_important = 1 THEN 1 ELSE 0 END) as important_count FROM emails WHERE date >= ? GROUP BY DATE(date) ORDER BY email_date", 
                conn, params=[cutoff_date]
            )
            
            return {
                'total_emails': total_emails,
                'important_emails': important_emails,
                'importance_rate': (important_emails / total_emails * 100) if total_emails > 0 else 0,
                'total_actions': total_actions,
                'pending_actions': pending_actions,
                'total_entities': total_entities,
                'total_threads': total_threads,
                'avg_emails_per_thread': round(avg_emails_per_thread, 1),
                'top_senders': senders_df.to_dict('records'),
                'daily_trend': daily_emails.to_dict('records'),
                'importance_trend': importance_trend.to_dict('records'),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting email overview: {e}")
            return {
                'total_emails': 0,
                'important_emails': 0,
                'importance_rate': 0,
                'total_actions': 0,
                'pending_actions': 0,
                'total_entities': 0,
                'total_threads': 0,
                'avg_emails_per_thread': 0,
                'top_senders': [],
                'daily_trend': [],
                'importance_trend': [],
                'period_days': days,
                'error': str(e)
            }
        finally:
            conn.close()
    
    def get_detailed_email_analysis(self, email_id: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific email"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get email details
            email_df = pd.read_sql_query(
                "SELECT * FROM emails WHERE id = ?", 
                conn, params=[email_id]
            )
            
            if email_df.empty:
                return {"error": "Email not found"}
            
            email = email_df.iloc[0].to_dict()
            
            # Parse JSON fields
            try:
                email['entities'] = json.loads(email['entities']) if email['entities'] else []
            except:
                email['entities'] = []
            
            try:
                email['action_items'] = json.loads(email['action_items']) if email['action_items'] else []
            except:
                email['action_items'] = []
            
            try:
                email['labels'] = json.loads(email['labels']) if email['labels'] else []
            except:
                email['labels'] = []
            
            # Get related action items
            actions_df = pd.read_sql_query(
                "SELECT * FROM action_items WHERE email_id = ?", 
                conn, params=[email_id]
            )
            
            # Get related entities
            entities_df = pd.read_sql_query(
                "SELECT * FROM graph_entities WHERE email_ids LIKE ?", 
                conn, params=[f'%{email_id}%']
            )
            
            # Get thread context
            thread_emails = pd.read_sql_query(
                "SELECT id, subject, from_email, date, is_important FROM emails WHERE thread_id = ? ORDER BY date", 
                conn, params=[email['thread_id']]
            )
            
            # Sentiment and complexity analysis
            analysis = self._analyze_email_content(email['body'], email['subject'])
            
            return {
                'email': email,
                'related_actions': actions_df.to_dict('records'),
                'related_entities': entities_df.to_dict('records'),
                'thread_context': thread_emails.to_dict('records'),
                'content_analysis': analysis,
                'entity_count': len(email['entities']),
                'action_count': len(email['action_items']),
                'thread_position': len(thread_emails[thread_emails['date'] <= email['date']]),
                'thread_length': len(thread_emails)
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed email analysis: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def get_action_items_analysis(self) -> Dict[str, Any]:
        """Get comprehensive action items analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get all action items
            actions_df = pd.read_sql_query(
                "SELECT * FROM action_items", conn
            )
            
            if actions_df.empty:
                return {
                    'total_actions': 0,
                    'by_status': {},
                    'by_priority': {},
                    'by_category': {},
                    'overdue_count': 0,
                    'recent_actions': []
                }
            
            # Status distribution
            status_dist = actions_df['status'].value_counts().to_dict()
            
            # Priority distribution
            priority_dist = actions_df['priority'].value_counts().to_dict()
            
            # Category distribution (if available)
            category_dist = {}
            if 'category' in actions_df.columns:
                category_dist = actions_df['category'].value_counts().to_dict()
            
            # Overdue analysis
            current_date = datetime.now().isoformat()
            overdue_count = 0
            
            if 'due_date' in actions_df.columns:
                try:
                    overdue_actions = actions_df[
                        (actions_df['due_date'].notna()) & 
                        (actions_df['due_date'] < current_date) &
                        (actions_df['status'] == 'pending')
                    ]
                    overdue_count = len(overdue_actions)
                except:
                    # Fallback if date comparison fails
                    overdue_count = 0
            
            # Recent actions
            try:
                # Convert created_date to datetime for sorting
                actions_df['created_date_dt'] = pd.to_datetime(actions_df['created_date'], errors='coerce')
                recent_actions = actions_df.nlargest(10, 'created_date_dt')[
                    ['description', 'priority', 'status', 'created_date']
                ].to_dict('records')
            except:
                # Fallback: just take the first 10 rows
                recent_actions = actions_df.head(10)[
                    ['description', 'priority', 'status', 'created_date']
                ].to_dict('records')
            
            # Weekly trend
            try:
                actions_df['created_week'] = pd.to_datetime(actions_df['created_date'], errors='coerce').dt.to_period('W')
                weekly_trend = actions_df.groupby('created_week').size().tail(8).to_dict()
            except:
                weekly_trend = {}
            
            return {
                'total_actions': len(actions_df),
                'by_status': status_dist,
                'by_priority': priority_dist,
                'by_category': category_dist,
                'overdue_count': overdue_count,
                'recent_actions': recent_actions,
                'weekly_trend': {str(k): v for k, v in weekly_trend.items()},
                'completion_rate': (status_dist.get('completed', 0) / len(actions_df) * 100) if len(actions_df) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting action items analysis: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def get_entity_analysis(self) -> Dict[str, Any]:
        """Get comprehensive entity analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            entities_df = pd.read_sql_query(
                "SELECT * FROM graph_entities", conn
            )
            
            if entities_df.empty:
                return {
                    'total_entities': 0,
                    'by_type': {},
                    'top_entities': [],
                    'entity_network': {}
                }
            
            # Type distribution
            type_dist = entities_df['type'].value_counts().to_dict()
            
            # Top entities by frequency
            entities_df['email_count'] = entities_df['email_ids'].apply(
                lambda x: len(json.loads(x)) if x else 0
            )
            
            top_entities = entities_df.nlargest(10, 'email_count')[
                ['name', 'type', 'email_count', 'description']
            ].to_dict('records')
            
            # Entity relationships
            relationships_df = pd.read_sql_query(
                "SELECT * FROM graph_relationships", conn
            )
            
            entity_network = {}
            if not relationships_df.empty:
                # Create network structure
                for _, rel in relationships_df.iterrows():
                    source = rel['source_entity']
                    target = rel['target_entity']
                    rel_type = rel['relationship_type']
                    
                    if source not in entity_network:
                        entity_network[source] = []
                    
                    entity_network[source].append({
                        'target': target,
                        'type': rel_type,
                        'strength': rel.get('strength', 0.5)
                    })
            
            return {
                'total_entities': len(entities_df),
                'by_type': type_dist,
                'top_entities': top_entities,
                'entity_network': entity_network,
                'total_relationships': len(relationships_df) if not relationships_df.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting entity analysis: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def get_sender_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed sender analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Sender statistics
            senders_df = pd.read_sql_query("""
                SELECT 
                    from_email,
                    COUNT(*) as total_emails,
                    SUM(CASE WHEN is_important = 1 THEN 1 ELSE 0 END) as important_emails,
                    AVG(CASE WHEN is_important = 1 THEN 1.0 ELSE 0.0 END) as importance_rate,
                    MIN(date) as first_email,
                    MAX(date) as last_email
                FROM emails 
                WHERE date >= ?
                GROUP BY from_email
                ORDER BY total_emails DESC
            """, conn, params=[cutoff_date])
            
            # Domain analysis
            senders_df['domain'] = senders_df['from_email'].apply(
                lambda x: x.split('@')[1] if '@' in x else 'unknown'
            )
            
            domain_stats = senders_df.groupby('domain').agg({
                'total_emails': 'sum',
                'important_emails': 'sum'
            }).reset_index()
            
            domain_stats['importance_rate'] = (
                domain_stats['important_emails'] / domain_stats['total_emails'] * 100
            ).round(1)
            
            # VIP senders (high importance rate and volume)
            vip_senders = senders_df[
                (senders_df['total_emails'] >= 3) & 
                (senders_df['importance_rate'] >= 0.5)
            ].sort_values('importance_rate', ascending=False)
            
            return {
                'total_senders': len(senders_df),
                'top_senders': senders_df.head(10).to_dict('records'),
                'domain_analysis': domain_stats.sort_values('total_emails', ascending=False).head(10).to_dict('records'),
                'vip_senders': vip_senders.head(10).to_dict('records'),
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting sender analysis: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def search_emails(self, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Advanced email search with filters"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Base query
            sql = """
                SELECT id, subject, from_email, date, is_important, summary
                FROM emails 
                WHERE 1=1
            """
            params = []
            
            # Text search
            if query:
                sql += " AND (subject LIKE ? OR body LIKE ? OR from_email LIKE ?)"
                search_term = f"%{query}%"
                params.extend([search_term, search_term, search_term])
            
            # Apply filters
            if filters:
                if filters.get('important_only'):
                    sql += " AND is_important = 1"
                
                if filters.get('date_from'):
                    sql += " AND date >= ?"
                    params.append(filters['date_from'])
                
                if filters.get('date_to'):
                    sql += " AND date <= ?"
                    params.append(filters['date_to'])
                
                if filters.get('sender'):
                    sql += " AND from_email LIKE ?"
                    params.append(f"%{filters['sender']}%")
            
            sql += " ORDER BY date DESC LIMIT 50"
            
            results_df = pd.read_sql_query(sql, conn, params=params)
            
            return {
                'results': results_df.to_dict('records'),
                'total_found': len(results_df),
                'query': query,
                'filters': filters or {}
            }
            
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def get_thread_analysis(self, thread_id: str) -> Dict[str, Any]:
        """Get detailed thread analysis"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get all emails in thread
            thread_df = pd.read_sql_query(
                "SELECT * FROM emails WHERE thread_id = ? ORDER BY date", 
                conn, params=[thread_id]
            )
            
            if thread_df.empty:
                return {"error": "Thread not found"}
            
            # Thread statistics
            total_emails = len(thread_df)
            important_emails = thread_df['is_important'].sum()
            participants = thread_df['from_email'].nunique()
            
            # Timeline analysis
            thread_df['date_parsed'] = pd.to_datetime(thread_df['date'])
            duration = (thread_df['date_parsed'].max() - thread_df['date_parsed'].min()).days
            
            # Response time analysis
            response_times = []
            for i in range(1, len(thread_df)):
                prev_time = thread_df.iloc[i-1]['date_parsed']
                curr_time = thread_df.iloc[i]['date_parsed']
                response_time = (curr_time - prev_time).total_seconds() / 3600  # hours
                response_times.append(response_time)
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Participant analysis
            participant_stats = thread_df.groupby('from_email').agg({
                'id': 'count',
                'is_important': 'sum'
            }).rename(columns={'id': 'email_count', 'is_important': 'important_count'})
            
            return {
                'thread_id': thread_id,
                'total_emails': total_emails,
                'important_emails': important_emails,
                'participants': participants,
                'duration_days': duration,
                'avg_response_time_hours': round(avg_response_time, 2),
                'emails': thread_df.to_dict('records'),
                'participant_stats': participant_stats.to_dict('index'),
                'subject': thread_df.iloc[0]['subject']
            }
            
        except Exception as e:
            logger.error(f"Error getting thread analysis: {e}")
            return {"error": str(e)}
        finally:
            conn.close()
    
    def _analyze_email_content(self, body: str, subject: str) -> Dict[str, Any]:
        """Analyze email content for sentiment, complexity, etc."""
        try:
            # Word count
            word_count = len(body.split())
            
            # Sentence count
            sentence_count = len(re.split(r'[.!?]+', body))
            
            # Average words per sentence
            avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
            
            # Question count
            question_count = body.count('?')
            
            # Urgency indicators
            urgency_words = ['urgent', 'asap', 'immediately', 'deadline', 'critical', 'important']
            urgency_score = sum(1 for word in urgency_words if word.lower() in body.lower())
            
            # Politeness indicators
            politeness_words = ['please', 'thank', 'appreciate', 'grateful', 'kindly']
            politeness_score = sum(1 for word in politeness_words if word.lower() in body.lower())
            
            # Complexity score (based on word length and sentence structure)
            long_words = [word for word in body.split() if len(word) > 6]
            complexity_score = (len(long_words) / word_count * 100) if word_count > 0 else 0
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_words_per_sentence': round(avg_words_per_sentence, 1),
                'question_count': question_count,
                'urgency_score': urgency_score,
                'politeness_score': politeness_score,
                'complexity_score': round(complexity_score, 1),
                'has_attachments': 'attachment' in body.lower(),
                'has_links': 'http' in body.lower() or 'www.' in body.lower()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing email content: {e}")
            return {}
    
    def get_uncertain_emails_for_review(self) -> List[Dict[str, Any]]:
        """Get uncertain emails that need human review"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            uncertain_df = pd.read_sql_query("""
                SELECT 
                    ue.id as uncertain_id,
                    ue.email_id,
                    ue.subject as uncertain_subject,
                    ue.from_email as uncertain_from,
                    ue.date as uncertain_date,
                    ue.classification_reason,
                    ue.analysis_data,
                    ue.user_feedback,
                    ue.created_date,
                    ue.reviewed,
                    ue.confidence_score,
                    e.subject as email_subject,
                    e.from_email as email_from,
                    e.body as email_body
                FROM uncertain_emails ue
                LEFT JOIN emails e ON ue.email_id = e.id
                WHERE ue.reviewed = 0
                ORDER BY ue.created_date DESC
            """, conn)
            
            return uncertain_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting uncertain emails: {e}")
            return []
        finally:
            conn.close()
    
    def get_processing_performance(self) -> Dict[str, Any]:
        """Get processing performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get processing stats if available
            try:
                stats_df = pd.read_sql_query(
                    "SELECT * FROM processing_stats ORDER BY date DESC LIMIT 10", 
                    conn
                )
                
                if not stats_df.empty:
                    latest_stats = stats_df.iloc[0].to_dict()
                    avg_processing_time = stats_df['processing_time'].mean() if 'processing_time' in stats_df.columns else 0
                    
                    return {
                        'latest_processing': latest_stats,
                        'avg_processing_time': round(avg_processing_time, 2),
                        'processing_history': stats_df.to_dict('records')
                    }
            except:
                pass
            
            # Fallback to basic stats
            total_emails = pd.read_sql_query("SELECT COUNT(*) as count FROM emails", conn).iloc[0]['count']
            processed_today = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM emails WHERE DATE(date) = DATE('now')", 
                conn
            ).iloc[0]['count']
            
            return {
                'total_emails_processed': total_emails,
                'processed_today': processed_today,
                'avg_processing_time': 0,
                'processing_history': []
            }
            
        except Exception as e:
            logger.error(f"Error getting processing performance: {e}")
            return {"error": str(e)}
        finally:
            conn.close()