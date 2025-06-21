"""
Comprehensive Email Processing Module
Integrates with the main folder utilities and YAML configuration
Uses the optimized single API call approach for efficient processing
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid
import time

# Third-party imports
import pandas as pd
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Local imports
from eaia.schemas import State, EmailData
from eaia.main.config import get_config
from eaia.main.triage import triage_input
from eaia.main.draft_response import draft_response
from eaia.main.human_inbox import save_email
from eaia.gmail import fetch_group_emails, mark_as_read
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig

# Configure logging
logger = logging.getLogger(__name__)

class EmailProcessor:
    """Main email processing class that integrates with main folder utilities"""
    
    def __init__(self, db_path: str = "email_assistant.db", chroma_path: str = "./chroma_db"):
        load_dotenv(override=True)
        
        # Validate OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        api_key = api_key.strip()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            request_timeout=30,
            max_retries=3,
            api_key=api_key
        )
        
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.setup_database()
        self.setup_chroma_db()
        
    def setup_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced emails table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                subject TEXT,
                from_email TEXT,
                to_email TEXT,
                date TEXT,
                body TEXT,
                labels TEXT,
                is_important BOOLEAN,
                summary TEXT,
                entities TEXT,
                action_items TEXT,
                confidence TEXT DEFAULT 'high',
                classification_reason TEXT,
                processed_date TEXT,
                processing_mode TEXT DEFAULT 'comprehensive'
            )
        ''')
        
        # Action items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT,
                description TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                created_date TEXT,
                due_date TEXT,
                assigned_to TEXT,
                category TEXT,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        # Enhanced entities table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_entities (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                description TEXT,
                email_ids TEXT,
                created_date TEXT,
                updated_date TEXT,
                importance_score REAL DEFAULT 0.5,
                frequency_count INTEGER DEFAULT 1
            )
        ''')
        
        # Enhanced relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_relationships (
                id TEXT PRIMARY KEY,
                source_entity TEXT,
                target_entity TEXT,
                relationship_type TEXT,
                description TEXT,
                strength REAL DEFAULT 0.5,
                email_ids TEXT,
                created_date TEXT,
                confidence REAL DEFAULT 0.5,
                FOREIGN KEY (source_entity) REFERENCES graph_entities (id),
                FOREIGN KEY (target_entity) REFERENCES graph_entities (id)
            )
        ''')
        
        # Uncertain emails for training
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uncertain_emails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT,
                subject TEXT,
                from_email TEXT,
                date TEXT,
                classification_reason TEXT,
                analysis_data TEXT,
                user_feedback TEXT DEFAULT NULL,
                created_date TEXT,
                reviewed BOOLEAN DEFAULT FALSE,
                confidence_score REAL,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        # Email status tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_status (
                email_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'unread',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                processed_by TEXT,
                processing_time REAL,
                api_calls_used INTEGER DEFAULT 1,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        # Processing statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processing_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                emails_processed INTEGER,
                important_emails INTEGER,
                entities_extracted INTEGER,
                relationships_created INTEGER,
                action_items_created INTEGER,
                total_api_calls INTEGER,
                processing_time REAL,
                mode TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def setup_chroma_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            
            # Email embeddings collection
            self.email_collection = self.chroma_client.get_or_create_collection(
                name="emails",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Entity embeddings collection
            self.entity_collection = self.chroma_client.get_or_create_collection(
                name="entities",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Relationship embeddings collection
            self.relationship_collection = self.chroma_client.get_or_create_collection(
                name="relationships",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize sentence transformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise
    
    async def comprehensive_email_analysis(self, email: EmailData, config: RunnableConfig) -> Dict[str, Any]:
        """Single comprehensive API call for complete email analysis"""
        prompt_config = get_config(config)
        
        comprehensive_prompt = f"""
        COMPREHENSIVE EMAIL ANALYSIS using Daniel's personal preferences:

        PERSONAL CONTEXT:
        - Full Name: {prompt_config['full_name']}
        - Background: {prompt_config['background']}
        - Email Preferences: {prompt_config.get('background_preferences', '')}
        - Timezone: {prompt_config.get('timezone', 'Europe/London')}

        TRIAGE CRITERIA:
        
        SKIP/IGNORE (triage_no):
        {prompt_config['triage_no']}
        
        NOTIFY ONLY (triage_notify):
        {prompt_config['triage_notify']}
        
        RESPOND/ACTION REQUIRED (triage_email):
        {prompt_config['triage_email']}

        EMAIL TO ANALYZE:
        Subject: {email.subject}
        From: {email.from_email}
        To: {email.to_email}
        Date: {email.date}
        Body: {email.body[:2000]}...

        PROVIDE COMPREHENSIVE ANALYSIS:
        
        1. TRIAGE DECISION with confidence level
        2. IMPORTANCE classification with reasoning
        3. DETAILED SUMMARY appropriate for importance level
        4. ENTITY EXTRACTION with types and descriptions
        5. ACTION ITEMS identification with priorities
        6. RELATIONSHIP MAPPING between entities
        7. RESPONSE RECOMMENDATION if needed

        Respond with ONLY this JSON structure:
        {{
            "triage_decision": "no|notify|email",
            "is_important": true|false,
            "confidence": "high|low",
            "classification_reason": "Detailed reasoning for this classification",
            "summary": "Comprehensive summary (detailed if important, brief if not)",
            "response_needed": true|false,
            "response_type": "reply|calendar_invite|question|ignore",
            "urgency": "low|medium|high|urgent",
            "entities": [
                {{
                    "name": "Entity Name",
                    "type": "PERSON|COMPANY|PROJECT|LOCATION|DATE|PRODUCT|CONCEPT|EVENT|TASK|OTHER",
                    "description": "Detailed description and context",
                    "importance": 0.1-1.0
                }}
            ],
            "action_items": [
                {{
                    "description": "Specific actionable task",
                    "priority": "low|medium|high|urgent",
                    "category": "task|meeting|deadline|follow_up|decision|other",
                    "due_date": "YYYY-MM-DD or null",
                    "assigned_to": "person or null"
                }}
            ],
            "relationships": [
                {{
                    "entity1": "Entity Name 1",
                    "entity2": "Entity Name 2",
                    "relationship": "WORKS_WITH|COLLABORATES|PART_OF|MANAGES|REPORTS_TO|RELATED_TO|MENTIONS|ATTENDS|ORGANIZES",
                    "description": "How they're related in this context",
                    "strength": 0.1-1.0
                }}
            ],
            "suggested_response": "Brief suggestion if response needed, null otherwise",
            "key_dates": ["YYYY-MM-DD list of important dates"],
            "follow_up_needed": true|false,
            "follow_up_date": "YYYY-MM-DD or null"
        }}
        """
        
        try:
            response_content = await self._make_llm_call_with_retry(comprehensive_prompt)
            
            # Clean response - remove markdown code blocks if present
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            
            # Validate and set defaults for required fields
            analysis.setdefault("triage_decision", "notify")
            analysis.setdefault("is_important", False)
            analysis.setdefault("confidence", "high")
            analysis.setdefault("entities", [])
            analysis.setdefault("action_items", [])
            analysis.setdefault("relationships", [])
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed in comprehensive analysis: {e}")
            # Return fallback analysis
            return {
                "triage_decision": "notify",
                "is_important": "important" in response_content.lower(),
                "confidence": "low",
                "classification_reason": "JSON parsing failed, fallback analysis",
                "summary": response_content[:200] + "..." if len(response_content) > 200 else response_content,
                "entities": [],
                "action_items": [],
                "relationships": []
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "triage_decision": "notify",
                "is_important": False,
                "confidence": "low",
                "classification_reason": f"Analysis error: {str(e)}",
                "summary": "Error in analysis",
                "entities": [],
                "action_items": [],
                "relationships": []
            }
    
    async def _make_llm_call_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Make LLM call with retry logic"""
        for attempt in range(max_retries):
            try:
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = (2 ** attempt) + (0.1 * attempt)
                logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    async def process_single_email(self, email: EmailData, config: RunnableConfig, store: BaseStore) -> EmailData:
        """Process a single email using comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Comprehensive analysis in single API call
            analysis = await self.comprehensive_email_analysis(email, config)
            
            # Update email with analysis results
            email.is_important = analysis.get("is_important", False)
            email.summary = analysis.get("summary", "")
            email.entities = [entity.get("name", "") for entity in analysis.get("entities", [])]
            email.action_items = [item.get("description", "") for item in analysis.get("action_items", [])]
            
            # Handle uncertain emails (low confidence)
            confidence = analysis.get("confidence", "high")
            if confidence == "low":
                await self._track_uncertain_email(email, analysis)
                logger.info(f"🤔 UNCERTAIN EMAIL flagged for review: {email.subject}")
            
            # Skip detailed processing for clearly unimportant emails
            triage_decision = analysis.get("triage_decision", "notify")
            if triage_decision == "no" and confidence == "high":
                logger.info(f"⏩ Skipping unimportant email: {email.subject}")
                skip_detailed = True
            else:
                skip_detailed = False
            
            # Store in database
            await self._save_email_to_db(email, analysis, confidence)
            
            # Store in vector database
            await self._store_email_in_chroma(email, analysis)
            
            if not skip_detailed:
                # Process entities and relationships
                await self._process_entities_from_analysis(email, analysis)
                await self._process_relationships_from_analysis(email, analysis)
                
                # Store action items
                await self._store_action_items(email, analysis)
            
            # Update processing statistics
            processing_time = time.time() - start_time
            await self._update_processing_stats(email, processing_time, 1)  # 1 API call used
            
            logger.info(f"✅ Email processed: {email.subject} (Important: {email.is_important}, "
                       f"Entities: {len(email.entities)}, Actions: {len(email.action_items)})")
            
        except Exception as e:
            logger.error(f"Error processing email {email.id}: {e}")
            # Still save basic email data even if processing fails
            await self._save_email_to_db(email, {}, "low")
            
        return email
    
    async def process_emails_batch(self, start_date: datetime, end_date: datetime, 
                                 config: RunnableConfig, store: BaseStore,
                                 quick_mode: bool = False) -> Dict[str, Any]:
        """Process a batch of emails using the main folder utilities"""
        logger.info(f"Starting email processing from {start_date} to {end_date}")
        
        # Fetch emails using Gmail integration
        try:
            # Import and use the proper email fetching function
            from email_ai_assistant import EmailAIAssistant
            
            # Create temporary assistant instance for email fetching
            temp_assistant = EmailAIAssistant()
            email_objects = temp_assistant.fetch_all_emails(start_date, end_date)
            
            if not email_objects:
                logger.info("No emails found in the specified date range")
                return {"processed": 0, "important": 0, "action_items": 0}
                
        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            return {"processed": 0, "important": 0, "action_items": 0, "error": str(e)}
        
        # Process each email
        processed_emails = []
        important_count = 0
        total_action_items = 0
        
        logger.info(f"🚀 OPTIMIZED: Processing {len(email_objects)} emails with 1 API call each")
        
        for i, email in enumerate(email_objects):
            logger.info(f"Processing email {i+1}/{len(email_objects)}: {email.subject}")
            
            processed_email = await self.process_single_email(email, config, store)
            processed_emails.append(processed_email)
            
            if processed_email.is_important:
                important_count += 1
            
            total_action_items += len(processed_email.action_items)
            
            # Add small delay to prevent rate limiting
            time.sleep(0.5)
        
        # Store batch processing statistics
        await self._store_batch_stats(len(processed_emails), important_count, total_action_items)
        
        logger.info(f"✅ Batch processing complete!")
        logger.info(f"   - Total emails processed: {len(processed_emails)}")
        logger.info(f"   - Important emails: {important_count}")
        logger.info(f"   - Total action items: {total_action_items}")
        logger.info(f"   - API calls used: {len(processed_emails)} (1 per email)")
        
        return {
            "processed": len(processed_emails),
            "important": important_count,
            "action_items": total_action_items,
            "emails": processed_emails
        }
    
    async def _track_uncertain_email(self, email: EmailData, analysis: Dict[str, Any]):
        """Track uncertain emails for user review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO uncertain_emails 
            (email_id, subject, from_email, date, classification_reason, analysis_data, 
             created_date, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            email.id,
            email.subject,
            email.from_email,
            email.date.isoformat(),
            analysis.get("classification_reason", ""),
            json.dumps(analysis),
            datetime.now().isoformat(),
            0.5  # Low confidence score
        ))
        
        conn.commit()
        conn.close()
    
    async def _save_email_to_db(self, email: EmailData, analysis: Dict[str, Any], confidence: str):
        """Save email with comprehensive analysis to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO emails 
            (id, thread_id, subject, from_email, to_email, date, body, labels, 
             is_important, summary, entities, action_items, confidence, 
             classification_reason, processed_date, processing_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            email.id,
            email.thread_id,
            email.subject,
            email.from_email,
            email.to_email,
            email.date.isoformat(),
            email.body,
            json.dumps(email.labels),
            email.is_important,
            email.summary,
            json.dumps(email.entities),
            json.dumps(email.action_items),
            confidence,
            analysis.get("classification_reason", ""),
            datetime.now().isoformat(),
            "comprehensive"
        ))
        
        # Update email status
        cursor.execute('''
            INSERT OR REPLACE INTO email_status 
            (email_id, status, updated_at, processed_by, api_calls_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            email.id,
            "processed",
            datetime.now().isoformat(),
            "comprehensive_processor",
            1
        ))
        
        conn.commit()
        conn.close()
    
    async def _store_email_in_chroma(self, email: EmailData, analysis: Dict[str, Any]):
        """Store email in ChromaDB for vector search"""
        email_text = f"Subject: {email.subject}\nFrom: {email.from_email}\nSummary: {email.summary}\nBody: {email.body[:1000]}"
        embedding = self.embedding_model.encode(email_text).tolist()
        
        self.email_collection.add(
            documents=[email_text],
            embeddings=[embedding],
            metadatas=[{
                "email_id": email.id,
                "subject": email.subject,
                "from_email": email.from_email,
                "date": email.date.isoformat(),
                "is_important": email.is_important,
                "confidence": analysis.get("confidence", "high"),
                "triage_decision": analysis.get("triage_decision", "notify")
            }],
            ids=[email.id]
        )
    
    async def _process_entities_from_analysis(self, email: EmailData, analysis: Dict[str, Any]):
        """Process entities from comprehensive analysis"""
        entities_data = analysis.get("entities", [])
        
        for entity_data in entities_data:
            entity_id = str(uuid.uuid4())
            entity_name = entity_data.get("name", "")
            entity_type = entity_data.get("type", "OTHER")
            entity_description = entity_data.get("description", "")
            importance = entity_data.get("importance", 0.5)
            
            if not entity_name:
                continue
            
            # Store in database
            await self._store_entity_in_db(entity_id, entity_name, entity_type, 
                                         entity_description, email.id, importance)
            
            # Store in ChromaDB
            entity_text = f"Entity: {entity_name}\nType: {entity_type}\nDescription: {entity_description}\nContext: {email.subject}"
            embedding = self.embedding_model.encode(entity_text).tolist()
            
            self.entity_collection.add(
                documents=[entity_text],
                embeddings=[embedding],
                metadatas=[{
                    "entity_id": entity_id,
                    "name": entity_name,
                    "type": entity_type,
                    "email_id": email.id,
                    "importance": importance
                }],
                ids=[entity_id]
            )
    
    async def _process_relationships_from_analysis(self, email: EmailData, analysis: Dict[str, Any]):
        """Process relationships from comprehensive analysis"""
        relationships_data = analysis.get("relationships", [])
        
        for rel_data in relationships_data:
            relationship_id = str(uuid.uuid4())
            entity1_name = rel_data.get("entity1", "")
            entity2_name = rel_data.get("entity2", "")
            relationship_type = rel_data.get("relationship", "RELATED_TO")
            description = rel_data.get("description", "")
            strength = rel_data.get("strength", 0.5)
            
            if not entity1_name or not entity2_name:
                continue
            
            # Store in database
            await self._store_relationship_in_db(relationship_id, entity1_name, entity2_name,
                                               relationship_type, description, strength, email.id)
            
            # Store in ChromaDB
            rel_text = f"Relationship: {entity1_name} {relationship_type} {entity2_name}\nDescription: {description}\nContext: {email.subject}"
            embedding = self.embedding_model.encode(rel_text).tolist()
            
            self.relationship_collection.add(
                documents=[rel_text],
                embeddings=[embedding],
                metadatas=[{
                    "relationship_id": relationship_id,
                    "entity1": entity1_name,
                    "entity2": entity2_name,
                    "type": relationship_type,
                    "email_id": email.id,
                    "strength": strength
                }],
                ids=[relationship_id]
            )
    
    async def _store_action_items(self, email: EmailData, analysis: Dict[str, Any]):
        """Store action items from comprehensive analysis"""
        action_items_data = analysis.get("action_items", [])
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for action_data in action_items_data:
            cursor.execute('''
                INSERT INTO action_items 
                (email_id, description, priority, category, due_date, assigned_to, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                email.id,
                action_data.get("description", ""),
                action_data.get("priority", "medium"),
                action_data.get("category", "task"),
                action_data.get("due_date"),
                action_data.get("assigned_to"),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    async def _store_entity_in_db(self, entity_id: str, name: str, entity_type: str, 
                                description: str, email_id: str, importance: float):
        """Store entity in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO graph_entities 
            (id, name, type, description, email_ids, created_date, updated_date, 
             importance_score, frequency_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entity_id,
            name,
            entity_type,
            description,
            json.dumps([email_id]),
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            importance,
            1
        ))
        
        conn.commit()
        conn.close()
    
    async def _store_relationship_in_db(self, relationship_id: str, entity1: str, entity2: str,
                                      relationship_type: str, description: str, 
                                      strength: float, email_id: str):
        """Store relationship in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO graph_relationships 
            (id, source_entity, target_entity, relationship_type, description, 
             strength, email_ids, created_date, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            relationship_id,
            entity1,
            entity2,
            relationship_type,
            description,
            strength,
            json.dumps([email_id]),
            datetime.now().isoformat(),
            strength
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_processing_stats(self, email: EmailData, processing_time: float, api_calls: int):
        """Update processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE email_status 
            SET processing_time = ?, api_calls_used = ?
            WHERE email_id = ?
        ''', (processing_time, api_calls, email.id))
        
        conn.commit()
        conn.close()
    
    async def _store_batch_stats(self, processed: int, important: int, action_items: int):
        """Store batch processing statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO processing_stats 
            (date, emails_processed, important_emails, action_items_created, 
             total_api_calls, mode)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            processed,
            important,
            action_items,
            processed,  # 1 API call per email
            "comprehensive"
        ))
        
        conn.commit()
        conn.close()
    
    def get_uncertain_emails(self) -> List[Dict]:
        """Get uncertain emails for review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email_id, subject, from_email, date, classification_reason, 
                   created_date, confidence_score
            FROM uncertain_emails
            WHERE reviewed = FALSE
            ORDER BY created_date DESC
        ''')
        
        uncertain_emails = []
        for row in cursor.fetchall():
            uncertain_emails.append({
                'id': row[0],
                'email_id': row[1],
                'subject': row[2],
                'from_email': row[3],
                'date': row[4],
                'classification_reason': row[5],
                'created_date': row[6],
                'confidence_score': row[7]
            })
        
        conn.close()
        return uncertain_emails
    
    def mark_uncertain_email_reviewed(self, uncertain_id: int, user_feedback: str):
        """Mark uncertain email as reviewed with user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE uncertain_emails 
            SET reviewed = TRUE, user_feedback = ?
            WHERE id = ?
        ''', (user_feedback, uncertain_id))
        
        conn.commit()
        conn.close()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get overall stats
        stats = {}
        stats['total_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails", conn).iloc[0]['count']
        stats['important_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM emails WHERE is_important = 1", conn).iloc[0]['count']
        stats['total_entities'] = pd.read_sql_query("SELECT COUNT(*) as count FROM graph_entities", conn).iloc[0]['count']
        stats['total_relationships'] = pd.read_sql_query("SELECT COUNT(*) as count FROM graph_relationships", conn).iloc[0]['count']
        stats['total_action_items'] = pd.read_sql_query("SELECT COUNT(*) as count FROM action_items", conn).iloc[0]['count']
        stats['pending_actions'] = pd.read_sql_query("SELECT COUNT(*) as count FROM action_items WHERE status = 'pending'", conn).iloc[0]['count']
        stats['uncertain_emails'] = pd.read_sql_query("SELECT COUNT(*) as count FROM uncertain_emails WHERE reviewed = FALSE", conn).iloc[0]['count']
        
        # Get recent processing stats
        recent_stats = pd.read_sql_query('''
            SELECT * FROM processing_stats 
            ORDER BY date DESC LIMIT 1
        ''', conn)
        
        if not recent_stats.empty:
            stats['last_processing'] = recent_stats.iloc[0].to_dict()
        
        conn.close()
        return stats