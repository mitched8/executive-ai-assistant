#!/usr/bin/env python3
"""
Email AI Assistant - A comprehensive system for processing emails with AI agents
and building a knowledge graph for intelligent task management.

Requirements:
pip install openai google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
pip install langgraph langchain-openai networkx python-dotenv
"""

import os
import json
import base64
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Any
import logging
from dataclasses import dataclass, asdict
import re

# Third-party imports
import openai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import chromadb
from chromadb.config import Settings
import uuid
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BATCH_SIZE = 10
OPENAI_MODEL = "gpt-4o"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_RETRIES = 3
RETRY_DELAY = 1  # Base delay in seconds

@dataclass
class EmailData:
    """Data structure for email information"""
    id: str
    thread_id: str
    subject: str
    from_email: str
    to_email: str
    date: datetime
    body: str
    labels: List[str]
    is_important: bool = False
    summary: str = ""
    entities: List[str] = None
    action_items: List[str] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.action_items is None:
            self.action_items = []

class AgentState(TypedDict):
    """State for LangGraph agents"""
    email: EmailData
    analysis: Dict[str, Any]
    is_important: bool
    summary: str
    entities: List[str]
    action_items: List[str]
    next_action: str

class EmailAIAssistant:
    """Main class for the Email AI Assistant system"""
    
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.gmail_service = None
        
        # Force reload environment variables to get fresh API key
        load_dotenv(override=True)
        
        # Validate OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")
        
        # Strip any whitespace from API key
        api_key = api_key.strip()
        
        self.llm = ChatOpenAI(
            model=OPENAI_MODEL, 
            temperature=0,
            request_timeout=30,
            max_retries=MAX_RETRIES,
            api_key=api_key  # Explicitly pass the clean API key
        )
        self.db_path = "email_assistant.db"
        self.chroma_path = "./chroma_db"
        self.setup_database()
        self.setup_chroma_db()
        self.setup_langgraph()
        
    def setup_database(self):
        """Initialize SQLite database for storing emails and action items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create emails table
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
                action_items TEXT
            )
        ''')
        
        # Create action items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT,
                description TEXT,
                priority TEXT,
                status TEXT DEFAULT 'pending',
                created_date TEXT,
                due_date TEXT,
                FOREIGN KEY (email_id) REFERENCES emails (id)
            )
        ''')
        
        # Create graph entities table for GraphRAG
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_entities (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                description TEXT,
                email_ids TEXT,
                created_date TEXT,
                updated_date TEXT
            )
        ''')
        
        # Create graph relationships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_relationships (
                id TEXT PRIMARY KEY,
                source_entity TEXT,
                target_entity TEXT,
                relationship_type TEXT,
                description TEXT,
                strength REAL,
                email_ids TEXT,
                created_date TEXT,
                FOREIGN KEY (source_entity) REFERENCES graph_entities (id),
                FOREIGN KEY (target_entity) REFERENCES graph_entities (id)
            )
        ''')
        
        # Create uncertain emails table for training feedback
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
                reviewed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def setup_chroma_db(self):
        """Initialize ChromaDB for vector storage and GraphRAG"""
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Create collections for different types of data
        try:
            self.email_collection = self.chroma_client.get_collection("emails")
        except:
            self.email_collection = self.chroma_client.create_collection(
                name="emails",
                metadata={"description": "Email content and metadata"}
            )
            
        try:
            self.entity_collection = self.chroma_client.get_collection("entities")
        except:
            self.entity_collection = self.chroma_client.create_collection(
                name="entities",
                metadata={"description": "Extracted entities from emails"}
            )
            
        try:
            self.relationship_collection = self.chroma_client.get_collection("relationships")
        except:
            self.relationship_collection = self.chroma_client.create_collection(
                name="relationships",
                metadata={"description": "Relationships between entities"}
            )
            
        logger.info("ChromaDB initialized successfully")
        
    def test_openai_connection(self):
        """Comprehensive OpenAI connection test"""
        print("\n🔍 OpenAI Connection Diagnostic Test")
        print("=" * 50)
        
        # 1. Check API Key
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("❌ No OpenAI API key found in environment variables")
            print("   Please set OPENAI_API_KEY in your .env file")
            return False
        
        masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        print(f"✅ API Key found: {masked_key}")
        
        # 2. Test basic connectivity
        print("\n🌐 Testing internet connectivity...")
        try:
            import urllib.request
            urllib.request.urlopen('https://api.openai.com', timeout=10)
            print("✅ Can reach OpenAI API endpoint")
        except Exception as e:
            print(f"❌ Cannot reach OpenAI API: {e}")
            return False
        
        # 3. Test with direct OpenAI client (not LangChain)
        print("\n🤖 Testing direct OpenAI API call...")
        try:
            import openai
            client = openai.OpenAI(
                api_key=api_key,
                timeout=15.0
            )
            
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Respond with exactly: TEST_OK"}],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            print(f"✅ Direct API call successful: '{result}'")
            
        except openai.AuthenticationError as e:
            print(f"❌ Authentication failed: {e}")
            print("   Your API key is invalid or expired")
            return False
        except openai.RateLimitError as e:
            print(f"❌ Rate limit exceeded: {e}")
            print("   You may have exceeded your usage quota")
            return False
        except openai.APIConnectionError as e:
            print(f"❌ Connection error: {e}")
            print("   Network connectivity issue")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print(f"   Error type: {type(e).__name__}")
            return False
        
        # 4. Test LangChain OpenAI wrapper
        print("\n🔗 Testing LangChain OpenAI wrapper...")
        try:
            from langchain_openai import ChatOpenAI
            test_llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                request_timeout=15,
                max_retries=1,
                api_key=api_key.strip()  # Strip whitespace
            )
            
            from langchain.schema import HumanMessage
            response = test_llm.invoke([HumanMessage(content="Say: LANGCHAIN_OK")])
            result = response.content.strip()
            print(f"✅ LangChain wrapper works: '{result}'")
            
        except Exception as e:
            print(f"❌ LangChain wrapper failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            return False
        
        print("\n🎉 All OpenAI connection tests passed!")
        return True
    
    def check_system_status(self):
        """Check system status and configuration"""
        status = {
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "gmail_credentials": os.path.exists(self.credentials_path),
            "gmail_token": os.path.exists(self.token_path),
            "database": os.path.exists(self.db_path),
            "chroma_db": os.path.exists(self.chroma_path)
        }
        
        print("\n📊 System Status Check")
        print("=" * 30)
        for component, status_ok in status.items():
            status_emoji = "✅" if status_ok else "❌"
            print(f"{status_emoji} {component}: {'OK' if status_ok else 'Missing/Not configured'}")
        
        # Run comprehensive OpenAI test
        openai_ok = self.test_openai_connection()
        status["openai_connection"] = openai_ok
            
        return status
        
    def setup_gmail_credentials(self):
        """Setup Gmail API credentials"""
        creds = None
        
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
                
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
                
        self.gmail_service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail service initialized successfully")
        
    def extract_email_body(self, payload):
        """Extract email body from Gmail API payload"""
        body = ""
        
        if payload.get('mimeType') == 'text/plain':
            data = payload.get('body', {}).get('data')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8')
        elif payload.get('mimeType') == 'text/html':
            data = payload.get('body', {}).get('data')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8')
                # Simple HTML tag removal
                body = re.sub('<[^<]+?>', '', body)
        elif payload.get('parts'):
            for part in payload['parts']:
                part_body = self.extract_email_body(part)
                if part_body:
                    body = part_body
                    break
                    
        return body.strip()
        
    def fetch_emails_batch(self, start_date: datetime, end_date: datetime, 
                          page_token: str = None) -> tuple[List[EmailData], str]:
        """Fetch a batch of emails between two dates"""
        if not self.gmail_service:
            self.setup_gmail_credentials()
            
        # Convert dates to Gmail query format
        start_str = start_date.strftime('%Y/%m/%d')
        end_str = end_date.strftime('%Y/%m/%d')
        query = f"after:{start_str} before:{end_str}"
        
        try:
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=query,
                maxResults=BATCH_SIZE,
                pageToken=page_token
            ).execute()
            
            messages = results.get('messages', [])
            next_page_token = results.get('nextPageToken')
            
            emails = []
            for message in messages:
                try:
                    msg = self.gmail_service.users().messages().get(
                        userId='me',
                        id=message['id']
                    ).execute()
                    
                    payload = msg['payload']
                    headers = payload.get('headers', [])
                    
                    # Extract email details
                    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
                    from_email = next((h['value'] for h in headers if h['name'] == 'From'), '')
                    to_email = next((h['value'] for h in headers if h['name'] == 'To'), '')
                    date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '')
                    
                    # Parse date
                    try:
                        email_date = datetime.strptime(date_str.split(' (')[0], '%a, %d %b %Y %H:%M:%S %z')
                        email_date = email_date.replace(tzinfo=None)  # Remove timezone for simplicity
                    except:
                        email_date = datetime.now()
                    
                    body = self.extract_email_body(payload)
                    labels = msg.get('labelIds', [])
                    
                    email_data = EmailData(
                        id=message['id'],
                        thread_id=msg['threadId'],
                        subject=subject,
                        from_email=from_email,
                        to_email=to_email,
                        date=email_date,
                        body=body,
                        labels=labels
                    )
                    
                    emails.append(email_data)
                    
                except Exception as e:
                    logger.error(f"Error processing email {message['id']}: {e}")
                    continue
                    
            return emails, next_page_token
            
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return [], None
            
    def fetch_all_emails(self, start_date: datetime, end_date: datetime) -> List[EmailData]:
        """Fetch all emails between two dates, handling pagination"""
        all_emails = []
        page_token = None
        
        logger.info(f"Fetching emails from {start_date} to {end_date}")
        
        while True:
            emails, next_token = self.fetch_emails_batch(start_date, end_date, page_token)
            all_emails.extend(emails)
            
            logger.info(f"Fetched {len(emails)} emails (total: {len(all_emails)})")
            
            if not next_token:
                break
                
            page_token = next_token
            
        logger.info(f"Total emails fetched: {len(all_emails)}")
        return all_emails
        
    def setup_langgraph(self):
        """Setup LangGraph workflow for email processing - OPTIMIZED"""
        workflow = StateGraph(AgentState)
        
        # OPTIMIZATION: Single comprehensive analysis node
        workflow.add_node("comprehensive_analysis", self.comprehensive_email_analysis)
        workflow.add_node("update_knowledge_graph", self.update_knowledge_graph)
        
        # Add edges
        workflow.add_edge("comprehensive_analysis", "update_knowledge_graph")
        workflow.add_edge("update_knowledge_graph", END)
        
        # Set entry point
        workflow.set_entry_point("comprehensive_analysis")
        
        self.workflow = workflow.compile()
        
    def make_llm_call_with_retry(self, prompt: str, max_retries: int = MAX_RETRIES) -> str:
        """Make LLM call with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to make LLM call after {max_retries} attempts: {e}")
                    raise e
                
                # Exponential backoff with jitter
                delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
    
    def comprehensive_email_analysis(self, state: AgentState) -> AgentState:
        """Smart two-stage email analysis in a single API call - MEGA OPTIMIZED"""
        email = state["email"]
        
        smart_prompt = f"""
        SMART EMAIL ANALYSIS - Confidence-based classification:

        Email Subject: {email.subject}
        From: {email.from_email}
        To: {email.to_email}
        Date: {email.date}
        Body: {email.body[:2000]}...

        STAGE 1 - RELEVANCE CHECK WITH CONFIDENCE:
        Classify the email into one of three categories:

        CLEARLY IMPORTANT (confidence: high):
        - Actionable tasks or requests
        - Important business information
        - Meeting invitations or scheduling
        - Project updates or deadlines
        - Personal correspondence requiring response

        CLEARLY UNIMPORTANT (confidence: high):
        - Spam, newsletters, automated notifications
        - Marketing emails, promotions
        - System notifications without action needed
        - Out-of-office replies

        UNCERTAIN (confidence: low):
        - Could be important but context is unclear
        - Mixed content (some important, some not)
        - Borderline cases where human review needed
        - When you're not confident in classification

        STAGE 2 - CONDITIONAL PROCESSING:
        - IF CLEARLY IMPORTANT: Full analysis with entities, actions, relationships
        - IF UNCERTAIN: Full analysis (better safe than sorry for training)
        - IF CLEARLY UNIMPORTANT: Minimal summary only

        Respond with ONLY this JSON:
        {{
            "is_important": true|false,
            "confidence": "high|low",
            "classification_reason": "Brief explanation of why this classification",
            "summary": "Summary (detailed if important/uncertain, brief if unimportant)",
            "entities": [
                {{
                    "name": "Entity Name",
                    "type": "PERSON|COMPANY|PROJECT|LOCATION|DATE|PRODUCT|CONCEPT|OTHER",
                    "description": "Brief description"
                }}
            ],
            "action_items": [
                "Specific action item 1"
            ],
            "relationships": [
                {{
                    "entity1": "Entity Name 1",
                    "entity2": "Entity Name 2", 
                    "relationship": "WORKS_WITH|COLLABORATES|PART_OF|RELATED_TO|MENTIONS",
                    "description": "How they're related"
                }}
            ]
        }}

        NOTE: 
        - is_important: true for CLEARLY IMPORTANT and UNCERTAIN emails
        - confidence: "low" for UNCERTAIN emails (flags for human review)
        - Only populate detailed arrays if is_important is true
        """
        
        try:
            response_content = self.make_llm_call_with_retry(smart_prompt)
            
            # Clean response - remove markdown code blocks if present
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            analysis = json.loads(cleaned_response)
            
            # Extract all information from the comprehensive response
            is_important = analysis.get("is_important", False)
            confidence = analysis.get("confidence", "high")
            classification_reason = analysis.get("classification_reason", "")
            
            state["is_important"] = is_important
            state["summary"] = analysis.get("summary", "")
            
            # Extract entity names for the simple entities list
            entities_data = analysis.get("entities", [])
            state["entities"] = [entity.get("name", "") for entity in entities_data if entity.get("name")]
            
            state["action_items"] = analysis.get("action_items", [])
            state["analysis"] = analysis  # Store full analysis for knowledge graph processing
            
            # Handle uncertain emails
            if confidence == "low":
                logger.info(f"🤔 UNCERTAIN EMAIL flagged for review: {email.subject}")
                self.track_uncertain_email(email, classification_reason, analysis)
                uncertainty_note = " [FLAGGED FOR REVIEW]"
            else:
                uncertainty_note = ""
            
            # Skip processing if clearly unimportant
            if not is_important and confidence == "high":
                logger.info(f"⏩ Skipping unimportant email: {email.subject}")
                state["skip_processing"] = True
            else:
                state["skip_processing"] = False
            
            logger.info(f"✅ Smart analysis complete: Important={is_important}, Confidence={confidence}, "
                       f"Entities={len(state['entities'])}, Actions={len(state['action_items'])}{uncertainty_note}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed in comprehensive analysis: {e}")
            # Fallback to text extraction
            state["is_important"] = "important" in response_content.lower() or "action" in response_content.lower()
            state["summary"] = response_content[:200] + "..." if len(response_content) > 200 else response_content
            state["entities"] = []
            state["action_items"] = []
            state["analysis"] = {}
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            state["is_important"] = False
            state["summary"] = "Error in analysis"
            state["entities"] = []
            state["action_items"] = []
            state["analysis"] = {}
            
        return state
    
    def track_uncertain_email(self, email: EmailData, classification_reason: str, analysis: dict):
        """Track uncertain emails for user review and training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO uncertain_emails 
            (email_id, subject, from_email, date, classification_reason, analysis_data, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            email.id,
            email.subject,
            email.from_email,
            email.date.isoformat(),
            classification_reason,
            json.dumps(analysis),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def get_uncertain_emails(self) -> List[Dict]:
        """Get all uncertain emails pending review"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email_id, subject, from_email, date, classification_reason, created_date
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
                'created_date': row[6]
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
        
    def triage_email(self, state: AgentState) -> AgentState:
        """Triage email to determine importance and generate summary"""
        email = state["email"]
        
        prompt = f"""
        Analyze this email and determine:
        1. Is this email important/actionable? (true/false)
        2. Provide a concise summary (2-3 sentences)
        
        Email Subject: {email.subject}
        From: {email.from_email}
        Date: {email.date}
        Body: {email.body[:1000]}...
        
        IMPORTANT: Respond ONLY with valid JSON format, no additional text:
        {{
            "is_important": true,
            "summary": "your summary here"
        }}
        """
        
        try:
            response_content = self.make_llm_call_with_retry(prompt)
            
            # Try to parse JSON, handle cases where response isn't valid JSON
            try:
                # Clean response - remove markdown code blocks if present
                cleaned_response = response_content.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                analysis = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract information from text response
                logger.warning(f"Non-JSON response in triage: {response_content[:100]}...")
                analysis = {
                    "is_important": "important" in response_content.lower() or "action" in response_content.lower(),
                    "summary": response_content[:200] + "..." if len(response_content) > 200 else response_content
                }
            
            state["is_important"] = analysis.get("is_important", False)
            state["summary"] = analysis.get("summary", "")
            state["analysis"] = analysis
            
        except Exception as e:
            logger.error(f"Error in triage: {e}")
            state["is_important"] = False
            state["summary"] = "Error in analysis"
            state["analysis"] = {}
            
        return state
        
    def extract_entities(self, state: AgentState) -> AgentState:
        """Extract entities from email content"""
        email = state["email"]
        
        prompt = f"""
        Extract key entities from this email including:
        - People names
        - Companies/Organizations
        - Dates and deadlines
        - Projects or topics
        - Locations
        - Important keywords
        
        Email Subject: {email.subject}
        Body: {email.body[:1500]}...
        
        IMPORTANT: Respond ONLY with a valid JSON array, no additional text:
        ["entity1", "entity2", "entity3"]
        """
        
        try:
            response_content = self.make_llm_call_with_retry(prompt)
            
            # Try to parse JSON, handle non-JSON responses
            try:
                # Clean response - remove markdown code blocks if present
                cleaned_response = response_content.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                entities = json.loads(cleaned_response)
                state["entities"] = entities if isinstance(entities, list) else []
            except json.JSONDecodeError:
                # Extract entities from text response if JSON parsing fails
                logger.warning(f"Non-JSON response in extract_entities: {response_content[:100]}...")
                # Simple text extraction - look for quoted strings or comma-separated values
                import re
                entities = re.findall(r'"([^"]*)"', response_content)
                if not entities:
                    entities = [item.strip() for item in response_content.split(',') if item.strip()]
                state["entities"] = entities[:10]  # Limit to 10 entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            state["entities"] = []
            
        return state
        
    def identify_action_items(self, state: AgentState) -> AgentState:
        """Identify action items from email content"""
        email = state["email"]
        
        prompt = f"""
        Identify specific action items from this email. Look for:
        - Tasks that need to be completed
        - Deadlines or time-sensitive items
        - Requests for information or responses
        - Meeting scheduling needs
        - Follow-up requirements
        
        Email Subject: {email.subject}
        From: {email.from_email}
        Body: {email.body[:1500]}...
        
        IMPORTANT: Respond ONLY with a valid JSON array, no additional text:
        ["action item 1", "action item 2"]
        
        If no action items, return: []
        """
        
        try:
            response_content = self.make_llm_call_with_retry(prompt)
            
            # Try to parse JSON, handle non-JSON responses
            try:
                # Clean response - remove markdown code blocks if present
                cleaned_response = response_content.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                action_items = json.loads(cleaned_response)
                state["action_items"] = action_items if isinstance(action_items, list) else []
            except json.JSONDecodeError:
                # Extract action items from text response if JSON parsing fails
                logger.warning(f"Non-JSON response in identify_action_items: {response_content[:100]}...")
                # Simple text extraction - look for quoted strings or list items
                import re
                action_items = re.findall(r'"([^"]*)"', response_content)
                if not action_items:
                    # Look for bullet points or numbered lists
                    lines = response_content.split('\n')
                    action_items = [line.strip() for line in lines if line.strip() and any(char in line for char in ['-', '*', '1.', '2.', '3.'])]
                state["action_items"] = action_items[:5]  # Limit to 5 action items
            
        except Exception as e:
            logger.error(f"Error identifying action items: {e}")
            state["action_items"] = []
            
        return state
        
    def update_knowledge_graph(self, state: AgentState) -> AgentState:
        """Update GraphRAG knowledge base with email information - OPTIMIZED"""
        email = state["email"]
        entities = state.get("entities", [])
        summary = state.get("summary", "")
        analysis = state.get("analysis", {})
        skip_processing = state.get("skip_processing", False)
        
        # Always store email metadata (even if unimportant)
        self.store_email_in_chroma(email, summary, state.get("is_important", False))
        
        # Skip detailed processing for clearly unimportant emails
        if skip_processing:
            logger.info(f"⏩ Skipped detailed knowledge graph processing for unimportant email")
            state["next_action"] = "complete"
            return state
        
        # OPTIMIZATION: Use comprehensive analysis data if available
        if analysis and "entities" in analysis:
            # Use the detailed entity data from comprehensive analysis
            processed_entities = self.process_comprehensive_entities(email, analysis.get("entities", []))
            
            # Process relationships if provided in the analysis
            if "relationships" in analysis:
                self.store_comprehensive_relationships(email, analysis.get("relationships", []), processed_entities)
            
            logger.info(f"✅ Knowledge graph updated using comprehensive analysis data")
        else:
            # Fallback to old method if comprehensive analysis failed
            quick_mode = getattr(self, 'quick_mode', False)
            if not quick_mode:
                processed_entities = self.process_entities(email, entities)
                self.generate_relationships(email, processed_entities)
        
        state["next_action"] = "complete"
        return state
        
    def store_email_in_chroma(self, email: EmailData, summary: str, is_important: bool):
        """Store email content and metadata in ChromaDB"""
        # Create document text for embedding
        doc_text = f"Subject: {email.subject}\nFrom: {email.from_email}\nSummary: {summary}\nContent: {email.body[:2000]}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(doc_text).tolist()
        
        # Store in ChromaDB
        self.email_collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            metadatas=[{
                "email_id": email.id,
                "thread_id": email.thread_id,
                "subject": email.subject,
                "from_email": email.from_email,
                "to_email": email.to_email,
                "date": email.date.isoformat(),
                "is_important": is_important,
                "summary": summary,
                "labels": json.dumps(email.labels)
            }],
            ids=[email.id]
        )
        
    def process_entities(self, email: EmailData, entities: List[str]) -> List[Dict]:
        """Process and store entities with enhanced information - OPTIMIZED"""
        if not entities:
            return []
            
        processed_entities = []
        
        # OPTIMIZATION: Process all entities in a single LLM call
        entities_batch_prompt = f"""
        Analyze these entities from an email and provide information for each:
        
        Email Subject: {email.subject}
        Email Context: {email.body[:500]}...
        
        Entities to analyze: {', '.join(entities)}
        
        For each entity, provide:
        1. A brief description (1-2 sentences)
        2. Type classification: PERSON, COMPANY, PROJECT, LOCATION, DATE, PRODUCT, CONCEPT, or OTHER
        
        Respond with valid JSON only:
        {{
            "entity_analysis": [
                {{"name": "entity1", "description": "...", "type": "PERSON"}},
                {{"name": "entity2", "description": "...", "type": "COMPANY"}}
            ]
        }}
        """
        
        try:
            response_content = self.make_llm_call_with_retry(entities_batch_prompt)
            
            # Clean response
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            batch_result = json.loads(cleaned_response)
            entity_analyses = batch_result.get("entity_analysis", [])
            
        except Exception as e:
            logger.warning(f"Batch entity processing failed, falling back to simple approach: {e}")
            entity_analyses = [{"name": entity, "description": f"Entity from email: {email.subject}", "type": "OTHER"} for entity in entities]
        
        # Process each entity with the batch results
        for i, entity in enumerate(entities):
            entity_id = str(uuid.uuid4())
            
            # Get analysis from batch or use defaults
            if i < len(entity_analyses):
                analysis = entity_analyses[i]
                entity_description = analysis.get("description", f"Entity from email: {email.subject}")
                entity_type = analysis.get("type", "OTHER")
            else:
                entity_description = f"Entity from email: {email.subject}"
                entity_type = "OTHER"
            
            # Create embedding for entity
            entity_text = f"Entity: {entity}\nDescription: {entity_description}\nContext: {email.subject}"
            embedding = self.embedding_model.encode(entity_text).tolist()
            
            # Store in ChromaDB
            self.entity_collection.add(
                documents=[entity_text],
                embeddings=[embedding],
                metadatas=[{
                    "entity_id": entity_id,
                    "name": entity,
                    "type": entity_type,
                    "description": entity_description,
                    "email_id": email.id,
                    "created_date": datetime.now().isoformat()
                }],
                ids=[entity_id]
            )
            
            # Store in SQL database
            self.store_entity_in_db(entity_id, entity, entity_type, entity_description, email.id)
            
            processed_entities.append({
                "id": entity_id,
                "name": entity,
                "type": entity_type,
                "description": entity_description
            })
            
        return processed_entities
    
    def process_comprehensive_entities(self, email: EmailData, entities_data: List[Dict]) -> List[Dict]:
        """Process entities from comprehensive analysis - NO ADDITIONAL API CALLS"""
        processed_entities = []
        
        for entity_data in entities_data:
            entity_id = str(uuid.uuid4())
            entity_name = entity_data.get("name", "")
            entity_type = entity_data.get("type", "OTHER")
            entity_description = entity_data.get("description", f"Entity from email: {email.subject}")
            
            if not entity_name:
                continue
            
            # Create embedding for entity
            entity_text = f"Entity: {entity_name}\nDescription: {entity_description}\nContext: {email.subject}"
            embedding = self.embedding_model.encode(entity_text).tolist()
            
            # Store in ChromaDB
            self.entity_collection.add(
                documents=[entity_text],
                embeddings=[embedding],
                metadatas=[{
                    "entity_id": entity_id,
                    "name": entity_name,
                    "type": entity_type,
                    "description": entity_description,
                    "email_id": email.id,
                    "created_date": datetime.now().isoformat()
                }],
                ids=[entity_id]
            )
            
            # Store in SQL database
            self.store_entity_in_db(entity_id, entity_name, entity_type, entity_description, email.id)
            
            processed_entities.append({
                "id": entity_id,
                "name": entity_name,
                "type": entity_type,
                "description": entity_description
            })
            
        logger.info(f"Processed {len(processed_entities)} entities from comprehensive analysis")
        return processed_entities
        
    def store_comprehensive_relationships(self, email: EmailData, relationships_data: List[Dict], entities: List[Dict]):
        """Store relationships from comprehensive analysis - NO ADDITIONAL API CALLS"""
        entity_lookup = {entity["name"]: entity for entity in entities}
        
        for rel_data in relationships_data:
            entity1_name = rel_data.get("entity1", "")
            entity2_name = rel_data.get("entity2", "")
            relationship_type = rel_data.get("relationship", "RELATED_TO")
            description = rel_data.get("description", "")
            
            # Find corresponding entities
            entity1 = entity_lookup.get(entity1_name)
            entity2 = entity_lookup.get(entity2_name)
            
            if entity1 and entity2:
                relationship = {
                    "relationship_type": relationship_type,
                    "description": description,
                    "strength": 0.8  # Default strength for comprehensive analysis
                }
                self.store_relationship(entity1, entity2, relationship, email)
        
        logger.info(f"Stored {len(relationships_data)} relationships from comprehensive analysis")
        
    def generate_entity_description(self, entity: str, email: EmailData) -> str:
        """Generate description for an entity using LLM"""
        prompt = f"""
        Based on this email context, provide a brief description of the entity "{entity}":
        
        Email Subject: {email.subject}
        From: {email.from_email}
        Content: {email.body[:500]}...
        
        Provide a 1-2 sentence description of what "{entity}" represents in this context.
        """
        
        try:
            response_content = self.make_llm_call_with_retry(prompt)
            return response_content.strip()
        except Exception as e:
            logger.error(f"Error generating entity description: {e}")
            return f"Entity mentioned in email: {email.subject}"
            
    def classify_entity_type(self, entity: str, email: EmailData) -> str:
        """Classify entity type using LLM"""
        prompt = f"""
        Classify the entity "{entity}" into one of these categories:
        - PERSON
        - COMPANY
        - PROJECT
        - LOCATION
        - DATE
        - PRODUCT
        - CONCEPT
        - OTHER
        
        Context from email:
        Subject: {email.subject}
        Content: {email.body[:300]}...
        
        Return only the category name.
        """
        
        try:
            response_content = self.make_llm_call_with_retry(prompt)
            entity_type = response_content.strip().upper()
            valid_types = ["PERSON", "COMPANY", "PROJECT", "LOCATION", "DATE", "PRODUCT", "CONCEPT", "OTHER"]
            return entity_type if entity_type in valid_types else "OTHER"
        except Exception as e:
            logger.error(f"Error classifying entity type: {e}")
            return "OTHER"
            
    def generate_relationships(self, email: EmailData, entities: List[Dict]):
        """Generate and store relationships between entities - OPTIMIZED"""
        if len(entities) < 2:
            return
            
        # OPTIMIZATION: Limit relationship generation to avoid too many API calls
        # Only generate relationships for the most important entities (max 5)
        # and only if there are clear relationships
        
        if len(entities) > 5:
            # Only process top 5 entities to limit API calls
            entities = entities[:5]
            logger.info(f"Limiting relationship generation to top 5 entities for email {email.id}")
        
        # OPTIMIZATION: Batch relationship identification 
        entity_pairs = []
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                entity_pairs.append((entity1, entity2))
        
        # Only process if we have a reasonable number of pairs (max 10)
        if len(entity_pairs) > 10:
            logger.info(f"Skipping relationship generation for email {email.id} - too many entity pairs ({len(entity_pairs)})")
            return
            
        # Process relationships in smaller batches to reduce API calls
        for entity1, entity2 in entity_pairs:
            try:
                relationship = self.identify_relationship(entity1, entity2, email)
                if relationship:
                    self.store_relationship(entity1, entity2, relationship, email)
            except Exception as e:
                logger.warning(f"Failed to generate relationship between {entity1['name']} and {entity2['name']}: {e}")
                continue
                    
    def identify_relationship(self, entity1: Dict, entity2: Dict, email: EmailData) -> Optional[Dict]:
        """Identify relationship between two entities using LLM"""
        prompt = f"""
        Analyze the relationship between "{entity1['name']}" and "{entity2['name']}" in this email context:
        
        Email Subject: {email.subject}
        Content: {email.body[:800]}...
        
        Entity 1: {entity1['name']} ({entity1['type']})
        Entity 2: {entity2['name']} ({entity2['type']})
        
        If there's a meaningful relationship, respond in JSON format:
        {{
            "relationship_type": "WORKS_WITH|PART_OF|RELATED_TO|MENTIONS|COLLABORATES|OTHER",
            "description": "brief description of the relationship",
            "strength": 0.1-1.0
        }}
        
        If no meaningful relationship exists, respond with: {{"relationship_type": null}}
        """
        
        try:
            response_content = self.make_llm_call_with_retry(prompt)
            
            # Clean response - remove markdown code blocks if present
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```
            cleaned_response = cleaned_response.strip()
            
            relationship_data = json.loads(cleaned_response)
            
            if relationship_data.get("relationship_type"):
                return relationship_data
            return None
            
        except Exception as e:
            logger.error(f"Error identifying relationship: {e}")
            return None
            
    def store_relationship(self, entity1: Dict, entity2: Dict, relationship: Dict, email: EmailData):
        """Store relationship in both ChromaDB and SQL database"""
        relationship_id = str(uuid.uuid4())
        
        # Create relationship text for embedding
        rel_text = f"Relationship: {entity1['name']} {relationship['relationship_type']} {entity2['name']}\nDescription: {relationship['description']}\nContext: {email.subject}"
        embedding = self.embedding_model.encode(rel_text).tolist()
        
        # Store in ChromaDB
        self.relationship_collection.add(
            documents=[rel_text],
            embeddings=[embedding],
            metadatas=[{
                "relationship_id": relationship_id,
                "source_entity": entity1['id'],
                "target_entity": entity2['id'],
                "relationship_type": relationship['relationship_type'],
                "description": relationship['description'],
                "strength": relationship['strength'],
                "email_id": email.id,
                "created_date": datetime.now().isoformat()
            }],
            ids=[relationship_id]
        )
        
        # Store in SQL database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO graph_relationships 
            (id, source_entity, target_entity, relationship_type, description, strength, email_ids, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            relationship_id,
            entity1['id'],
            entity2['id'],
            relationship['relationship_type'],
            relationship['description'],
            relationship['strength'],
            json.dumps([email.id]),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def store_entity_in_db(self, entity_id: str, name: str, entity_type: str, description: str, email_id: str):
        """Store entity in SQL database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if entity already exists
        cursor.execute('SELECT id, email_ids FROM graph_entities WHERE name = ? AND type = ?', (name, entity_type))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing entity
            existing_email_ids = json.loads(existing[1])
            if email_id not in existing_email_ids:
                existing_email_ids.append(email_id)
                cursor.execute('''
                    UPDATE graph_entities 
                    SET email_ids = ?, updated_date = ?
                    WHERE id = ?
                ''', (json.dumps(existing_email_ids), datetime.now().isoformat(), existing[0]))
        else:
            # Insert new entity
            cursor.execute('''
                INSERT INTO graph_entities 
                (id, name, type, description, email_ids, created_date, updated_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entity_id,
                name,
                entity_type,
                description,
                json.dumps([email_id]),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
        
    def process_email(self, email: EmailData) -> EmailData:
        """Process a single email through the LangGraph workflow"""
        initial_state = {
            "email": email,
            "analysis": {},
            "is_important": False,
            "summary": "",
            "entities": [],
            "action_items": [],
            "next_action": ""
        }
        
        try:
            result = self.workflow.invoke(initial_state)
            
            # Update email object with results
            email.is_important = result.get("is_important", False)
            email.summary = result.get("summary", "")
            email.entities = result.get("entities", [])
            email.action_items = result.get("action_items", [])
            
        except Exception as e:
            logger.error(f"Error processing email {email.id}: {e}")
            
        return email
        
    def save_email_to_db(self, email: EmailData):
        """Save email data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO emails 
            (id, thread_id, subject, from_email, to_email, date, body, labels, 
             is_important, summary, entities, action_items)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            json.dumps(email.action_items)
        ))
        
        # Save action items
        for action_item in email.action_items:
            cursor.execute('''
                INSERT INTO action_items (email_id, description, priority, created_date)
                VALUES (?, ?, ?, ?)
            ''', (email.id, action_item, 'medium', datetime.now().isoformat()))
            
        conn.commit()
        conn.close()
        
    def vector_search_emails(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search emails using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in email collection
            results = self.email_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            for i, doc in enumerate(results['documents'][0]):
                search_results.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'type': 'email'
                })
                
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
            
    def vector_search_entities(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search entities using vector similarity"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            results = self.entity_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            for i, doc in enumerate(results['documents'][0]):
                search_results.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'type': 'entity'
                })
                
            return search_results
            
        except Exception as e:
            logger.error(f"Error in entity search: {e}")
            return []
            
    def vector_search_relationships(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search relationships using vector similarity"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            results = self.relationship_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            for i, doc in enumerate(results['documents'][0]):
                search_results.append({
                    'document': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'type': 'relationship'
                })
                
            return search_results
            
        except Exception as e:
            logger.error(f"Error in relationship search: {e}")
            return []
            
    def graphrag_search(self, query: str, n_results: int = 10) -> Dict[str, List]:
        """Comprehensive GraphRAG search across all collections"""
        # Search across all collections
        email_results = self.vector_search_emails(query, n_results//3)
        entity_results = self.vector_search_entities(query, n_results//3)
        relationship_results = self.vector_search_relationships(query, n_results//3)
        
        # Combine and rank results
        all_results = email_results + entity_results + relationship_results
        all_results.sort(key=lambda x: x['distance'])
        
        return {
            'emails': email_results,
            'entities': entity_results,
            'relationships': relationship_results,
            'combined': all_results[:n_results]
        }
        
    def get_action_items(self) -> List[Dict]:
        """Get all pending action items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ai.id, ai.email_id, ai.description, ai.priority, ai.status, 
                   ai.created_date, ai.due_date, e.subject, e.from_email
            FROM action_items ai
            JOIN emails e ON ai.email_id = e.id
            WHERE ai.status = 'pending'
            ORDER BY ai.created_date DESC
        ''')
        
        action_items = []
        for row in cursor.fetchall():
            action_items.append({
                'id': row[0],
                'email_id': row[1],
                'description': row[2],
                'priority': row[3],
                'status': row[4],
                'created_date': row[5],
                'due_date': row[6],
                'email_subject': row[7],
                'from_email': row[8]
            })
            
        conn.close()
        return action_items
        
    def add_action_item(self, description: str, priority: str = 'medium', 
                       due_date: str = None, email_id: str = None):
        """Add a new action item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO action_items (email_id, description, priority, created_date, due_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (email_id, description, priority, datetime.now().isoformat(), due_date))
        
        conn.commit()
        conn.close()
        
    def update_action_item_status(self, action_id: int, status: str):
        """Update action item status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE action_items SET status = ? WHERE id = ?
        ''', (status, action_id))
        
        conn.commit()
        conn.close()
        
    def search_knowledge_graph(self, query: str) -> List[Dict]:
        """Search GraphRAG knowledge base for relevant information"""
        # Use GraphRAG search for comprehensive results
        search_results = self.graphrag_search(query, n_results=10)
        
        # Format results for display
        formatted_results = []
        
        for result in search_results['combined']:
            formatted_result = {
                'type': result['type'],
                'content': result['document'][:200] + "..." if len(result['document']) > 200 else result['document'],
                'metadata': result['metadata'],
                'relevance_score': 1.0 - result['distance']  # Convert distance to relevance
            }
            
            if result['type'] == 'email':
                formatted_result['title'] = result['metadata'].get('subject', 'No Subject')
                formatted_result['from'] = result['metadata'].get('from_email', 'Unknown')
            elif result['type'] == 'entity':
                formatted_result['title'] = result['metadata'].get('name', 'Unknown Entity')
                formatted_result['entity_type'] = result['metadata'].get('type', 'Unknown')
            elif result['type'] == 'relationship':
                formatted_result['title'] = result['metadata'].get('relationship_type', 'Unknown Relationship')
                formatted_result['description'] = result['metadata'].get('description', '')
                
            formatted_results.append(formatted_result)
            
        return formatted_results
        
    def chat_interface(self):
        """Interactive chat interface for the assistant"""
        print("🤖 Email AI Assistant - Chat Interface")
        print("Commands: /actions, /search <query>, /add <task>, /complete <id>, /quit")
        print("-" * 60)
        
        while True:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            elif user_input.startswith('/actions'):
                action_items = self.get_action_items()
                if action_items:
                    print(f"\n📋 You have {len(action_items)} pending action items:")
                    for item in action_items:
                        print(f"  [{item['id']}] {item['description']}")
                        print(f"      From: {item['email_subject']} ({item['from_email']})")
                        print(f"      Priority: {item['priority']} | Created: {item['created_date'][:10]}")
                else:
                    print("\n✅ No pending action items!")
                    
            elif user_input.startswith('/search '):
                query = user_input[8:]
                results = self.search_knowledge_graph(query)
                if results:
                    print(f"\n🔍 Found {len(results)} relevant items:")
                    for i, result in enumerate(results[:5], 1):  # Limit to top 5
                        relevance = f"{result['relevance_score']:.2f}"
                        print(f"\n  {i}. [{result['type'].upper()}] {result['title']} (Relevance: {relevance})")
                        
                        if result['type'] == 'email':
                            print(f"     From: {result['from']}")
                            print(f"     Content: {result['content']}")
                        elif result['type'] == 'entity':
                            print(f"     Type: {result['entity_type']}")
                            print(f"     Content: {result['content']}")
                        elif result['type'] == 'relationship':
                            print(f"     Description: {result.get('description', 'N/A')}")
                            print(f"     Content: {result['content']}")
                else:
                    print(f"\n❌ No results found for '{query}'")
                    
            elif user_input.startswith('/add '):
                task = user_input[5:]
                self.add_action_item(task)
                print(f"✅ Added action item: {task}")
                
            elif user_input.startswith('/complete '):
                try:
                    action_id = int(user_input[10:])
                    self.update_action_item_status(action_id, 'completed')
                    print(f"✅ Marked action item {action_id} as completed")
                except ValueError:
                    print("❌ Invalid action item ID")
                    
            else:
                # General conversation
                prompt = f"""
                User question: {user_input}
                
                You are an AI assistant that helps manage emails and tasks. 
                You have access to a knowledge graph of emails and can help with:
                - Finding information from emails
                - Managing action items
                - Providing summaries and insights
                
                Respond helpfully and conversationally.
                """
                
                try:
                    response_content = self.make_llm_call_with_retry(prompt)
                    print(f"\n🤖 Assistant: {response_content}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
    def run_full_pipeline(self, start_date: datetime, end_date: datetime, quick_mode: bool = False):
        """Run the complete email processing pipeline"""
        logger.info("Starting email processing pipeline...")
        
        if quick_mode:
            logger.info("🚀 QUICK MODE: Faster processing with reduced knowledge graph generation")
            self.quick_mode = True
        else:
            logger.info("🔍 FULL MODE: Complete processing with detailed knowledge graph")
            self.quick_mode = False
        
        # Step 1: Fetch emails
        emails = self.fetch_all_emails(start_date, end_date)
        
        if not emails:
            logger.info("No emails found in the specified date range")
            return
            
        # Step 2: Process emails through AI agents
        logger.info("Processing emails through AI agents...")
        processed_emails = []
        
        estimated_calls = len(emails) * 1  # MEGA OPTIMIZED: Only 1 call per email!
        logger.info(f"🚀 MEGA OPTIMIZED: Only ~{estimated_calls} API calls total (1 comprehensive call per email)!")
        
        for i, email in enumerate(emails):
            logger.info(f"Processing email {i+1}/{len(emails)}: {email.subject}")
            processed_email = self.process_email(email)
            processed_emails.append(processed_email)
            
            # Save to database
            self.save_email_to_db(processed_email)
            
            # Add delay between emails to help with rate limiting
            time.sleep(0.5)
            
        # Step 3: GraphRAG data is automatically saved during processing
        mode_text = "basic" if quick_mode else "detailed"
        logger.info(f"GraphRAG knowledge base updated ({mode_text} mode)...")
        
        # Step 4: Summary
        important_emails = [e for e in processed_emails if e.is_important]
        total_action_items = sum(len(e.action_items) for e in processed_emails)
        
        logger.info(f"Pipeline complete!")
        logger.info(f"  - Total emails processed: {len(processed_emails)}")
        logger.info(f"  - Important emails: {len(important_emails)}")
        logger.info(f"  - Total action items identified: {total_action_items}")
        logger.info(f"  - Processing mode: {'Quick (basic knowledge graph)' if quick_mode else 'Full (detailed knowledge graph)'}")
        
        return processed_emails

def main():
    """Main function to run the Email AI Assistant"""
    # Initialize the assistant
    assistant = EmailAIAssistant()
    
    print("🚀 Email AI Assistant")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Process emails from date range")
        print("2. View action items") 
        print("3. Review uncertain emails")
        print("4. Chat with assistant")
        print("5. Check system status")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            # Get date range from user
            print("\nEnter date range for email processing:")
            start_str = input("Start date (YYYY-MM-DD): ").strip()
            end_str = input("End date (YYYY-MM-DD): ").strip()
            
            try:
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
                
                assistant.run_full_pipeline(start_date, end_date)
                
            except ValueError:
                print("❌ Invalid date format. Please use YYYY-MM-DD")
                
        elif choice == '2':
            action_items = assistant.get_action_items()
            if action_items:
                print(f"\n📋 You have {len(action_items)} pending action items:")
                for item in action_items:
                    print(f"  [{item['id']}] {item['description']}")
                    print(f"      From: {item['email_subject']}")
                    print(f"      Priority: {item['priority']} | Created: {item['created_date'][:10]}")
            else:
                print("\n✅ No pending action items!")
                
        elif choice == '3':
            # Review uncertain emails
            uncertain_emails = assistant.get_uncertain_emails()
            if uncertain_emails:
                print(f"\n🤔 You have {len(uncertain_emails)} uncertain emails to review:")
                for i, email in enumerate(uncertain_emails[:5], 1):  # Show first 5
                    print(f"\n  {i}. [{email['id']}] {email['subject']}")
                    print(f"     From: {email['from_email']}")
                    print(f"     Reason: {email['classification_reason']}")
                    print(f"     Date: {email['date'][:10]}")
                
                print(f"\n💡 Use chat interface to provide feedback on these emails for training.")
            else:
                print("\n✅ No uncertain emails pending review!")
            
        elif choice == '4':
            assistant.chat_interface()
            
        elif choice == '5':
            assistant.check_system_status()
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid option. Please select 1-6.")

if __name__ == "__main__":
    main() 