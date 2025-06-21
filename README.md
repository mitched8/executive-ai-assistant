# Email AI Assistant

A comprehensive system for processing emails with AI agents and building a knowledge graph for intelligent task management.

## Features

1. **Gmail Integration**: Connects to Gmail API to fetch emails between specified dates
2. **AI-Powered Email Triage**: Uses GPT-4o to analyze emails and determine importance
3. **GraphRAG Knowledge Base**: Vector-based knowledge graph using ChromaDB for semantic search
4. **Action Item Detection**: Automatically identifies tasks and action items from emails
5. **Interactive Chat Interface**: Conversational AI for managing tasks and searching emails
6. **Batch Processing**: Handles large volumes of emails in batches of 100

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Gmail API Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API
4. Create credentials (OAuth 2.0 Client ID)
5. Download the credentials JSON file and save it as `credentials.json` in the project directory

### 4. Run the Application

```bash
python email_ai_assistant.py
```

## Usage

### Main Menu Options

1. **Process emails from date range**: Fetch and analyze emails between two dates
2. **View action items**: See all pending tasks identified from emails
3. **Chat with assistant**: Interactive conversation about your emails and tasks
4. **Exit**: Close the application

### Chat Commands

- `/actions` - View all pending action items
- `/search <query>` - Search the knowledge graph for relevant information
- `/add <task>` - Add a new action item manually
- `/complete <id>` - Mark an action item as completed
- `/quit` - Exit the chat interface

## System Architecture

### Components

1. **EmailAIAssistant**: Main orchestrator class
2. **LangGraph Workflow**: AI agent pipeline with four stages:
   - **Triage**: Determine email importance and generate summary
   - **Extract Entities**: Identify people, companies, dates, projects
   - **Identify Actions**: Find actionable tasks and deadlines
   - **Update Knowledge Graph**: Store relationships and information

3. **SQLite Database**: Persistent storage for:
   - Email metadata and content
   - Action items with status tracking
   - Graph entities and relationships metadata

4. **ChromaDB Vector Database**: Semantic search and storage for:
   - Email content embeddings
   - Entity embeddings with descriptions
   - Relationship embeddings for graph traversal

### Data Flow

```
Gmail API → Email Fetching → AI Processing → GraphRAG (ChromaDB + SQLite) → Vector Search
                                ↓
                         Action Item Detection → Task Management
```

## Database Schema

### Tables

- **emails**: Email metadata, content, and AI analysis results
- **action_items**: Tasks with priority, status, and due dates
- **graph_entities**: Extracted entities with types and descriptions
- **graph_relationships**: Relationships between entities with strength scores

### ChromaDB Collections

- **emails**: Vector embeddings of email content for semantic search
- **entities**: Vector embeddings of entities with contextual descriptions
- **relationships**: Vector embeddings of entity relationships

## AI Processing Pipeline

Each email goes through a 4-stage LangGraph workflow:

1. **Triage Agent**: 
   - Analyzes email importance
   - Generates concise summary
   - Uses GPT-4o for decision making

2. **Entity Extraction Agent**:
   - Identifies people, companies, dates
   - Extracts project names and keywords
   - Creates structured entity list

3. **Action Item Agent**:
   - Finds tasks and deadlines
   - Identifies follow-up requirements
   - Detects meeting scheduling needs

4. **GraphRAG Agent**:
   - Stores emails as vector embeddings
   - Creates entity embeddings with AI-generated descriptions
   - Establishes semantic relationships between entities
   - Updates vector database for semantic search

## GraphRAG Structure

### Entity Types
- **PERSON**: Individuals mentioned in emails
- **COMPANY**: Organizations and businesses
- **PROJECT**: Work projects and initiatives
- **LOCATION**: Places and addresses
- **DATE**: Important dates and deadlines
- **PRODUCT**: Products and services
- **CONCEPT**: Abstract concepts and topics
- **OTHER**: Miscellaneous entities

### Relationship Types
- **WORKS_WITH**: Professional collaboration
- **PART_OF**: Hierarchical relationships
- **RELATED_TO**: General associations
- **MENTIONS**: Simple mentions in context
- **COLLABORATES**: Active collaboration
- **OTHER**: Miscellaneous relationships

## Customization

### Modifying AI Prompts

Edit the prompt templates in the agent methods:
- `triage_email()`: Email importance analysis
- `extract_entities()`: Entity extraction logic
- `identify_action_items()`: Action item detection

### Adding New Node Types

Extend the knowledge graph by:
1. Adding new node types in `update_knowledge_graph()`
2. Creating new relationship types
3. Updating database schema if needed

### Batch Size Configuration

Modify `BATCH_SIZE` constant to change email processing batch size (default: 100).

## Error Handling

The system includes comprehensive error handling:
- Gmail API rate limiting and errors
- AI model failures with fallback responses
- Database connection issues
- Email parsing errors

## Performance Considerations

- Processes emails in batches to manage memory usage
- Uses SQLite for efficient local storage
- Implements pagination for large email volumes
- Caches knowledge graph in memory for fast access

## Security Notes

- OAuth 2.0 for secure Gmail access
- Local storage of credentials and data
- No email content sent to external services except OpenAI
- API keys stored in environment variables

## Troubleshooting

### Common Issues

1. **Gmail API Quota Exceeded**: Wait for quota reset or request increase
2. **OpenAI API Errors**: Check API key and billing status
3. **Database Locked**: Close other connections to SQLite file
4. **Memory Issues**: Reduce batch size for large email volumes

### Logs

The system uses Python logging to track:
- Email processing progress
- AI agent execution
- Database operations
- Error conditions

## Future Enhancements

- Email response generation
- Calendar integration for scheduling
- Advanced search with vector embeddings
- Multi-user support
- Web interface
- Integration with task management tools

## License

MIT License - see LICENSE file for details.
