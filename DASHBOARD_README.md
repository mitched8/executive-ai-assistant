# 📧 Email AI Assistant Dashboard

A comprehensive Streamlit-based dashboard for managing your intelligent email processing system.

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
python run_dashboard.py
```

### Option 2: Direct Streamlit Launch
```bash
pip install -r dashboard_requirements.txt
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 📱 Dashboard Features

### 🏠 Dashboard Overview
- **Key Metrics**: Total emails, important emails, entities, action items
- **Trends**: Email processing over time with interactive charts
- **Quick Actions**: One-click navigation to main features

### ⚡ Process Emails
- **Date Range Selection**: Choose specific periods to process
- **Processing Options**: Quick mode for faster processing
- **Real-time Status**: Live updates during email processing
- **Results Summary**: Immediate feedback on processing results

### 📊 Email Analysis
- **Smart Filtering**: Filter by importance, date range, search terms
- **Detailed View**: Expandable email cards with full analysis
- **Entity Display**: See extracted entities and their types
- **Action Items**: View action items directly from emails

### 📋 Action Items Management
- **Status Tracking**: Pending, in-progress, completed actions
- **Priority Management**: High, medium, low priority filtering
- **Interactive Updates**: Change status directly in the dashboard
- **Email Context**: See which email generated each action

### 🧠 Knowledge Graph Explorer
- **Entity Browser**: Explore all extracted entities
- **Relationship Mapping**: Visualize connections between entities
- **Search Functionality**: Find specific people, organizations, topics
- **Analytics**: Entity type distribution and connection analysis

### ⚙️ Settings
- **System Status**: Check API connections and database health
- **Configuration**: Adjust processing parameters
- **Database Management**: Clear data, export/import options

## 🎯 Key Benefits

- **Single Interface**: Manage all email AI functions in one place
- **Real-time Processing**: Start email processing with one click
- **Visual Analytics**: Interactive charts and graphs
- **Action Management**: Track and update tasks from emails
- **Knowledge Discovery**: Explore relationships and insights
- **Mobile Friendly**: Responsive design works on all devices

## 🔧 Technical Details

- **Framework**: Streamlit for rapid web app development
- **Database**: SQLite for local data storage
- **Visualization**: Plotly for interactive charts
- **Processing**: Integrated with your existing EmailAIAssistant
- **Real-time**: Background processing with status updates

## 📊 Dashboard Sections

### Navigation
The sidebar provides easy navigation between all dashboard sections:
- Dashboard Overview (home page with metrics)
- Process Emails (trigger email processing)
- Email Analysis (browse and search processed emails)
- Action Items (manage tasks and to-dos)
- Knowledge Graph (explore entities and relationships)
- Settings (configuration and system status)

### Data Flow
1. **Process**: Use "Process Emails" to analyze new messages
2. **Analyze**: Review results in "Email Analysis"
3. **Act**: Manage tasks in "Action Items"
4. **Explore**: Discover insights in "Knowledge Graph"
5. **Configure**: Adjust settings as needed

## 🚨 Troubleshooting

### Dashboard Won't Start
- Ensure all requirements are installed: `pip install -r dashboard_requirements.txt`
- Check Python version (3.8+ recommended)
- Verify database file exists: `email_assistant.db`

### No Data Showing
- Run email processing first using the "Process Emails" page
- Check your Gmail API credentials are configured
- Verify OpenAI API key is set in `.env` file

### Performance Issues
- Reduce date ranges for large email volumes
- Use "Quick Mode" for faster processing
- Clear old data if database becomes too large

## 💡 Tips

- **Start Small**: Process a few days of emails first to test
- **Use Filters**: Leverage filtering options to find specific emails
- **Regular Processing**: Set up regular processing schedules
- **Action Management**: Keep action items updated for best results
- **Explore Relationships**: Use the knowledge graph to discover insights

## 🔒 Privacy & Security

- All data is stored locally in your SQLite database
- No data is sent to external services except OpenAI for processing
- Gmail access is read-only for email processing
- You can clear all data anytime from the Settings page

---

**Happy Email Processing!** 🎉

For issues or questions, check the main project README or create an issue. 