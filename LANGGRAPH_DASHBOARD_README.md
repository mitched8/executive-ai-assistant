# 🚀 LangGraph Enhanced Dashboard

A comprehensive web interface that brings all LangGraph CLI functionality into a modern, user-friendly dashboard.

## ✨ Features Overview

### 📊 **Overview Tab**
- **Real-time metrics**: Total emails, important emails, unread count, action items
- **System status monitoring**: Database, OpenAI API, Gmail API, ChromaDB health
- **Processing performance**: Last run statistics, processing times, success rates
- **Recent activity feed**: Latest emails, actions, and system events

### 🔄 **Workflows Tab**
- **LangGraph workflow management**: Visual status of all 4 graph workflows
  - Main Email Processing Graph
  - Cron Email Ingestion Graph  
  - General Reflection Graph
  - Multi-Aspect Reflection Graph
- **Manual workflow triggers**: Start/stop workflows with custom parameters
- **Workflow history**: Track recent runs, durations, and success rates
- **Parameter configuration**: Batch sizes, confidence thresholds, learning rates

### ✅ **Approval Queue Tab**
- **Human-in-the-loop interface**: Review uncertain emails requiring approval
- **Detailed email analysis**: Full content, AI reasoning, suggested actions
- **Decision interface**: Approve, reject, or modify AI decisions
- **Feedback collection**: Capture reasoning for AI learning
- **Batch operations**: Handle multiple emails efficiently

### 🧠 **AI Learning Tab**
- **Learning statistics**: Track AI improvement over time
- **Reflection management**: Trigger learning from user feedback
- **Prompt engineering**: Edit and manage system prompts
- **Learning categories**: Tone & style, content, scheduling, background knowledge
- **Performance tracking**: Monitor accuracy improvements

### 📈 **Analytics Tab**
- **Processing metrics**: Volume trends, classification distribution
- **Performance analytics**: Processing times, confidence trends
- **Interactive charts**: Time series, pie charts, trend analysis
- **Customizable time ranges**: 24 hours to 90 days
- **Export capabilities**: Download data and reports

### ⚙️ **Management Tab**
- **Database management**: Table statistics, optimization, backups
- **System configuration**: API settings, rate limits, model parameters
- **LangGraph server integration**: Connection testing and monitoring
- **System logs**: Real-time log viewing with filtering
- **Health diagnostics**: Comprehensive system status checks

## 🎛️ **Sidebar Controls**

### 📊 **System Status**
Real-time health monitoring for all system components:
- ✅/❌ Database connection
- ✅/❌ OpenAI API status  
- ✅/❌ Gmail API connectivity
- ✅/❌ ChromaDB availability

### 🔄 **Workflow Controls**
One-click workflow management:
- **📥 Ingest Emails**: Fetch new emails from Gmail
- **🔄 Process Queue**: Run main email processing workflow
- **🧠 Trigger Learning**: Start AI reflection and improvement
- **⏰ Cron Management**: Enable/disable automated processing

### ⚙️ **Configuration**
Customizable system parameters:
- **Email Lookback**: Time window for processing (10-1440 minutes)
- **Auto-approval**: Enable automatic approval for high-confidence emails
- **AI Learning**: Toggle reflection and learning features

### ⚡ **Quick Actions**
Instant system operations:
- **🔍 Refresh Data**: Update all dashboard data
- **🧹 Clear Cache**: Reset session cache
- **📊 Export Data**: Download dashboard data as JSON

## 🚀 **Getting Started**

### **Option 1: Simple Launch**
```bash
python run_langgraph_dashboard.py
```

### **Option 2: Direct Streamlit**
```bash
streamlit run dashboard_langgraph.py --server.port 8502
```

### **Option 3: Background Service**
```bash
nohup python run_langgraph_dashboard.py > dashboard.log 2>&1 &
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GMAIL_CREDENTIALS_PATH="./credentials.json"
export DATABASE_PATH="./email_assistant.db"
export CHROMA_DB_PATH="./chroma_db"
```

### **LangGraph Integration**
The dashboard can integrate with a running LangGraph server:

1. **Start LangGraph Server** (optional):
   ```bash
   langgraph up
   ```

2. **Configure Server URL** in Management tab:
   - Default: `http://127.0.0.1:2024`
   - Test connection using the "🔍 Test Connection" button

## 📊 **Dashboard vs LangGraph CLI**

### **✅ Fully Implemented Features**
| Feature | Dashboard | LangGraph CLI |
|---------|-----------|---------------|
| Email Processing Monitoring | ✅ | ✅ |
| Workflow Status Tracking | ✅ | ✅ |
| Human Approval Queue | ✅ | ✅ |
| AI Learning Interface | ✅ | ✅ |
| System Configuration | ✅ | ✅ |
| Analytics & Reporting | ✅ | ❌ |
| Database Management | ✅ | ❌ |

### **⚠️ Simplified Features**
| Feature | Dashboard Implementation | LangGraph CLI |
|---------|-------------------------|---------------|
| Workflow Orchestration | Manual triggers + status monitoring | Full async orchestration |
| State Management | Database-backed simulation | Persistent thread state |
| Human Interrupts | Approval queue with polling | Native interrupt/resume |
| Complex Routing | Step-by-step wizards | Conditional graph execution |

### **❌ Not Implemented**
- Real-time workflow interrupts (use polling instead)
- Complex conditional graph routing (simplified workflows)
- Automatic cron job scheduling (manual triggers available)

## 🎯 **Best Practices**

### **Workflow Management**
1. **Monitor system health** before starting workflows
2. **Process approval queue** regularly for optimal AI learning
3. **Use manual triggers** for controlled email processing
4. **Review analytics** to optimize processing parameters

### **AI Learning**
1. **Provide detailed feedback** when approving/rejecting emails
2. **Run learning sessions** after significant feedback collection
3. **Monitor learning statistics** to track improvement
4. **Customize prompts** based on your specific needs

### **Performance Optimization**
1. **Adjust batch sizes** based on system capacity
2. **Set appropriate confidence thresholds** for auto-approval
3. **Monitor processing times** and optimize parameters
4. **Regular database maintenance** for optimal performance

## 🔍 **Troubleshooting**

### **Common Issues**

**Dashboard won't start:**
```bash
# Check if port is in use
lsof -i :8502

# Kill existing process
kill -9 $(lsof -t -i:8502)

# Try different port
streamlit run dashboard_langgraph.py --server.port 8503
```

**Database errors:**
```bash
# Check database file exists
ls -la email_assistant.db

# Test database connection
sqlite3 email_assistant.db ".tables"

# Recreate if corrupted
python -c "from eaia.main.email_processor import EmailProcessor; EmailProcessor().setup_database()"
```

**Missing data:**
```bash
# Run integration test
python test_integration.py

# Process some emails
python scripts/run_ingest.py --minutes-since 60
```

## 📈 **Monitoring & Maintenance**

### **Health Checks**
- Monitor system status indicators in sidebar
- Check processing performance metrics regularly
- Review error logs in Management tab
- Test API connections periodically

### **Data Management**
- **Regular backups**: Use "💾 Backup Database" in Management tab
- **Database optimization**: Run "📊 Optimize Database" weekly
- **Log rotation**: Monitor log files and rotate as needed
- **Performance tuning**: Adjust parameters based on analytics

## 🤝 **Integration with Existing System**

The dashboard seamlessly integrates with your existing email processing system:

- **Uses same database**: `email_assistant.db`
- **Same configuration**: `eaia/main/config.yaml`
- **Compatible with CLI tools**: Can run alongside existing scripts
- **Shared ChromaDB**: Same vector database for consistency

## 🎉 **What's New**

This enhanced dashboard provides significant improvements over the basic dashboard:

### **🆕 New Features**
- **LangGraph workflow visualization and control**
- **Human approval queue with detailed review interface**
- **AI learning and reflection management**
- **Comprehensive analytics with interactive charts**
- **Advanced system management and configuration**
- **Real-time health monitoring**

### **🔧 Enhanced Functionality**
- **Modern, responsive UI with custom CSS**
- **Comprehensive error handling and logging**
- **Session state management for better UX**
- **Parallel data loading for improved performance**
- **Export capabilities for data analysis**

### **📊 Better Analytics**
- **Processing performance trends**
- **AI confidence tracking**
- **Email classification insights**
- **System usage patterns**

## 🔮 **Future Enhancements**

Potential future improvements:
- **Real-time WebSocket integration** for live updates
- **LangGraph server auto-deployment** and management
- **Advanced workflow builder** with drag-and-drop interface
- **Multi-user support** with role-based access
- **API endpoints** for external integrations
- **Mobile-responsive design** for smartphone access

---

**🚀 Ready to revolutionize your email management? Launch the dashboard and experience the power of LangGraph in a beautiful, user-friendly interface!** 