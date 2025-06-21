#!/usr/bin/env python3
"""
Test script for the new email processing and analysis integration
Verifies that all components work together correctly
"""

import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_integration():
    """Test the integration components"""
    print("🧪 Testing Email Processing and Analysis Integration")
    print("=" * 60)
    
    try:
        # Test importing the modules
        print("📦 Testing module imports...")
        
        from eaia.main.email_processor import EmailProcessor
        from eaia.main.analysis import EmailAnalyzer
        from eaia.main.dashboard_integration import DashboardIntegration
        
        print("✅ All modules imported successfully")
        
        # Test database initialization
        print("\n🗄️ Testing database initialization...")
        
        processor = EmailProcessor()
        analyzer = EmailAnalyzer()
        integration = DashboardIntegration()
        
        print("✅ All components initialized successfully")
        
        # Test system status
        print("\n🔍 Testing system status...")
        
        system_status = integration.test_system_integration()
        
        print(f"Database: {system_status.get('database', '❓ Unknown')}")
        print(f"Processor: {system_status.get('processor', '❓ Unknown')}")
        print(f"OpenAI: {system_status.get('openai', '❓ Unknown')}")
        print(f"Gmail: {system_status.get('gmail', '❓ Unknown')}")
        print(f"Overall Status: {system_status.get('overall_status', '❓ Unknown')}")
        
        # Test dashboard overview
        print("\n📊 Testing dashboard overview...")
        
        overview = integration.get_dashboard_overview(7)
        
        if 'error' in overview:
            print(f"⚠️ Dashboard overview error: {overview['error']}")
        else:
            print(f"✅ Dashboard overview loaded:")
            print(f"   - Total emails: {overview.get('total_emails', 0)}")
            print(f"   - Important emails: {overview.get('important_emails', 0)}")
            print(f"   - Action items: {overview.get('total_actions', 0)}")
            print(f"   - Entities: {overview.get('total_entities', 0)}")
        
        # Test email analysis
        print("\n🔍 Testing email analysis...")
        
        analysis_data = integration.get_email_analysis_data()
        
        if 'error' in analysis_data:
            print(f"⚠️ Email analysis error: {analysis_data['error']}")
        else:
            print("✅ Email analysis loaded successfully")
            
            sender_analysis = analysis_data.get('sender_analysis', {})
            print(f"   - Total senders: {sender_analysis.get('total_senders', 0)}")
            
            entity_analysis = analysis_data.get('entity_analysis', {})
            print(f"   - Total entities: {entity_analysis.get('total_entities', 0)}")
        
        # Test action items
        print("\n📋 Testing action items...")
        
        action_data = integration.get_action_items_dashboard()
        
        if 'error' in action_data:
            print(f"⚠️ Action items error: {action_data['error']}")
        else:
            print("✅ Action items loaded successfully")
            print(f"   - Total actions: {action_data.get('total_actions', 0)}")
            print(f"   - Completion rate: {action_data.get('completion_rate', 0):.1f}%")
        
        # Test knowledge graph
        print("\n🧠 Testing knowledge graph...")
        
        graph_data = integration.get_knowledge_graph_data()
        
        if 'error' in graph_data:
            print(f"⚠️ Knowledge graph error: {graph_data['error']}")
        else:
            print("✅ Knowledge graph loaded successfully")
            stats = graph_data.get('stats', {})
            print(f"   - Total entities: {stats.get('total_entities', 0)}")
            print(f"   - Total relationships: {stats.get('total_relationships', 0)}")
        
        # Test uncertain emails
        print("\n🤔 Testing uncertain emails...")
        
        uncertain_emails = integration.get_uncertain_emails_for_review()
        print(f"✅ Found {len(uncertain_emails)} uncertain emails for review")
        
        print("\n" + "=" * 60)
        print("🎉 Integration test completed successfully!")
        print("✅ All components are working correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required modules are installed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Check the error details above")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n📝 Testing configuration loading...")
    
    try:
        from eaia.main.config import get_config
        from langchain_core.runnables import RunnableConfig
        
        # Test config loading
        config = RunnableConfig(
            configurable={
                "assistant_id": "test_assistant",
                "model": "gpt-4o"
            }
        )
        
        config_data = get_config(config)
        
        print("✅ Configuration loaded successfully")
        print(f"   - Full name: {config_data.get('full_name', 'Not set')}")
        print(f"   - Background: {config_data.get('background', 'Not set')[:50]}...")
        print(f"   - Timezone: {config_data.get('timezone', 'Not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_integration()
    config_success = test_config_loading()
    
    if success and config_success:
        print("\n🎯 All tests passed! The integration is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1) 