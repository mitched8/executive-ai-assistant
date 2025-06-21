#!/usr/bin/env python3
"""
Comprehensive Test Suite for Email AI Assistant
Tests all components of the email processing pipeline
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Force reload environment
load_dotenv(override=True)

from email_ai_assistant import EmailAIAssistant, EmailData, AgentState

def create_test_email():
    """Create a test email for testing"""
    return EmailData(
        id="test_email_001",
        thread_id="test_thread_001",
        subject="Important: Project Alpha Deadline Approaching",
        from_email="john.smith@company.com",
        to_email="daniel@example.com",
        date=datetime.now(),
        body="""
        Hi Daniel,
        
        I hope this email finds you well. I wanted to remind you that the Project Alpha 
        deadline is coming up next Friday, June 21st. We need to complete the following tasks:
        
        1. Finalize the user interface design
        2. Complete the database integration 
        3. Write unit tests for all components
        4. Prepare the deployment documentation
        
        Could you please confirm that you'll be able to finish the UI design by Wednesday?
        Also, we should schedule a meeting with Sarah Johnson from the QA team to review
        the testing requirements.
        
        Thanks,
        John Smith
        Project Manager
        Company XYZ
        """,
        labels=["INBOX", "IMPORTANT"]
    )

def test_openai_connection(assistant):
    """Test basic OpenAI connection"""
    print("\n🔗 Testing OpenAI Connection")
    print("-" * 40)
    
    try:
        response = assistant.make_llm_call_with_retry("Say exactly: CONNECTION_TEST_OK")
        if "CONNECTION_TEST_OK" in response:
            print("✅ OpenAI connection working")
            return True
        else:
            print(f"❌ Unexpected response: {response}")
            return False
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_email_triage(assistant):
    """Test email triage functionality"""
    print("\n📋 Testing Email Triage")
    print("-" * 40)
    
    test_email = create_test_email()
    
    # Create initial state
    state = {
        "email": test_email,
        "analysis": {},
        "is_important": False,
        "summary": "",
        "entities": [],
        "action_items": [],
        "next_action": ""
    }
    
    try:
        result_state = assistant.triage_email(state)
        
        print(f"✅ Email triaged successfully")
        print(f"   Important: {result_state['is_important']}")
        print(f"   Summary: {result_state['summary'][:100]}...")
        
        return result_state['is_important'] and len(result_state['summary']) > 0
        
    except Exception as e:
        print(f"❌ Triage failed: {e}")
        return False

def test_entity_extraction(assistant):
    """Test entity extraction functionality"""
    print("\n🏷️  Testing Entity Extraction")
    print("-" * 40)
    
    test_email = create_test_email()
    
    state = {
        "email": test_email,
        "entities": []
    }
    
    try:
        result_state = assistant.extract_entities(state)
        entities = result_state['entities']
        
        print(f"✅ Entities extracted: {len(entities)} found")
        for i, entity in enumerate(entities[:5], 1):
            print(f"   {i}. {entity}")
        
        # Check if we found expected entities
        expected_entities = ["John Smith", "Project Alpha", "Company XYZ", "Sarah Johnson"]
        found_expected = any(expected in str(entities) for expected in expected_entities)
        
        return len(entities) > 0 and found_expected
        
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
        return False

def test_action_item_identification(assistant):
    """Test action item identification"""
    print("\n✅ Testing Action Item Identification")
    print("-" * 40)
    
    test_email = create_test_email()
    
    state = {
        "email": test_email,
        "action_items": []
    }
    
    try:
        result_state = assistant.identify_action_items(state)
        action_items = result_state['action_items']
        
        print(f"✅ Action items identified: {len(action_items)} found")
        for i, item in enumerate(action_items, 1):
            print(f"   {i}. {item}")
        
        # Check if we found action items related to the tasks
        action_text = " ".join(action_items).lower()
        has_relevant_actions = any(keyword in action_text 
                                 for keyword in ["ui", "design", "meeting", "confirm", "deadline"])
        
        return len(action_items) > 0 and has_relevant_actions
        
    except Exception as e:
        print(f"❌ Action item identification failed: {e}")
        return False

def test_full_email_processing(assistant):
    """Test full email processing pipeline"""
    print("\n🔄 Testing Full Email Processing Pipeline")
    print("-" * 40)
    
    test_email = create_test_email()
    
    try:
        processed_email = assistant.process_email(test_email)
        
        print(f"✅ Email processed successfully")
        print(f"   Subject: {processed_email.subject}")
        print(f"   Important: {processed_email.is_important}")
        print(f"   Summary: {processed_email.summary[:100]}...")
        print(f"   Entities: {len(processed_email.entities)} found")
        print(f"   Action Items: {len(processed_email.action_items)} found")
        
        # Validate results
        success = (
            len(processed_email.summary) > 0 and
            len(processed_email.entities) > 0 and
            processed_email.is_important is not None
        )
        
        return success
        
    except Exception as e:
        print(f"❌ Full email processing failed: {e}")
        return False

def test_database_operations(assistant):
    """Test database save and retrieve operations"""
    print("\n💾 Testing Database Operations")
    print("-" * 40)
    
    test_email = create_test_email()
    test_email.is_important = True
    test_email.summary = "Test summary for database"
    test_email.entities = ["Test Entity 1", "Test Entity 2"]
    test_email.action_items = ["Test action item"]
    
    try:
        # Save to database
        assistant.save_email_to_db(test_email)
        print("✅ Email saved to database")
        
        # Try to retrieve action items
        action_items = assistant.get_action_items()
        print(f"✅ Retrieved {len(action_items)} action items from database")
        
        return True
        
    except Exception as e:
        print(f"❌ Database operations failed: {e}")
        return False

def test_vector_search(assistant):
    """Test vector search functionality"""
    print("\n🔍 Testing Vector Search")
    print("-" * 40)
    
    try:
        # Test email search
        results = assistant.vector_search_emails("project deadline", n_results=3)
        print(f"✅ Email vector search returned {len(results)} results")
        
        # Test comprehensive search
        search_results = assistant.graphrag_search("project management", n_results=5)
        print(f"✅ GraphRAG search returned results for: {list(search_results.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return False

def run_comprehensive_tests():
    """Run all tests"""
    print("🧪 COMPREHENSIVE EMAIL AI ASSISTANT TEST SUITE")
    print("=" * 60)
    
    try:
        # Initialize assistant
        print("📝 Initializing EmailAIAssistant...")
        assistant = EmailAIAssistant()
        print("✅ Assistant initialized successfully")
        
        # Run all tests
        tests = [
            ("OpenAI Connection", test_openai_connection),
            ("Email Triage", test_email_triage),
            ("Entity Extraction", test_entity_extraction),
            ("Action Item Identification", test_action_item_identification),
            ("Full Email Processing", test_full_email_processing),
            ("Database Operations", test_database_operations),
            ("Vector Search", test_vector_search),
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func(assistant)
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name}")
        
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Your Email AI Assistant is fully functional!")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Check the errors above.")
            
        return passed == total
        
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {e}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1) 