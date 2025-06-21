#!/usr/bin/env python3
"""
Standalone OpenAI Connection Test Script
Run this to diagnose OpenAI API connection issues
"""

import os
import sys
from dotenv import load_dotenv

def test_openai_connection():
    """Test OpenAI connection step by step"""
    
    # Load environment variables
    load_dotenv()
    
    print("🔍 OpenAI Connection Diagnostic Test")
    print("=" * 50)
    
    # 1. Check API Key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    # Show masked key
    if len(api_key) > 12:
        masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    else:
        masked_key = "***"
    print(f"✅ API Key found: {masked_key}")
    print(f"   Key length: {len(api_key)} characters")
    
    # 2. Test internet connectivity
    print("\n🌐 Testing internet connectivity...")
    try:
        import urllib.request
        response = urllib.request.urlopen('https://api.openai.com', timeout=10)
        print(f"✅ Can reach OpenAI API (Status: {response.getcode()})")
    except Exception as e:
        error_str = str(e).lower()
        if "certificate verify failed" in error_str or "ssl" in error_str:
            print(f"❌ SSL Certificate Error: {e}")
            print("   This is a common macOS issue!")
            print("   🔧 QUICK FIX: Run this command in Terminal:")
            print("   /Applications/Python*/Install\\ Certificates.command")
            print("   (Replace * with your Python version, e.g., 3.11)")
            print("   ")
            print("   Or run: python fix_ssl_certificates.py")
            return False
        else:
            print(f"❌ Cannot reach OpenAI API: {e}")
            print("   Check your internet connection")
            return False
    
    # 3. Test OpenAI package
    print("\n📦 Testing OpenAI package...")
    try:
        import openai
        print(f"✅ OpenAI package imported (version: {openai.__version__})")
    except ImportError as e:
        print(f"❌ Cannot import OpenAI: {e}")
        print("   Run: pip install openai")
        return False
    
    # 4. Test direct API call
    print("\n🤖 Testing direct OpenAI API call...")
    try:
        client = openai.OpenAI(
            api_key=api_key,
            timeout=20.0
        )
        
        # Simple test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say exactly: CONNECTION_TEST_OK"}],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ API call successful!")
        print(f"   Response: '{result}'")
        print(f"   Usage: {response.usage}")
        
    except openai.AuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("   Your API key is invalid, expired, or malformed")
        return False
    except openai.RateLimitError as e:
        print(f"❌ Rate limit exceeded: {e}")
        print("   You've exceeded your usage quota or rate limits")
        return False
    except openai.APIConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("   Network issue or API is down")
        return False
    except openai.BadRequestError as e:
        print(f"❌ Bad request: {e}")
        print("   Request format issue")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test LangChain wrapper
    print("\n🔗 Testing LangChain OpenAI wrapper...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        
        test_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            request_timeout=20,
            max_retries=1,
            api_key=api_key
        )
        
        response = test_llm.invoke([HumanMessage(content="Say: LANGCHAIN_TEST_OK")])
        result = response.content.strip()
        print(f"✅ LangChain wrapper works!")
        print(f"   Response: '{result}'")
        
    except Exception as e:
        print(f"❌ LangChain wrapper failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All connection tests passed!")
    print("   Your OpenAI setup is working correctly.")
    return True

if __name__ == "__main__":
    print("OpenAI Connection Test")
    print("Make sure you have a .env file with OPENAI_API_KEY=your_key")
    print()
    
    success = test_openai_connection()
    
    if success:
        print("\n✅ Connection test successful!")
        print("   You can now run the email assistant.")
    else:
        print("\n❌ Connection test failed!")
        print("   Fix the issues above before running the email assistant.")
        sys.exit(1) 