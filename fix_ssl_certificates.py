#!/usr/bin/env python3
"""
SSL Certificate Fix Script for macOS
Helps resolve SSL certificate verification issues
"""

import os
import sys
import subprocess
import ssl
import urllib.request

def check_ssl_certificates():
    """Check and fix SSL certificate issues"""
    print("🔒 SSL Certificate Diagnostic and Fix")
    print("=" * 50)
    
    # 1. Check current SSL context
    print("📋 Current SSL Configuration:")
    try:
        context = ssl.create_default_context()
        print(f"✅ Default SSL context created")
        print(f"   CA certs file: {context.get_ca_certs()[:2] if context.get_ca_certs() else 'None'}")
    except Exception as e:
        print(f"❌ SSL context error: {e}")
    
    # 2. Check Python version and path
    print(f"\n🐍 Python Information:")
    print(f"   Version: {sys.version}")
    print(f"   Executable: {sys.executable}")
    
    # 3. Check if we're on macOS
    if sys.platform != "darwin":
        print("⚠️  This script is designed for macOS. Your OS may have different SSL setup.")
        return False
    
    print("\n🍎 macOS SSL Certificate Fixes:")
    
    # Fix 1: Install certificates using Python's built-in script
    print("\n1️⃣ Running Python certificate installer...")
    try:
        # This is the standard way to install certificates on macOS Python
        cert_script = "/Applications/Python*/Install Certificates.command"
        import glob
        cert_scripts = glob.glob(cert_script)
        if cert_scripts:
            print(f"   Found certificate installer: {cert_scripts[0]}")
            print("   🔧 Please run this command in Terminal:")
            print(f"   {cert_scripts[0]}")
        else:
            # Alternative method
            print("   Certificate installer not found, trying alternative...")
            try:
                import certifi
                print(f"   ✅ Certifi package found: {certifi.where()}")
            except ImportError:
                print("   Installing certifi package...")
                subprocess.run([sys.executable, "-m", "pip", "install", "certifi"], check=True)
                import certifi
                print(f"   ✅ Certifi installed: {certifi.where()}")
    except Exception as e:
        print(f"   ❌ Error with certificate installer: {e}")
    
    # Fix 2: Update system certificates
    print("\n2️⃣ System certificate update:")
    print("   🔧 Run these commands in Terminal:")
    print("   brew update && brew upgrade ca-certificates")
    print("   # OR if no Homebrew:")
    print("   /Applications/Python*/Install\\ Certificates.command")
    
    # Fix 3: Test with certificates
    print("\n3️⃣ Testing with certificate bundle...")
    try:
        import certifi
        import ssl
        
        # Create SSL context with certifi certificates
        context = ssl.create_default_context(cafile=certifi.where())
        
        # Test connection
        req = urllib.request.Request('https://api.openai.com')
        response = urllib.request.urlopen(req, context=context, timeout=10)
        print(f"   ✅ Connection successful with certifi! Status: {response.getcode()}")
        return True
        
    except Exception as e:
        print(f"   ❌ Still failing with certifi: {e}")
    
    # Fix 4: Temporary workaround (not recommended for production)
    print("\n4️⃣ Temporary SSL bypass (FOR TESTING ONLY):")
    print("   ⚠️  This is NOT secure and should only be used for testing!")
    
    return False

def test_openai_with_ssl_fix():
    """Test OpenAI connection with SSL certificate fixes"""
    print("\n🤖 Testing OpenAI with SSL fixes...")
    
    try:
        import certifi
        import ssl
        import openai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not api_key:
            print("❌ No API key found")
            return False
        
        # Create OpenAI client with proper SSL context
        client = openai.OpenAI(
            api_key=api_key,
            timeout=20.0
        )
        
        # Test API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say: SSL_FIX_SUCCESS"}],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI API call successful: '{result}'")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False

if __name__ == "__main__":
    print("SSL Certificate Fix for OpenAI Connection")
    print()
    
    # Step 1: Diagnose SSL issues
    ssl_fixed = check_ssl_certificates()
    
    print("\n" + "="*50)
    print("🛠️  RECOMMENDED FIXES:")
    print("="*50)
    
    print("\n1. Install Python certificates (MOST IMPORTANT):")
    print("   Open Terminal and run:")
    print("   /Applications/Python*/Install\\ Certificates.command")
    print("   (Replace * with your Python version)")
    
    print("\n2. Alternative - Install certifi:")
    print("   pip install --upgrade certifi")
    
    print("\n3. If using Homebrew:")
    print("   brew update && brew upgrade ca-certificates")
    
    print("\n4. Update macOS:")
    print("   System Preferences > Software Update")
    
    print("\n5. After fixes, test again:")
    print("   python test_openai_connection.py")
    
    # Step 2: Test OpenAI if SSL seems fixed
    if ssl_fixed:
        test_openai_with_ssl_fix()
    
    print("\n💡 If issues persist, you may need to:")
    print("   - Restart your terminal/IDE")
    print("   - Recreate your virtual environment")
    print("   - Check corporate firewall settings") 