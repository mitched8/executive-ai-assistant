#!/usr/bin/env python3
"""
Database Management Script
Simple utility to clear or manage the email assistant database
"""

import sqlite3
import os

def clear_all_data():
    """Clear all data from the database"""
    if not os.path.exists('email_assistant.db'):
        print("❌ Database file 'email_assistant.db' not found!")
        return
    
    try:
        conn = sqlite3.connect('email_assistant.db')
        cursor = conn.cursor()
        
        # Get current counts
        cursor.execute("SELECT COUNT(*) FROM emails")
        email_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM action_items")
        action_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM graph_entities")
        entity_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM graph_relationships")
        rel_count = cursor.fetchone()[0]
        
        print(f"📊 Current database contents:")
        print(f"   📧 Emails: {email_count}")
        print(f"   📋 Action Items: {action_count}")
        print(f"   🏷️  Entities: {entity_count}")
        print(f"   🔗 Relationships: {rel_count}")
        
        if email_count == 0 and action_count == 0 and entity_count == 0 and rel_count == 0:
            print("✅ Database is already empty!")
            return
        
        # Confirm deletion
        confirm = input("\n⚠️  Are you sure you want to delete ALL data? (type 'yes' to confirm): ")
        
        if confirm.lower() != 'yes':
            print("❌ Operation cancelled")
            return
        
        # Clear all tables
        print("🧹 Clearing database...")
        
        cursor.execute("DELETE FROM graph_relationships")
        print("   ✅ Cleared relationships")
        
        cursor.execute("DELETE FROM graph_entities")
        print("   ✅ Cleared entities")
        
        cursor.execute("DELETE FROM action_items")
        print("   ✅ Cleared action items")
        
        cursor.execute("DELETE FROM emails")
        print("   ✅ Cleared emails")
        
        # Also clear uncertain emails if table exists
        try:
            cursor.execute("DELETE FROM uncertain_emails")
            print("   ✅ Cleared uncertain emails")
        except:
            pass
        
        conn.commit()
        conn.close()
        
        print("\n🎉 All data cleared successfully!")
        print("💡 You can now process fresh emails")
        
    except Exception as e:
        print(f"❌ Error clearing database: {str(e)}")

def show_database_stats():
    """Show current database statistics"""
    if not os.path.exists('email_assistant.db'):
        print("❌ Database file 'email_assistant.db' not found!")
        return
    
    try:
        conn = sqlite3.connect('email_assistant.db')
        cursor = conn.cursor()
        
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM emails")
        email_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM emails WHERE is_important = 1")
        important_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM action_items")
        action_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM action_items WHERE status = 'pending'")
        pending_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM graph_entities")
        entity_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM graph_relationships")
        rel_count = cursor.fetchone()[0]
        
        print("📊 Database Statistics")
        print("=" * 30)
        print(f"📧 Total Emails: {email_count}")
        print(f"⭐ Important Emails: {important_count}")
        print(f"📋 Total Action Items: {action_count}")
        print(f"⏳ Pending Actions: {pending_count}")
        print(f"🏷️  Entities: {entity_count}")
        print(f"🔗 Relationships: {rel_count}")
        
        # Show recent emails
        cursor.execute("SELECT subject, from_email, date FROM emails ORDER BY date DESC LIMIT 5")
        recent_emails = cursor.fetchall()
        
        if recent_emails:
            print(f"\n📧 Recent Emails:")
            for subject, from_email, date in recent_emails:
                print(f"   • {subject[:40]}... from {from_email}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {str(e)}")

if __name__ == "__main__":
    print("🗄️  Email AI Assistant Database Manager")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Show database statistics")
        print("2. Clear all data")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            show_database_stats()
        elif choice == "2":
            clear_all_data()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.") 