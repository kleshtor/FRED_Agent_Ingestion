#!/usr/bin/env python3
"""
Simple test script for the FRED Query Agent
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the src directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
        print("Please create a .env file in the project root with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=your_key_here")
        return False
    
    print("✅ Environment variables are properly configured")
    return True

async def test_query_agent():
    """Test the query agent with sample queries"""
    
    if not check_environment():
        return
    
    try:
        from helper_functions.query_agent import QueryAgent
        
        print("🚀 Initializing FRED Query Agent...")
        agent = QueryAgent()
        print("✅ Agent initialized successfully!")
        
        # Test queries
        test_queries = [
            "Show me GDP data for the US"
        ]
        
        print("\n" + "="*60)
        print("🧪 TESTING QUERY AGENT")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            print("-" * 40)
            
            try:
                response = await agent.run_query(query)
                print(f"🤖 Agent Response:\n{response}")
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            print("-" * 40)
        
        print("\n✅ Query agent testing completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function"""
    print("🧪 FRED Query Agent Test Suite")
    print("="*60)
    
    asyncio.run(test_query_agent())

if __name__ == "__main__":
    main() 