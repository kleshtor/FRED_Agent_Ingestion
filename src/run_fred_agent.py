#!/usr/bin/env python3
"""
Script to run the original FRED agent to populate the database with time series data
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Load environment variables
load_dotenv()

def main():
    """Run the FRED agent to populate database"""
    try:
        from helper_functions.FRED_AGENT_SDK import FredAgent
        
        print("🚀 Running FRED Agent to populate database...")
        
        # Initialize and run the agent
        config_path = os.path.join("helper_functions", "FRED.yaml")
        agent = FredAgent(config_path=config_path)
        agent.run()
        
        print("✅ FRED Agent completed successfully!")
        print("Database and Excel files are now populated with time series data.")
        
    except Exception as e:
        print(f"❌ Error running FRED agent: {e}")

if __name__ == "__main__":
    main() 