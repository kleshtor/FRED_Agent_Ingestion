#!/usr/bin/env python3
"""
Script to check the current database table structure and parameters.
This will help us understand what's actually in the database.
"""

import os
import yaml
import sys
import psycopg2

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_config() -> dict:
    """Load configuration from config file."""
    current_dir = os.path.dirname(__file__)  # auxiliary_files directory
    config_path = os.path.join(current_dir, '..', 'helper_functions', 'config.yaml')
    config_path = os.path.abspath(config_path)  # Resolve to absolute path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_database_structure():
    """Check the current database table structure"""
    try:
        # Load configuration
        config = load_config()
        db_config = config.get("database", {})
        
        print("🔍 Checking database structure...")
        print(f"Host: {db_config.get('host', 'localhost')}")
        print(f"Database: {db_config.get('dbname', 'N/A')}")
        print(f"Port: {db_config.get('port', 5432)}")
        print(f"User: {db_config.get('user', 'N/A')}")
        
        # Connect to database
        conn = psycopg2.connect(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            dbname=db_config.get("dbname"),
            user=db_config.get("user"),
            password=db_config.get("password")
        )
        
        with conn.cursor() as cur:
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'indicator_embeddings'
                );
            """)
            table_exists = cur.fetchone()[0]
            print(f"\n📋 Table exists: {table_exists}")
            
            if table_exists:
                # Get table structure
                cur.execute("""
                    SELECT column_name, data_type, udt_name, character_maximum_length
                    FROM information_schema.columns 
                    WHERE table_name = 'indicator_embeddings'
                    ORDER BY ordinal_position;
                """)
                columns = cur.fetchall()
                
                print("\n📊 Table Structure:")
                print("-" * 60)
                for col in columns:
                    print(f"Column: {col[0]}")
                    print(f"  Data Type: {col[1]}")
                    print(f"  UDT Name: {col[2]}")
                    if col[3]:
                        print(f"  Max Length: {col[3]}")
                    print()
                
                # Check specifically for embedding column
                cur.execute("""
                    SELECT column_name, data_type, udt_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'indicator_embeddings' AND column_name = 'embedding';
                """)
                embedding_col = cur.fetchone()
                
                if embedding_col:
                    print(f"🎯 Embedding Column Details:")
                    print(f"  Column Name: {embedding_col[0]}")
                    print(f"  Data Type: {embedding_col[1]}")
                    print(f"  UDT Name: {embedding_col[2]}")
                else:
                    print("❌ No embedding column found!")
                
                # Check for any existing data
                cur.execute("SELECT COUNT(*) FROM indicator_embeddings;")
                count = cur.fetchone()[0]
                print(f"\n📈 Current record count: {count}")
                
                if count > 0:
                    # Check a sample embedding dimension
                    cur.execute("SELECT embedding FROM indicator_embeddings LIMIT 1;")
                    sample = cur.fetchone()
                    if sample:
                        embedding_length = len(sample[0])
                        print(f"🔢 Sample embedding dimension: {embedding_length}")
                
            else:
                print("❌ Table 'indicator_embeddings' does not exist!")
        
        conn.close()
        print("\n✅ Database check completed!")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE STRUCTURE CHECKER")
    print("=" * 60)
    check_database_structure()