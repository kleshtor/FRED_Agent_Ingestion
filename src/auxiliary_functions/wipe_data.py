#!/usr/bin/env python3
"""
Script to wipe FRED data and embeddings for a complete fresh start.
This can remove:
1. All embeddings from the PostgreSQL database
2. All Excel output files and data dictionary
"""

import os
import yaml
import sys
import shutil
from typing import Dict

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.core_utils import SRC_DIR


def load_config() -> Dict:
    """
    Load configuration from config file.
    
    Returns:
        Configuration dictionary
    """
    # Get the path to the config file relative to the current script location
    current_dir = os.path.dirname(__file__)  # auxiliary_files directory
    config_path = os.path.join(current_dir, '..', 'helper_functions', 'config.yaml')
    config_path = os.path.abspath(config_path)  # Resolve to absolute path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def wipe_embeddings_database() -> bool:
    """
    Wipe all embeddings from the PostgreSQL database.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load configuration
        config = load_config()
        db_config = config.get("database", {})
        
        print("Connecting to PostgreSQL database...")
        print(f"Host: {db_config.get('host', 'localhost')}")
        print(f"Database: {db_config.get('dbname', 'N/A')}")
        
        # Initialize database connection
        embedding_store = PostgresEmbeddingStore(db_config)
        
        # Get count before deletion
        with embedding_store.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM indicator_embeddings;")
            count_before = cur.fetchone()[0]
        
        print(f"Found {count_before} embeddings in the database.")
        
        if count_before == 0:
            print("Database is already empty.")
        else:
            # Delete all embeddings
            print("Deleting all embeddings...")
            with embedding_store.conn.cursor() as cur:
                cur.execute("DELETE FROM indicator_embeddings;")
                deleted_count = cur.rowcount
            
            print(f"✅ Successfully deleted {deleted_count} embeddings from the database.")
            
            # Verify deletion
            with embedding_store.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM indicator_embeddings;")
                count_after = cur.fetchone()[0]
            
            print(f"Database now contains {count_after} embeddings.")
        
        # Close connection
        embedding_store.close()
        return True
        
    except Exception as e:
        print(f"❌ Error wiping embeddings database: {e}")
        return False


def wipe_excel_files() -> bool:
    """
    Wipe all Excel output files and the data dictionary.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get the path to excel-output directory relative to the current script location
        current_dir = os.path.dirname(__file__)  # auxiliary_files directory
        excel_output_dir = os.path.join(current_dir, '..', 'excel-output')
        excel_output_dir = os.path.abspath(excel_output_dir)  # Resolve to absolute path
        
        if not os.path.exists(excel_output_dir):
            print("Excel output directory doesn't exist - nothing to delete.")
            return True
        
        # List files to be deleted
        files_to_delete = []
        for file in os.listdir(excel_output_dir):
            if file.endswith(('.xlsx', '.xls')):
                files_to_delete.append(file)
        
        if not files_to_delete:
            print("No Excel files found to delete.")
            return True
        
        print(f"Found {len(files_to_delete)} Excel files to delete:")
        for file in files_to_delete:
            print(f"  - {file}")
        
        # Delete all Excel files
        print("Deleting Excel files...")
        deleted_count = 0
        for file in files_to_delete:
            file_path = os.path.join(excel_output_dir, file)
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"  ✅ Deleted: {file}")
            except Exception as e:
                print(f"  ❌ Failed to delete {file}: {e}")
        
        print(f"✅ Successfully deleted {deleted_count} out of {len(files_to_delete)} Excel files.")
        return True
        
    except Exception as e:
        print(f"❌ Error wiping Excel files: {e}")
        return False


def get_user_choice() -> str:
    """
    Get user's choice for what to wipe.
    
    Returns:
        User's choice: 'embeddings', 'excel', 'both', or 'cancel'
    """
    print("\nWhat would you like to wipe?")
    print("1. Embeddings database only")
    print("2. Excel output files only") 
    print("3. Both embeddings and Excel files (complete fresh start)")
    print("4. Cancel")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            return 'embeddings'
        elif choice == '2':
            return 'excel'
        elif choice == '3':
            return 'both'
        elif choice == '4':
            return 'cancel'
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def confirm_action(action: str) -> bool:
    """
    Confirm the user wants to proceed with the action.
    
    Args:
        action: Description of what will be deleted
        
    Returns:
        True if user confirms, False otherwise
    """
    print(f"\n⚠️  WARNING: This will {action}!")
    print("This action cannot be undone.")
    
    confirm = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
    return confirm in ['yes', 'y']


def main():
    """
    Main function to execute the data wipe utility.
    """
    print("=" * 70)
    print("FRED DATA & EMBEDDINGS WIPE UTILITY")
    print("=" * 70)
    
    # Get user's choice
    choice = get_user_choice()
    
    if choice == 'cancel':
        print("Operation cancelled by user.")
        return
    
    # Confirm action
    action_descriptions = {
        'embeddings': 'delete ALL embeddings from the database',
        'excel': 'delete ALL Excel output files and data dictionary',
        'both': 'delete ALL embeddings AND all Excel output files'
    }
    
    if not confirm_action(action_descriptions[choice]):
        print("Operation cancelled by user.")
        return
    
    print(f"\nStarting {choice} wipe operation...\n")
    
    success = True
    
    # Execute based on choice
    if choice in ['embeddings', 'both']:
        print("🗄️  WIPING EMBEDDINGS DATABASE")
        print("-" * 40)
        success &= wipe_embeddings_database()
        print()
    
    if choice in ['excel', 'both']:
        print("📊 WIPING EXCEL OUTPUT FILES")
        print("-" * 40)
        success &= wipe_excel_files()
        print()
    
    # Final status
    if success:
        print("✅ Data wipe completed successfully!")
        
        if choice == 'both':
            print("\n🎯 Complete fresh start ready!")
            print("You can now re-run your FRED data ingestion process")
            print("to rebuild everything from scratch.")
        elif choice == 'embeddings':
            print("\nYou can now re-run the embedding generation process")
            print("to rebuild embeddings from your existing Excel files.")
        elif choice == 'excel':
            print("\nYou can now re-run your FRED data collection process")
            print("to rebuild the Excel files and data dictionary.")
    else:
        print("❌ Data wipe encountered errors!")
        print("Please check the error messages above and try again.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main() 