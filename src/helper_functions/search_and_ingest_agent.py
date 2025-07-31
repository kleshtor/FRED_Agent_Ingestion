import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import yaml
from pathlib import Path
from dotenv import load_dotenv

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.llm_utils import LLMClient
from helper_functions.path_config import SRC_DIR
from helper_functions.fred_operations import (
    process_country_terms, process_dictionary_embeddings,
    load_existing_data_dictionary, ensure_complete_date_range,
    STANDARD_START_DATE
)

# ===== SEARCH AND INGEST AGENT =====

class SearchAndIngestAgent:
    def __init__(self, config_path: str, prompt_path: str = "llm_prompts.yaml"):
        self.config_path = config_path
        self.prompt_path = prompt_path

        self._load_environment()
        self._load_config()
        self.embedding_store = PostgresEmbeddingStore(self.db_config)

        self.start_date = "1900-01-01"
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def _load_environment(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in .env")

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.api_key = config.get("api_key")
        self.countries = config.get("countries", [])
        self.search_terms = [t['term'] for t in config.get("search_terms", [])]
        # Use SRC_DIR to ensure excel-output is created inside src/
        self.output_path = os.path.join(SRC_DIR, config.get("output_path", "excel-output/fred_macro_data.xlsx"))
        self.dict_file_path = os.path.join(SRC_DIR, "excel-output", "FRED_DataDictionary.xlsx")
        self.db_config = config.get("database", {})

        if not self.api_key:
            raise ValueError("FRED API key is missing in config")

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Main execution method that processes all countries and generates Excel files.
        Uses LLM-normalized country names for consistent file naming.
        """
        from helper_functions.llm_utils import LLMClient
        llm_client = LLMClient()
        
        all_dict_rows = []

        for user_country in self.countries:
            print(f"\nProcessing: {user_country}")
            
            # Get LLM-normalized country name for consistent file naming
            normalized_country = llm_client.normalize_country(user_country)

            data_by_freq, dict_rows = process_country_terms(
                country=user_country,
                terms=self.search_terms,
                api_key=self.api_key,
                end_date=self.end_date,
                output_base=self.output_path,
                dict_file_path=self.dict_file_path
            )

            for freq_code, df in data_by_freq.items():
                if not df.empty:
                    freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
                    # Use simplified naming format: {normalized_country}_{frequency}.xlsx
                    filename = os.path.join(SRC_DIR, "excel-output", f"{normalized_country.replace(' ', '_')}_{freq_label}.xlsx")

                    complete_df = ensure_complete_date_range(df, freq_code, self.end_date)
                    complete_df.to_excel(filename, index=True)
                    print(f"Saved: {filename}")

            all_dict_rows.extend(dict_rows)

        self._update_data_dictionary(all_dict_rows)
        self._generate_embeddings()

    def _update_data_dictionary(self, dict_rows):
        """Update the FRED data dictionary with new indicators"""
        print(f"      📝 **DICTIONARY UPDATE PROCESS**")
        
        if not dict_rows:
            print(f"         ⏭️ No new indicators to add to dictionary")
            return

        print(f"         📊 Processing {len(dict_rows)} new indicators...")
        print(f"         📁 Dictionary file: {os.path.basename(self.dict_file_path)}")
        
        # Load existing dictionary
        if os.path.exists(self.dict_file_path):
            existing = pd.read_excel(self.dict_file_path)
            print(f"         📚 Loaded existing dictionary: {len(existing)} entries")
        else:
            existing = pd.DataFrame()
            print(f"         📄 Creating new dictionary file")
        
        new = pd.DataFrame(dict_rows)
        print(f"         ➕ New entries to add: {len(new)}")

        if not existing.empty:
            print(f"         🔗 Merging with existing data...")
            combined = pd.concat([existing, new], ignore_index=True).drop_duplicates("Indicator ID", keep="last")
            print(f"         📊 Combined dictionary: {len(existing)} + {len(new)} = {len(combined)} total")
            if len(combined) < len(existing) + len(new):
                duplicates_removed = len(existing) + len(new) - len(combined)
                print(f"         🔄 Removed {duplicates_removed} duplicate(s)")
        else:
            combined = new
            print(f"         📄 Creating new dictionary with {len(combined)} entries")

        print(f"         💾 Saving dictionary to Excel...")
        combined.to_excel(self.dict_file_path, index=False)
        print(f"         ✅ Dictionary successfully updated: {len(new)} new entries added")

    def _generate_embeddings(self):
        """Generate embeddings for the data dictionary"""
        print(f"      🧠 **EMBEDDING GENERATION PROCESS**")
        print(f"         🔄 Processing dictionary for vector embeddings...")
        print(f"         📁 Dictionary file: {os.path.basename(self.dict_file_path)}")
        
        stats = process_dictionary_embeddings(
            dict_file_path=self.dict_file_path,
            embedding_store=self.embedding_store
        )
        
        print(f"         ✅ Embedding generation complete")
        print(f"         📊 Results: {stats}")
        return stats

    def ingest_specific_indicators(self, query: str, country: str = "USA") -> dict:
        """
        Targeted ingestion method for workflow delegation from other agents.
        
        Args:
            query: Search query for economic indicators
            country: Target country (defaults to USA)
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        from helper_functions.llms_utils import LLMClient
        llm_client = LLMClient()
        
        print(f"\n🔄 **SEARCH AND INGEST AGENT REQUEST**")
        print(f"📝 Query: '{query}'")
        print(f"🌍 Country: '{country}'")
        print(f"🎯 Purpose: Targeted ingestion for QueryAgent workflow")
        print("=" * 80)
        
        try:
            # Step 1: Country normalization
            print("🧠 **STEP 1: COUNTRY NORMALIZATION**")
            print(f"   📍 Original country: '{country}'")
            normalized_country = llm_client.normalize_country(country)
            print(f"   ✅ Normalized country: '{normalized_country}'")
            print(f"   💡 This ensures consistent file naming and data organization")
            
            # Step 2: Search term preparation  
            print(f"\n🔍 **STEP 2: SEARCH TERM PREPARATION**")
            search_terms = [query.strip()]
            print(f"   📝 Original query: '{query}'")
            print(f"   🎯 Search terms to use: {search_terms}")
            print(f"   📋 Strategy: Will search FRED for each term combined with country name")
            
            # Step 3: Load existing data dictionary
            print(f"\n📚 **STEP 3: CHECKING EXISTING DATA**")
            existing_dict = load_existing_data_dictionary(self.dict_file_path)
            print(f"   📖 Dictionary path: {self.dict_file_path}")
            print(f"   📊 Existing indicators in database: {len(existing_dict)}")
            print(f"   🔍 Will skip indicators that already exist locally")
            
            # Step 4: Main processing
            print(f"\n⚙️ **STEP 4: PROCESSING WITH FRED API**")
            print(f"   🌐 Starting targeted FRED search and ingestion...")
            print(f"   📅 Date range: {STANDARD_START_DATE} to {self.end_date}")
            
            data_by_freq, dict_rows = process_country_terms(
                country=country,
                terms=search_terms,
                api_key=self.api_key,
                end_date=self.end_date,
                output_base=self.output_path,
                dict_file_path=self.dict_file_path
            )
            
            # Step 5: Save Excel files
            print(f"\n💾 **STEP 5: CREATING EXCEL FILES**")
            excel_files_created = []
            for freq_code, df in data_by_freq.items():
                if not df.empty:
                    freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
                    filename = os.path.join(SRC_DIR, "excel-output", f"{normalized_country.replace(' ', '_')}_{freq_label}.xlsx")
                    
                    print(f"   📊 Processing {freq_label} data...")
                    print(f"      📈 Time series count: {len(df.columns)}")
                    print(f"      📅 Data points: {len(df)} rows")
                    
                    complete_df = ensure_complete_date_range(df, freq_code, self.end_date)
                    complete_df.to_excel(filename, index=True)
                    excel_files_created.append(os.path.basename(filename))
                    print(f"      ✅ Saved: {os.path.basename(filename)}")
                else:
                    freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
                    print(f"   ⏭️ Skipping {freq_label}: No data found")
            
            # Step 6: Update dictionary
            print(f"\n📝 **STEP 6: UPDATING DATA DICTIONARY**")
            if dict_rows:
                print(f"   📊 New indicators to add: {len(dict_rows)}")
                self._update_data_dictionary(dict_rows)
                print(f"   ✅ Dictionary updated successfully")
                for row in dict_rows:
                    print(f"      📈 Added: {row['Indicator Name']} ({row['Indicator ID']})")
            else:
                print(f"   ⏭️ No new indicators to add (all found indicators already exist)")
                
            # Step 7: Generate embeddings
            print(f"\n🧠 **STEP 7: GENERATING EMBEDDINGS**")
            print(f"   🔄 Creating vector embeddings for semantic search...")
            embedding_stats = process_dictionary_embeddings(
                dict_file_path=self.dict_file_path,
                embedding_store=self.embedding_store
            )
            print(f"   ✅ Embedding generation complete")
            print(f"      📊 New embeddings: {embedding_stats.get('processed', 0)}")
            print(f"      ⏭️ Skipped (already exist): {embedding_stats.get('skipped', 0)}")
            print(f"      ❌ Errors: {embedding_stats.get('errors', 0)}")
            
            # Prepare result summary
            result = {
                "success": True,
                "query": query,
                "country": country,
                "normalized_country": normalized_country,
                "new_indicators": len(dict_rows),
                "excel_files": excel_files_created,
                "embedding_stats": embedding_stats,
                "message": f"Successfully ingested {len(dict_rows)} new indicators for {normalized_country}"
            }
            
            print(f"\n✅ **SEARCH AND INGEST AGENT COMPLETE**")
            print("=" * 80)
            print(f"📊 **FINAL SUMMARY:**")
            print(f"   🎯 Query processed: '{query}'")
            print(f"   🌍 Country: {country} → {normalized_country}")
            print(f"   📈 New indicators added: {len(dict_rows)}")
            print(f"   📁 Excel files created/updated: {len(excel_files_created)}")
            if excel_files_created:
                for file in excel_files_created:
                    print(f"      📄 {file}")
            print(f"   🧠 New embeddings generated: {embedding_stats.get('processed', 0)}")
            print(f"   ✅ Status: SUCCESS - Data ready for QueryAgent")
            print("=" * 80)
            
            return result
            
        except Exception as e:
            print(f"\n❌ **SEARCH AND INGEST AGENT FAILED**")
            print("=" * 80)
            print(f"🚨 **ERROR DETAILS:**")
            print(f"   📝 Query: '{query}'")
            print(f"   🌍 Country: '{country}'")
            print(f"   💥 Error: {str(e)}")
            print(f"   🔧 Recommendation: Check FRED API connectivity and query format")
            print("=" * 80)
            
            error_result = {
                "success": False,
                "query": query,
                "country": country,
                "error": str(e),
                "message": f"Failed to ingest indicators: {str(e)}"
            }
            return error_result