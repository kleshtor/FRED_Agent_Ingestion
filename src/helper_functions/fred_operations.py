import os
import pandas as pd
from pandas_datareader import data as web
from fredapi import Fred
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import openai
import yaml
from pathlib import Path
from dotenv import load_dotenv

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.core_utils import LLMClient, SRC_DIR

# ===== FRED CONSTANTS =====

STANDARD_START_DATE = "1900-01-01"
VALID_FREQ = {"Monthly": "M", "Quarterly": "Q", "Annual": "A"}
llm_client = LLMClient()


# ===== FRED API OPERATIONS =====

def search_fred_series(search_term: str, api_key: str, max_results: int = 15) -> pd.DataFrame:
    try:
        fred = Fred(api_key=api_key)
        results = fred.search(search_term)
        if results is None or results.empty:
            return pd.DataFrame(columns=["id", "title", "frequency", "units"])
        return results.reset_index().head(max_results)[["id", "title", "frequency", "units"]]
    except Exception as e:
        print(f"FRED search error: {e}")
        return pd.DataFrame(columns=["id", "title", "frequency", "units"])

def fetch_series_data(series_id: str, label: str, start_date: str, end_date: str, freq_code: str) -> pd.DataFrame:
    try:
        df = web.DataReader(series_id, 'fred', start_date, end_date)
        df.index = pd.to_datetime(df.index)
        df.rename(columns={series_id: label}, inplace=True)

        if freq_code == "M":
            date_index = pd.date_range(start=start_date, end=end_date, freq='MS')
        elif freq_code == "Q":
            date_index = pd.date_range(start=start_date, end=end_date, freq='QS')
        elif freq_code == "A":
            date_index = pd.date_range(start=start_date, end=end_date, freq='YS')
        else:
            return pd.DataFrame()

        full_df = pd.DataFrame(index=date_index)
        full_df[label] = df[label].reindex(date_index)
        return full_df
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.DataFrame(columns=[label])

def load_existing_data_dictionary(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            print(f"Failed to load dictionary: {e}")
    return pd.DataFrame()

def indicator_exists_in_dictionary(indicator_id: str, dictionary: pd.DataFrame) -> bool:
    if dictionary.empty:
        return False
    return indicator_id in dictionary.get("Indicator ID", pd.Series([])).values

def load_existing_excel_data(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path, index_col=0, parse_dates=True)
            return df.sort_index()
        except Exception as e:
            print(f"Failed to load existing file {file_path}: {e}")
    return pd.DataFrame()

def merge_time_series_data(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new
    if new.empty:
        return existing
    merged = pd.concat([existing, new], axis=1)
    return merged.loc[:, ~merged.columns.duplicated(keep='last')].sort_index()

def ensure_complete_date_range(df: pd.DataFrame, freq_code: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(STANDARD_START_DATE)
    end = pd.to_datetime(end_date)
    if freq_code == "M":
        idx = pd.date_range(start=start, end=end, freq='MS')
    elif freq_code == "Q":
        idx = pd.date_range(start=start, end=end, freq='QS')
    elif freq_code == "A":
        idx = pd.date_range(start=start, end=end, freq='YS')
    else:
        return df
    return df.reindex(idx)

def process_country_terms(country: str, terms: List[str], api_key: str,
                          end_date: str, output_base: str,
                          dict_file_path: str) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """
    Process search terms for a specific country and return data organized by frequency.
    
    Args:
        country: Original country name from YAML config
        terms: List of economic indicator search terms
        api_key: FRED API key
        end_date: End date for data collection
        output_base: Base output path (used for directory structure)
        dict_file_path: Path to data dictionary file
        
    Returns:
        Tuple of (data_by_frequency_dict, dictionary_rows_list)
    """
    print(f"   🔍 **FRED SEARCH AND PROCESSING DETAILS**")
    
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    start_date = STANDARD_START_DATE
    normalized_country = llm_client.normalize_country(country)
    existing_dict = load_existing_data_dictionary(dict_file_path)

    print(f"      📅 Date range: {start_date} to {end_date}")
    print(f"      🌍 Normalized country: {normalized_country}")
    
    data_by_freq = {"M": pd.DataFrame(), "Q": pd.DataFrame(), "A": pd.DataFrame()}
    dictionary_rows = []

    # Load existing data using simplified naming format: {normalized_country}_{frequency}
    print(f"      💾 Loading existing Excel files...")
    for freq_code in data_by_freq:
        freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
        filename = os.path.join(SRC_DIR, "excel-output", f"{normalized_country.replace(' ', '_')}_{freq_label}.xlsx")
        existing_data = load_existing_excel_data(filename)
        data_by_freq[freq_code] = existing_data
        if not existing_data.empty:
            print(f"         📈 Loaded {freq_label}: {len(existing_data.columns)} series, {len(existing_data)} rows")
        else:
            print(f"         📄 {freq_label}: No existing data (will create new)")

    total_indicators_processed = 0
    total_new_indicators = 0
    total_skipped_existing = 0
    total_skipped_invalid = 0

    for term_idx, term in enumerate(terms, 1):
        print(f"\n      🔍 **SEARCH TERM {term_idx}/{len(terms)}: '{term}'**")
        query = f"{term} {normalized_country}"
        print(f"         🌐 FRED Query: '{query}'")
        
        print(f"         ⏳ Searching FRED database...")
        results = search_fred_series(query, api_key)
        
        if results.empty:
            print(f"         ❌ No results found for '{query}'")
            continue
            
        print(f"         ✅ Found {len(results)} potential indicators")
        print(f"         🔄 Processing each indicator...")

        for result_idx, (_, row) in enumerate(results.iterrows(), 1):
            freq = row.get("frequency")
            title = row.get("title")
            series_id = row.get("id")
            units = row.get("units", "")
            
            print(f"\n            📊 **INDICATOR {result_idx}/{len(results)}**")
            print(f"               🏷️ ID: {series_id}")
            print(f"               📝 Title: {title}")
            print(f"               📅 Frequency: {freq}")
            print(f"               📏 Units: {units}")
            
            freq_code = VALID_FREQ.get(freq)
            if not freq_code or not title or not series_id:
                print(f"               ❌ Skipping: Missing required data or unsupported frequency")
                total_skipped_invalid += 1
                total_indicators_processed += 1
                continue
                
            label = title.split('(')[0].strip().replace(" ", "_") + f"_{freq_code}"
            
            # Check if already exists in local data
            if label in data_by_freq[freq_code].columns:
                print(f"               ⏭️ Skipping: Already exists in local {freq} data")
                total_skipped_existing += 1
                total_indicators_processed += 1
                continue
                
            # Check if already exists in dictionary
            if indicator_exists_in_dictionary(series_id, existing_dict):
                print(f"               ⏭️ Skipping: Already exists in dictionary")
                total_skipped_existing += 1
                total_indicators_processed += 1
                continue

            print(f"               📈 Fetching time series data from FRED...")
            new_df = fetch_series_data(series_id, label, start_date, end_date, freq_code)
            
            if new_df.empty:
                print(f"               ⚠️ Warning: No time series data available")
                total_skipped_invalid += 1
                total_indicators_processed += 1
                continue
                
            print(f"               ✅ Data fetched: {len(new_df)} observations")
            print(f"               🔗 Merging with existing {freq} data...")
            
            data_by_freq[freq_code] = merge_time_series_data(data_by_freq[freq_code], new_df)
            
            print(f"               🧠 Generating description using LLM...")
            description = llm_client.generate_description(title, freq, normalized_country)
            print(f"               ✅ Description generated ({len(description)} characters)")
            
            dictionary_rows.append({
                "Indicator Name": title,
                "Indicator ID": series_id,
                "Description": description,
                "Frequency": freq_code,
                "Geography": normalized_country
            })
            
            total_new_indicators += 1
            total_indicators_processed += 1
            print(f"               🎉 Successfully processed and added!")

    print(f"\n      📊 **PROCESSING COMPLETE**")
    print(f"         🔢 Total indicators examined: {total_indicators_processed}")
    print(f"         ✅ New indicators added: {total_new_indicators}")
    print(f"         ⏭️ Skipped (already exist): {total_skipped_existing}")
    print(f"         ❌ Skipped (invalid/no data): {total_skipped_invalid}")
    
    for freq_code, df in data_by_freq.items():
        freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
        if not df.empty:
            print(f"         📈 {freq_label} dataset: {len(df.columns)} series, {len(df)} time periods")
        else:
            print(f"         📭 {freq_label} dataset: Empty")

    return data_by_freq, dictionary_rows

def process_dictionary_embeddings(dict_file_path: str, embedding_store, model: Optional[str] = None) -> dict:
    """
    Process dictionary entries and create embeddings for descriptions.
    
    Args:
        dict_file_path: Path to the Excel dictionary file
        embedding_store: PostgreSQL embedding store instance
        model: Embedding model to use (if None, uses config default)
        
    Returns:
        Dictionary with processing statistics
    """
    # Load configuration to get default embedding model
    if model is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = config.get("embedding", {}).get("model", "text-embedding-3-large")
    
    df = pd.read_excel(dict_file_path)
    stats = {"total": len(df), "processed": 0, "skipped": 0, "errors": 0}

    client = openai.OpenAI()

    for _, row in df.iterrows():
        ind_id = row.get("Indicator ID", "").strip()
        ind_name = row.get("Indicator Name", "").strip()
        desc = row.get("Description", "").strip()

        if not ind_id or not desc or desc == "Description not available":
            stats["skipped"] += 1
            continue

        if embedding_store.embedding_exists(ind_id):
            stats["skipped"] += 1
            continue

        try:
            emb = client.embeddings.create(input=desc, model=model).data[0].embedding
            success = embedding_store.save_embedding(ind_id, ind_name, desc, emb, model=model)
            stats["processed"] += int(success)
        except Exception as e:
            print(f"Embedding error for {ind_id}: {e}")
            stats["errors"] += 1

    print(f"Embedding Summary → Processed: {stats['processed']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
    return stats


# ===== FRED AGENT SDK =====

class FredAgent:
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
        print(f"\n🔄 **FRED AGENT INGESTION REQUEST**")
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
            
            print(f"\n✅ **FRED AGENT INGESTION COMPLETE**")
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
            print(f"\n❌ **FRED AGENT INGESTION FAILED**")
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