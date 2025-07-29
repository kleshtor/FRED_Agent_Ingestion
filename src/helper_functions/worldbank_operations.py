import os
import pandas as pd
import wbgapi as wb
from typing import List, Dict, Tuple
from datetime import datetime
import yaml
from pathlib import Path
from dotenv import load_dotenv

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.core_utils import LLMClient

STANDARD_START_YEAR = 1960
VALID_FREQ = {"Annual": "A"}  # World Bank data is typically annual
llm_client = LLMClient()


# ===== WORLD BANK API OPERATIONS =====

def search_worldbank_series(search_term: str, max_results: int = 15) -> pd.DataFrame:
    """Search World Bank indicators by term using improved wbgapi approach"""
    try:
        print(f"🔍 Searching World Bank for: '{search_term}'")
        
        # Search indicators using wbgapi
        results = wb.series.info(q=search_term)
        
        # Convert Featureset to DataFrame using .items
        df = pd.DataFrame(results.items).head(max_results)
        
        if df.empty:
            print(f"   ❌ No results found for '{search_term}'")
            return pd.DataFrame(columns=["id", "name", "frequency", "units"])
        
        print(f"   ✅ Found {len(df)} indicators")
        
        # Rename 'value' column to 'name' for consistency
        if 'value' in df.columns:
            df = df.rename(columns={'value': 'name'})
        
        # Add frequency and units columns (World Bank is typically annual)
        df['frequency'] = 'Annual'
        df['units'] = 'Various'  # World Bank doesn't provide standardized units in search
        
        # Ensure we have the required columns
        required_cols = ["id", "name", "frequency", "units"]
        for col in required_cols:
            if col not in df.columns:
                if col == 'name' and 'value' in df.columns:
                    df['name'] = df['value']
                else:
                    df[col] = 'N/A'
        
        # Select and return required columns
        result_df = df[required_cols].copy()
        
        # Clean up names - remove extra parentheses and long descriptions
        result_df['name'] = result_df['name'].apply(lambda x: str(x).split('(')[0].strip() if pd.notna(x) else 'N/A')
        
        print(f"   📊 Returning {len(result_df)} cleaned results")
        return result_df
        
    except Exception as e:
        print(f"   ❌ Error searching World Bank indicators: {e}")
        return pd.DataFrame(columns=["id", "name", "frequency", "units"])

def fetch_worldbank_data(indicator_id: str, label: str, countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch World Bank data for a specific indicator and countries using improved approach"""
    try:
        print(f"📊 Fetching World Bank data for {indicator_id}")
        print(f"   🌍 Countries: {countries}")
        print(f"   📅 Years: {start_year}-{end_year}")
        
        all_data = pd.DataFrame()
        
        for country in countries:
            try:
                print(f"   🔄 Processing {country}...")
                
                # Fetch all available data (no time parameter - API works better this way)
                data = wb.data.DataFrame(
                    series=indicator_id, 
                    economy=country
                )
                
                if data is not None and not data.empty:
                    # Filter to desired year range manually
                    year_cols = [col for col in data.columns 
                               if col.startswith('YR') and 
                               start_year <= int(col[2:]) <= end_year]
                    
                    if year_cols:
                        filtered_data = data[year_cols]
                        
                        # Create time series with proper datetime index
                        country_series = pd.Series(dtype=float)
                        for col in year_cols:
                            year = int(col[2:])  # Remove 'YR' prefix
                            date_index = pd.to_datetime(f"{year}-01-01")
                            value = filtered_data[col].iloc[0] if not filtered_data[col].empty else None
                            if pd.notna(value):
                                country_series[date_index] = float(value)
                        
                        # Add to combined dataframe with country-specific column name
                        column_name = f"{country}_{label}"
                        if not country_series.empty:
                            country_df = pd.DataFrame({column_name: country_series})
                            if all_data.empty:
                                all_data = country_df
                            else:
                                all_data = pd.concat([all_data, country_df], axis=1)
                            
                            print(f"      ✅ {country}: {len(country_series.dropna())} years of data")
                        else:
                            print(f"      ⚠️  {country}: No valid data points")
                    else:
                        print(f"      ⚠️  {country}: No data for {start_year}-{end_year}")
                else:
                    print(f"      ❌ {country}: No data returned from API")
                        
            except Exception as e:
                print(f"      ❌ Error fetching {country}: {e}")
                continue
        
        if not all_data.empty:
            all_data = all_data.sort_index()
            print(f"   ✅ Successfully fetched data: {all_data.shape[0]} rows × {all_data.shape[1]} columns")
        else:
            print(f"   ❌ No data fetched for indicator {indicator_id}")
        
        return all_data
        
    except Exception as e:
        print(f"   ❌ Error fetching World Bank data for {indicator_id}: {e}")
        return pd.DataFrame()

def load_existing_data_dictionary(path: str) -> pd.DataFrame:
    """Load existing World Bank data dictionary"""
    if os.path.exists(path):
        try:
            return pd.read_excel(path)
        except Exception as e:
            print(f"Failed to load dictionary: {e}")
    return pd.DataFrame()

def indicator_exists_in_dictionary(indicator_id: str, dictionary: pd.DataFrame) -> bool:
    """Check if indicator already exists in dictionary"""
    if dictionary.empty:
        return False
    return indicator_id in dictionary.get("Indicator ID", pd.Series([])).values

def load_existing_excel_data(file_path: str) -> pd.DataFrame:
    """Load existing Excel data file"""
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path, index_col=0, parse_dates=True)
            return df.sort_index()
        except Exception as e:
            print(f"Failed to load existing file {file_path}: {e}")
    return pd.DataFrame()

def merge_time_series_data(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge existing and new time series data"""
    if existing.empty:
        return new
    if new.empty:
        return existing
    merged = pd.concat([existing, new], axis=1)
    return merged.loc[:, ~merged.columns.duplicated(keep='last')].sort_index()

def ensure_complete_date_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Ensure complete date range for World Bank data (annual)"""
    if df.empty:
        return df
    
    start = pd.to_datetime(f"{start_year}-01-01")
    end = pd.to_datetime(f"{end_year}-01-01")
    idx = pd.date_range(start=start, end=end, freq='YS')
    return df.reindex(idx)

def process_country_terms(country: str, terms: List[str], countries: List[str],
                          start_year: int, end_year: int, output_base: str,
                          dict_file_path: str) -> Tuple[Dict[str, pd.DataFrame], List[Dict]]:
    """Process search terms for World Bank data extraction"""
    data_by_freq = {"A": pd.DataFrame()}  # World Bank is primarily annual
    dict_rows = []
    
    existing_dict = load_existing_data_dictionary(dict_file_path)
    
    for term_obj in terms:
        term = term_obj['term']
        print(f"\n🔍 Searching World Bank for: {term}")
        
        # Search for indicators
        search_results = search_worldbank_series(term, max_results=10)
        
        if search_results.empty:
            print(f"No results found for {term}")
            continue
        
        print(f"Found {len(search_results)} indicators for {term}")
        
        # Process each indicator
        for _, row in search_results.iterrows():
            indicator_id = row['id']
            indicator_name = row['name']
            
            # Skip if already exists
            if indicator_exists_in_dictionary(indicator_id, existing_dict):
                print(f"Skipping {indicator_id} (already exists)")
                continue
            
            # Create label
            clean_name = str(indicator_name).split('(')[0].strip().replace(' ', '_').replace(',', '').replace(':', '')[:30]
            label = f"{clean_name}_A"
            
            # Fetch data
            print(f"Fetching data for {indicator_id}: {indicator_name[:50]}...")
            ts_data = fetch_worldbank_data(indicator_id, label, countries, start_year, end_year)
            
            if not ts_data.empty:
                # Merge with existing data
                data_by_freq["A"] = merge_time_series_data(data_by_freq["A"], ts_data)
                
                # Add to dictionary
                dict_rows.append({
                    "Indicator ID": indicator_id,
                    "Indicator Name": indicator_name,
                    "Geography": country,
                    "Frequency": "Annual",
                    "Units": row.get('units', 'Various'),
                    "Source": "World Bank",
                    "Last Updated": datetime.now().strftime("%Y-%m-%d")
                })
                
                print(f"✅ Added {indicator_id} with {ts_data.shape[1]} country series")
            else:
                print(f"⚠️  No data available for {indicator_id}")
    
    return data_by_freq, dict_rows

def process_dictionary_embeddings(dict_file_path: str, embedding_store=None) -> Dict:
    """Process World Bank dictionary embeddings using PostgreSQL store"""
    try:
        if not os.path.exists(dict_file_path):
            return {"error": "Dictionary file not found"}
        
        df = pd.read_excel(dict_file_path)
        
        if df.empty:
            return {"processed": 0, "errors": 0}
        
        # Use PostgreSQL store if no store provided
        if embedding_store is None:
            from helper_functions.postgres_store import PostgresEmbeddingStore
            # Load World Bank config for database connection
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), "worldbank_config.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            db_config = config.get("database", {})
            embedding_store = PostgresEmbeddingStore(db_config)
            should_close = True
        else:
            should_close = False
        
        processed = 0
        errors = 0
        
        for _, row in df.iterrows():
            try:
                # Create embedding text
                text = f"{row['Indicator Name']} {row.get('Units', '')} {row.get('Geography', '')}"
                
                # Store embedding
                success = embedding_store.store_embedding(
                    text=text,
                    metadata={
                        "indicator_id": row['Indicator ID'],
                        "indicator_name": row['Indicator Name'],
                        "geography": row.get('Geography', ''),
                        "frequency": row.get('Frequency', 'Annual'),
                        "source": "World Bank"
                    }
                )
                
                if success:
                    processed += 1
                else:
                    errors += 1
                
            except Exception as e:
                print(f"Error processing {row.get('Indicator ID', 'unknown')}: {e}")
                errors += 1
        
        if should_close:
            embedding_store.close()
        
        return {"processed": processed, "errors": errors}
        
    except Exception as e:
        print(f"Error processing embeddings: {e}")
        return {"error": str(e)}


# ===== WORLD BANK AGENT SDK =====

class WorldBankAgent:
    def __init__(self, config_path: str, prompt_path: str = "prompts.yaml"):
        self.config_path = config_path
        self.prompt_path = prompt_path

        self._load_environment()
        self._load_config()
        self.embedding_store = PostgresEmbeddingStore(self.db_config)

        self.start_year = STANDARD_START_YEAR
        self.end_year = datetime.today().year

    def _load_environment(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in .env")

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.countries = config.get("countries", [])
        self.search_terms = [t for t in config.get("search_terms", [])]
        self.start_year = config.get("start_year", STANDARD_START_YEAR)
        self.end_year = config.get("end_year", datetime.today().year)
        self.output_path = config.get("output_path", "excel-output/worldbank_macro_data.xlsx")
        self.dict_file_path = "excel-output/WorldBank_DataDictionary.xlsx"
        self.db_config = config.get("database", {})

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
                countries=self.countries,  # Pass all countries for multi-country data
                start_year=self.start_year,
                end_year=self.end_year,
                output_base=self.output_path,
                dict_file_path=self.dict_file_path
            )

            for freq_code, df in data_by_freq.items():
                if not df.empty:
                    freq_label = {"A": "Annual"}[freq_code]
                    # Use simplified naming format: {normalized_country}_{frequency}.xlsx
                    filename = f"excel-output/{normalized_country.replace(' ', '_')}_{freq_label}.xlsx"

                    complete_df = ensure_complete_date_range(df, self.start_year, self.end_year)
                    complete_df.to_excel(filename, index=True)
                    print(f"Saved: {filename}")

            all_dict_rows.extend(dict_rows)

        self._update_data_dictionary(all_dict_rows)
        self._generate_embeddings()

    def _update_data_dictionary(self, dict_rows):
        """Update World Bank data dictionary"""
        if not dict_rows:
            print("No new indicators to update in dictionary.")
            return

        existing = pd.read_excel(self.dict_file_path) if os.path.exists(self.dict_file_path) else pd.DataFrame()
        new = pd.DataFrame(dict_rows)

        if not existing.empty:
            combined = pd.concat([existing, new], ignore_index=True).drop_duplicates("Indicator ID", keep="last")
        else:
            combined = new

        combined.to_excel(self.dict_file_path, index=False)
        print(f"Updated dictionary with {len(new)} new entries.")

    def _generate_embeddings(self):
        """Generate embeddings for World Bank data dictionary"""
        stats = process_dictionary_embeddings(
            dict_file_path=self.dict_file_path,
            embedding_store=self.embedding_store
        )
        print("Embedding Generation Complete:", stats) 