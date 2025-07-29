import os
import pandas as pd
from typing import Optional, Dict, Tuple, List
import yaml
import openai
from dotenv import load_dotenv

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.core_utils import SRC_DIR, LLMClient


# ===== CONFIGURATION UTILITIES =====

def load_worldbank_config() -> Dict:
    """
    Load configuration from worldbank_config.yaml file.
    
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(SRC_DIR, "helper_functions", "worldbank_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ===== WORLD BANK DATA EXTRACTION =====

def get_worldbank_indicator_metadata(indicator_id: str) -> Optional[Dict]:
    """
    Get indicator metadata from World Bank data dictionary.
    
    Args:
        indicator_id: The World Bank indicator ID
        
    Returns:
        Dictionary containing metadata or None if not found
    """
    dict_file_path = os.path.join(SRC_DIR, "excel-output", "WorldBank_DataDictionary.xlsx")
    
    if not os.path.exists(dict_file_path):
        print(f"World Bank data dictionary not found: {dict_file_path}")
        return None
    
    try:
        df = pd.read_excel(dict_file_path)
        
        # Find the indicator
        indicator_row = df[df['Indicator ID'] == indicator_id]
        
        if indicator_row.empty:
            print(f"Indicator ID '{indicator_id}' not found in World Bank dictionary")
            return None
        
        # Convert to dictionary
        metadata = indicator_row.iloc[0].to_dict()
        print(f"Found World Bank indicator: {metadata.get('Indicator Name', 'N/A')}")
        return metadata
        
    except Exception as e:
        print(f"Error reading World Bank data dictionary: {e}")
        return None


def construct_worldbank_excel_filename(geography: str, frequency: str) -> str:
    """
    Construct the Excel filename for World Bank data based on geography and frequency.
    Uses World Bank country code normalization.
    
    Args:
        geography: Country/geography name or code
        frequency: Frequency (e.g., 'Annual', 'Monthly', 'Quarterly')
        
    Returns:
        Excel filename path using World Bank naming format: {worldbank_country_code}_{frequency}.xlsx
    """
    # Normalize to World Bank country code (e.g., 'Canada' -> 'CAN')
    wb_country_code = LLMClient().normalize_worldbank_country(geography)
    
    # Construct filename using World Bank format
    filename = os.path.join(SRC_DIR, "excel-output", f"{wb_country_code}_{frequency}.xlsx")
    return filename


def find_worldbank_indicator_column(excel_file: str, indicator_name: str, frequency: str) -> Optional[str]:
    """
    Find the correct column name for a World Bank indicator in an Excel file.
    
    Args:
        excel_file: Path to the Excel file
        indicator_name: The indicator name to search for
        frequency: The frequency (for label matching)
        
    Returns:
        Column name if found, None otherwise
    """
    try:
        df = pd.read_excel(excel_file, index_col=0)
        
        # Create potential column name patterns for World Bank data
        # World Bank columns typically use country codes + indicator abbreviations
        clean_name = str(indicator_name).split('(')[0].strip().replace(' ', '_').replace(',', '').replace(':', '')[:30]
        
        # Try different column name patterns
        potential_names = [
            f"{clean_name}_A",  # Standard World Bank format
            clean_name,
            indicator_name,
            # Try with different country prefixes if geography info is available
        ]
        
        # Look for exact matches first
        for col_name in df.columns:
            if col_name in potential_names:
                print(f"Found exact column match: {col_name}")
                return col_name
        
        # Try partial matches
        for col_name in df.columns:
            for potential in potential_names:
                if potential.lower() in col_name.lower() or col_name.lower() in potential.lower():
                    print(f"Found partial column match: {col_name}")
                    return col_name
        
        print(f"Available columns in {excel_file}: {list(df.columns)}")
        print(f"Could not find column for indicator: {indicator_name}")
        return None
        
    except Exception as e:
        print(f"Error reading Excel file {excel_file}: {e}")
        return None


def extract_worldbank_time_series_data(excel_file: str, column_name: str) -> Optional[pd.Series]:
    """
    Extract time series data from World Bank Excel file for a specific column.
    
    Args:
        excel_file: Path to the Excel file
        column_name: Name of the column to extract
        
    Returns:
        Time series data as pandas Series or None if extraction fails
    """
    try:
        df = pd.read_excel(excel_file, index_col=0)
        
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in {excel_file}")
            return None
        
        # Extract the time series
        time_series = df[column_name].dropna()
        
        # Ensure datetime index
        if not isinstance(time_series.index, pd.DatetimeIndex):
            time_series.index = pd.to_datetime(time_series.index)
        
        print(f"Extracted {len(time_series)} data points from {excel_file}")
        return time_series
        
    except Exception as e:
        print(f"Error extracting time series from {excel_file}: {e}")
        return None


def display_worldbank_indicator_info(metadata: Dict, time_series: pd.Series):
    """
    Display World Bank indicator information and basic statistics.
    
    Args:
        metadata: Indicator metadata dictionary
        time_series: Time series data
    """
    print("\n" + "="*60)
    print("WORLD BANK INDICATOR INFORMATION")
    print("="*60)
    print(f"Indicator ID: {metadata.get('Indicator ID', 'N/A')}")
    print(f"Name: {metadata.get('Indicator Name', 'N/A')}")
    print(f"Geography: {metadata.get('Geography', 'N/A')}")
    print(f"Frequency: {metadata.get('Frequency', 'N/A')}")
    print(f"Units: {metadata.get('Units', 'N/A')}")
    print(f"Source: World Bank")
    print(f"Last Updated: {metadata.get('Last Updated', 'N/A')}")
    
    print("\nTIME SERIES STATISTICS:")
    print(f"Data Points: {len(time_series)}")
    print(f"Date Range: {time_series.index.min().strftime('%Y-%m-%d')} to {time_series.index.max().strftime('%Y-%m-%d')}")
    print(f"Latest Value: {time_series.iloc[-1]:.4f} ({time_series.index[-1].strftime('%Y-%m-%d')})")
    print(f"Mean: {time_series.mean():.4f}")
    print(f"Std Dev: {time_series.std():.4f}")
    print("="*60)


def extract_worldbank_data(indicator_id: str) -> Optional[Tuple[Dict, pd.Series]]:
    """
    Main function to extract World Bank time series data for a given Indicator ID.
    Uses World Bank-specific file naming conventions.
    
    Args:
        indicator_id: The World Bank indicator ID to extract data for
        
    Returns:
        Tuple of (metadata_dict, time_series) or None if extraction fails
    """
    print(f"Extracting World Bank data for Indicator ID: '{indicator_id}'")
    
    # Step 1: Get indicator metadata from World Bank data dictionary
    print("Loading World Bank indicator metadata...")
    metadata = get_worldbank_indicator_metadata(indicator_id)
    
    if not metadata:
        return None
    
    # Step 2: Determine file location based on metadata
    geography = metadata.get('Geography', '')
    frequency = metadata.get('Frequency', 'Annual')
    indicator_name = metadata.get('Indicator Name', '')
    
    if not all([geography, frequency, indicator_name]):
        print("Missing required metadata fields.")
        return None
    
    # Step 3: Construct Excel filename using World Bank naming convention
    excel_file = construct_worldbank_excel_filename(geography, frequency)
    print(f"Looking for World Bank data in: {excel_file}")
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"World Bank Excel file not found: {excel_file}")
        return None
    
    # Step 4: Find the correct column name
    print("Finding indicator column in World Bank Excel file...")
    column_name = find_worldbank_indicator_column(excel_file, indicator_name, frequency)
    
    if not column_name:
        return None
    
    # Step 5: Extract time series data
    print(f"Extracting World Bank time series data from column: {column_name}")
    time_series = extract_worldbank_time_series_data(excel_file, column_name)
    
    if time_series is None:
        return None
    
    # Step 6: Display results
    display_worldbank_indicator_info(metadata, time_series)
    
    return metadata, time_series


def create_worldbank_dataframe_preview(metadata: Dict, time_series: pd.Series, preview_rows: int = 10) -> str:
    """
    Create a formatted preview of World Bank time series data for display.
    
    Args:
        metadata: Indicator metadata dictionary
        time_series: Time series data
        preview_rows: Number of rows to show in preview
        
    Returns:
        Formatted string representation of the data preview
    """
    try:
        # Create header
        preview = f"\n{'='*80}\n"
        preview += f"WORLD BANK DATA PREVIEW: {metadata.get('Indicator Name', 'N/A')}\n"
        preview += f"{'='*80}\n"
        preview += f"Indicator ID: {metadata.get('Indicator ID', 'N/A')}\n"
        preview += f"Geography: {metadata.get('Geography', 'N/A')}\n"
        preview += f"Frequency: {metadata.get('Frequency', 'N/A')}\n"
        preview += f"Units: {metadata.get('Units', 'N/A')}\n"
        preview += f"Source: World Bank\n"
        preview += f"Total Data Points: {len(time_series)}\n"
        preview += f"Date Range: {time_series.index.min().strftime('%Y-%m-%d')} to {time_series.index.max().strftime('%Y-%m-%d')}\n\n"
        
        # Create data preview table
        preview += f"RECENT DATA (Last {min(preview_rows, len(time_series))} observations):\n"
        preview += f"{'-'*50}\n"
        preview += f"{'Date':<12} {'Value':<15} {'Change':<10}\n"
        preview += f"{'-'*50}\n"
        
        # Show most recent data
        recent_data = time_series.tail(preview_rows)
        for i, (date, value) in enumerate(recent_data.items()):
            # Calculate change from previous period if available
            if i > 0:
                prev_value = recent_data.iloc[i-1]
                change = ((value - prev_value) / prev_value * 100) if prev_value != 0 else 0
                change_str = f"{change:+.2f}%"
            else:
                change_str = "N/A"
            
            preview += f"{date.strftime('%Y-%m-%d'):<12} {value:<15.4f} {change_str:<10}\n"
        
        preview += f"{'-'*50}\n"
        preview += f"Latest Value: {time_series.iloc[-1]:.4f} ({time_series.index[-1].strftime('%Y-%m-%d')})\n"
        preview += f"Mean: {time_series.mean():.4f}\n"
        preview += f"Standard Deviation: {time_series.std():.4f}\n"
        preview += f"{'='*80}\n"
        
        return preview
        
    except Exception as e:
        return f"Error creating preview: {e}"


# ===== SEARCH AND QUERY OPERATIONS =====

def create_query_embedding(query: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Convert a text query into an embedding vector using OpenAI.
    
    Args:
        query: The text query to convert to embedding
        model: The OpenAI embedding model to use
        
    Returns:
        List of floats representing the embedding vector
    """
    # Load environment variables for OpenAI API key
    load_dotenv()
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        # Create embedding for the query
        response = client.embeddings.create(input=query, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding for query '{query}': {e}")
        return []


def display_search_results(results: List[Dict], query: str) -> None:
    """
    Display search results in a formatted way in the terminal.
    
    Args:
        results: List of search result dictionaries
        query: The original query string
    """
    print(f"\n{'='*80}")
    print(f"WORLD BANK SEARCH RESULTS FOR: '{query}'")
    print(f"{'='*80}")
    
    if not results:
        print("No matching indicators found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n--- MATCH {i} ---")
        print(f"Indicator ID: {result['indicator_id']}")
        print(f"Name: {result['indicator_name']}")
        print(f"Similarity Score: {result['similarity']:.4f}")
        print(f"Geography: {result.get('geography', 'N/A')}")
        print(f"Source: World Bank")
        print("-" * 60)


def search_worldbank_database(query: str) -> List[Dict]:
    """
    Main function to search for similar World Bank economic indicators based on a text query.
    Now uses PostgreSQL with proper World Bank source filtering.
    
    Args:
        query: The search query string
        
    Returns:
        List of top 3 matching indicators with similarity scores
    """
    print(f"Processing World Bank query: '{query}'")
    
    # Create embedding from query
    print("Converting query to embedding...")
    query_embedding = create_query_embedding(query)
    
    if not query_embedding:
        print("Failed to create embedding for query.")
        return []
    
    # Load World Bank configuration for database connection
    print("Connecting to PostgreSQL embedding database...")
    config = load_worldbank_config()
    db_config = config.get("database", {})
    embedding_store = PostgresEmbeddingStore(db_config)
    
    try:
        # Search for similar embeddings with World Bank source filter
        print("Searching for similar World Bank indicators...")
        results = embedding_store.search_similar_embeddings(
            query_embedding, 
            top_k=3, 
            filter_metadata={"source": "World Bank"}
        )
        
        # Display results
        display_search_results(results, query)
        
        return results
        
    except Exception as e:
        print(f"Error during World Bank search: {e}")
        return []
    finally:
        # Close database connection
        embedding_store.close() 