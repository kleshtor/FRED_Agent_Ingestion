import os
import pandas as pd
from typing import Optional, Dict, Tuple, List
import yaml
import openai
from dotenv import load_dotenv

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.core_utils import SRC_DIR, LLMClient

# Initialize LLM client for country normalization
llm_client = LLMClient()


# ===== CONFIGURATION UTILITIES =====

def load_config() -> Dict:
    """
    Load configuration from config.yaml file.
    
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(SRC_DIR, "helper_functions", "config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ===== SEARCH AND QUERY OPERATIONS =====

def create_query_embedding(query: str, model: Optional[str] = None) -> List[float]:
    """
    Convert a text query into an embedding vector using OpenAI. 
    
    Args:
        query: The text query to convert to embedding
        model: The OpenAI embedding model to use (if None, uses config default)
        
    Returns:
        List of floats representing the embedding vector
    """
    # Load environment variables for OpenAI API key
    load_dotenv()
    
    # Load configuration to get default embedding model
    if model is None:
        config = load_config()
        model = config.get("embedding", {}).get("model", "text-embedding-3-large")
    
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
    print(f"SEARCH RESULTS FOR: '{query}'")
    print(f"{'='*80}")
    
    if not results:
        print("No matching indicators found.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"\n--- MATCH {i} ---")
        print(f"Indicator ID: {result['indicator_id']}")
        print(f"Name: {result['indicator_name']}")
        print(f"Similarity Score: {result['similarity']:.4f}")
        print(f"Description: {result['description']}")
        print("-" * 60)


def search_and_query(query: str) -> List[Dict]:
    """
    Main function to search for similar economic indicators based on a text query.
    
    Args:
        query: The search query string
        
    Returns:
        List of top 3 matching indicators with similarity scores
    """
    print(f"Processing query: '{query}'")
    
    # Load configuration
    config = load_config()
    db_config = config.get("database", {})
    
    # Create embedding from query
    print("Converting query to embedding...")
    query_embedding = create_query_embedding(query)
    
    if not query_embedding:
        print("Failed to create embedding for query.")
        return []
    
    # Initialize database connection
    print("Connecting to embedding database...")
    embedding_store = PostgresEmbeddingStore(db_config)
    
    try:
        # Search for similar embeddings
        print("Searching for similar indicators...")
        results = embedding_store.search_similar_embeddings(query_embedding, top_k=3)
        
        # Display results
        display_search_results(results, query)
        
        return results
        
    except Exception as e:
        print(f"Error during search: {e}")
        return []
    finally:
        # Close database connection
        embedding_store.close()


# ===== DATA EXTRACTION OPERATIONS =====

def get_indicator_metadata(indicator_id: str) -> Optional[Dict]:
    """
    Get metadata for an indicator from the data dictionary.
    
    Args:
        indicator_id: The FRED indicator ID to look up
        
    Returns:
        Dictionary with indicator metadata or None if not found
    """
    # Use path relative to the src directory
    dict_file_path = os.path.join(SRC_DIR, "excel-output", "FRED_DataDictionary.xlsx")
    
    try:
        # Load the data dictionary
        df = pd.read_excel(dict_file_path)
        
        # Find the indicator by ID
        indicator_row = df[df['Indicator ID'] == indicator_id]
        
        if indicator_row.empty:
            print(f"Indicator ID '{indicator_id}' not found in data dictionary.")
            return None
            
        # Convert to dictionary
        metadata = indicator_row.iloc[0].to_dict()
        return metadata
        
    except Exception as e:
        print(f"Error loading data dictionary: {e}")
        return None


def construct_excel_filename(geography: str, frequency: str) -> str:
    """
    Construct the Excel filename based on geography and frequency.
    Uses LLM normalization to match the naming convention used during ingestion.
    
    Args:
        geography: Country/geography name
        frequency: Frequency code (M, Q, A)
        
    Returns:
        Excel filename path using simplified naming format: {normalized_geography}_{frequency}.xlsx
    """
    # Map frequency codes to labels
    freq_labels = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}
    freq_label = freq_labels.get(frequency, frequency)
    
    # Use LLM normalization to match ingestion naming convention
    normalized_geography = llm_client.normalize_country(geography).replace(" ", "_")
    
    # Construct filename using simplified format (no fred_macro_data prefix)
    filename = os.path.join(SRC_DIR, "excel-output", f"{normalized_geography}_{freq_label}.xlsx")
    return filename


def find_indicator_column(excel_file: str, indicator_name: str, frequency: str) -> Optional[str]:
    """
    Find the column name for an indicator in the Excel file.
    
    Args:
        excel_file: Path to the Excel file
        indicator_name: Name of the indicator
        frequency: Frequency code (M, Q, A)
        
    Returns:
        Column name if found, None otherwise
    """
    try:
        # Load the Excel file to get column names
        df = pd.read_excel(excel_file, nrows=0)  # Only load headers
        columns = df.columns.tolist()
        
        # Expected column pattern: {indicator_name_without_parentheses}_{frequency}
        # Clean the indicator name (remove parentheses and normalize)
        clean_name = indicator_name.split('(')[0].strip().replace(" ", "_")
        expected_column = f"{clean_name}_{frequency}"
        
        # Look for exact match first
        if expected_column in columns:
            return expected_column
            
        # Look for partial matches (in case of slight naming differences)
        for col in columns:
            if clean_name.lower() in col.lower() and frequency in col:
                return col
                
        print(f"Column for indicator '{indicator_name}' not found in {excel_file}")
        print(f"Expected: {expected_column}")
        print(f"Available columns: {columns}")
        return None
        
    except Exception as e:
        print(f"Error reading Excel file {excel_file}: {e}")
        return None


def extract_time_series_data(excel_file: str, column_name: str) -> Optional[pd.Series]:
    """
    Extract time series data for a specific column from Excel file.
    
    Args:
        excel_file: Path to the Excel file
        column_name: Name of the column to extract
        
    Returns:
        Pandas Series with time series data or None if error
    """
    try:
        # Load the Excel file with date index
        df = pd.read_excel(excel_file, index_col=0, parse_dates=True)
        
        # Check if column exists
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in {excel_file}")
            return None
            
        # Extract the time series
        time_series = df[column_name].dropna()  # Remove NaN values
        
        return time_series
        
    except Exception as e:
        print(f"Error extracting data from {excel_file}: {e}")
        return None


def display_indicator_info(metadata: Dict, time_series: pd.Series) -> None:
    """
    Display information about the extracted indicator data.
    
    Args:
        metadata: Indicator metadata dictionary
        time_series: Time series data
    """
    print(f"\n{'='*80}")
    print(f"INDICATOR DATA EXTRACTED")
    print(f"{'='*80}")
    print(f"Indicator ID: {metadata.get('Indicator ID', 'N/A')}")
    print(f"Name: {metadata.get('Indicator Name', 'N/A')}")
    print(f"Description: {metadata.get('Description', 'N/A')}")
    print(f"Frequency: {metadata.get('Frequency', 'N/A')}")
    print(f"Geography: {metadata.get('Geography', 'N/A')}")
    print(f"\nData Range: {time_series.index[0]} to {time_series.index[-1]}")
    print(f"Total Observations: {len(time_series)}")
    print(f"Non-null Observations: {time_series.count()}")
    
    # Show recent data points
    print(f"\nRecent Data (Last 5 observations):")
    print("-" * 40)
    recent_data = time_series.tail(5)
    for date, value in recent_data.items():
        print(f"{date.strftime('%Y-%m-%d')}: {value}")
    print("-" * 80)


def extract_data(indicator_id: str) -> Optional[Tuple[Dict, pd.Series]]:
    """
    Main function to extract time series data for a given Indicator ID.
    
    Args:
        indicator_id: The FRED indicator ID to extract data for
        
    Returns:
        Tuple of (metadata_dict, time_series) or None if extraction fails
    """
    print(f"Extracting data for Indicator ID: '{indicator_id}'")
    
    # Step 1: Get indicator metadata from data dictionary
    print("Loading indicator metadata...")
    metadata = get_indicator_metadata(indicator_id)
    
    if not metadata:
        return None
    
    # Step 2: Determine file location based on metadata
    geography = metadata.get('Geography', '')
    frequency = metadata.get('Frequency', '')
    indicator_name = metadata.get('Indicator Name', '')
    
    if not all([geography, frequency, indicator_name]):
        print("Missing required metadata fields.")
        return None
    
    # Step 3: Construct Excel filename
    excel_file = construct_excel_filename(geography, frequency)
    print(f"Looking for data in: {excel_file}")
    
    # Check if file exists
    if not os.path.exists(excel_file):
        print(f"Excel file not found: {excel_file}")
        return None
    
    # Step 4: Find the correct column name
    print("Finding indicator column in Excel file...")
    column_name = find_indicator_column(excel_file, indicator_name, frequency)
    
    if not column_name:
        return None
    
    # Step 5: Extract time series data
    print(f"Extracting time series data from column: {column_name}")
    time_series = extract_time_series_data(excel_file, column_name)
    
    if time_series is None:
        return None
    
    # Step 6: Display results
    display_indicator_info(metadata, time_series)
    
    return metadata, time_series 