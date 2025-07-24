import os
import yaml
import openai
from typing import List, Dict
from dotenv import load_dotenv

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.path_config import SRC_DIR


def load_config() -> Dict:
    """
    Load configuration from FRED.yaml file.
    
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(SRC_DIR, "helper_functions", "FRED.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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