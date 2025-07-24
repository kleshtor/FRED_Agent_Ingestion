import os
import yaml
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool
from agents.tracing import set_tracing_disabled
from pydantic import BaseModel

# Local imports
from helper_functions.search_and_query import search_and_query
from helper_functions.extract_data import extract_data
from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.FRED_helper import (
    search_fred_series, fetch_series_data, process_dictionary_embeddings,
    load_existing_data_dictionary, indicator_exists_in_dictionary,
    merge_time_series_data, ensure_complete_date_range
)
from helper_functions.LLM_utils import LLMClient
from helper_functions.path_config import SRC_DIR, PROMPT_YAML_PATH


class SearchDatabaseArgs(BaseModel):
    """Arguments for database search tool"""
    query: str
    max_attempts: int = 3


class ExtractDataArgs(BaseModel):
    """Arguments for data extraction tool"""
    indicator_id: str


class IngestFromFredArgs(BaseModel):
    """Arguments for FRED ingestion tool"""
    query: str
    country: str = "USA"


class DisplayDataframeArgs(BaseModel):
    """Arguments for dataframe display tool"""
    indicator_id: str
    preview_rows: int = 10


class QueryAgent:
    """
    An intelligent query agent that can search local database, extract data,
    and ingest new data from FRED as needed.
    """
    
    def __init__(self, config_path: str = None, prompt_path: str = None):
        # Disable tracing to clean up output
        set_tracing_disabled(True)
        
        # Load environment variables
        load_dotenv()
        
        # Set up paths
        self.config_path = config_path or os.path.join(SRC_DIR, "helper_functions", "FRED.yaml")
        self.prompt_path = prompt_path or PROMPT_YAML_PATH
        
        # Load configuration and prompts
        self._load_config()
        self._load_prompts()
        
        # Initialize components
        self.embedding_store = PostgresEmbeddingStore(self.db_config)
        self.llm_client = LLMClient(self.prompt_path)
        
        # Create the agent with tools
        self.agent = self._create_agent()
    
    def _load_config(self):
        """Load FRED configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.api_key = config.get("api_key")
        self.db_config = config.get("database", {})
        self.dict_file_path = os.path.join(SRC_DIR, "excel-output", "FRED_DataDictionary.xlsx")
        
        if not self.api_key:
            raise ValueError("FRED API key is missing in config")
    
    def _load_prompts(self):
        """Load agent prompts from YAML"""
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        self.agent_config = prompts.get("query_agent", {})
        self.max_search_attempts = self.agent_config.get("max_search_attempts", 3)
    
    def _create_agent(self) -> Agent:
        """Create the OpenAI Agent with tools"""
        
        @function_tool
        def search_database(query: str, max_attempts: int = 3) -> str:
            """
            Search the local database for economic indicators matching the query.
            Returns detailed results with similarity scores for the agent to evaluate.
            """
            return self._search_database_impl(query, max_attempts)
        
        @function_tool
        def extract_data(indicator_id: str) -> str:
            """
            Extract time series data for a specific indicator ID.
            Returns formatted data with metadata.
            """
            return self._extract_data_impl(indicator_id)
        
        @function_tool
        def display_dataframe_preview(indicator_id: str, preview_rows: int = 10) -> str:
            """
            Display a preview of the time series data for a specific indicator.
            Shows both metadata and a sample of the actual data in a formatted table.
            """
            return self._display_dataframe_preview_impl(indicator_id, preview_rows)
        
        @function_tool
        def ingest_from_fred(query: str, country: str = "USA") -> str:
            """
            Search FRED database and ingest new economic indicators with full time series data.
            Returns status of ingestion process.
            """
            return self._ingest_from_fred_impl(query, country)
        
        # Get agent instructions from YAML configuration
        agent_instructions = self.agent_config.get("QueryAgent_Instructions", "")
        
        agent = Agent(
            name="FRED Query Agent",
            instructions=agent_instructions,
            tools=[search_database, extract_data, display_dataframe_preview, ingest_from_fred]
        )
        
        return agent
    
    def _search_database_impl(self, query: str, max_attempts: int = 3) -> str:
        """Implementation of database search - returns all results for agent evaluation"""
        try:
            print(f"🔍 Searching database for: '{query}'")
            
            # Try different variations of the query
            search_attempts = 0
            all_results = []
            
            # Generate search variations using LLM
            variations = self._generate_search_variations(query)
            
            for variation in variations[:max_attempts]:
                search_attempts += 1
                print(f"  Attempt {search_attempts}: '{variation}'")
                
                results = search_and_query(variation)
                if results:
                    all_results.extend(results)
            
            # Remove duplicates and sort by similarity
            unique_results = {}
            for result in all_results:
                indicator_id = result['indicator_id']
                if indicator_id not in unique_results or result['similarity'] > unique_results[indicator_id]['similarity']:
                    unique_results[indicator_id] = result
            
            sorted_results = sorted(unique_results.values(), key=lambda x: x['similarity'], reverse=True)
            
            if sorted_results:
                response = f"Found {len(sorted_results)} indicators from database search:\n\n"
                for i, result in enumerate(sorted_results[:5], 1):  # Show top 5 results
                    response += f"{i}. **{result['indicator_name']}** (ID: {result['indicator_id']})\n"
                    response += f"   Similarity Score: {result['similarity']:.3f}\n"
                    response += f"   Description: {result['description']}\n\n"
                
                response += "Please evaluate these indicators and determine if any are relevant to the user's query. "
                response += "If relevant, use the display_dataframe_preview tool to show the data to the user."
                return response
            else:
                return f"No indicators found in database after searching {search_attempts} variations. Consider using FRED ingestion to find new data."
                
        except Exception as e:
            return f"❌ Error searching database: {str(e)}"
    
    def _extract_data_impl(self, indicator_id: str) -> str:
        """Implementation of data extraction"""
        try:
            print(f"📊 Extracting data for indicator: {indicator_id}")
            
            result = extract_data(indicator_id)
            if result is None:
                return f"❌ Failed to extract data for indicator '{indicator_id}'. Please check the indicator ID."
            
            metadata, time_series = result
            
            # Format the response
            response = f"✅ **Data Extracted Successfully**\n\n"
            response += f"**Indicator:** {metadata.get('Indicator Name', 'N/A')}\n"
            response += f"**ID:** {metadata.get('Indicator ID', 'N/A')}\n"
            response += f"**Geography:** {metadata.get('Geography', 'N/A')}\n"
            response += f"**Frequency:** {metadata.get('Frequency', 'N/A')}\n"
            response += f"**Description:** {metadata.get('Description', 'N/A')}\n\n"
            
            response += f"**Data Range:** {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}\n"
            response += f"**Total Observations:** {len(time_series)}\n"
            response += f"**Non-null Values:** {time_series.count()}\n\n"
            
            # Show recent data
            response += "**Recent Data (Last 5 observations):**\n"
            recent_data = time_series.tail(5)
            for date, value in recent_data.items():
                response += f"- {date.strftime('%Y-%m-%d')}: {value}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error extracting data: {str(e)}"
    
    def _display_dataframe_preview_impl(self, indicator_id: str, preview_rows: int = 10) -> str:
        """Implementation of dataframe preview display"""
        try:
            print(f"📊 Displaying preview for indicator: {indicator_id}")
            
            # Extract the data first
            result = extract_data(indicator_id)
            if result is None:
                return f"❌ Failed to extract data for indicator '{indicator_id}'. Please check the indicator ID."
            
            metadata, time_series = result
            
            # Create a DataFrame for better display
            df = pd.DataFrame({
                'Date': time_series.index,
                'Value': time_series.values
            })
            
            # Format the response with metadata and data preview
            response = f"📊 **Data Preview: {metadata.get('Indicator Name', 'N/A')}**\n"
            response += "=" * 80 + "\n\n"
            
            # Metadata section
            response += f"**Indicator ID:** {metadata.get('Indicator ID', 'N/A')}\n"
            response += f"**Geography:** {metadata.get('Geography', 'N/A')}\n"
            response += f"**Frequency:** {metadata.get('Frequency', 'N/A')}\n"
            response += f"**Description:** {metadata.get('Description', 'N/A')}\n\n"
            
            # Data summary
            response += f"**Data Summary:**\n"
            response += f"- Range: {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}\n"
            response += f"- Total Observations: {len(time_series)}\n"
            response += f"- Non-null Values: {time_series.count()}\n"
            response += f"- Latest Value: {time_series.iloc[-1]:.2f} ({time_series.index[-1].strftime('%Y-%m-%d')})\n\n"
            
            # Data preview table
            response += f"**Data Preview (Last {preview_rows} observations):**\n"
            response += "-" * 50 + "\n"
            
            # Get last N rows for preview
            preview_data = df.tail(preview_rows)
            
            # Format as a nice table
            response += f"{'Date':<12} {'Value':<15}\n"
            response += "-" * 27 + "\n"
            
            for _, row in preview_data.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d')
                value_str = f"{row['Value']:.2f}" if pd.notna(row['Value']) else "N/A"
                response += f"{date_str:<12} {value_str:<15}\n"
            
            response += "-" * 50 + "\n"
            
            # Statistics
            response += f"\n**Quick Statistics:**\n"
            response += f"- Mean: {time_series.mean():.2f}\n"
            response += f"- Median: {time_series.median():.2f}\n"
            response += f"- Min: {time_series.min():.2f} ({time_series.idxmin().strftime('%Y-%m-%d')})\n"
            response += f"- Max: {time_series.max():.2f} ({time_series.idxmax().strftime('%Y-%m-%d')})\n"
            response += f"- Standard Deviation: {time_series.std():.2f}\n"
            
            response += "\n" + "=" * 80
            
            return response
            
        except Exception as e:
            return f"❌ Error displaying dataframe preview: {str(e)}"
    
    def _ingest_from_fred_impl(self, query: str, country: str = "USA") -> str:
        """Implementation of FRED data ingestion with full time series data"""
        try:
            print(f"🌐 Ingesting from FRED: '{query}' for {country}")
            
            # Rephrase query for FRED search
            fred_query = self._rephrase_for_fred(query)
            print(f"  Rephrased query: '{fred_query}'")
            
            # Search FRED
            results = search_fred_series(fred_query, self.api_key, max_results=10)
            
            if results.empty:
                return f"❌ No results found in FRED for query: '{fred_query}'"
            
            # Process and ingest the results with full time series data
            ingested_count = 0
            normalized_country = self.llm_client.normalize_country(country)
            existing_dict = load_existing_data_dictionary(self.dict_file_path)
            
            new_indicators = []
            
            # Initialize data storage by frequency
            data_by_freq = {"M": {}, "Q": {}, "A": {}}
            
            for _, row in results.iterrows():
                series_id = row.get("id")
                title = row.get("title")
                frequency = row.get("frequency")
                
                if not all([series_id, title, frequency]):
                    continue
                
                # Check if already exists
                if indicator_exists_in_dictionary(series_id, existing_dict):
                    continue
                
                # Generate description
                description = self.llm_client.generate_description(title, frequency, normalized_country)
                
                # Map frequency
                freq_map = {"Monthly": "M", "Quarterly": "Q", "Annual": "A"}
                freq_code = freq_map.get(frequency)
                
                if freq_code:
                    # Fetch the actual time series data
                    print(f"  Fetching time series for {series_id}: {title}")
                    
                    start_date = "1900-01-01"
                    end_date = datetime.today().strftime("%Y-%m-%d")
                    
                    # Create label for the series
                    label = title.split('(')[0].strip().replace(" ", "_") + f"_{freq_code}"
                    
                    # Fetch the time series data
                    time_series_df = fetch_series_data(series_id, label, start_date, end_date, freq_code)
                    
                    if not time_series_df.empty:
                        # Store in data_by_freq structure
                        if freq_code not in data_by_freq:
                            data_by_freq[freq_code] = {}
                        data_by_freq[freq_code][label] = time_series_df
                    
                    new_indicators.append({
                        "Indicator Name": title,
                        "Indicator ID": series_id,
                        "Description": description,
                        "Frequency": freq_code,
                        "Geography": normalized_country
                    })
                    ingested_count += 1
            
            if new_indicators:
                # Update dictionary
                new_df = pd.DataFrame(new_indicators)
                
                if os.path.exists(self.dict_file_path):
                    existing_df = pd.read_excel(self.dict_file_path)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df
                
                combined_df.to_excel(self.dict_file_path, index=False)
                
                # Create Excel files with time series data
                excel_files_created = []
                for freq_code, series_dict in data_by_freq.items():
                    if series_dict:  # Only if we have data for this frequency
                        freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
                        filename = f"excel-output/{normalized_country.replace(' ', '_')}_{freq_label}.xlsx"
                        full_path = os.path.join(SRC_DIR, filename)
                        
                        # Load existing data if file exists
                        if os.path.exists(full_path):
                            existing_data = pd.read_excel(full_path, index_col=0, parse_dates=True)
                        else:
                            existing_data = pd.DataFrame()
                        
                        # Merge all new series for this frequency
                        combined_data = existing_data
                        for label, series_df in series_dict.items():
                            combined_data = merge_time_series_data(combined_data, series_df)
                        
                        # Ensure complete date range and save
                        if not combined_data.empty:
                            complete_data = ensure_complete_date_range(combined_data, freq_code, end_date)
                            complete_data.to_excel(full_path, index=True)
                            excel_files_created.append(filename)
                            print(f"  Created/Updated: {filename}")
                
                # Generate embeddings
                stats = process_dictionary_embeddings(self.dict_file_path, self.embedding_store)
                
                response = f"✅ **FRED Ingestion Successful**\n\n"
                response += f"**Query:** '{fred_query}'\n"
                response += f"**Results Found:** {len(results)}\n"
                response += f"**New Indicators Added:** {ingested_count}\n"
                response += f"**Excel Files Created/Updated:** {len(excel_files_created)}\n"
                if excel_files_created:
                    response += f"**Files:** {', '.join(excel_files_created)}\n"
                response += f"**Embeddings Processed:** {stats.get('processed', 0)}\n\n"
                response += "You can now search the database again for better matches!"
                
                return response
            else:
                return f"❌ No new indicators found or all already exist in database."
                
        except Exception as e:
            return f"❌ Error during FRED ingestion: {str(e)}"
    
    def _generate_search_variations(self, query: str) -> List[str]:
        """Generate search variations using LLM"""
        try:
            template = self.agent_config.get("search_variations_template", "")
            prompt = template.format(query=query)
            
            # Use LLM to generate variations
            response = self.llm_client.client.chat.completions.create(
                model=self.agent_config["model_config"].get("model", "gpt-4o"),
                temperature=self.agent_config["model_config"].get("temperature", 0.3),
                max_tokens=self.agent_config["model_config"].get("max_tokens", 150),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            # Always include original query
            if query not in variations:
                variations.insert(0, query)
            
            return variations[:3]  # Max 3 variations
            
        except Exception as e:
            print(f"Error generating variations: {e}")
            return [query]  # Fallback to original query
    
    def _rephrase_for_fred(self, query: str) -> str:
        """Rephrase query for FRED database search"""
        try:
            template = self.agent_config.get("fred_query_rephrase_template", "")
            prompt = template.format(query=query)
            
            response = self.llm_client.client.chat.completions.create(
                model=self.agent_config["model_config"].get("model", "gpt-4o"),
                temperature=self.agent_config["model_config"].get("temperature", 0.3),
                max_tokens=self.agent_config["model_config"].get("max_tokens", 150),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            rephrased = response.choices[0].message.content.strip()
            return rephrased if rephrased else query
            
        except Exception as e:
            print(f"Error rephrasing query: {e}")
            return query  # Fallback to original query
    
    async def run_query(self, user_query: str) -> str:
        """
        Main method to process user queries
        """
        print(f"\n🤖 Processing query: '{user_query}'")
        
        try:
            result = await Runner.run(self.agent, user_query)
            return result.final_output
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def run_query_sync(self, user_query: str) -> str:
        """
        Synchronous version of run_query - works in Jupyter notebooks
        """
        print(f"\n🤖 Processing query: '{user_query}'")
        
        try:
            # Check if we're in a Jupyter environment with an existing event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, we're in Jupyter/IPython with an active loop
                import nest_asyncio
                nest_asyncio.apply()
                result = Runner.run_sync(self.agent, user_query)
            except RuntimeError:
                # No running loop, we can use normal sync method
                result = Runner.run_sync(self.agent, user_query)
            
            return result.final_output
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"


# Main function for testing
async def main():
    """Test the query agent"""
    
    # Example queries to test
    test_queries = [
        "Show me unemployment data for the US",
        "I need inflation statistics",
        "What's the GDP growth rate?",
        "Employment trends in manufacturing"
    ]
    
    # Initialize agent
    agent = QueryAgent()
    
    print("🚀 FRED Query Agent initialized!")
    print("=" * 60)
    
    # Interactive mode
    while True:
        try:
            user_input = input("\n💬 Enter your query (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process the query
            response = await agent.run_query(user_input)
            print(f"\n🤖 Agent Response:\n{response}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 