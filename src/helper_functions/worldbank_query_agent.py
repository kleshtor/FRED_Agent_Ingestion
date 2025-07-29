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
from helper_functions.worldbank_data_operations import (
    search_worldbank_database, extract_worldbank_data, create_worldbank_dataframe_preview
)
from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.worldbank_operations import (
    search_worldbank_series, fetch_worldbank_data, process_dictionary_embeddings,
    load_existing_data_dictionary, indicator_exists_in_dictionary,
    merge_time_series_data, ensure_complete_date_range
)
from helper_functions.core_utils import LLMClient, SRC_DIR, PROMPT_YAML_PATH


class SearchDatabaseArgs(BaseModel):
    """Arguments for World Bank database search tool"""
    query: str
    max_attempts: int = 3


class ExtractDataArgs(BaseModel):
    """Arguments for World Bank data extraction tool"""
    indicator_id: str


class DisplayPreviewArgs(BaseModel):
    """Arguments for World Bank data preview tool"""
    indicator_id: str
    preview_rows: int = 10


class IngestFromWorldBankArgs(BaseModel):
    """Arguments for World Bank data ingestion tool"""
    query: str
    countries: List[str] = ["USA"]


class WorldBankQueryAgent:
    """
    An intelligent query agent that can search local database, extract data,
    and ingest new data from World Bank as needed.
    """
    
    def __init__(self, config_path: str = None, prompt_path: str = None):
        print("🔧 Initializing WorldBankQueryAgent...")
        
        # Disable tracing to clean up output
        set_tracing_disabled(True)
        print("   ✓ Tracing disabled for cleaner output")
        
        # Load environment variables
        load_dotenv()
        print("   ✓ Environment variables loaded")
        
        # Set up paths
        self.config_path = config_path or os.path.join(SRC_DIR, "helper_functions", "worldbank_config.yaml")
        self.prompt_path = prompt_path or PROMPT_YAML_PATH
        print(f"   ✓ Config path: {self.config_path}")
        print(f"   ✓ Prompt path: {self.prompt_path}")
        
        # Load configuration and prompts
        self._load_config()
        self._load_prompts()
        
        # Initialize components
        print("   🔗 Connecting to PostgreSQL embedding store...")
        self.embedding_store = PostgresEmbeddingStore(self.db_config)
        print("   ✓ PostgreSQL database connection established")
        
        print("   🤖 Initializing LLM client...")
        self.llm_client = LLMClient(self.prompt_path)
        print("   ✓ LLM client ready")
        
        # Create the agent with tools
        print("   🛠️ Creating agent with tools...")
        self.agent = self._create_agent()
        print("   ✓ Agent created with 4 tools: search_database, extract_data, display_dataframe_preview, ingest_from_worldbank")
        
        print("✅ WorldBankQueryAgent initialization complete!")

    def _load_config(self):
        """Load configuration from worldbank_config.yaml"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.countries = config.get("countries", ["USA"])
            self.search_terms = config.get("search_terms", [])
            self.start_year = config.get("start_year", 1960)
            self.end_year = config.get("end_year", 2023)
            self.output_path = config.get("output_path", "excel-output/worldbank_macro_data.xlsx")
            self.dict_file_path = "excel-output/WorldBank_DataDictionary.xlsx"
            self.db_config = config.get("database", {})
            
            print(f"   ✓ Configuration loaded from {self.config_path}")
            print(f"   ✓ Countries: {self.countries}")
            print(f"   ✓ Search terms: {len(self.search_terms)} terms")
            print(f"   ✓ Date range: {self.start_year}-{self.end_year}")
            
        except Exception as e:
            print(f"   ❌ Error loading config: {e}")
            raise

    def _load_prompts(self):
        """Load prompts from YAML file"""
        try:
            with open(self.prompt_path, 'r') as f:
                prompts = yaml.safe_load(f)
            
            self.agent_config = prompts.get("worldbank_query_agent", {})
            print(f"   ✓ Agent prompts loaded from {self.prompt_path}")
            
        except Exception as e:
            print(f"   ❌ Error loading prompts: {e}")
            # Set default config if file doesn't exist
            self.agent_config = {}

    def _create_agent(self) -> Agent:
        """Create the OpenAI Agent with World Bank tools"""
        
        @function_tool
        def search_database(query: str, max_attempts: int = 3) -> str:
            """
            Search the local World Bank database for economic indicators matching the query.
            Returns detailed results with similarity scores for the agent to evaluate.
            """
            print(f"\n🔍 TOOL CALLED: search_database(query='{query}', max_attempts={max_attempts})")
            print("   📋 Tool Purpose: Find World Bank economic indicators in local database using semantic search")
            return self._search_database_impl(query, max_attempts)
        
        @function_tool
        def extract_data(indicator_id: str) -> str:
            """
            Extract time series data for a specific World Bank indicator ID.
            Returns formatted data with metadata.
            """
            print(f"\n📊 TOOL CALLED: extract_data(indicator_id='{indicator_id}')")
            print("   📋 Tool Purpose: Extract raw time series data and metadata for specific World Bank indicator")
            return self._extract_data_impl(indicator_id)
        
        @function_tool
        def display_dataframe_preview(indicator_id: str, preview_rows: int = 10) -> str:
            """
            Display a preview of the World Bank time series data for a specific indicator.
            Shows both metadata and a sample of the actual data in a formatted table.
            """
            print(f"\n📈 TOOL CALLED: display_dataframe_preview(indicator_id='{indicator_id}', preview_rows={preview_rows})")
            print("   📋 Tool Purpose: Create formatted preview of World Bank time series data for user display")
            return self._display_dataframe_preview_impl(indicator_id, preview_rows)
        
        @function_tool
        def ingest_from_worldbank(query: str, countries: List[str] = None) -> str:
            """
            Search World Bank database and ingest new economic indicators with full time series data.
            Returns status of ingestion process.
            """
            if countries is None:
                countries = self.countries
            print(f"\n🌐 TOOL CALLED: ingest_from_worldbank(query='{query}', countries={countries})")
            print("   📋 Tool Purpose: Search World Bank database for new indicators and ingest them locally")
            return self._ingest_from_worldbank_impl(query, countries)
        
        # Get agent instructions from YAML configuration
        agent_instructions = self.agent_config.get("WorldBankQueryAgent_Instructions", 
            """You are an intelligent economic data query agent for World Bank data. Your primary goal is to help users find and extract economic indicators from the World Bank database.

You have access to several tools:
1. search_database - Search the local database for World Bank economic indicators
2. extract_data - Extract time series data for specific indicators  
3. display_dataframe_preview - Display a formatted preview of time series data
4. ingest_from_worldbank - Search and ingest new data from World Bank database

Your workflow should be:
1. First, search the local database for indicators matching the user's query
2. INTELLIGENTLY EVALUATE the search results - look at the indicator names, descriptions, and similarity scores to determine if any are relevant to the user's request
3. If you find relevant indicators (even with moderate similarity), extract the data AND display a preview using display_dataframe_preview
4. If no relevant indicators are found, use the World Bank ingestion tool to search for new data
5. After ingestion, search the database again and evaluate the new results
6. Present results with data previews or apologize if no relevant data is found""")
        
        print(f"   ✓ Agent instructions loaded ({len(agent_instructions)} characters)")
        
        agent = Agent(
            name="World Bank Query Agent",
            instructions=agent_instructions,
            tools=[search_database, extract_data, display_dataframe_preview, ingest_from_worldbank]
        )
        
        return agent

    def _search_database_impl(self, query: str, max_attempts: int = 3) -> str:
        """Implementation of World Bank database search - returns all results for agent evaluation"""
        try:
            print(f"🔍 Starting World Bank database search for: '{query}'")
            print(f"   🎯 Strategy: Will try up to {max_attempts} query variations using LLM-generated alternatives")
            
            # Try different variations of the query
            search_attempts = 0
            all_results = []
            
            # Generate search variations using LLM
            print("   🧠 Generating search variations using LLM...")
            variations = self._generate_search_variations(query)
            print(f"   ✓ Generated {len(variations)} variations: {variations}")
            
            for variation in variations[:max_attempts]:
                search_attempts += 1
                print(f"\n   🔍 Search Attempt {search_attempts}/{max_attempts}: '{variation}'")
                
                results = search_worldbank_database(variation)
                if results:
                    print(f"      ✓ Found {len(results)} indicators for this variation")
                    all_results.extend(results)
                else:
                    print("      ❌ No results for this variation")
            
            # Remove duplicates based on indicator_id
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['indicator_id'] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result['indicator_id'])
            
            print(f"\n   📊 Search Summary:")
            print(f"      Total attempts: {search_attempts}")
            print(f"      Raw results: {len(all_results)}")
            print(f"      Unique results: {len(unique_results)}")
            
            if not unique_results:
                return "❌ No World Bank indicators found in local database. Consider using ingest_from_worldbank to search for new data."
            
            # Format results for agent evaluation
            result_text = f"✅ Found {len(unique_results)} unique World Bank indicators:\n\n"
            
            for i, result in enumerate(unique_results, 1):
                result_text += f"[{i}] Indicator ID: {result['indicator_id']}\n"
                result_text += f"    Name: {result['indicator_name']}\n"
                result_text += f"    Similarity Score: {result['similarity']:.4f}\n"
                result_text += f"    Geography: {result.get('geography', 'N/A')}\n"
                result_text += f"    Source: World Bank\n\n"
            
            result_text += "💡 EVALUATION GUIDANCE:\n"
            result_text += "- Similarity scores > 0.8 are excellent matches\n"
            result_text += "- Similarity scores > 0.6 are good matches worth considering\n"
            result_text += "- Similarity scores > 0.4 may be relevant depending on context\n"
            result_text += "- Consider the indicator names and descriptions when evaluating relevance\n"
            
            return result_text
            
        except Exception as e:
            error_msg = f"❌ Error during World Bank database search: {e}"
            print(f"   🚨 {error_msg}")
            return error_msg

    def _generate_search_variations(self, query: str) -> List[str]:
        """Generate search variations using LLM"""
        try:
            variations = self.llm_client.generate_worldbank_search_variations(query)
            # Always include the original query
            if query not in variations:
                variations.insert(0, query)
            return variations
        except Exception as e:
            print(f"   ⚠️  Error generating variations: {e}")
            return [query]  # Fallback to original query

    def _extract_data_impl(self, indicator_id: str) -> str:
        """Implementation of World Bank data extraction"""
        try:
            print(f"📊 Starting World Bank data extraction for: '{indicator_id}'")
            
            result = extract_worldbank_data(indicator_id)
            
            if result is None:
                return f"❌ Failed to extract data for World Bank indicator '{indicator_id}'. Check if the indicator exists in the database."
            
            metadata, time_series = result
            
            # Create summary
            summary = f"✅ Successfully extracted World Bank data for '{indicator_id}'\n\n"
            summary += f"Indicator: {metadata.get('Indicator Name', 'N/A')}\n"
            summary += f"Geography: {metadata.get('Geography', 'N/A')}\n"
            summary += f"Frequency: {metadata.get('Frequency', 'Annual')}\n"
            summary += f"Data Points: {len(time_series)}\n"
            summary += f"Date Range: {time_series.index.min().strftime('%Y')} to {time_series.index.max().strftime('%Y')}\n"
            summary += f"Latest Value: {time_series.iloc[-1]:.2f} ({time_series.index[-1].strftime('%Y')})\n\n"
            summary += "💡 Use display_dataframe_preview to see a formatted table of the data."
            
            return summary
            
        except Exception as e:
            error_msg = f"❌ Error extracting World Bank data for '{indicator_id}': {e}"
            print(f"   🚨 {error_msg}")
            return error_msg

    def _display_dataframe_preview_impl(self, indicator_id: str, preview_rows: int = 10) -> str:
        """Implementation of World Bank data preview display"""
        try:
            print(f"📈 Creating World Bank data preview for: '{indicator_id}'")
            
            result = extract_worldbank_data(indicator_id)
            
            if result is None:
                return f"❌ Failed to create preview for World Bank indicator '{indicator_id}'. Check if the indicator exists in the database."
            
            metadata, time_series = result
            
            # Create formatted preview
            preview = create_worldbank_dataframe_preview(metadata, time_series, preview_rows)
            
            return preview
            
        except Exception as e:
            error_msg = f"❌ Error creating World Bank data preview for '{indicator_id}': {e}"
            print(f"   🚨 {error_msg}")
            return error_msg

    def _ingest_from_worldbank_impl(self, query: str, countries: List[str]) -> str:
        """Implementation of World Bank data ingestion with full time series data"""
        try:
            print(f"🌐 Starting World Bank ingestion process...")
            print(f"   🎯 Query: '{query}'")
            print(f"   🌍 Countries: {countries}")
            print(f"   📋 Goal: Find new economic indicators from World Bank database")
            
            # Rephrase query for World Bank search
            print("   🧠 Step 1: Rephrasing query for World Bank database using LLM...")
            wb_query = self._rephrase_for_worldbank(query, countries)
            print(f"   ✅ Original query: '{query}'")
            print(f"   ✅ Target countries: {countries}")
            print(f"   ✅ World Bank-optimized query: '{wb_query}'")
            
            # Search World Bank
            print(f"   🔍 Step 2: Searching World Bank database...")
            results = search_worldbank_series(wb_query, max_results=10)
            
            # If no results, try fallback queries
            if results.empty:
                print(f"   ⚠️  No results for LLM query: '{wb_query}'")
                print("   🔄 Trying fallback queries...")
                
                # Generate fallback queries based on original query
                fallback_queries = []
                original_lower = query.lower()
                
                if 'gdp' in original_lower:
                    fallback_queries.extend(['GDP per capita', 'GDP growth', 'gross domestic product'])
                if 'inflation' in original_lower:
                    fallback_queries.extend(['inflation', 'consumer price index', 'price level'])
                if 'unemployment' in original_lower or 'employment' in original_lower:
                    fallback_queries.extend(['unemployment rate', 'employment', 'labor force'])
                if 'population' in original_lower:
                    fallback_queries.extend(['population growth', 'population total'])
                if 'trade' in original_lower:
                    fallback_queries.extend(['trade', 'exports', 'imports'])
                
                # If no specific fallbacks, use generic terms
                if not fallback_queries:
                    # Extract key economic terms from the original query
                    key_terms = [word for word in original_lower.split() 
                               if word not in ['per', 'capita', 'rate', 'growth', 'data', 'for', 'in', 'of', 'the']]
                    fallback_queries = key_terms[:3]  # Use first 3 key terms
                
                # Try each fallback query
                for fallback in fallback_queries[:3]:  # Limit to 3 attempts
                    print(f"      🔍 Trying fallback: '{fallback}'")
                    results = search_worldbank_series(fallback, max_results=10)
                    if not results.empty:
                        print(f"      ✅ Found {len(results)} results with fallback query!")
                        break
                    else:
                        print(f"      ❌ No results for: '{fallback}'")
            
            if results.empty:
                error_msg = f"❌ No results found in World Bank for query: '{wb_query}' or fallback queries"
                print(f"   🚨 {error_msg}")
                print("   💡 Try rephrasing your query or using more general economic terms")
                return error_msg
            
            print(f"   ✅ Found {len(results)} potential indicators in World Bank")
            
            # Process and ingest the results with full time series data
            print("   🔄 Step 3: Processing and filtering results...")
            ingested_count = 0
            existing_dict = load_existing_data_dictionary(self.dict_file_path)
            
            print(f"   📚 Existing dictionary has {len(existing_dict)} indicators")
            
            new_indicators = []
            
            for _, row in results.iterrows():
                indicator_id = row['id']
                indicator_name = row['name']
                
                # Skip if already exists
                if indicator_exists_in_dictionary(indicator_id, existing_dict):
                    print(f"      ⏭️  Skipping {indicator_id} (already exists)")
                    continue
                
                print(f"      🔄 Processing {indicator_id}: {indicator_name[:50]}...")
                
                # Create label
                clean_name = str(indicator_name).split('(')[0].strip().replace(' ', '_').replace(',', '').replace(':', '')[:30]
                label = f"{clean_name}_A"
                
                # Fetch data for all countries
                ts_data = fetch_worldbank_data(indicator_id, label, countries, self.start_year, self.end_year)
                
                if not ts_data.empty:
                    # Ensure excel-output directory exists
                    import os
                    os.makedirs("excel-output", exist_ok=True)
                    
                    # Save to appropriate Excel files
                    for country in countries:
                        normalized_country = self.llm_client.normalize_worldbank_country(country)
                        filename = f"excel-output/{normalized_country.replace(' ', '_')}_Annual.xlsx"
                        
                        # Load existing data or create new
                        from helper_functions.worldbank_operations import load_existing_excel_data
                        existing_data = load_existing_excel_data(filename) if os.path.exists(filename) else pd.DataFrame()
                        
                        # Get country-specific columns
                        country_columns = [col for col in ts_data.columns if col.startswith(f"{country}_")]
                        if country_columns:
                            country_data = ts_data[country_columns]
                            merged_data = merge_time_series_data(existing_data, country_data)
                            complete_data = ensure_complete_date_range(merged_data, self.start_year, self.end_year)
                            complete_data.to_excel(filename, index=True)
                    
                    # Add to dictionary
                    new_indicators.append({
                        "Indicator ID": indicator_id,
                        "Indicator Name": indicator_name,
                        "Geography": ", ".join(countries),
                        "Frequency": "Annual",
                        "Units": row.get('units', 'Various'),
                        "Source": "World Bank",
                        "Last Updated": datetime.now().strftime("%Y-%m-%d")
                    })
                    
                    ingested_count += 1
                    print(f"      ✅ Successfully ingested {indicator_id}")
                else:
                    print(f"      ⚠️  No data available for {indicator_id}")
            
            # Update dictionary
            if new_indicators:
                self._update_dictionary(new_indicators)
                self._generate_embeddings()
            
            success_msg = f"✅ World Bank ingestion complete!\n"
            success_msg += f"   📊 Processed {len(results)} potential indicators\n"
            success_msg += f"   ✅ Successfully ingested {ingested_count} new indicators\n"
            success_msg += f"   📚 Updated data dictionary with {len(new_indicators)} entries\n\n"
            success_msg += f"💡 Try searching the database again to find the newly ingested indicators."
            
            print(f"   {success_msg}")
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Error during World Bank ingestion: {e}"
            print(f"   🚨 {error_msg}")
            return error_msg

    def _rephrase_for_worldbank(self, query: str, countries: List[str]) -> str:
        """Rephrase query for World Bank search using LLM"""
        try:
            return self.llm_client.rephrase_for_worldbank(query, countries)
        except Exception as e:
            print(f"   ⚠️  Error rephrasing query: {e}")
            return query  # Fallback to original query

    def _update_dictionary(self, new_indicators: List[Dict]):
        """Update World Bank data dictionary"""
        if not new_indicators:
            return
        
        # Ensure excel-output directory exists
        import os
        os.makedirs("excel-output", exist_ok=True)
        
        existing = pd.read_excel(self.dict_file_path) if os.path.exists(self.dict_file_path) else pd.DataFrame()
        new = pd.DataFrame(new_indicators)
        
        if not existing.empty:
            combined = pd.concat([existing, new], ignore_index=True).drop_duplicates("Indicator ID", keep="last")
        else:
            combined = new
        
        combined.to_excel(self.dict_file_path, index=False)
        print(f"   ✓ Updated dictionary with {len(new_indicators)} new entries")

    def _generate_embeddings(self):
        """Generate embeddings for World Bank data dictionary"""
        try:
            stats = process_dictionary_embeddings(
                dict_file_path=self.dict_file_path,
                embedding_store=self.embedding_store
            )
            print(f"   ✓ Embedding generation complete: {stats}")
        except Exception as e:
            print(f"   ⚠️  Error generating embeddings: {e}")

    async def run(self, query: str) -> str:
        """Run the World Bank query agent with a user query"""
        try:
            print(f"\n🚀 Running World Bank Query Agent with query: '{query}'")
            
            # Use the correct Runner API pattern from FRED agent
            result = await Runner.run(self.agent, query)
            
            return result.final_output
            
        except Exception as e:
            error_msg = f"❌ Error running World Bank agent: {e}"
            print(error_msg)
            return error_msg

    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'embedding_store'):
                self.embedding_store.close()
            print("✅ WorldBankQueryAgent resources cleaned up")
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")

    def run_query_sync(self, query: str) -> str:
        """Synchronous wrapper for running World Bank queries - matches FRED pattern"""
        print(f"\n🚀 **STARTING WORLD BANK QUERY PROCESSING (SYNC MODE)**")
        print(f"📝 User Query: '{query}'")
        print(f"🤖 Delegating to OpenAI Agent for intelligent processing...")
        print("=" * 80)
        
        try:
            # Check if we're in a Jupyter environment with an existing event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, we're in Jupyter/IPython with an active loop
                print("   🔄 Detected Jupyter environment - applying nest_asyncio patch...")
                import nest_asyncio
                nest_asyncio.apply()
                result = Runner.run_sync(self.agent, query)
            except RuntimeError:
                # No running loop, we can use normal sync method
                print("   ✅ Standard Python environment - using normal sync method...")
                result = Runner.run_sync(self.agent, query)
            
            print("\n" + "=" * 80)
            print("🎯 **WORLD BANK QUERY PROCESSING COMPLETED**")
            print(f"📋 Final Response Length: {len(result.final_output)} characters")
            print("✅ Agent successfully processed the query and generated response")
            
            return result.final_output
        except Exception as e:
            error_msg = f"❌ Error processing World Bank query: {str(e)}"
            print(f"\n🚨 **WORLD BANK QUERY PROCESSING FAILED**")
            print(f"❌ Error: {error_msg}")
            return error_msg


# Convenience function for running the agent
async def run_worldbank_query(query: str, config_path: str = None, prompt_path: str = None) -> str:
    """
    Convenience function to run a World Bank query
    
    Args:
        query: The user's query
        config_path: Path to worldbank_config.yaml (optional)
        prompt_path: Path to prompts.yaml (optional)
        
    Returns:
        Agent response as string
    """
    agent = None
    try:
        agent = WorldBankQueryAgent(config_path, prompt_path)
        result = await agent.run(query)
        return result
    finally:
        if agent:
            agent.close()


# Synchronous wrapper for convenience
def run_worldbank_query_sync(query: str, config_path: str = None, prompt_path: str = None) -> str:
    """
    Synchronous wrapper for running World Bank queries
    
    Args:
        query: The user's query
        config_path: Path to worldbank_config.yaml (optional)
        prompt_path: Path to prompts.yaml (optional)
        
    Returns:
        Agent response as string
    """
    return asyncio.run(run_worldbank_query(query, config_path, prompt_path)) 