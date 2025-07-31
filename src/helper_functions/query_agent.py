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
from helper_functions.data_operations import search_and_query, extract_data
from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.fred_operations import (
    search_fred_series, fetch_series_data, process_dictionary_embeddings,
    load_existing_data_dictionary, indicator_exists_in_dictionary,
    merge_time_series_data, ensure_complete_date_range
)
from helper_functions.llm_utils import LLMClient
from helper_functions.path_config import SRC_DIR, PROMPT_YAML_PATH


class SearchDatabaseArgs(BaseModel):
    """Arguments for database search tool"""
    query: str
    max_attempts: int = 3


class ExtractDataArgs(BaseModel):
    """Arguments for data extraction tool"""
    indicator_id: str


class DelegateToSearchAndIngestAgentArgs(BaseModel):
    """Arguments for delegating to Search and Ingest Agent"""
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
    
    def __init__(self, config_path: str = None, prompt_path: str = None, verbose: bool = None):
        print("🔧 Initializing QueryAgent...")
        
        # Set up paths
        self.config_path = config_path or os.path.join(SRC_DIR, "helper_functions", "config.yaml")
        self.prompt_path = prompt_path or PROMPT_YAML_PATH
        print(f"   ✓ Config path: {self.config_path}")
        print(f"   ✓ Prompt path: {self.prompt_path}")
        
        # Load configuration first to get verbosity setting
        self._load_config()
        
        # Determine verbosity: explicit parameter overrides config, otherwise use config
        if verbose is not None:
            self.verbose = verbose
            print(f"   🔍 Verbose mode set to: {verbose} (from parameter)")
        else:
            self.verbose = self.agent_config.get("verbose", False)
            print(f"   🔍 Verbose mode set to: {self.verbose} (from config)")
        
        # Configure tracing based on verbose mode
        if self.verbose:
            print("   🔍 Verbose mode enabled - agent thinking will be visible")
            # Don't disable tracing - let it show the agent's thought process
        else:
            # Disable tracing to clean up output
            set_tracing_disabled(True)
            print("   ✓ Tracing disabled for cleaner output")
        
        # Load environment variables
        load_dotenv()
        print("   ✓ Environment variables loaded")
        
        # Load prompts
        self._load_prompts()
        
        # Initialize components
        print("   🔗 Connecting to PostgreSQL embedding store...")
        self.embedding_store = PostgresEmbeddingStore(self.db_config)
        print("   ✓ Database connection established")
        
        print("   🤖 Initializing LLM client...")
        self.llm_client = LLMClient(self.prompt_path)
        print("   ✓ LLM client ready")
        
        # Initialize Search and Ingest Agent for workflow delegation
        print("   🏗️ Initializing Search and Ingest Agent for delegation...")
        from helper_functions.search_and_ingest_agent import SearchAndIngestAgent
        self.search_and_ingest_agent = SearchAndIngestAgent(self.config_path, self.prompt_path)
        print("   ✓ Search and Ingest Agent ready for workflow delegation")
        
        # Create the agent with tools
        print("   🛠️ Creating agent with tools...")
        self.agent = self._create_agent()
        print("   ✓ Agent created with 4 tools: search_database, extract_data, display_dataframe_preview, delegate_to_search_and_ingest_agent")
        
        print("✅ QueryAgent initialization complete!")
    
    def _load_config(self):
        """Load FRED configuration"""
        print("   📄 Loading FRED configuration...")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.api_key = config.get("api_key")
        self.db_config = config.get("database", {})
        self.agent_config = config.get("agent", {})
        self.dict_file_path = os.path.join(SRC_DIR, "excel-output", "FRED_DataDictionary.xlsx")
        
        print(f"   ✓ FRED API key loaded: {'***' + self.api_key[-4:] if self.api_key else 'MISSING'}")
        print(f"   ✓ Database config: {self.db_config.get('host', 'localhost')}:{self.db_config.get('port', 5432)}")
        print(f"   ✓ Dictionary path: {self.dict_file_path}")
        
        if not self.api_key:
            raise ValueError("FRED API key is missing in config")
    
    def _load_prompts(self):
        """Load agent prompts from YAML"""
        print("   📝 Loading agent prompts...")
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        self.prompt_config = prompts.get("query_agent", {})
        self.max_search_attempts = self.prompt_config.get("max_search_attempts", 3)
        
        print(f"   ✓ Agent prompts loaded")
        print(f"   ✓ Max search attempts: {self.max_search_attempts}")
        print(f"   ✓ Available prompt templates: {list(self.prompt_config.keys())}")
    
    def _create_agent(self) -> Agent:
        """Create the OpenAI Agent with tools"""
        
        @function_tool
        def search_database(query: str, max_attempts: int = 3) -> str:
            """
            Search the local database for economic indicators matching the query.
            Returns detailed results with similarity scores for the agent to evaluate.
            """
            print(f"\n🔍 TOOL CALLED: search_database(query='{query}', max_attempts={max_attempts})")
            print("   📋 Tool Purpose: Find economic indicators in local database using semantic search")
            return self._search_database_implementation(query, max_attempts)
        
        @function_tool
        def extract_data(indicator_id: str) -> str:
            """
            Extract time series data for a specific indicator ID.
            Returns formatted data with metadata.
            """
            print(f"\n📊 TOOL CALLED: extract_data(indicator_id='{indicator_id}')")
            print("   📋 Tool Purpose: Extract raw time series data and metadata for specific indicator")
            return self._extract_data_implementation(indicator_id)
        
        @function_tool
        def display_dataframe_preview(indicator_id: str, preview_rows: int = 10) -> str:
            """
            Display a preview of the time series data for a specific indicator.
            Shows both metadata and a sample of the actual data in a formatted table.
            """
            print(f"\n📈 TOOL CALLED: display_dataframe_preview(indicator_id='{indicator_id}', preview_rows={preview_rows})")
            print("   📋 Tool Purpose: Create formatted preview of time series data for user display")
            return self._display_dataframe_preview_implementation(indicator_id, preview_rows)
        
        @function_tool
        def delegate_to_search_and_ingest_agent(query: str, country: str = "USA") -> str:
            """
            Delegate to Search and Ingest Agent to search and ingest new economic indicators from FRED database.
            Use this when no relevant indicators are found in the local database search.
            After delegation, you should search the database again to find the newly ingested data.
            """
            print(f"\n🤝 TOOL CALLED: delegate_to_search_and_ingest_agent(query='{query}', country='{country}')")
            print("   📋 Tool Purpose: Delegate to Search and Ingest Agent for specialized data ingestion")
            return self._delegate_to_search_and_ingest_agent_implementation(query, country)
        
        # Get agent instructions from YAML configuration
        agent_instructions = self.prompt_config.get("QueryAgent_Instructions", "")
        print(f"   ✓ Agent instructions loaded ({len(agent_instructions)} characters)")
        
        agent = Agent(
            name="FRED Query Agent",
            instructions=agent_instructions,
            tools=[search_database, extract_data, display_dataframe_preview, delegate_to_search_and_ingest_agent]
        )
        
        return agent
    
    def _search_database_implementation(self, query: str, max_attempts: int = 3) -> str:
        """Implementation of database search - returns all results for agent evaluation"""
        try:
            print(f"🔍 Starting database search for: '{query}'")
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
                
                results = search_and_query(variation)
                if results:
                    print(f"      ✓ Found {len(results)} indicators for this variation")
                    all_results.extend(results)
                else:
                    print("      ❌ No results for this variation")
            
            print(f"\n   📊 Search Summary: Found {len(all_results)} total results across all variations")
            
            # Remove duplicates and sort by similarity
            print("   🔄 Processing results: removing duplicates and sorting by similarity...")
            unique_results = {}
            for result in all_results:
                indicator_id = result['indicator_id']
                if indicator_id not in unique_results or result['similarity'] > unique_results[indicator_id]['similarity']:
                    unique_results[indicator_id] = result
            
            sorted_results = sorted(unique_results.values(), key=lambda x: x['similarity'], reverse=True)
            print(f"   ✓ After deduplication: {len(sorted_results)} unique indicators")
            
            if sorted_results:
                print(f"   📈 Top similarity scores: {[f'{r['similarity']:.3f}' for r in sorted_results[:3]]}")
                
                response = f"🎯 **SEARCH RESULTS ANALYSIS**\n"
                response += f"Found {len(sorted_results)} unique indicators from {search_attempts} search variations.\n\n"
                
                response += "**EVALUATION CRITERIA FOR AGENT:**\n"
                response += "- Look beyond similarity scores - focus on indicator names and descriptions\n"
                response += "- Consider if indicator semantically matches user intent\n"
                response += "- Moderate scores (0.5-0.7) can still be highly relevant\n"
                response += "- Examples: 'UNRATE' is clearly unemployment data, 'CPIAUCSL' is clearly inflation data\n\n"
                
                response += "**TOP CANDIDATES:**\n"
                for i, result in enumerate(sorted_results[:5], 1):  # Show top 5 results
                    response += f"{i}. **{result['indicator_name']}** (ID: {result['indicator_id']})\n"
                    response += f"   📊 Similarity Score: {result['similarity']:.3f}\n"
                    response += f"   📝 Description: {result['description']}\n"
                    
                    # Add reasoning hints for the agent
                    name_lower = result['indicator_name'].lower()
                    desc_lower = result['description'].lower()
                    query_lower = query.lower()
                    
                    reasoning_hints = []
                    if any(word in name_lower for word in ['unemployment', 'jobless', 'employment']):
                        if any(word in query_lower for word in ['unemployment', 'jobless', 'employment', 'job']):
                            reasoning_hints.append("Strong name match for employment-related query")
                    
                    if any(word in name_lower for word in ['price', 'inflation', 'cpi']):
                        if any(word in query_lower for word in ['inflation', 'price', 'cost', 'cpi']):
                            reasoning_hints.append("Strong name match for inflation-related query")
                    
                    if any(word in name_lower for word in ['gdp', 'product', 'output']):
                        if any(word in query_lower for word in ['gdp', 'growth', 'product', 'output', 'economy']):
                            reasoning_hints.append("Strong name match for GDP/growth-related query")
                    
                    if reasoning_hints:
                        response += f"   💡 Relevance Hints: {'; '.join(reasoning_hints)}\n"
                    
                    response += "\n"
                
                response += "**AGENT DECISION REQUIRED:**\n"
                response += "Evaluate the above indicators intelligently. If any seem relevant to the user's query "
                response += "(based on name/description, not just similarity score), use display_dataframe_preview "
                response += "to show the data. If none are relevant, consider using delegate_to_fred_agent to find new data."
                
                print(f"   ✅ Returning comprehensive results to agent for intelligent evaluation")
                return response
            else:
                print("   ❌ No indicators found after all search attempts")
                response = f"❌ **NO RESULTS FOUND**\n"
                response += f"Searched {search_attempts} variations of the query but found no matching indicators.\n"
                response += f"Variations tried: {variations}\n\n"
                response += "**RECOMMENDATION:** Use delegate_to_fred_agent tool to search FRED database for new data."
                return response
                
        except Exception as e:
            error_msg = f"❌ Error during database search: {str(e)}"
            print(f"   🚨 {error_msg}")
            return error_msg
    
    def _extract_data_implementation(self, indicator_id: str) -> str:
        """Implementation of data extraction"""
        try:
            print(f"📊 Extracting data for indicator: {indicator_id}")
            print("   🔍 Step 1: Loading indicator metadata from data dictionary...")
            
            result = extract_data(indicator_id)
            if result is None:
                error_msg = f"❌ Failed to extract data for indicator '{indicator_id}'"
                print(f"   🚨 {error_msg}")
                print("   💡 Possible reasons: Indicator not in dictionary, Excel file missing, or column not found")
                return f"{error_msg}. Please check the indicator ID."
            
            metadata, time_series = result
            print(f"   ✅ Successfully extracted {len(time_series)} observations")
            print(f"   📅 Data range: {time_series.index[0]} to {time_series.index[-1]}")
            
            # Format the response
            response = f"✅ **DATA EXTRACTION SUCCESSFUL**\n\n"
            response += f"**Indicator:** {metadata.get('Indicator Name', 'N/A')}\n"
            response += f"**ID:** {metadata.get('Indicator ID', 'N/A')}\n"
            response += f"**Geography:** {metadata.get('Geography', 'N/A')}\n"
            response += f"**Frequency:** {metadata.get('Frequency', 'N/A')}\n"
            response += f"**Description:** {metadata.get('Description', 'N/A')}\n\n"
            
            response += f"**Data Summary:**\n"
            response += f"- Range: {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}\n"
            response += f"- Total Observations: {len(time_series)}\n"
            response += f"- Non-null Values: {time_series.count()}\n\n"
            
            # Show recent data
            response += "**Recent Data (Last 5 observations):**\n"
            recent_data = time_series.tail(5)
            for date, value in recent_data.items():
                response += f"- {date.strftime('%Y-%m-%d')}: {value}\n"
            
            print("   ✅ Data extraction completed successfully")
            return response
            
        except Exception as e:
            error_msg = f"❌ Error extracting data: {str(e)}"
            print(f"   🚨 {error_msg}")
            return error_msg
    
    def _display_dataframe_preview_implementation(self, indicator_id: str, preview_rows: int = 10) -> str:
        """Implementation of dataframe preview display"""
        try:
            print(f"📈 Creating formatted preview for indicator: {indicator_id}")
            print(f"   🎯 Purpose: Generate user-friendly data visualization with {preview_rows} rows")
            
            # Extract the data first
            print("   🔍 Step 1: Extracting underlying data...")
            result = extract_data(indicator_id)
            if result is None:
                error_msg = f"❌ Failed to extract data for indicator '{indicator_id}'"
                print(f"   🚨 {error_msg}")
                return f"{error_msg}. Please check the indicator ID."
            
            metadata, time_series = result
            print(f"   ✅ Data extracted: {len(time_series)} observations")
            
            # Create a DataFrame for better display
            print("   📊 Step 2: Creating DataFrame for display formatting...")
            df = pd.DataFrame({
                'Date': time_series.index,
                'Value': time_series.values
            })
            print(f"   ✅ DataFrame created with {len(df)} rows")
            
            # Calculate statistics
            print("   📈 Step 3: Computing statistical summary...")
            stats = {
                'mean': time_series.mean(),
                'median': time_series.median(),
                'min': time_series.min(),
                'max': time_series.max(),
                'std': time_series.std(),
                'min_date': time_series.idxmin(),
                'max_date': time_series.idxmax()
            }
            print(f"   ✅ Statistics computed: mean={stats['mean']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
            
            # Format the response with metadata and data preview
            print("   🎨 Step 4: Formatting comprehensive display...")
            response = f"📊 **DATA PREVIEW: {metadata.get('Indicator Name', 'N/A')}**\n"
            response += "=" * 80 + "\n\n"
            
            # Metadata section
            response += f"**INDICATOR METADATA:**\n"
            response += f"- ID: {metadata.get('Indicator ID', 'N/A')}\n"
            response += f"- Geography: {metadata.get('Geography', 'N/A')}\n"
            response += f"- Frequency: {metadata.get('Frequency', 'N/A')}\n"
            response += f"- Description: {metadata.get('Description', 'N/A')}\n\n"
            
            # Data summary
            response += f"**DATA SUMMARY:**\n"
            response += f"- Time Range: {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}\n"
            response += f"- Total Observations: {len(time_series):,}\n"
            response += f"- Non-null Values: {time_series.count():,}\n"
            response += f"- Latest Value: {time_series.iloc[-1]:.2f} ({time_series.index[-1].strftime('%Y-%m-%d')})\n\n"
            
            # Data preview table
            response += f"**DATA PREVIEW (Last {preview_rows} observations):**\n"
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
            response += f"\n**STATISTICAL SUMMARY:**\n"
            response += f"- Mean: {stats['mean']:.2f}\n"
            response += f"- Median: {stats['median']:.2f}\n"
            response += f"- Minimum: {stats['min']:.2f} ({stats['min_date'].strftime('%Y-%m-%d')})\n"
            response += f"- Maximum: {stats['max']:.2f} ({stats['max_date'].strftime('%Y-%m-%d')})\n"
            response += f"- Standard Deviation: {stats['std']:.2f}\n"
            
            response += "\n" + "=" * 80
            
            print("   ✅ Comprehensive preview formatted successfully")
            print(f"   📋 Preview includes: metadata, {len(preview_data)} data points, and 5 key statistics")
            return response
            
        except Exception as e:
            error_msg = f"❌ Error displaying dataframe preview: {str(e)}"
            print(f"   🚨 {error_msg}")
            return error_msg
    
    def _delegate_to_search_and_ingest_agent_implementation(self, query: str, country: str = "USA") -> str:
        """Implementation of workflow delegation to Search and Ingest Agent"""
        try:
            print(f"🤝 **DELEGATING TO SEARCH AND INGEST AGENT**")
            print(f"   📝 Query: '{query}'") 
            print(f"   🌍 Country: '{country}'")
            print(f"   🎯 Purpose: Specialized ingestion workflow")
            print("   ⏳ Handing off to Search and Ingest Agent...")
            
            # Delegate to Search and Ingest Agent
            result = self.search_and_ingest_agent.ingest_specific_indicators(query, country)
            
            if result["success"]:
                response = f"✅ **SEARCH AND INGEST AGENT DELEGATION SUCCESSFUL**\n\n"
                response += f"**DELEGATION SUMMARY:**\n"
                response += f"- Query: '{result['query']}'\n"
                response += f"- Country: '{result['country']}' → '{result['normalized_country']}'\n"
                response += f"- New Indicators Added: {result['new_indicators']}\n"
                response += f"- Excel Files Created/Updated: {len(result['excel_files'])}\n"
                if result['excel_files']:
                    response += f"- Files: {', '.join(result['excel_files'])}\n"
                response += f"- New Embeddings: {result['embedding_stats'].get('processed', 0)}\n\n"
                response += "**IMPORTANT NEXT STEP:**\n"
                response += "✅ **The database now contains new indicators! Search the database again** "
                response += "using the search_database tool to find the newly ingested data and answer the user's query."
                
                print("🎉 **DELEGATION COMPLETED SUCCESSFULLY**")
                print("📋 QueryAgent should now search database again for newly ingested data")
                return response
            else:
                error_response = f"❌ **SEARCH AND INGEST AGENT DELEGATION FAILED**\n\n"
                error_response += f"**ERROR DETAILS:**\n"
                error_response += f"- Query: '{result['query']}'\n"
                error_response += f"- Country: '{result['country']}'\n"
                error_response += f"- Error: {result['error']}\n"
                error_response += f"- Message: {result['message']}\n\n"
                error_response += "**RECOMMENDATION:**\n"
                error_response += "Try rephrasing the query with more specific economic terms or check if FRED has data for the requested country."
                
                print("❌ **DELEGATION FAILED**")
                return error_response
                
        except Exception as e:
            error_msg = f"❌ **CRITICAL ERROR IN DELEGATION**: {str(e)}"
            print(f"🚨 {error_msg}")
            return error_msg
    
    def _generate_search_variations(self, query: str) -> List[str]:
        """Generate search variations using LLM"""
        try:
            print(f"   🧠 Generating search variations for: '{query}'")
            template = self.prompt_config.get("search_variations_template", "")
            prompt = template.format(query=query)
            
            # Use LLM to generate variations
            response = self.llm_client.client.chat.completions.create(
                model=self.prompt_config["model_config"].get("model", "gpt-4o"),
                temperature=self.prompt_config["model_config"].get("temperature", 0.3),
                max_tokens=self.prompt_config["model_config"].get("max_tokens", 150),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            variations = response.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip()]
            
            # Always include original query
            if query not in variations:
                variations.insert(0, query)
            
            final_variations = variations[:3]  # Max 3 variations
            print(f"   ✅ Generated {len(final_variations)} variations: {final_variations}")
            return final_variations
            
        except Exception as e:
            print(f"   ⚠️ Error generating variations: {e}")
            print("   🔄 Falling back to original query only")
            return [query]  # Fallback to original query
    
    def _rephrase_for_fred(self, query: str, country: str = "USA") -> str:
        """Rephrase query for FRED database search with country context"""
        try:
            print(f"   🧠 Rephrasing query for FRED: '{query}' (country: {country})")
            template = self.prompt_config.get("fred_query_rephrase_template", "")
            prompt = template.format(query=query, country=country)
            
            response = self.llm_client.client.chat.completions.create(
                model=self.prompt_config["model_config"].get("model", "gpt-4o"),
                temperature=self.prompt_config["model_config"].get("temperature", 0.3),
                max_tokens=self.prompt_config["model_config"].get("max_tokens", 150),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            rephrased = response.choices[0].message.content.strip()
            final_query = rephrased if rephrased else f"{query} {country}"
            print(f"   ✅ Rephrased to: '{final_query}'")
            return final_query
            
        except Exception as e:
            print(f"   ⚠️ Error rephrasing query: {e}")
            print("   🔄 Falling back to original query with country")
            return f"{query} {country}"  # Fallback to original query with country
    
    async def run_query(self, user_query: str) -> str:
        """
        Main method to process user queries
        """
        print(f"\n🚀 **STARTING QUERY PROCESSING**")
        print(f"📝 User Query: '{user_query}'")
        print(f"🤖 Delegating to OpenAI Agent for intelligent processing...")
        print("=" * 80)
        
        try:
            result = await Runner.run(self.agent, user_query)
            
            print("\n" + "=" * 80)
            print("🎯 **QUERY PROCESSING COMPLETED**")
            print(f"📋 Final Response Length: {len(result.final_output)} characters")
            print("✅ Agent successfully processed the query and generated response")
            
            return result.final_output
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            print(f"\n🚨 **QUERY PROCESSING FAILED**")
            print(f"❌ Error: {error_msg}")
            return error_msg
    
    def run_query_sync(self, user_query: str) -> str:
        """
        Synchronous version of run_query - works in Jupyter notebooks
        """
        print(f"\n🚀 **STARTING QUERY PROCESSING (SYNC MODE)**")
        print(f"📝 User Query: '{user_query}'")
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
                result = Runner.run_sync(self.agent, user_query)
            except RuntimeError:
                # No running loop, we can use normal sync method
                print("   ✅ Standard Python environment - using normal sync method...")
                result = Runner.run_sync(self.agent, user_query)
            
            print("\n" + "=" * 80)
            print("🎯 **QUERY PROCESSING COMPLETED**")
            print(f"📋 Final Response Length: {len(result.final_output)} characters")
            print("✅ Agent successfully processed the query and generated response")
            
            return result.final_output
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            print(f"\n🚨 **QUERY PROCESSING FAILED**")
            print(f"❌ Error: {error_msg}")
            return error_msg


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
    
    # Initialize agent (verbosity controlled by config.yaml)
    agent = QueryAgent()
    
    print("🚀 FRED Query Agent initialized!")
    if agent.verbose:
        print("🔍 VERBOSE MODE: You will see the agent's thinking process and tool calls")
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