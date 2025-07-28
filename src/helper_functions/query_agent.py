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
from helper_functions.core_utils import LLMClient, SRC_DIR, PROMPT_YAML_PATH


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
        print("🔧 Initializing QueryAgent...")
        
        # Disable tracing to clean up output
        set_tracing_disabled(True)
        print("   ✓ Tracing disabled for cleaner output")
        
        # Load environment variables
        load_dotenv()
        print("   ✓ Environment variables loaded")
        
        # Set up paths
        self.config_path = config_path or os.path.join(SRC_DIR, "helper_functions", "config.yaml")
        self.prompt_path = prompt_path or PROMPT_YAML_PATH
        print(f"   ✓ Config path: {self.config_path}")
        print(f"   ✓ Prompt path: {self.prompt_path}")
        
        # Load configuration and prompts
        self._load_config()
        self._load_prompts()
        
        # Initialize components
        print("   🔗 Connecting to PostgreSQL embedding store...")
        self.embedding_store = PostgresEmbeddingStore(self.db_config)
        print("   ✓ Database connection established")
        
        print("   🤖 Initializing LLM client...")
        self.llm_client = LLMClient(self.prompt_path)
        print("   ✓ LLM client ready")
        
        # Create the agent with tools
        print("   🛠️ Creating agent with tools...")
        self.agent = self._create_agent()
        print("   ✓ Agent created with 4 tools: search_database, extract_data, display_dataframe_preview, ingest_from_fred")
        
        print("✅ QueryAgent initialization complete!")
    
    def _load_config(self):
        """Load FRED configuration"""
        print("   📄 Loading FRED configuration...")
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.api_key = config.get("api_key")
        self.db_config = config.get("database", {})
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
        
        self.agent_config = prompts.get("query_agent", {})
        self.max_search_attempts = self.agent_config.get("max_search_attempts", 3)
        
        print(f"   ✓ Agent prompts loaded")
        print(f"   ✓ Max search attempts: {self.max_search_attempts}")
        print(f"   ✓ Available prompt templates: {list(self.agent_config.keys())}")
    
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
            return self._search_database_impl(query, max_attempts)
        
        @function_tool
        def extract_data(indicator_id: str) -> str:
            """
            Extract time series data for a specific indicator ID.
            Returns formatted data with metadata.
            """
            print(f"\n📊 TOOL CALLED: extract_data(indicator_id='{indicator_id}')")
            print("   📋 Tool Purpose: Extract raw time series data and metadata for specific indicator")
            return self._extract_data_impl(indicator_id)
        
        @function_tool
        def display_dataframe_preview(indicator_id: str, preview_rows: int = 10) -> str:
            """
            Display a preview of the time series data for a specific indicator.
            Shows both metadata and a sample of the actual data in a formatted table.
            """
            print(f"\n📈 TOOL CALLED: display_dataframe_preview(indicator_id='{indicator_id}', preview_rows={preview_rows})")
            print("   📋 Tool Purpose: Create formatted preview of time series data for user display")
            return self._display_dataframe_preview_impl(indicator_id, preview_rows)
        
        @function_tool
        def ingest_from_fred(query: str, country: str = "USA") -> str:
            """
            Search FRED database and ingest new economic indicators with full time series data.
            Returns status of ingestion process.
            """
            print(f"\n🌐 TOOL CALLED: ingest_from_fred(query='{query}', country='{country}')")
            print("   📋 Tool Purpose: Search FRED database for new indicators and ingest them locally")
            return self._ingest_from_fred_impl(query, country)
        
        # Get agent instructions from YAML configuration
        agent_instructions = self.agent_config.get("QueryAgent_Instructions", "")
        print(f"   ✓ Agent instructions loaded ({len(agent_instructions)} characters)")
        
        agent = Agent(
            name="FRED Query Agent",
            instructions=agent_instructions,
            tools=[search_database, extract_data, display_dataframe_preview, ingest_from_fred]
        )
        
        return agent
    
    def _search_database_impl(self, query: str, max_attempts: int = 3) -> str:
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
                response += "to show the data. If none are relevant, consider using ingest_from_fred to find new data."
                
                print(f"   ✅ Returning comprehensive results to agent for intelligent evaluation")
                return response
            else:
                print("   ❌ No indicators found after all search attempts")
                response = f"❌ **NO RESULTS FOUND**\n"
                response += f"Searched {search_attempts} variations of the query but found no matching indicators.\n"
                response += f"Variations tried: {variations}\n\n"
                response += "**RECOMMENDATION:** Use ingest_from_fred tool to search FRED database for new data."
                return response
                
        except Exception as e:
            error_msg = f"❌ Error during database search: {str(e)}"
            print(f"   🚨 {error_msg}")
            return error_msg
    
    def _extract_data_impl(self, indicator_id: str) -> str:
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
    
    def _display_dataframe_preview_impl(self, indicator_id: str, preview_rows: int = 10) -> str:
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
    
    def _ingest_from_fred_impl(self, query: str, country: str = "USA") -> str:
        """Implementation of FRED data ingestion with full time series data"""
        try:
            print(f"🌐 Starting FRED ingestion process...")
            print(f"   🎯 Query: '{query}'")
            print(f"   🌍 Country: {country}")
            print(f"   📋 Goal: Find new economic indicators from FRED database")
            
            # Rephrase query for FRED search
            print("   🧠 Step 1: Rephrasing query for FRED database using LLM...")
            fred_query = self._rephrase_for_fred(query, country)
            print(f"   ✅ Original query: '{query}'")
            print(f"   ✅ Target country: '{country}'")
            print(f"   ✅ FRED-optimized query: '{fred_query}'")
            
            # Search FRED
            print(f"   🔍 Step 2: Searching FRED database...")
            results = search_fred_series(fred_query, self.api_key, max_results=10)
            
            if results.empty:
                error_msg = f"❌ No results found in FRED for query: '{fred_query}'"
                print(f"   🚨 {error_msg}")
                print("   💡 Try rephrasing your query or using more general economic terms")
                return error_msg
            
            print(f"   ✅ Found {len(results)} potential indicators in FRED")
            
            # Process and ingest the results with full time series data
            print("   🔄 Step 3: Processing and filtering results...")
            ingested_count = 0
            normalized_country = self.llm_client.normalize_country(country)
            existing_dict = load_existing_data_dictionary(self.dict_file_path)
            
            print(f"   ✅ Target country normalized to: '{normalized_country}'")
            print(f"   📚 Existing dictionary has {len(existing_dict)} indicators")
            
            new_indicators = []
            
            # Initialize data storage by frequency
            data_by_freq = {"M": {}, "Q": {}, "A": {}}
            
            for idx, row in results.iterrows():
                series_id = row.get("id")
                title = row.get("title")
                frequency = row.get("frequency")
                
                print(f"\n   📊 Processing indicator {idx+1}/{len(results)}: {series_id}")
                print(f"      📝 Title: {title}")
                print(f"      📅 Frequency: {frequency}")
                
                if not all([series_id, title, frequency]):
                    print("      ❌ Skipping: Missing required fields")
                    continue
                
                # Check if already exists
                if indicator_exists_in_dictionary(series_id, existing_dict):
                    print("      ⏭️ Skipping: Already exists in local database")
                    continue
                
                # Generate description
                print("      🧠 Generating description using LLM...")
                description = self.llm_client.generate_description(title, frequency, normalized_country)
                print(f"      ✅ Description generated ({len(description)} chars)")
                
                # Map frequency
                freq_map = {"Monthly": "M", "Quarterly": "Q", "Annual": "A"}
                freq_code = freq_map.get(frequency)
                
                if freq_code:
                    # Fetch the actual time series data
                    print(f"      📈 Fetching time series data from FRED...")
                    
                    start_date = "1900-01-01"
                    end_date = datetime.today().strftime("%Y-%m-%d")
                    
                    # Create label for the series
                    label = title.split('(')[0].strip().replace(" ", "_") + f"_{freq_code}"
                    
                    # Fetch the time series data
                    time_series_df = fetch_series_data(series_id, label, start_date, end_date, freq_code)
                    
                    if not time_series_df.empty:
                        print(f"      ✅ Time series data fetched: {len(time_series_df)} observations")
                        # Store in data_by_freq structure
                        if freq_code not in data_by_freq:
                            data_by_freq[freq_code] = {}
                        data_by_freq[freq_code][label] = time_series_df
                    else:
                        print("      ⚠️ Warning: No time series data available")
                    
                    new_indicators.append({
                        "Indicator Name": title,
                        "Indicator ID": series_id,
                        "Description": description,
                        "Frequency": freq_code,
                        "Geography": normalized_country
                    })
                    ingested_count += 1
                    print(f"      ✅ Indicator processed successfully")
                else:
                    print(f"      ❌ Skipping: Unsupported frequency '{frequency}'")
            
            print(f"\n   📊 Processing complete: {ingested_count} new indicators ready for ingestion")
            
            if new_indicators:
                print("   💾 Step 4: Updating local database...")
                
                # Update dictionary
                new_df = pd.DataFrame(new_indicators)
                
                if os.path.exists(self.dict_file_path):
                    existing_df = pd.read_excel(self.dict_file_path)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    print(f"      ✅ Updated existing dictionary: {len(existing_df)} → {len(combined_df)} indicators")
                else:
                    combined_df = new_df
                    print(f"      ✅ Created new dictionary with {len(combined_df)} indicators")
                
                combined_df.to_excel(self.dict_file_path, index=False)
                
                # Create Excel files with time series data
                print("   📁 Step 5: Creating/updating Excel files with time series data...")
                excel_files_created = []
                for freq_code, series_dict in data_by_freq.items():
                    if series_dict:  # Only if we have data for this frequency
                        freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
                        filename = f"excel-output/{normalized_country.replace(' ', '_')}_{freq_label}.xlsx"
                        full_path = os.path.join(SRC_DIR, filename)
                        
                        print(f"      📊 Processing {freq_label} data ({len(series_dict)} series)...")
                        
                        # Load existing data if file exists
                        if os.path.exists(full_path):
                            existing_data = pd.read_excel(full_path, index_col=0, parse_dates=True)
                            print(f"         📂 Loaded existing file with {len(existing_data.columns)} columns")
                        else:
                            existing_data = pd.DataFrame()
                            print("         📄 Creating new file")
                        
                        # Merge all new series for this frequency
                        combined_data = existing_data
                        for label, series_df in series_dict.items():
                            combined_data = merge_time_series_data(combined_data, series_df)
                            print(f"         ✅ Merged series: {label}")
                        
                        # Ensure complete date range and save
                        if not combined_data.empty:
                            complete_data = ensure_complete_date_range(combined_data, freq_code, end_date)
                            complete_data.to_excel(full_path, index=True)
                            excel_files_created.append(filename)
                            print(f"         💾 Saved: {filename} ({len(complete_data)} rows, {len(complete_data.columns)} columns)")
                
                # Generate embeddings
                print("   🧠 Step 6: Generating embeddings for new indicators...")
                stats = process_dictionary_embeddings(self.dict_file_path, self.embedding_store)
                print(f"      ✅ Embeddings processed: {stats.get('processed', 0)} new, {stats.get('skipped', 0)} skipped")
                
                response = f"✅ **FRED INGESTION COMPLETED SUCCESSFULLY**\n\n"
                response += f"**INGESTION SUMMARY:**\n"
                response += f"- Original Query: '{query}'\n"
                response += f"- FRED Query: '{fred_query}'\n"
                response += f"- FRED Results Found: {len(results)}\n"
                response += f"- New Indicators Added: {ingested_count}\n"
                response += f"- Excel Files Created/Updated: {len(excel_files_created)}\n"
                if excel_files_created:
                    response += f"- Files: {', '.join(excel_files_created)}\n"
                response += f"- New Embeddings Generated: {stats.get('processed', 0)}\n\n"
                response += "**NEXT STEPS:**\n"
                response += "The database now contains new economic indicators. You should search the database "
                response += "again to find better matches for the user's original query."
                
                print("   🎉 FRED ingestion completed successfully!")
                return response
            else:
                warning_msg = f"⚠️ No new indicators found - all {len(results)} results already exist in database"
                print(f"   {warning_msg}")
                return warning_msg
                
        except Exception as e:
            error_msg = f"❌ Error during FRED ingestion: {str(e)}"
            print(f"   🚨 {error_msg}")
            return error_msg
    
    def _generate_search_variations(self, query: str) -> List[str]:
        """Generate search variations using LLM"""
        try:
            print(f"   🧠 Generating search variations for: '{query}'")
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
            template = self.agent_config.get("fred_query_rephrase_template", "")
            prompt = template.format(query=query, country=country)
            
            response = self.llm_client.client.chat.completions.create(
                model=self.agent_config["model_config"].get("model", "gpt-4o"),
                temperature=self.agent_config["model_config"].get("temperature", 0.3),
                max_tokens=self.agent_config["model_config"].get("max_tokens", 150),
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