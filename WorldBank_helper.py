import pandas as pd
import wbgapi as wb
import os
import sys
import yaml
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

def search_wb_indicators(search_term: str, max_results: int = 15) -> pd.DataFrame:
    """Search World Bank indicators by term"""
    try:
        # Search indicators
        results = wb.series.info(q=search_term)
        
        # Convert Featureset to DataFrame using .items
        df = pd.DataFrame(results.items).head(max_results)
        
        if df.empty:
            return pd.DataFrame({"id": [], "name": []})
        
        # Rename 'value' column to 'name' for consistency
        if 'value' in df.columns:
            df = df.rename(columns={'value': 'name'})
        
        # Select available columns
        available_cols = []
        for col in ["id", "name"]:
            if col in df.columns:
                available_cols.append(col)
        
        if available_cols:
            return pd.DataFrame(df[available_cols])
        else:
            return pd.DataFrame(df)
    
    except Exception as e:
        print(f"Error searching indicators: {e}")
        return pd.DataFrame({"id": [], "name": []})

def display_indicator_options(results: pd.DataFrame):
    """Display search results in a readable format"""
    print("\nSearch Results:")
    for idx, row in results.iterrows():
        name_val = row.get('name', row.get('value', 'N/A'))
        if name_val is not None:
            name = str(name_val)[:80] + "..." if len(str(name_val)) > 80 else str(name_val)
        else:
            name = 'N/A'
        print(f"[{idx}] {name} | ID: {row['id']}")

def get_user_selection(results: pd.DataFrame) -> List[Dict]:
    """Get user selection of indicators to fetch"""
    try:
        selection = input("\nEnter the indices of the indicators to fetch (comma-separated): ")
        selected_indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
        selected = []
        
        for idx in selected_indices:
            if idx < len(results):
                row = results.iloc[idx]
                # Create clean label from name
                name = row.get('name', row.get('value', f'Indicator_{idx}'))
                label = str(name).split('(')[0].strip().replace(' ', '_').replace(',', '').replace(':', '')[:30]
                selected.append({
                    "id": row['id'],
                    "label": f"{label}_A",  # World Bank data is typically annual
                    "name": str(name)
                })
        
        return selected
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted. Please use get_user_selection_manual() instead.")
        return []

def get_user_selection_manual(results: pd.DataFrame, selected_indices: List[int]) -> List[Dict]:
    """Manual selection for notebook environments - pass indices directly"""
    selected = []
    
    for idx in selected_indices:
        if idx < len(results):
            row = results.iloc[idx]
            # Create clean label from name
            name = row.get('name', row.get('value', f'Indicator_{idx}'))
            label = str(name).split('(')[0].strip().replace(' ', '_').replace(',', '').replace(':', '')[:30]
            selected.append({
                "id": row['id'],
                "label": f"{label}_A",  # World Bank data is typically annual
                "name": str(name)
            })
    
    return selected

def fetch_selected_indicators(indicator_list: List[Dict], countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch selected World Bank indicators for specified countries and date range"""
    all_data = pd.DataFrame()
    
    for indicator in indicator_list:
        try:
            print(f"Fetching {indicator['name'][:50]}...")
            
            # Fetch data one country at a time without time parameter (API works better this way)
            country_data = []
            for country in countries:
                try:
                    # Fetch all available data (no time parameter)
                    data = wb.data.DataFrame(
                        series=indicator['id'], 
                        economy=country
                    )
                    
                    if data is not None and not data.empty:
                        # Filter to desired year range manually
                        year_cols = [col for col in data.columns 
                                   if col.startswith('YR') and 
                                   start_year <= int(col[2:]) <= end_year]
                        
                        if year_cols:
                            filtered_data = data[year_cols]
                            
                            # Rename columns to include country and indicator
                            new_columns = {}
                            for col in filtered_data.columns:
                                year = col[2:]  # Remove 'YR' prefix
                                new_columns[col] = f"{country}_{indicator['label']}_{year}"
                            
                            filtered_data = filtered_data.rename(columns=new_columns)
                            country_data.append(filtered_data)
                            print(f"  ✅ {country}: {len(year_cols)} years of data")
                        else:
                            print(f"  ⚠️  {country}: No data for {start_year}-{end_year}")
                
                except Exception as e:
                    print(f"  ❌ Failed for {country}: {e}")
            
            # Combine data for this indicator
            if country_data:
                indicator_data = pd.concat(country_data, axis=1)
                
                # Concatenate with existing data
                if all_data.empty:
                    all_data = indicator_data
                else:
                    all_data = pd.concat([all_data, indicator_data], axis=1)
        
        except Exception as e:
            print(f"Failed to fetch {indicator['id']}: {e}")
    
    return all_data

def get_country_info(country_codes: List[str]) -> pd.DataFrame:
    """Get information about specified countries"""
    try:
        # Get country information one by one
        countries_data = []
        for country in country_codes:
            country_info = wb.economy.info(country)
            if hasattr(country_info, 'items'):
                df_country = pd.DataFrame(country_info.items)
                if not df_country.empty:
                    countries_data.append(df_country.iloc[0])
        
        if countries_data:
            df = pd.DataFrame(countries_data)
            # Select useful columns
            useful_cols = []
            for col in ['id', 'value', 'region', 'incomeLevel', 'lendingType']:
                if col in df.columns:
                    useful_cols.append(col)
            
            if useful_cols:
                return pd.DataFrame(df[useful_cols])
            else:
                return pd.DataFrame(df)
        else:
            return pd.DataFrame({"country": country_codes})
    
    except Exception as e:
        print(f"Error getting country info: {e}")
        return pd.DataFrame({"country": country_codes})

def setup_working_directory():
    """
    Verify current working directory and ensure all required files are present.
    Since all files are now in the same folder, this is simplified.
    """
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Check for required files in current directory
    required_files = ["WorldBank.yaml", "WorldBank_helper.py", "WorldBank_Extractor.py"]
    missing_files = [f for f in required_files if not (current_dir / f).exists()]
    
    if missing_files:
        print(f"⚠️  Warning: Required files not found: {missing_files}")
        print(f"   Current directory contents: {[p.name for p in current_dir.iterdir()]}")
    else:
        print("✅ All required files found in current directory")
    
    print(f"✅ Working directory verified: {current_dir}")
    return current_dir


def import_helper_functions():
    """Import helper functions after setting up the working directory."""
    try:
        from WorldBank_helper import (
            search_wb_indicators,
            display_indicator_options,
            get_user_selection_manual,
            fetch_selected_indicators,
            get_country_info
        )
        print("✅ Helper functions imported successfully")
        return {
            'search_wb_indicators': search_wb_indicators,
            'display_indicator_options': display_indicator_options,
            'get_user_selection_manual': get_user_selection_manual,
            'fetch_selected_indicators': fetch_selected_indicators,
            'get_country_info': get_country_info
        }
    except ImportError as e:
        print(f"❌ Error importing helper functions: {e}")
        return None


def clean_and_recreate_excel_output():
    """
    Create excel-output directory in current folder if it doesn't exist, and clear it if it does.
    """
    print("\n📁 Setting up excel-output directory...")
    
    # Get current working directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Define the excel-output directory in current folder
    excel_output_dir = current_dir / "excel-output"
    
    print(f"Target directory: {excel_output_dir}")
    
    # Handle excel-output directory
    if excel_output_dir.exists():
        try:
            # First try to clear the contents of the directory
            files_in_dir = list(excel_output_dir.glob("*"))
            if files_in_dir:
                print(f"🧹 Clearing {len(files_in_dir)} files from existing directory...")
                for file_path in files_in_dir:
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                        elif file_path.is_dir():
                            shutil.rmtree(file_path)
                        print(f"   ✅ Removed: {file_path.name}")
                    except Exception as e:
                        print(f"   ⚠️  Could not remove {file_path.name}: {e}")
                
                print(f"✅ Cleared directory contents")
            else:
                print(f"✅ Directory already empty")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not clear directory contents: {e}")
            # Try to remove the entire directory as fallback
            try:
                shutil.rmtree(excel_output_dir)
                print(f"🗑️ Removed entire directory as fallback")
            except Exception as e2:
                print(f"⚠️  Could not remove directory either: {e2}")
                print(f"📁 Will work with existing directory")
    
    # Ensure the directory exists
    try:
        excel_output_dir.mkdir(parents=True, exist_ok=True)
        if not excel_output_dir.exists():
            print(f"❌ Failed to create directory")
            return False
            
        print(f"📁 Directory ready: {excel_output_dir}")
        
        # Verify the directory status
        files_in_dir = list(excel_output_dir.glob("*"))
        if files_in_dir:
            print(f"⚠️  Directory contains {len(files_in_dir)} files:")
            for file in files_in_dir[:5]:  # Show first 5 files
                print(f"   - {file.name}")
            if len(files_in_dir) > 5:
                print(f"   ... and {len(files_in_dir) - 5} more files")
        else:
            print(f"✅ Directory is clean (no files)")
        
        print(f"📍 Absolute path: {excel_output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating directory: {e}")
        return False


def extract_yaml_instructions():
    """Extract configuration from YAML file."""
    yaml_file = Path.cwd() / "WorldBank.yaml"
    
    if not yaml_file.exists():
        print(f"❌ Error: WorldBank.yaml not found in current directory")
        print(f"   Looking for: {yaml_filse}")
        return None, None, None, None,
    
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            search_terms = config["search_terms"]
            countries = config["countries"]
            start_year = config.get("start_year", 2000)
            end_year = config.get("end_year", 2025)
            output_format = config.get("output_format", "excel")
            output_path = config.get("output_path", "outputs/world_bank_data.xlsx")
            print(f"\n✅ Instructions retrieved from yaml file: {yaml_file}")
            return search_terms, countries, start_year, end_year
    except Exception as e:
        print(f"❌ Error reading YAML file: {e}")
        return None, None, None, None


def search_result_getter(search_terms: list):
    """Get search results for all search terms."""
    search_results = {}
    for item in search_terms:
        print(f"\n==== Searching: {item['term']} ====")
        results = search_wb_indicators(item['term'])
        display_indicator_options(results)
        search_results[item['term']] = results
    return search_results 

def get_user_indicator_selections(search_results: dict) -> list:
    """
    Prompt user to select indicators from each search term's results.
    Returns a list of selected indicators with their metadata.
    """
    all_selected = []
    
    print(f"\n🎯 INDICATOR SELECTION")
    print("=" * 50)
    
    for term, results in search_results.items():
        if results.empty:
            print(f"\n⚠️  No results found for '{term}' - skipping")
            continue
            
        print(f"\n{'='*50}")
        print(f"SELECTING INDICATORS FOR: {term.upper()}")
        print(f"{'='*50}")
        
        # Display the options again for this specific term
        display_indicator_options(results)
        
        # Get user input with retry logic
        while True:
            try:
                selection = input(f"\nEnter the indices for '{term}' (comma-separated, or 'skip' to skip): ").strip()
                
                if selection.lower() in ['skip', 's', '']:
                    print(f"Skipping '{term}'")
                    break
                
                # Parse the selection
                selected_indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
                
                if not selected_indices:
                    print("No valid indices entered. Try again or type 'skip'.")
                    continue
                
                # Validate indices
                invalid_indices = [i for i in selected_indices if i >= len(results)]
                if invalid_indices:
                    print(f"Invalid indices: {invalid_indices}. Max index is {len(results)-1}")
                    continue
                
                # Process the selection using the manual selection function
                selected = get_user_selection_manual(results, selected_indices)
                all_selected.extend(selected)
                print(f"✅ Selected for '{term}': {[s['label'] for s in selected]}")
                break
                
            except (ValueError, KeyboardInterrupt):
                print("Invalid input. Enter numbers separated by commas, or 'skip'.")
                continue
            except EOFError:
                print("\nInput interrupted. Skipping this term.")
                break
    
    return all_selected


def display_final_selection(all_selected: list):
    """Display the final selection summary."""
    print(f"\n🎯 FINAL SELECTION SUMMARY")
    print("=" * 50)
    print(f"Total indicators selected: {len(all_selected)}")
    
    if all_selected:
        for idx, indicator in enumerate(all_selected, 1):
            # Truncate long names for display
            name = indicator['name'][:60] + "..." if len(indicator['name']) > 60 else indicator['name']
            print(f"  {idx}. {name}")
            print(f"     ID: {indicator['id']} | Label: {indicator['label']}")
    else:
        print("No indicators selected.")
    
    return len(all_selected) > 0


def fetch_indicators_separately(all_selected: list, countries: list, start_year: int, end_year: int) -> dict:
    """
    Fetch each indicator separately and return a dictionary of DataFrames.
    Each indicator gets its own DataFrame.
    """
    indicator_dataframes = {}
    
    print(f"\n📊 FETCHING DATA")
    print("=" * 50)
    print(f"Fetching {len(all_selected)} indicators for {len(countries)} countries ({start_year}-{end_year})")
    
    for idx, indicator in enumerate(all_selected, 1):
        print(f"\n[{idx}/{len(all_selected)}] Fetching: {indicator['name'][:50]}...")
        
        try:
            # Fetch data for this single indicator
            data = fetch_selected_indicators([indicator], countries, start_year, end_year)
            
            if not data.empty:
                # Store with a clean key name
                key_name = f"{indicator['label']}_{indicator['id']}"
                indicator_dataframes[key_name] = data
                
                print(f"✅ Success: {data.shape[0]} rows × {data.shape[1]} columns")
                print(f"   Countries with data: {data.shape[0]}")
                print(f"   Years covered: {data.shape[1]} columns")
                
                # Show a preview of the data
                if not data.empty:
                    print(f"   Preview: {list(data.columns)[:3]}{'...' if len(data.columns) > 3 else ''}")
            else:
                print(f"⚠️  No data returned for this indicator")
                
        except Exception as e:
            print(f"❌ Error fetching {indicator['name']}: {e}")
    
    print(f"\n📈 FETCH SUMMARY")
    print("=" * 30)
    print(f"Successfully fetched: {len(indicator_dataframes)} out of {len(all_selected)} indicators")
    
    if indicator_dataframes:
        print(f"\nDataFrame keys:")
        for key in indicator_dataframes.keys():
            df = indicator_dataframes[key]
            print(f"  • {key}: {df.shape[0]}×{df.shape[1]} ({df.shape[0]*df.shape[1]} data points)")
    
    return indicator_dataframes


def display_dataframe_previews(indicator_dataframes: dict):
    """Display previews of each fetched DataFrame."""
    if not indicator_dataframes:
        print("No DataFrames to preview.")
        return
    
    print(f"\n📋 DATA PREVIEWS")
    print("=" * 50)
    
    for key, df in indicator_dataframes.items():
        print(f"\n🔍 {key}")
        print("-" * 40)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if not df.empty:
            # Show column names (first few)
            print(f"Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
            
            # Show index (countries)
            print(f"Countries: {list(df.index)}")
            
            # Show first few values from first few columns
            preview_cols = min(3, df.shape[1])
            preview_rows = min(3, df.shape[0])
            print(f"Sample data (first {preview_rows} countries, first {preview_cols} columns):")
            print(df.iloc[:preview_rows, :preview_cols].to_string())
        else:
            print("DataFrame is empty") 

def detect_data_frequency(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect the frequency of data (yearly, quarterly, monthly, daily) using regex patterns.
    Returns tuple of (frequency, pattern_example)
    """
    if df.empty:
        return "unknown", "no_data"
    
    # Get a sample of column names to analyze
    sample_columns = list(df.columns)[:10]  # Check first 10 columns
    
    # Define regex patterns for different frequencies
    patterns = {
        'yearly': [
            r'_(\d{4})$',           # Ends with _YYYY
            r'_A_(\d{4})$',         # Ends with _A_YYYY (annual)
            r'YR(\d{4})$',          # Ends with YRYYYY
        ],
        'quarterly': [
            r'_(\d{4})Q([1-4])$',   # Ends with _YYYYQ1, _YYYYQ2, etc.
            r'_Q([1-4])_(\d{4})$',  # Ends with _Q1_YYYY, etc.
            r'(\d{4})-Q([1-4])$',   # Ends with YYYY-Q1, etc.
        ],
        'monthly': [
            r'_(\d{4})M(\d{1,2})$', # Ends with _YYYYM1, _YYYYM12, etc.
            r'_(\d{4})-(\d{1,2})$', # Ends with _YYYY-1, _YYYY-12, etc.
            r'M(\d{1,2})_(\d{4})$', # Ends with M1_YYYY, M12_YYYY, etc.
        ],
        'daily': [
            r'_(\d{4})-(\d{1,2})-(\d{1,2})$',  # Ends with _YYYY-MM-DD
            r'(\d{4})(\d{2})(\d{2})$',         # Ends with YYYYMMDD
        ]
    }
    
    # Test each pattern against sample columns
    detected_patterns = {}
    
    for frequency, regex_list in patterns.items():
        matches = 0
        example_match = None
        
        for regex_pattern in regex_list:
            for col in sample_columns:
                match = re.search(regex_pattern, str(col))
                if match:
                    matches += 1
                    if not example_match:
                        example_match = col
            
            if matches > 0:
                detected_patterns[frequency] = {
                    'count': matches,
                    'example': example_match,
                    'pattern': regex_pattern
                }
                break  # Found matching pattern for this frequency
    
    # Determine the most likely frequency
    if detected_patterns:
        # Get frequency with most matches
        best_frequency = max(detected_patterns.keys(), 
                           key=lambda x: detected_patterns[x]['count'])
        example = detected_patterns[best_frequency]['example']
        return best_frequency, example
    else:
        return "unknown", sample_columns[0] if sample_columns else "no_columns"


def extract_date_from_column(column_name: str, frequency: str) -> str:
    """
    Extract date/period information from column name based on detected frequency.
    """
    if frequency == "yearly":
        # Look for 4-digit year
        match = re.search(r'(\d{4})', column_name)
        if match:
            return match.group(1)
    
    elif frequency == "quarterly":
        # Look for YYYY and Q1-Q4
        year_match = re.search(r'(\d{4})', column_name)
        quarter_match = re.search(r'Q([1-4])', column_name)
        if year_match and quarter_match:
            return f"{year_match.group(1)}-Q{quarter_match.group(1)}"
    
    elif frequency == "monthly":
        # Look for YYYY and MM
        year_match = re.search(r'(\d{4})', column_name)
        month_match = re.search(r'M?(\d{1,2})', column_name)
        if year_match and month_match:
            month = month_match.group(1).zfill(2)  # Pad with zero if needed
            return f"{year_match.group(1)}-{month}"
    
    elif frequency == "daily":
        # Look for YYYY-MM-DD pattern
        match = re.search(r'(\d{4})-?(\d{1,2})-?(\d{1,2})', column_name)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    # Fallback - return the part after last underscore
    parts = column_name.split('_')
    return parts[-1] if parts else column_name


def transform_to_long_format(df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
    """
    Transform wide format DataFrame to long format with columns: Country, Date, Value.
    """
    if df.empty:
        return pd.DataFrame({'Country': [], 'Date': [], 'Value': [], 'Indicator': []})
    
    # Detect data frequency
    frequency, example = detect_data_frequency(df)
    print(f"  📊 Detected frequency: {frequency} (example: {example})")
    
    # Prepare data for transformation
    long_data = []
    
    # Get all countries (index)
    countries = df.index.tolist()
    
    # Process each column (which represents country-date combinations)
    for col in df.columns:
        # Extract country code from column name (usually the first part)
        col_parts = col.split('_')
        country_code = col_parts[0] if col_parts else 'Unknown'
        
        # Extract date/period from column name
        date_period = extract_date_from_column(col, frequency)
        
        # Get values for this column
        for country in countries:
            value = df.loc[country, col]
            if pd.notna(value):  # Only include non-null values
                long_data.append({
                    'Country': country,
                    'Date': date_period,
                    'Value': value,
                    'Indicator': indicator_name,
                    'Source_Country': country_code  # The country this column was originally for
                })
    
    # Create DataFrame
    long_df = pd.DataFrame(long_data)
    
    # Filter to only include rows where Country matches Source_Country
    # (to avoid duplicate/incorrect country-data combinations)
    if not long_df.empty and 'Source_Country' in long_df.columns:
        long_df = long_df[long_df['Country'] == long_df['Source_Country']]
        long_df = long_df.drop('Source_Country', axis=1)
    
    # Sort by Country and Date
    if not long_df.empty and len(long_df) > 0 and isinstance(long_df, pd.DataFrame):
        long_df = long_df.sort_values(['Country', 'Date']).reset_index(drop=True)
    
    return long_df


def transform_all_dataframes(indicator_dataframes: dict) -> dict:
    """
    Transform all DataFrames from wide to long format.
    Returns a dictionary with separate DataFrames for each indicator.
    """
    if not indicator_dataframes:
        print("No DataFrames to transform.")
        return {}
    
    print(f"\n🔄 TRANSFORMING DATA TO LONG FORMAT")
    print("=" * 50)
    
    transformed_dfs = {}
    
    for key, df in indicator_dataframes.items():
        print(f"\n🔄 Transforming: {key}")
        
        # Extract indicator name from key (remove ID part)
        indicator_name = key.split('_')[:-1]  # Remove the ID part
        indicator_name = '_'.join(indicator_name)
        
        # Transform to long format
        long_df = transform_to_long_format(df, indicator_name)
        
        if not long_df.empty:
            # Remove the 'Indicator' column and rename 'Value' to the indicator name
            long_df = long_df.drop('Indicator', axis=1)
            long_df = long_df.rename(columns={'Value': indicator_name})
            
            transformed_dfs[key] = long_df
            print(f"  ✅ Success: {long_df.shape[0]} rows × {long_df.shape[1]} columns")
            print(f"  📈 Countries: {long_df['Country'].nunique()}")
            print(f"  📅 Date range: {long_df['Date'].min()} to {long_df['Date'].max()}")
            print(f"  📊 Non-null values: {long_df[indicator_name].notna().sum()}")
        else:
            print(f"  ⚠️  No data after transformation")
    
    print(f"\n✅ TRANSFORMATION COMPLETE")
    print("=" * 30)
    print(f"Successfully transformed: {len(transformed_dfs)} out of {len(indicator_dataframes)} indicators")
    
    return transformed_dfs


def display_transformed_previews(transformed_dfs: dict):
    """Display previews of the separate transformed DataFrames."""
    if not transformed_dfs:
        print("No transformed DataFrames to preview.")
        return
    
    print(f"\n📋 TRANSFORMED DATA PREVIEWS")
    print("=" * 50)
    
    for key, df in transformed_dfs.items():
        print(f"\n🔍 {key}")
        print("-" * 40)
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
            print(f"Countries: {sorted(df['Country'].unique())}")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Sample data (first 5 rows):")
            print(df.head().to_string())
        else:
            print("DataFrame is empty") 

def determine_resolution_priority() -> dict:
    """Define resolution priority from highest to lowest frequency."""
    return {
        'daily': 1,
        'weekly': 2, 
        'monthly': 3,
        'quarterly': 4,
        'yearly': 5
    }


def identify_highest_resolution(transformed_dfs: dict) -> str:
    """
    Identify the highest resolution (most frequent) time period across all DataFrames.
    """
    print(f"\n🔍 IDENTIFYING HIGHEST RESOLUTION")
    print("=" * 40)
    
    resolution_priority = determine_resolution_priority()
    detected_resolutions = {}
    
    for key, df in transformed_dfs.items():
        if not df.empty and 'Date' in df.columns:
            # Analyze the date format to determine resolution
            sample_dates = df['Date'].dropna().unique()[:5]
            
            for date_str in sample_dates:
                date_str = str(date_str)
                
                # Check patterns
                if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):  # YYYY-MM-DD
                    detected_resolutions[key] = 'daily'
                    break
                elif re.match(r'^\d{4}-W\d{1,2}$', date_str):  # YYYY-W52
                    detected_resolutions[key] = 'weekly' 
                    break
                elif re.match(r'^\d{4}-\d{2}$', date_str):  # YYYY-MM
                    detected_resolutions[key] = 'monthly'
                    break
                elif re.match(r'^\d{4}-Q[1-4]$', date_str):  # YYYY-Q1
                    detected_resolutions[key] = 'quarterly'
                    break
                elif re.match(r'^\d{4}$', date_str):  # YYYY
                    detected_resolutions[key] = 'yearly'
                    break
            
            if key not in detected_resolutions:
                detected_resolutions[key] = 'unknown'
    
    print("Detected resolutions:")
    for key, resolution in detected_resolutions.items():
        print(f"  • {key}: {resolution}")
    
    # Find highest resolution (lowest priority number)
    if detected_resolutions:
        highest_res = min(detected_resolutions.values(), 
                         key=lambda x: resolution_priority.get(x, 999))
        print(f"\n✅ Highest resolution: {highest_res}")
        return highest_res
    else:
        print(f"\n⚠️  No resolutions detected, defaulting to yearly")
        return 'yearly'


def generate_time_periods(start_year: int, end_year: int, resolution: str) -> list:
    """
    Generate all time periods for the given resolution between start and end years.
    """
    periods = []
    
    for year in range(start_year, end_year + 1):
        if resolution == 'yearly':
            periods.append(str(year))
        elif resolution == 'quarterly':
            for quarter in range(1, 5):
                periods.append(f"{year}-Q{quarter}")
        elif resolution == 'monthly':
            for month in range(1, 13):
                periods.append(f"{year}-{month:02d}")
        elif resolution == 'weekly':
            # Approximate 52 weeks per year
            for week in range(1, 53):
                periods.append(f"{year}-W{week}")
        elif resolution == 'daily':
            # Generate all days (simplified - just month ends)
            import calendar
            for month in range(1, 13):
                days_in_month = calendar.monthrange(year, month)[1]
                for day in range(1, days_in_month + 1):
                    periods.append(f"{year}-{month:02d}-{day:02d}")
    
    return periods


def decompose_to_higher_resolution(df: pd.DataFrame, current_resolution: str, 
                                 target_resolution: str, indicator_name: str) -> pd.DataFrame:
    """
    Decompose a DataFrame from lower resolution to higher resolution.
    """
    if current_resolution == target_resolution:
        return df.copy()
    
    print(f"  🔄 Decomposing {current_resolution} → {target_resolution}")
    
    decomposed_data = []
    
    for _, row in df.iterrows():
        country = row['Country']
        date_str = str(row['Date'])
        value = row[indicator_name]
        
        # Parse the current date
        if current_resolution == 'yearly':
            year = int(date_str)
            
            if target_resolution == 'quarterly':
                for quarter in range(1, 5):
                    decomposed_data.append({
                        'Country': country,
                        'Date': f"{year}-Q{quarter}",
                        indicator_name: value
                    })
            elif target_resolution == 'monthly':
                for month in range(1, 13):
                    decomposed_data.append({
                        'Country': country,
                        'Date': f"{year}-{month:02d}",
                        indicator_name: value
                    })
            elif target_resolution == 'weekly':
                for week in range(1, 53):
                    decomposed_data.append({
                        'Country': country,
                        'Date': f"{year}-W{week}",
                        indicator_name: value
                    })
        
        elif current_resolution == 'quarterly':
            # Parse YYYY-Q1 format
            year = int(date_str.split('-Q')[0])
            quarter = int(date_str.split('-Q')[1])
            
            if target_resolution == 'monthly':
                # Each quarter has 3 months
                start_month = (quarter - 1) * 3 + 1
                for month_offset in range(3):
                    month = start_month + month_offset
                    decomposed_data.append({
                        'Country': country,
                        'Date': f"{year}-{month:02d}",
                        indicator_name: value
                    })
            elif target_resolution == 'weekly':
                # Each quarter has ~13 weeks
                start_week = (quarter - 1) * 13 + 1
                for week_offset in range(13):
                    week = start_week + week_offset
                    if week <= 52:  # Don't exceed 52 weeks
                        decomposed_data.append({
                            'Country': country,
                            'Date': f"{year}-W{week}",
                            indicator_name: value
                        })
        
        elif current_resolution == 'monthly':
            # Parse YYYY-MM format
            year, month = date_str.split('-')
            year, month = int(year), int(month)
            
            if target_resolution == 'weekly':
                # Each month has ~4.3 weeks, approximate
                weeks_in_month = 4 if month != 12 else 5  # Simplified
                start_week = (month - 1) * 4 + 1
                for week_offset in range(weeks_in_month):
                    week = start_week + week_offset
                    if week <= 52:
                        decomposed_data.append({
                            'Country': country,
                            'Date': f"{year}-W{week}",
                            indicator_name: value
                        })
    
    return pd.DataFrame(decomposed_data)


def merge_dataframes_by_resolution(transformed_dfs: dict) -> pd.DataFrame:
    """
    Merge all DataFrames by first decomposing them to the highest resolution.
    """
    if not transformed_dfs:
        print("No DataFrames to merge.")
        return pd.DataFrame()
    
    print(f"\n🔗 MERGING DATAFRAMES BY RESOLUTION")
    print("=" * 50)
    
    # Identify highest resolution
    target_resolution = identify_highest_resolution(transformed_dfs)
    
    # Decompose each DataFrame to target resolution
    decomposed_dfs = []
    
    for key, df in transformed_dfs.items():
        if df.empty:
            continue
            
        print(f"\n📊 Processing: {key}")
        
        # Get indicator name (3rd column, after Country and Date)
        indicator_cols = [col for col in df.columns if col not in ['Country', 'Date']]
        if not indicator_cols:
            print(f"  ⚠️  No indicator column found, skipping")
            continue
            
        indicator_name = indicator_cols[0]
        
        # Detect current resolution
        current_resolution = 'yearly'  # Default
        if not df.empty and 'Date' in df.columns:
            sample_date = str(df['Date'].iloc[0])
            if re.match(r'^\d{4}-Q[1-4]$', sample_date):
                current_resolution = 'quarterly'
            elif re.match(r'^\d{4}-\d{2}$', sample_date):
                current_resolution = 'monthly'
            elif re.match(r'^\d{4}-W\d{1,2}$', sample_date):
                current_resolution = 'weekly'
            elif re.match(r'^\d{4}-\d{2}-\d{2}$', sample_date):
                current_resolution = 'daily'
        
        print(f"  📅 Current resolution: {current_resolution}")
        
        # Decompose to target resolution
        decomposed_df = decompose_to_higher_resolution(
            df, current_resolution, target_resolution, indicator_name
        )
        
        if not decomposed_df.empty:
            decomposed_dfs.append(decomposed_df)
            print(f"  ✅ Decomposed: {decomposed_df.shape[0]} rows")
        else:
            print(f"  ⚠️  No data after decomposition")
    
    # Merge all decomposed DataFrames
    if decomposed_dfs:
        print(f"\n🔗 MERGING {len(decomposed_dfs)} DECOMPOSED DATAFRAMES")
        print("=" * 40)
        
        merged_df = decomposed_dfs[0]
        for df in decomposed_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on=['Country', 'Date'], how='outer')
            print(f"  ✅ Merged: {merged_df.shape}")
        
        # Sort by Country and Date
        try:
            merged_df = merged_df.sort_values(['Country', 'Date']).reset_index(drop=True)
        except Exception:
            pass  # Keep original order if sorting fails
        
        print(f"\n✅ FINAL MERGED DATAFRAME")
        print(f"  Shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
        print(f"  Resolution: {target_resolution}")
        print(f"  Countries: {merged_df['Country'].nunique()}")
        print(f"  Time periods: {merged_df['Date'].nunique()}")
        
        return merged_df
    else:
        print("❌ No DataFrames to merge after decomposition")
        return pd.DataFrame()


def display_merged_preview(merged_df: pd.DataFrame):
    """Display preview of the final merged DataFrame."""
    if merged_df.empty:
        print("No merged DataFrame to preview.")
        return
    
    print(f"\n📋 MERGED DATA PREVIEW")
    print("=" * 50)
    
    print(f"Shape: {merged_df.shape[0]} rows × {merged_df.shape[1]} columns")
    print(f"Columns: {list(merged_df.columns)}")
    print(f"Countries: {sorted(merged_df['Country'].unique())}")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    
    # Show sample data
    print(f"\nSample data (first 10 rows):")
    print(merged_df.head(10).to_string())
    
    # Show data summary
    print(f"\nData Summary:")
    print(f"  • Total observations: {merged_df.shape[0]}")
    print(f"  • Countries: {merged_df['Country'].nunique()}")
    print(f"  • Time periods: {merged_df['Date'].nunique()}")
    
    # Show non-null counts for each indicator
    indicator_cols = [col for col in merged_df.columns if col not in ['Country', 'Date']]
    print(f"  • Indicator data availability:")
    for col in indicator_cols:
        non_null_count = merged_df[col].notna().sum()
        total_count = len(merged_df)
        percentage = (non_null_count / total_count) * 100
        print(f"    - {col}: {non_null_count}/{total_count} ({percentage:.1f}%)")


def export_to_csv(df: pd.DataFrame, filename: str = "") -> str:
    """
    Export DataFrame to CSV file in the excel-output folder.
    
    Args:
        df: DataFrame to export
        filename: Optional custom filename (without extension)
    
    Returns:
        str: Path to the created CSV file
    """
    if df.empty:
        print("❌ Cannot export empty DataFrame")
        return ""
    
    print(f"\n💾 EXPORTING TO CSV")
    print("=" * 30)
    
    # Generate filename if not provided
    if not filename:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"worldbank_data_{timestamp}"
    
    # Ensure filename has .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Create full path
    output_dir = os.path.join(os.getcwd(), 'excel-output')
    file_path = os.path.join(output_dir, filename)
    
    try:
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        # Verify file was created
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ CSV exported successfully!")
            print(f"  📁 File: {filename}")
            print(f"  📍 Location: {file_path}")
            print(f"  📊 Size: {file_size:,} bytes")
            print(f"  📈 Data: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show first few rows of what was exported
            print(f"\n📋 Exported data preview:")
            print(df.head(3).to_string())
            
            return file_path
        else:
            print(f"❌ File was not created at {file_path}")
            return ""
            
    except Exception as e:
        print(f"❌ Error exporting to CSV: {str(e)}")
        return "" 