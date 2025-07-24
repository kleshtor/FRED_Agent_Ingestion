import pandas as pd
from pandas_datareader import data as web
from fredapi import Fred
from typing import List, Dict, Tuple
from datetime import datetime
import os
import openai

from helper_functions.LLM_utils import *
from helper_functions.postgres_store import *

STANDARD_START_DATE = "1900-01-01"
VALID_FREQ = {"Monthly": "M", "Quarterly": "Q", "Annual": "A"}
llm_client = LLMClient()

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
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    start_date = STANDARD_START_DATE
    normalized_country = llm_client.normalize_country(country)
    existing_dict = load_existing_data_dictionary(dict_file_path)

    data_by_freq = {"M": pd.DataFrame(), "Q": pd.DataFrame(), "A": pd.DataFrame()}
    dictionary_rows = []

    # Load existing data using simplified naming format: {normalized_country}_{frequency}
    for freq_code in data_by_freq:
        freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
        filename = f"excel-output/{normalized_country.replace(' ', '_')}_{freq_label}.xlsx"
        data_by_freq[freq_code] = load_existing_excel_data(filename)

    for term in terms:
        query = f"{term} {normalized_country}"
        results = search_fred_series(query, api_key)

        for _, row in results.iterrows():
            freq = row.get("frequency")
            title = row.get("title")
            series_id = row.get("id")
            freq_code = VALID_FREQ.get(freq)
            if not freq_code or not title or not series_id:
                continue
            label = title.split('(')[0].strip().replace(" ", "_") + f"_{freq_code}"
            if label in data_by_freq[freq_code].columns:
                continue

            new_df = fetch_series_data(series_id, label, start_date, end_date, freq_code)
            data_by_freq[freq_code] = merge_time_series_data(data_by_freq[freq_code], new_df)

            if not indicator_exists_in_dictionary(series_id, existing_dict):
                description = llm_client.generate_description(title, freq, normalized_country)
                dictionary_rows.append({
                    "Indicator Name": title,
                    "Indicator ID": series_id,
                    "Description": description,
                    "Frequency": freq_code,
                    "Geography": normalized_country
                })

    return data_by_freq, dictionary_rows

def process_dictionary_embeddings(dict_file_path: str, embedding_store, model: str = "text-embedding-3-small") -> dict:
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
