import os
import yaml
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from helper_functions.postgres_store import PostgresEmbeddingStore
from helper_functions.FRED_helper import *

class FredAgent:
    def __init__(self, config_path: str, prompt_path: str = "llm_prompts.yaml"):
        self.config_path = config_path
        self.prompt_path = prompt_path

        self._load_environment()
        self._load_config()
        self.embedding_store = PostgresEmbeddingStore(self.db_config)

        self.start_date = "1900-01-01"
        self.end_date = datetime.today().strftime("%Y-%m-%d")

    def _load_environment(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in .env")

    def _load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.api_key = config.get("api_key")
        self.countries = config.get("countries", [])
        self.search_terms = [t['term'] for t in config.get("search_terms", [])]
        self.output_path = config.get("output_path", "excel-output/fred_macro_data.xlsx")
        self.dict_file_path = "excel-output/FRED_DataDictionary.xlsx"
        self.db_config = config.get("database", {})

        if not self.api_key:
            raise ValueError("FRED API key is missing in config")

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
                api_key=self.api_key,
                end_date=self.end_date,
                output_base=self.output_path,
                dict_file_path=self.dict_file_path
            )

            for freq_code, df in data_by_freq.items():
                if not df.empty:
                    freq_label = {"M": "Monthly", "Q": "Quarterly", "A": "Annual"}[freq_code]
                    # Use simplified naming format: {normalized_country}_{frequency}.xlsx
                    filename = f"excel-output/{normalized_country.replace(' ', '_')}_{freq_label}.xlsx"

                    complete_df = ensure_complete_date_range(df, freq_code, self.end_date)
                    complete_df.to_excel(filename, index=True)
                    print(f"Saved: {filename}")

            all_dict_rows.extend(dict_rows)

        self._update_data_dictionary(all_dict_rows)
        self._generate_embeddings()

    def _update_data_dictionary(self, dict_rows):
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
        stats = process_dictionary_embeddings(
            dict_file_path=self.dict_file_path,
            embedding_store=self.embedding_store
        )
        print("Embedding Generation Complete:", stats)
