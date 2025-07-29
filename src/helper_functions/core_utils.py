import os
import openai
import yaml
import re
from pathlib import Path
from typing import List

# ===== PATH CONFIGURATION =====
# Contains absolute paths of all the important folders in code

PROJECT_DIR = str(Path(__file__).parents[2])

# Data folder
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
RAW_STATIC_DATA_DIR = os.path.join(DATA_DIR, "raw", "static")
INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "intermediate")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# src folder
SRC_DIR = os.path.join(PROJECT_DIR, "src")

# output folder
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
OUTPUT_INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "Intermediate Results")
OUTPUT_FINAL_RESULT_DIR = os.path.join(OUTPUT_DIR, "Final Results")
OUTPUT_TRAINING_RESULT_DIR = os.path.join(OUTPUT_DIR, "Training Outputs")

# Log folder
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# Config folder
CONFIG_DIR = os.path.join(PROJECT_DIR, "config")

# Path to prompts.yaml
PROMPT_YAML_PATH = os.path.join(SRC_DIR, "helper_functions", "prompts.yaml")


# ===== LLM CLIENT =====

class LLMClient:
    def __init__(self, prompt_file: str = None):
        self.prompts = self._load_prompts(prompt_file or PROMPT_YAML_PATH)
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _load_prompts(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def generate_description(self, title: str, frequency: str, country: str) -> str:
        cfg = self.prompts.get("economic_indicators", {})
        prompt = cfg.get("user_prompt_template", "").format(
            title=title, frequency=frequency.lower(), country=country
        )

        try:
            response = self.client.chat.completions.create(
                model=cfg["model_config"].get("model", "gpt-4o"),
                temperature=cfg["model_config"].get("temperature", 0.2),
                max_tokens=cfg["model_config"].get("max_tokens", 100),
                messages=[
                    {"role": "system", "content": cfg.get("system_instructions", "")},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Description generation failed: {e}")
            return "Description not available"

    def normalize_country(self, user_input: str) -> str:
        cfg = self.prompts.get("country_resolution", {})
        try:
            response = self.client.chat.completions.create(
                model=cfg["model_config"].get("model", "gpt-4o"),
                temperature=cfg["model_config"].get("temperature", 0),
                max_tokens=cfg["model_config"].get("max_tokens", 30),
                messages=[
                    {"role": "system", "content": cfg.get("system_instructions", "")},
                    {"role": "user", "content": user_input.strip()}
                ]
            )
            return re.sub(r"[^\w\s,\-]", "", response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Country normalization failed: {e}")
            return user_input.strip()

    def normalize_worldbank_country(self, user_input: str) -> str:
        """Normalize country names to World Bank 3-letter country codes"""
        # Common country name to World Bank code mappings
        country_mappings = {
            'united states': 'USA',
            'usa': 'USA',
            'america': 'USA',
            'us': 'USA',
            'china': 'CHN',
            'germany': 'DEU',
            'japan': 'JPN',
            'united kingdom': 'GBR',
            'uk': 'GBR',
            'britain': 'GBR',
            'canada': 'CAN',
            'france': 'FRA',
            'italy': 'ITA',
            'spain': 'ESP',
            'brazil': 'BRA',
            'india': 'IND',
            'russia': 'RUS',
            'south korea': 'KOR',
            'australia': 'AUS',
            'mexico': 'MEX',
            'netherlands': 'NLD',
            'switzerland': 'CHE',
            'sweden': 'SWE',
            'norway': 'NOR',
            'denmark': 'DNK',
            'finland': 'FIN',
            'belgium': 'BEL',
            'austria': 'AUT',
            'ireland': 'IRL',
            'portugal': 'PRT',
            'greece': 'GRC',
            'poland': 'POL',
            'turkey': 'TUR',
            'south africa': 'ZAF',
            'argentina': 'ARG',
            'chile': 'CHL',
            'colombia': 'COL',
            'peru': 'PER',
            'thailand': 'THA',
            'malaysia': 'MYS',
            'singapore': 'SGP',
            'indonesia': 'IDN',
            'philippines': 'PHL',
            'vietnam': 'VNM',
            'egypt': 'EGY',
            'israel': 'ISR',
            'saudi arabia': 'SAU',
            'uae': 'ARE',
            'united arab emirates': 'ARE',
            'new zealand': 'NZL',
            'czech republic': 'CZE',
            'hungary': 'HUN',
            'slovakia': 'SVK',
            'slovenia': 'SVN',
            'croatia': 'HRV',
            'estonia': 'EST',
            'latvia': 'LVA',
            'lithuania': 'LTU'
        }
        
        # Normalize input
        normalized_input = user_input.strip().lower()
        
        # Check direct mapping
        if normalized_input in country_mappings:
            return country_mappings[normalized_input]
        
        # Check if it's already a 3-letter code
        if len(normalized_input) == 3 and normalized_input.upper() in country_mappings.values():
            return normalized_input.upper()
        
        # Fallback to original normalization or return as-is
        try:
            cfg = self.prompts.get("country_resolution", {})
            if cfg:
                response = self.client.chat.completions.create(
                    model=cfg["model_config"].get("model", "gpt-4o"),
                    temperature=cfg["model_config"].get("temperature", 0),
                    max_tokens=cfg["model_config"].get("max_tokens", 30),
                    messages=[
                        {"role": "system", "content": "Convert country names to World Bank 3-letter country codes (e.g., 'Canada' -> 'CAN', 'United States' -> 'USA')"},
                        {"role": "user", "content": user_input.strip()}
                    ]
                )
                result = response.choices[0].message.content.strip().upper()
                return result if len(result) == 3 else user_input.strip().upper()
            else:
                return user_input.strip().upper()
        except Exception as e:
            print(f"World Bank country normalization failed: {e}")
            return user_input.strip().upper()

    def generate_search_variations(self, query: str) -> List[str]:
        """Generate search variations for FRED database queries"""
        cfg = self.prompts.get("query_agent", {})
        template = cfg.get("search_variations_template", "")
        prompt = template.format(query=query)
        
        try:
            response = self.client.chat.completions.create(
                model=cfg["model_config"].get("model", "gpt-4o"),
                temperature=cfg["model_config"].get("temperature", 0.3),
                max_tokens=cfg["model_config"].get("max_tokens", 150),
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
            print(f"Error generating search variations: {e}")
            return [query]  # Fallback to original query

    def rephrase_for_fred(self, query: str, country: str) -> str:
        """Rephrase query for FRED database search"""
        cfg = self.prompts.get("query_agent", {})
        template = cfg.get("fred_query_rephrase_template", "")
        prompt = template.format(query=query, country=country)
        
        try:
            response = self.client.chat.completions.create(
                model=cfg["model_config"].get("model", "gpt-4o"),
                temperature=cfg["model_config"].get("temperature", 0.3),
                max_tokens=cfg["model_config"].get("max_tokens", 150),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error rephrasing for FRED: {e}")
            return query  # Fallback to original query

    def generate_worldbank_search_variations(self, query: str) -> List[str]:
        """Generate search variations for World Bank database queries"""
        cfg = self.prompts.get("worldbank_query_agent", {})
        template = cfg.get("worldbank_search_variations_template", "")
        prompt = template.format(query=query)
        
        try:
            response = self.client.chat.completions.create(
                model=cfg["model_config"].get("model", "gpt-4o"),
                temperature=cfg["model_config"].get("temperature", 0.3),
                max_tokens=cfg["model_config"].get("max_tokens", 150),
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
            print(f"Error generating World Bank search variations: {e}")
            return [query]  # Fallback to original query

    def rephrase_for_worldbank(self, query: str, countries: List[str]) -> str:
        """Rephrase query for World Bank database search"""
        cfg = self.prompts.get("worldbank_query_agent", {})
        template = cfg.get("worldbank_query_rephrase_template", "")
        prompt = template.format(query=query, countries=countries)
        
        try:
            response = self.client.chat.completions.create(
                model=cfg["model_config"].get("model", "gpt-4o"),
                temperature=cfg["model_config"].get("temperature", 0.3),
                max_tokens=cfg["model_config"].get("max_tokens", 150),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error rephrasing for World Bank: {e}")
            return query  # Fallback to original query 