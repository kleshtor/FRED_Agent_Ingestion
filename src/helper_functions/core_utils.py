import os
import openai
import yaml
import re
from pathlib import Path

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