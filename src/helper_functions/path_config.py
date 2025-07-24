import os
# Contains absolute paths of all the important folders in code

PROJECT_DIR = (str(os.path.realpath(__file__)))[:-36]

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
