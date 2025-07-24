# FRED Query Agent

An intelligent agentic framework built with OpenAI's Agents SDK for querying economic data from the FRED database.

## Overview

The Query Agent provides a conversational interface to:
1. Search the local database for economic indicators
2. Extract time series data for specific indicators  
3. Ingest new data from FRED when local data is insufficient
4. Present results in a user-friendly format

## Features

- **Smart Search**: Uses semantic search with embeddings to find relevant indicators
- **Auto-Ingestion**: Automatically searches FRED and ingests new data when needed
- **Query Variations**: Generates multiple search variations to improve match quality
- **Similarity Threshold**: Only presents high-quality matches (similarity ≥ 0.7)
- **Interactive Interface**: Conversational agent that explains its actions

## Setup

### 1. Install Dependencies

The required dependencies are already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Configuration

The agent uses existing configuration files:
- `FRED.yaml` - FRED API key and database config
- `prompts.yaml` - Agent prompts and instructions

## Usage

### Interactive Mode

```python
from helper_functions.query_agent import QueryAgent
import asyncio

async def main():
    agent = QueryAgent()
    
    # Process a query
    response = await agent.run_query("Show me unemployment data for the US")
    print(response)

asyncio.run(main())
```

### Direct Script Execution

```bash
cd src
python helper_functions/query_agent.py
```

This will start an interactive session where you can enter queries.

### Test Script

```bash
cd src  
python test_query_agent.py
```

## Agent Workflow

1. **Database Search**: First searches local database for matching indicators
2. **Quality Check**: Filters results by similarity threshold (≥ 0.7)
3. **Data Extraction**: If good matches found, extracts and presents data
4. **FRED Ingestion**: If no good matches, searches FRED for new data
5. **Re-search**: After ingestion, searches database again for better matches
6. **Results**: Presents final results or apologizes if no relevant data found

## Available Tools

The agent has access to three main tools:

### `search_database(query, max_attempts=3)`
- Searches local database using semantic similarity
- Tries up to 3 query variations
- Returns formatted results with similarity scores

### `extract_data(indicator_id)`  
- Extracts time series data for specific indicator
- Returns metadata and recent data points
- Handles missing data gracefully

### `ingest_from_fred(query, country="USA")`
- Searches FRED database for new indicators
- Ingests new data and generates embeddings
- Updates local database and dictionary

## Example Queries

- "Show me unemployment data for the US"
- "I need inflation statistics"
- "What's the GDP growth rate?"
- "Employment trends in manufacturing"
- "Consumer price index data"

## Configuration

### Similarity Threshold
Default: 0.7 (configurable in `prompts.yaml`)

### Max Search Attempts  
Default: 3 (configurable in `prompts.yaml`)

### Model Settings
- Model: gpt-4o
- Temperature: 0.3
- Max Tokens: 150

## Error Handling

The agent handles various error conditions:
- Missing API keys
- Database connection issues
- FRED API errors
- Invalid indicator IDs
- Empty search results

## Architecture

```
QueryAgent
├── OpenAI Agents SDK (agent framework)
├── PostgreSQL + pgvector (embeddings storage)
├── FRED API (data source)
├── LLM Client (query processing)
└── Existing tools (search, extract, ingest)
```

## Extending the Agent

To add new capabilities:

1. Define a new function with `@function_tool` decorator
2. Add the function to the agent's tools list
3. Update prompts in `prompts.yaml` if needed
4. Test with various query types

## Troubleshooting

### "OpenAI API key not set"
- Create `.env` file with `OPENAI_API_KEY=your_key`
- Ensure `.env` is in project root directory

### "FRED API key missing"
- Check `FRED.yaml` has valid `api_key` field
- Verify FRED API key is active

### "Database connection failed"
- Ensure PostgreSQL is running
- Check database config in `FRED.yaml`
- Verify pgvector extension is installed

### "No embeddings found"
- Run the main FRED agent first to populate database
- Check if `FRED_DataDictionary.xlsx` exists
- Verify embeddings were generated successfully 