import os
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from typing import List, Optional, Dict

class PostgresEmbeddingStore:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self._connect()
        self._ensure_schema()

    def _connect(self):
        self.conn = psycopg2.connect(
            host=self.db_config.get("host", "localhost"),
            port=self.db_config.get("port", 5432),
            dbname=self.db_config.get("dbname"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password")
        )
        self.conn.autocommit = True

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;

                CREATE TABLE IF NOT EXISTS indicator_embeddings (
                    indicator_id TEXT PRIMARY KEY,
                    indicator_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding VECTOR(1536) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    source TEXT DEFAULT 'FRED',
                    geography TEXT,
                    frequency TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                -- Add columns if they don't exist (for backward compatibility)
                ALTER TABLE indicator_embeddings 
                ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'FRED';
                
                ALTER TABLE indicator_embeddings 
                ADD COLUMN IF NOT EXISTS geography TEXT;
                
                ALTER TABLE indicator_embeddings 
                ADD COLUMN IF NOT EXISTS frequency TEXT;
            """)
            print("PostgreSQL schema ensured.")

    def embedding_exists(self, indicator_id: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM indicator_embeddings WHERE indicator_id = %s", (indicator_id,))
            return cur.fetchone() is not None

    def save_embedding(self, indicator_id: str, indicator_name: str, description: str,
                       embedding: List[float], model: str = "text-embedding-3-small") -> bool:
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO indicator_embeddings (
                        indicator_id, indicator_name, description, embedding, embedding_model, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (indicator_id) DO UPDATE SET
                        indicator_name = EXCLUDED.indicator_name,
                        description = EXCLUDED.description,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = EXCLUDED.updated_at;
                """, (
                    indicator_id,
                    indicator_name,
                    description,
                    embedding,
                    model,
                    datetime.utcnow()
                ))
                return True
            except Exception as e:
                print(f"Error saving embedding {indicator_id}: {e}")
                return False

    def store_embedding(self, text: str, metadata: Dict, model: str = "text-embedding-3-small") -> bool:
        """
        Store embedding with metadata support for both FRED and World Bank data
        
        Args:
            text: The text to create embedding from
            metadata: Dictionary containing indicator metadata
            model: The embedding model to use
            
        Returns:
            True if successful, False otherwise
        """
        import openai
        
        try:
            # Create embedding
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(input=text, model=model)
            embedding = response.data[0].embedding
            
            # Store with metadata
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO indicator_embeddings (
                        indicator_id, indicator_name, description, embedding, embedding_model, 
                        source, geography, frequency, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (indicator_id) DO UPDATE SET
                        indicator_name = EXCLUDED.indicator_name,
                        description = EXCLUDED.description,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        source = EXCLUDED.source,
                        geography = EXCLUDED.geography,
                        frequency = EXCLUDED.frequency,
                        updated_at = EXCLUDED.updated_at;
                """, (
                    metadata.get("indicator_id"),
                    metadata.get("indicator_name"),
                    text,  # Use the text as description
                    embedding,
                    model,
                    metadata.get("source", "FRED"),
                    metadata.get("geography"),
                    metadata.get("frequency"),
                    datetime.utcnow()
                ))
                return True
                
        except Exception as e:
            print(f"Error storing embedding for {metadata.get('indicator_id', 'unknown')}: {e}")
            return False

    def save_embeddings_batch(self, records: List[Dict], model: str = "text-embedding-3-small") -> int:
        if not records:
            return 0
        rows = [(
            r["indicator_id"],
            r["indicator_name"],
            r["description"],
            r["embedding"],
            model,
            datetime.utcnow()
        ) for r in records]

        with self.conn.cursor() as cur:
            try:
                execute_values(cur, """
                    INSERT INTO indicator_embeddings (
                        indicator_id, indicator_name, description, embedding, embedding_model, updated_at
                    ) VALUES %s
                    ON CONFLICT (indicator_id) DO UPDATE SET
                        indicator_name = EXCLUDED.indicator_name,
                        description = EXCLUDED.description,
                        embedding = EXCLUDED.embedding,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = EXCLUDED.updated_at;
                """, rows)
                print(f"Batch inserted {len(rows)} embeddings.")
                return len(rows)
            except Exception as e:
                print(f"Batch insertion failed: {e}")
                return 0

    def close(self):
        if self.conn:
            self.conn.close()
            print("🔌 PostgreSQL connection closed.")

    def search_similar_embeddings(self, query_embedding: List[float], top_k: int = 3, 
                                  filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for the most similar embeddings using cosine distance.
        
        Args:
            query_embedding: The embedding vector to search for
            top_k: Number of top matches to return
            filter_metadata: Optional dictionary to filter results by metadata (e.g., {"source": "World Bank"})
            
        Returns:
            List of dictionaries containing similar indicators with similarity scores
        """
        with self.conn.cursor() as cur:
            try:
                # Convert query embedding to proper vector format
                query_vector = f"[{','.join(map(str, query_embedding))}]"
                
                # Build WHERE clause and parameters separately
                where_clause = ""
                filter_params = []
                
                if filter_metadata:
                    conditions = []
                    for key, value in filter_metadata.items():
                        if key in ['source', 'geography', 'frequency']:
                            conditions.append(f"{key} = %s")
                            filter_params.append(value)
                    
                    if conditions:
                        where_clause = "WHERE " + " AND ".join(conditions)
                
                # Build the SQL query
                base_query = f"""
                    SELECT 
                        indicator_id,
                        indicator_name,
                        description,
                        source,
                        geography,
                        frequency,
                        embedding <=> %s::vector AS distance,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM indicator_embeddings 
                    {where_clause}
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """
                
                # Build final parameters: 3 query_vectors + filter_params + limit
                final_params = [query_vector, query_vector, query_vector] + filter_params + [top_k]
                
                cur.execute(base_query, final_params)
                results = cur.fetchall()
                
                # Convert results to list of dictionaries
                similar_indicators = []
                for row in results:
                    similar_indicators.append({
                        "indicator_id": row[0],
                        "indicator_name": row[1], 
                        "description": row[2],
                        "source": row[3],
                        "geography": row[4],
                        "frequency": row[5],
                        "distance": float(row[6]),
                        "similarity": float(row[7])
                    })
                
                return similar_indicators
                
            except Exception as e:
                print(f"Error during similarity search: {e}")
                return []
