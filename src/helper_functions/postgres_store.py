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
                    embedding VECTOR(3072) NOT NULL,
                    embedding_model TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
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

    def search_similar_embeddings(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        Search for the most similar embeddings using cosine distance.
        
        Args:
            query_embedding: The embedding vector to search for
            top_k: Number of top matches to return
            
        Returns:
            List of dictionaries containing similar indicators with similarity scores
        """
        with self.conn.cursor() as cur:
            try:
                # Convert query embedding to proper vector format
                query_vector = f"[{','.join(map(str, query_embedding))}]"
                
                # Use cosine distance for similarity search with explicit casting
                cur.execute("""
                    SELECT 
                        indicator_id,
                        indicator_name,
                        description,
                        embedding <=> %s::vector AS distance,
                        1 - (embedding <=> %s::vector) AS similarity
                    FROM indicator_embeddings 
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s;
                """, (query_vector, query_vector, query_vector, top_k))
                
                results = cur.fetchall()
                
                # Convert results to list of dictionaries
                similar_indicators = []
                for row in results:
                    similar_indicators.append({
                        "indicator_id": row[0],
                        "indicator_name": row[1], 
                        "description": row[2],
                        "distance": float(row[3]),
                        "similarity": float(row[4])
                    })
                
                return similar_indicators
                
            except Exception as e:
                print(f"Error during similarity search: {e}")
                return []
