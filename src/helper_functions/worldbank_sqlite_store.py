#!/usr/bin/env python3
"""
SQLite-based embedding store for World Bank data
Simple, reliable alternative to PostgreSQL vector store
"""

import sqlite3
import json
import os
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import openai
from dotenv import load_dotenv

class WorldBankSQLiteStore:
    """SQLite-based embedding store for World Bank indicators"""
    
    def __init__(self, db_path: str = "worldbank_embeddings.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database and create tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS worldbank_embeddings (
                indicator_id TEXT PRIMARY KEY,
                indicator_name TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding TEXT NOT NULL,  -- JSON string of embedding vector
                source TEXT DEFAULT 'World Bank',
                geography TEXT,
                frequency TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        print("✅ SQLite World Bank embedding store initialized")
    
    def store_embedding(self, text: str, metadata: Dict, model: str = "text-embedding-3-small") -> bool:
        """Store embedding with metadata"""
        load_dotenv()
        
        try:
            # Create embedding
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(input=text, model=model)
            embedding = response.data[0].embedding
            
            # Store in SQLite
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO worldbank_embeddings (
                    indicator_id, indicator_name, description, embedding, 
                    source, geography, frequency, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.get("indicator_id"),
                metadata.get("indicator_name"),
                text,
                json.dumps(embedding),  # Store as JSON string
                metadata.get("source", "World Bank"),
                metadata.get("geography"),
                metadata.get("frequency"),
                datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False
    
    def search_similar_embeddings(self, query_embedding: List[float], top_k: int = 3, 
                                  filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar embeddings using cosine similarity"""
        try:
            cursor = self.conn.cursor()
            
            # Build WHERE clause for filtering
            where_clause = "WHERE 1=1"
            params = []
            
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key in ['source', 'geography', 'frequency']:
                        where_clause += f" AND {key} = ?"
                        params.append(value)
            
            # Get all embeddings
            cursor.execute(f"""
                SELECT indicator_id, indicator_name, description, embedding, 
                       source, geography, frequency
                FROM worldbank_embeddings 
                {where_clause}
            """, params)
            
            results = cursor.fetchall()
            
            if not results:
                return []
            
            # Calculate cosine similarities
            similarities = []
            query_vector = np.array(query_embedding)
            
            for row in results:
                stored_embedding = json.loads(row[3])  # Parse JSON embedding
                stored_vector = np.array(stored_embedding)
                
                # Cosine similarity
                dot_product = np.dot(query_vector, stored_vector)
                norm_query = np.linalg.norm(query_vector)
                norm_stored = np.linalg.norm(stored_vector)
                
                if norm_query > 0 and norm_stored > 0:
                    similarity = dot_product / (norm_query * norm_stored)
                else:
                    similarity = 0.0
                
                similarities.append({
                    "indicator_id": row[0],
                    "indicator_name": row[1],
                    "description": row[2],
                    "source": row[4],
                    "geography": row[5],
                    "frequency": row[6],
                    "similarity": float(similarity),
                    "distance": float(1 - similarity)
                })
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error during SQLite similarity search: {e}")
            return []
    
    def count_embeddings(self, source: str = "World Bank") -> int:
        """Count embeddings by source"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM worldbank_embeddings WHERE source = ?", (source,))
        return cursor.fetchone()[0]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔌 SQLite World Bank store closed")


# Test function
def test_worldbank_sqlite_store():
    """Test the SQLite store functionality"""
    print("🧪 Testing World Bank SQLite Store...")
    
    store = WorldBankSQLiteStore("test_worldbank.db")
    
    # Test storing an embedding
    test_metadata = {
        "indicator_id": "TEST.GDP.PCAP",
        "indicator_name": "Test GDP per capita",
        "source": "World Bank",
        "geography": "Test Country",
        "frequency": "Annual"
    }
    
    success = store.store_embedding("Test GDP per capita indicator", test_metadata)
    print(f"✅ Store test: {'PASS' if success else 'FAIL'}")
    
    # Test counting
    count = store.count_embeddings()
    print(f"✅ Count test: {count} embeddings")
    
    store.close()
    
    # Clean up test file
    if os.path.exists("test_worldbank.db"):
        os.remove("test_worldbank.db")
    
    print("🎉 SQLite store test complete!")


if __name__ == "__main__":
    test_worldbank_sqlite_store() 