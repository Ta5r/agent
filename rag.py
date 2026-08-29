import os
import json
import time
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()

KB_DIR = os.path.join("data", "knowledge_base")
CACHE_PATH = os.path.join("data", "rag_cache.json")

# In-memory index of chunks: list of {"text": str, "source": str, "embedding": list}
_vector_index = []

def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client()
    except Exception:
        return None

def extract_pdf_chunks(pdf_path: str, chunk_size: int = 500, overlap: int = 100):
    """Extracts text from a PDF and chunks it with overlap."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    # Simple character-based sliding window chunking
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
        
    return chunks

def build_or_load_index():
    """Reads PDF files, processes them, caches embeddings, and loads into memory."""
    global _vector_index
    _vector_index = []
    
    os.makedirs(KB_DIR, exist_ok=True)
    
    # Load cache if it exists
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass
            
    pdf_files = [f for f in os.listdir(KB_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("RAG: No PDFs found in", KB_DIR)
        return
        
    client = get_genai_client()
    cache_updated = False
    
    for filename in pdf_files:
        pdf_path = os.path.join(KB_DIR, filename)
        mtime = os.path.getmtime(pdf_path)
        
        # Check if cache is valid for this file
        if filename in cache and cache[filename].get("mtime") == mtime:
            print(f"RAG: Loading '{filename}' from cache...")
            for chunk_data in cache[filename].get("chunks", []):
                _vector_index.append({
                    "text": chunk_data["text"],
                    "source": filename,
                    "embedding": chunk_data["embedding"]
                })
            continue
            
        # File is new or changed - need to re-index
        if not client:
            print(f"RAG: GEMINI_API_KEY is not set. Cannot index new file '{filename}'.")
            continue
            
        print(f"RAG: Indexing new/modified file '{filename}'...")
        chunks = extract_pdf_chunks(pdf_path)
        
        if not chunks:
            continue
            
        # Batch embed the chunks
        embedded_chunks = []
        try:
            # We call embed_content in batches or one-by-one. One-by-one is simple and robust
            for chunk in chunks:
                response = client.models.embed_content(
                    model="gemini-embedding-2",
                    contents=chunk
                )
                embedding = response.embeddings[0].values
                embedded_chunks.append({
                    "text": chunk,
                    "embedding": embedding
                })
                # Prevent rate-limiting just in case
                time.sleep(0.1)
                
            # Save to memory index
            for ec in embedded_chunks:
                _vector_index.append({
                    "text": ec["text"],
                    "source": filename,
                    "embedding": ec["embedding"]
                })
                
            # Update cache
            cache[filename] = {
                "mtime": mtime,
                "chunks": embedded_chunks
            }
            cache_updated = True
            
        except APIError as e:
            print(f"RAG Error: Failed to embed '{filename}' due to API error: {e}")
        except Exception as e:
            print(f"RAG Error: Unexpected error while indexing '{filename}': {e}")
            
    if cache_updated:
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
            print("RAG: Embeddings cache updated on disk.")
        except Exception as e:
            print(f"RAG Error: Failed to save cache: {e}")

def cosine_similarity(v1, v2):
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    dot = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def search_knowledge_base(query: str, top_k: int = 2):
    """Searches the indexed chunks for matches to the query using cosine similarity."""
    global _vector_index
    
    if not _vector_index:
        # Try loading index first
        build_or_load_index()
        if not _vector_index:
            return []
            
    client = get_genai_client()
    if not client:
        print("RAG Search Warning: GEMINI_API_KEY is not set. RAG search disabled.")
        return []
        
    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=query
        )
        query_vector = response.embeddings[0].values
    except Exception as e:
        print(f"RAG Search Error: Failed to embed query: {e}")
        return []
        
    results = []
    for item in _vector_index:
        sim = cosine_similarity(query_vector, item["embedding"])
        results.append({
            "text": item["text"],
            "source": item["source"],
            "score": sim
        })
        
    # Sort by similarity score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
