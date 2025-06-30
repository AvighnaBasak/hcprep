import json
from pprint import pprint
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load your SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and accurate

def load_chunks(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def rank_chunks(chunks, query, score_threshold=0.2, min_chunks=10):
    # Filter out empty chunks
    filtered_chunks = [c for c in chunks if c['content'].strip()]
    texts = [c['content'] for c in filtered_chunks]

    # Get embeddings for chunks and query
    chunk_embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

    # Compute cosine similarities
    scores = util.cos_sim(query_embedding, chunk_embeddings).squeeze().cpu().numpy()

    # Rank chunks by score
    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = [i for i in sorted_indices if scores[i] > score_threshold]

    # Ensure a minimum number of chunks
    if len(selected_indices) < min_chunks:
        selected_indices = sorted_indices[:min_chunks]

    ranked = []
    for rank, idx in enumerate(selected_indices, start=1):
        chunk = filtered_chunks[idx].copy()
        chunk['rank'] = rank
        chunk['score'] = float(scores[idx])
        ranked.append(chunk)

    return ranked

# ----------- Main -----------------
if __name__ == "__main__":
    chunks = load_chunks("generic_chunks.json")
    query = "summarize the text"
    ranked_chunks = rank_chunks(chunks, query, score_threshold=0.2, min_chunks=10)

    print(f"Selected {len(ranked_chunks)} chunks for query: '{query}'\n")
    pprint(ranked_chunks)

    with open("ranked_chunks.json", "w", encoding="utf-8") as f:
        json.dump(ranked_chunks, f, indent=2)

    print("\n✅ Saved ranked chunks to ranked_chunks.json")
