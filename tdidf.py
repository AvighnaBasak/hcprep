import json
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def load_chunks(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def rank_chunks(chunks, query, score_threshold=0.05, min_chunks=5):
    #------empty chunk removal
    filtered_chunks = [c for c in chunks if c['content'].strip()]
    texts = [c['content'] for c in filtered_chunks]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()

    #-----reaarrange decending order
    sorted_indices = np.argsort(scores)[::-1]

    #selecting the chunks above the threshhold
    selected_indices = [i for i in sorted_indices if scores[i] > score_threshold]

    #min chunks---
    if len(selected_indices) < min_chunks:
        selected_indices = sorted_indices[:min_chunks]

    ranked = []
    for rank, idx in enumerate(selected_indices, start=1):
        chunk = filtered_chunks[idx].copy()
        chunk['rank'] = rank
        chunk['score'] = float(scores[idx])
        ranked.append(chunk)

    return ranked

#main--
if __name__ == "__main__":
    chunks = load_chunks("generic_chunks.json")

    query = "summarize the whole text"

    ranked_chunks = rank_chunks(chunks, query, score_threshold=0.06, min_chunks=10)

    print(f"Selected {len(ranked_chunks)} chunks for query: '{query}'\n")
    pprint(ranked_chunks)

    with open("ranked_chunks.json", "w", encoding="utf-8") as f:
        json.dump(ranked_chunks, f, indent=2)

    print("\nSaved ranked chunks to ranked_chunks.json")
