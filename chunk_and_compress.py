import re
import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ---- Text Cleaning ----
def clean(text):
    return text.strip().replace('\n', ' ').replace('\r', '').strip()

# ---- Chunking Functions ----
def chunk_play_format(lines):
    chunks = []
    current_act = None
    current_scene = None
    buffer = []
    act_id = 0
    scene_id = 0

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Detect ACT lines like "ACT I"
        if re.match(r'^ACT\s+[IVXLC]+$', line_clean, re.IGNORECASE):
            # Save previous scene if any
            if current_scene and buffer:
                content = clean("\n".join(buffer))
                chunks.append({
                    "id": f"{act_id}.{scene_id}",
                    "heading": f"{current_act} - {current_scene}",
                    "depth": 2,
                    "parent_id": str(act_id),
                    "content": content
                })
                buffer = []
                current_scene = None

            act_id += 1
            scene_id = 0
            current_act = line_clean
            chunks.append({
                "id": str(act_id),
                "heading": current_act,
                "depth": 1,
                "parent_id": None,
                "content": ""
            })

        # Detect SCENE lines like "SCENE I. A street."
        elif re.match(r'^SCENE\s+[\w\d\.]+', line_clean, re.IGNORECASE):
            # Save previous scene
            if current_scene and buffer:
                content = clean("\n".join(buffer))
                chunks.append({
                    "id": f"{act_id}.{scene_id}",
                    "heading": f"{current_act} - {current_scene}",
                    "depth": 2,
                    "parent_id": str(act_id),
                    "content": content
                })
                buffer = []

            scene_id += 1
            current_scene = line_clean

        else:
            buffer.append(line_clean)

    # Save last scene
    if current_scene and buffer:
        content = clean("\n".join(buffer))
        chunks.append({
            "id": f"{act_id}.{scene_id}",
            "heading": f"{current_act} - {current_scene}",
            "depth": 2,
            "parent_id": str(act_id),
            "content": content
        })

    return chunks

def chunk_generic_format(lines, max_chunk_lines=30):
    chunks = []
    buffer = []
    chunk_id = 1

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        buffer.append(line_clean)
        if len(buffer) >= max_chunk_lines:
            content = clean("\n".join(buffer))
            chunks.append({
                "id": str(chunk_id),
                "heading": f"Chunk {chunk_id}",
                "depth": 1,
                "parent_id": None,
                "content": content
            })
            chunk_id += 1
            buffer = []

    if buffer:
        content = clean("\n".join(buffer))
        chunks.append({
            "id": str(chunk_id),
            "heading": f"Chunk {chunk_id}",
            "depth": 1,
            "parent_id": None,
            "content": content
        })

    return chunks

# ---- Save Final Output ----
def save_compressed_output(chunks, output_path="final_autoencoder_compressed.txt"):
    print("\n🧾 Final sections being saved:")
    formatted = []
    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if content:
            formatted.append(content)
            print(f"- {chunk['heading']} | {len(content)} chars")
    merged = "\n\n".join(formatted)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)
    print(f"\n✅ Final compressed text saved to {output_path}.")

# ---- Embedding & Selection ----
def process_chunks(chunks, rounds=1, top_k_ratio=0.2):
    print(f"\n🚀 Autoencoder Compression | Rounds: {rounds}, Top {int(top_k_ratio * 100)}% each round")
    texts = [chunk["content"] for chunk in chunks]
    print(f"📥 Loading SentenceTransformer autoencoder model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for r in range(rounds):
        print(f"\n🔁 Round {r + 1}/{rounds}")
        print("📌 Encoding chunks...")
        embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)
        k = max(1, int(len(embeddings) * top_k_ratio))
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)
        selected = []
        for label in set(labels):
            idxs = [i for i, lb in enumerate(labels) if lb == label]
            center = km.cluster_centers_[label]
            best_i = max(idxs, key=lambda i: cosine_similarity([embeddings[i]], [center])[0][0])
            selected.append(best_i)
        selected = sorted(selected)
        texts = [texts[i] for i in selected]
        chunks = [chunks[i] for i in selected]
        print(f"📊 Selected {len(chunks)} chunks for round {r + 1}")
    return chunks

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and compress a text file using autoencoder-based clustering.")
    parser.add_argument("--input", type=str, default="test.txt", help="Input text file to chunk and compress")
    parser.add_argument("--rounds", type=int, default=1, help="Number of refinement rounds")
    parser.add_argument("--ratio", type=float, default=0.2, help="Top-K chunk ratio per round (default=0.2)")
    parser.add_argument("--max_chunk_lines", type=int, default=30, help="Max lines per chunk for generic chunking")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    play_chunks = chunk_play_format(lines)
    if len(play_chunks) > 3:
        chunks = play_chunks
        print(f"✅ Detected play format. Chunked into {len(chunks)} ACT/SCENE sections.")
    else:
        chunks = chunk_generic_format(lines, max_chunk_lines=args.max_chunk_lines)
        print(f"⚙️ No ACT/SCENE detected. Using generic chunking... {len(chunks)} chunks created.")

    # Show preview
    print("\nPreview of first 2 chunks:")
    print(json.dumps(chunks[:2], indent=2))

    # Compress
    compressed_chunks = process_chunks(chunks, rounds=args.rounds, top_k_ratio=args.ratio)
    save_compressed_output(compressed_chunks) 