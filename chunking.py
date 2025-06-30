import re
import json
from pprint import pprint

def clean(text):
    return text.strip().replace('\n', ' ').replace('\r', '').strip()

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

if __name__ == "__main__":
    file_path = "test.txt"

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Try play format first
    play_chunks = chunk_play_format(lines)

    # If we got multiple ACT/SCENE chunks, use it
    if len(play_chunks) > 3:
        chunks = play_chunks
        print(f"✅ Detected play format. Chunked into {len(chunks)} ACT/SCENE sections.")
    else:
        # Fallback to generic format
        chunks = chunk_generic_format(lines)
        print(f"⚙️ No ACT/SCENE detected. Using generic chunking... {len(chunks)} chunks created.")

    # Show preview
    pprint(chunks[:2])

    # Save chunks
    with open("generic_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("\n✅ Saved chunks to generic_chunks.json")
