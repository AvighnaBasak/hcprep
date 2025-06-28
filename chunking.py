import re
import json
from pprint import pprint

def clean(text):
    return text.strip().replace('\n', ' ').replace('\r', '').strip()

def chunk_format(filepath):
    chunks = []
    current_act = None
    current_scene = None
    buffer = []
    act_id = 0
    scene_id = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        #-------detect-act------------------------------
        if re.match(r'^ACT\s+[IVXLC]+$', line_clean, re.IGNORECASE):
            if current_scene and buffer:
                #----------------save the previos scene----------------------
                chunks.append({
                    "id": f"{act_id}.{scene_id}",
                    "heading": f"{current_act} - {current_scene}",
                    "depth": 2,
                    "parent_id": f"{act_id}",
                    "content": "\n".join([l.strip() for l in buffer]).strip()
                })
                buffer = []
                current_scene = None

            act_id += 1
            scene_id = 0
            current_act = line_clean
            chunks.append({
                "id": f"{act_id}",
                "heading": current_act,
                "depth": 1,
                "parent_id": None,
                "content": ""  
            })

        elif re.match(r'^SCENE\s+[\w\d\.]+', line_clean, re.IGNORECASE):
            if current_scene and buffer:
                
                chunks.append({
                    "id": f"{act_id}.{scene_id}",
                    "heading": f"{current_act} - {current_scene}",
                    "depth": 2,
                    "parent_id": f"{act_id}",
                    "content": "\n".join([l.strip() for l in buffer]).strip()
                })
                buffer = []

            scene_id += 1
            current_scene = line_clean

        else:
            buffer.append(line)

    #-------------- last scene-------------------------
    if current_scene and buffer:
        chunks.append({
            "id": f"{act_id}.{scene_id}",
            "heading": f"{current_act} - {current_scene}",
            "depth": 2,
            "parent_id": f"{act_id}",
            "content": "\n".join([l.strip() for l in buffer]).strip()
        })

    return chunks

#-------main fucn-----------------
if __name__ == "__main__":
    file_path = "test.txt" 
    chunks = chunk_format(file_path)

    if not chunks:
        print("No chunks detected.")
    else:
        print(f" Chunked {len(chunks)} sections from {file_path}")
        pprint(chunks[:2])

        with open("generic_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print("\nSaved to generic_chunks.json")
