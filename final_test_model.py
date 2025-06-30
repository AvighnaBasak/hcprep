import requests
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def call_llama3(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Failed to generate summary: {response.status_code}, {response.text}")

def summarize_file(input_path: str, output_path: str, refined: bool = False):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if refined:
        prompt = (
            "The following text contains multiple scene-level summaries of a play. "
            "Please refine, compress, and combine them into a single high-quality overall summary "
            "that captures the main events, themes, and character arcs clearly and concisely:\n\n"
            + text
        )
    else:
        prompt = (
            "Summarize the following long text in clear, concise bullet points or paragraphs "
            "that capture the key ideas, events, and characters:\n\n"
            + text
        )

    print(f"⏳ Summarizing {input_path}...")
    summary = call_llama3(prompt)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    summarize_file("final_hcprep_compressed.txt", "summary_with_hcprep.txt", refined=True)
    summarize_file("test.txt", "summary_without_hcprep.txt", refined=False)
