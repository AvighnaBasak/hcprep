#this is the model.py file
#it is not part of the pipeline, it is used to summarize the text file using the LLaMA3 model

import argparse
import requests
import os

# Define your summarization prompt format
def format_prompt(content):
    return f"""Summarize the following text in modern English in about 800 words. Do not include any introduction, commentary, or references to the summary or the source. Only output the summary itself, as direct and concise as possible.

{content}

"""

# Send request to local LLaMA3 (Ollama)
def summarize_text(content):
    prompt = format_prompt(content)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    data = response.json()
    return data["response"].strip()

def summarize_file(input_path, output_path):
    print(f"📖 Reading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"🧠 Summarizing {input_path} ... (this may take a while for large files)")
    summary = summarize_text(content)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✅ Done! Summary saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Summarize a text file using LLaMA3.")
    parser.add_argument("--input", type=str, help="Input file to summarize (e.g., test.txt or final_autoencoder_compressed.txt)")
    parser.add_argument("--output", type=str, help="Output file for summary (e.g., summary.txt or hcprepsummary.txt)")
    args = parser.parse_args()

    if args.input and args.output:
        summarize_file(args.input, args.output)
        return

    # Default behavior: summarize both files if no args
    did_any = False
    if os.path.exists("test.txt"):
        summarize_file("test.txt", "summary.txt")
        did_any = True
    if os.path.exists("final_autoencoder_compressed.txt"):
        summarize_file("final_autoencoder_compressed.txt", "hcprepsummary.txt")
        did_any = True
    if not did_any:
        print("No input files found. Please provide --input and --output arguments, or ensure test.txt or final_autoencoder_compressed.txt exist in the current directory.")

if __name__ == "__main__":
    main()
































'''import json
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


def load_chunks(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
llm = Ollama(model="gemma3:1b")

prompt_template = PromptTemplate(
    input_variables=["content"],
    template="Summarize the following Shakespearean passage in modern English:\n\n{content}\n\nSummary:")


def summarize_chunk(content):
    prompt = prompt_template.format(content=content)
    return llm.invoke(prompt).strip()


def main():
    chunks = load_chunks("ranked_chunks.json")

    summaries = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}...")
        summary = summarize_chunk(chunk['content'])
        summaries.append(f"Summary {i} - {chunk['heading']}:\n{summary}\n")

    full_summary = "\n".join(summaries)

    with open("final_hcprep_compressed.txt", "w", encoding="utf-8") as f:
        f.write(full_summary)

    print("✅ Done! Saved to final_hcprep_compressed.txt")

if __name__ == "__main__":
    main()'''