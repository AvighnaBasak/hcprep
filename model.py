import json
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

    with open("combined_summary.txt", "w", encoding="utf-8") as f:
        f.write(full_summary)

    print("✅ Done! Saved to combined_summary.txt")

if __name__ == "__main__":
    main()