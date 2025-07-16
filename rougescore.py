from rouge_score import rouge_scorer

# Read summaries
with open("test_summary.txt", "r", encoding="utf-8") as f:
    ref_summary = f.read()

with open("summary.txt", "r", encoding="utf-8") as f:
    llm_summary = f.read()

with open("hcprepsummary.txt", "r", encoding="utf-8") as f:
    compressed_summary = f.read()

# Initialize scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Compare
print("\n🔍 LLM Summary vs Human:")
llm_scores = scorer.score(ref_summary, llm_summary)
for k, v in llm_scores.items():
    print(f"{k}: Precision={v.precision:.3f}, Recall={v.recall:.3f}, F1={v.fmeasure:.3f}")

print("\n🔍 Compressed Summary (HCPreP) vs Human:")
compressed_scores = scorer.score(ref_summary, compressed_summary)
for k, v in compressed_scores.items():
    print(f"{k}: Precision={v.precision:.3f}, Recall={v.recall:.3f}, F1={v.fmeasure:.3f}")
