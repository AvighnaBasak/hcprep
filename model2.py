import json
import re
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Required libraries for evaluations
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import textstat
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
except ImportError as e:
    print(f"Missing required libraries. Please install:")
    print("pip install sentence-transformers scikit-learn nltk textstat")
    raise e

def load_chunks(json_path: str) -> List[Dict]:
    """Load chunks from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class ObjectiveEvaluator:
    """Objective evaluation metrics for text summarization."""
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_content_words(self, text: str) -> Set[str]:
        """Extract content words (non-stopwords, stemmed)."""
        words = word_tokenize(self.clean_text(text))
        content_words = set()
        for word in words:
            if word not in self.stop_words and len(word) > 2:
                content_words.add(self.stemmer.stem(word))
        return content_words
    
    def evaluate_all_metrics(self, original: str, summary: str) -> Dict[str, float]:
        """Run all evaluation metrics."""
        evaluations = {}
        
        # 1. Word Count Ratio
        evaluations['word_count_ratio'] = float(self.word_count_ratio(original, summary))
        
        # 2. Semantic Similarity (Sentence Embeddings)
        evaluations['semantic_similarity'] = float(self.semantic_similarity(original, summary))
        
        # 3. TF-IDF Cosine Similarity
        evaluations['tfidf_similarity'] = float(self.tfidf_similarity(original, summary))
        
        # 4. Content Word Overlap
        evaluations['content_word_overlap'] = float(self.content_word_overlap(original, summary))
        
        # 5. N-gram Overlap (Bigrams)
        evaluations['bigram_overlap'] = float(self.ngram_overlap(original, summary, n=2))
        
        # 6. N-gram Overlap (Trigrams)
        evaluations['trigram_overlap'] = float(self.ngram_overlap(original, summary, n=3))
        
        # 7. Sentence Structure Preservation
        evaluations['sentence_structure_score'] = float(self.sentence_structure_preservation(original, summary))
        
        # 8. Readability Improvement
        evaluations['readability_improvement'] = float(self.readability_improvement(original, summary))
        
        # 9. Information Density
        evaluations['information_density'] = float(self.information_density(original, summary))
        
        # 10. Key Entity Preservation
        evaluations['entity_preservation'] = float(self.entity_preservation(original, summary))
        
        # Calculate overall score
        evaluations['overall_score'] = float(np.mean(list(evaluations.values())))
        
        return evaluations
    
    def word_count_ratio(self, original: str, summary: str) -> float:
        """Evaluate word count reduction ratio (0-1, higher is better compression)."""
        orig_words = len(original.split())
        summ_words = len(summary.split())
        
        if orig_words == 0:
            return 0.0
        
        ratio = summ_words / orig_words
        # Score: 1.0 for 30-70% of original length, decreasing outside this range
        if 0.3 <= ratio <= 0.7:
            return 1.0
        elif ratio < 0.3:
            return ratio / 0.3  # Penalize over-compression
        else:
            return max(0.0, 1.0 - (ratio - 0.7) / 0.3)  # Penalize under-compression
    
    def semantic_similarity(self, original: str, summary: str) -> float:
        """Calculate semantic similarity using sentence embeddings."""
        try:
            orig_embedding = self.sentence_model.encode([original])
            summ_embedding = self.sentence_model.encode([summary])
            similarity = cosine_similarity(orig_embedding, summ_embedding)[0][0]
            return float(max(0.0, similarity))  # Ensure non-negative and JSON serializable
        except Exception:
            return 0.0
    
    def tfidf_similarity(self, original: str, summary: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([original, summary])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(max(0.0, similarity))
        except Exception:
            return 0.0
    
    def content_word_overlap(self, original: str, summary: str) -> float:
        """Calculate overlap of content words."""
        orig_words = self.get_content_words(original)
        summ_words = self.get_content_words(summary)
        
        if len(orig_words) == 0:
            return 0.0
        
        intersection = len(orig_words.intersection(summ_words))
        return intersection / len(orig_words)
    
    def ngram_overlap(self, original: str, summary: str, n: int = 2) -> float:
        """Calculate n-gram overlap."""
        def get_ngrams(text: str, n: int) -> Set[tuple]:
            words = word_tokenize(self.clean_text(text))
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        orig_ngrams = get_ngrams(original, n)
        summ_ngrams = get_ngrams(summary, n)
        
        if len(orig_ngrams) == 0:
            return 0.0
        
        intersection = len(orig_ngrams.intersection(summ_ngrams))
        return intersection / len(orig_ngrams)
    
    def sentence_structure_preservation(self, original: str, summary: str) -> float:
        """Evaluate preservation of sentence structure patterns."""
        try:
            orig_sentences = sent_tokenize(original)
            summ_sentences = sent_tokenize(summary)
            
            # Calculate average sentence length similarity
            orig_avg_len = np.mean([len(sent.split()) for sent in orig_sentences])
            summ_avg_len = np.mean([len(sent.split()) for sent in summ_sentences])
            
            if orig_avg_len == 0:
                return 0.0
            
            length_similarity = 1.0 - abs(orig_avg_len - summ_avg_len) / orig_avg_len
            return float(max(0.0, length_similarity))
        except Exception:
            return 0.0
    
    def readability_improvement(self, original: str, summary: str) -> float:
        """Measure readability improvement (lower reading level is better)."""
        try:
            orig_grade = textstat.flesch_kincaid_grade(original)
            summ_grade = textstat.flesch_kincaid_grade(summary)
            
            # Score based on improvement (summary should be easier to read)
            if orig_grade <= summ_grade:
                return 0.5  # No improvement
            else:
                improvement = (orig_grade - summ_grade) / orig_grade
                return min(1.0, 0.5 + improvement)
        except Exception:
            return 0.5
    
    def information_density(self, original: str, summary: str) -> float:
        """Calculate information density (unique content words per word)."""
        orig_words = original.split()
        summ_words = summary.split()
        
        orig_unique_content = len(self.get_content_words(original))
        summ_unique_content = len(self.get_content_words(summary))
        
        if len(orig_words) == 0 or len(summ_words) == 0:
            return 0.0
        
        orig_density = orig_unique_content / len(orig_words)
        summ_density = summ_unique_content / len(summ_words)
        
        # Score: 1.0 if summary has higher or equal density
        if summ_density >= orig_density:
            return 1.0
        else:
            return summ_density / orig_density
    
    def entity_preservation(self, original: str, summary: str) -> float:
        """Evaluate preservation of named entities (capitalized words)."""
        def extract_entities(text: str) -> Set[str]:
            words = word_tokenize(text)
            entities = set()
            for word in words:
                if word.istitle() and len(word) > 2:
                    entities.add(word.lower())
            return entities
        
        orig_entities = extract_entities(original)
        summ_entities = extract_entities(summary)
        
        if len(orig_entities) == 0:
            return 1.0  # No entities to preserve
        
        preserved = len(orig_entities.intersection(summ_entities))
        return preserved / len(orig_entities)

class ShakespeareanSummarizer:
    def __init__(self, model_name: str = "gemma3:1b"):
        self.llm = Ollama(model=model_name)
        self.evaluator = ObjectiveEvaluator()
        
        # Enhanced detailed prompt for first iteration (Shakespearean text input)
        self.detailed_prompt = PromptTemplate(
            input_variables=["content", "heading"],
            template="""You are an expert in Shakespearean literature. Provide a comprehensive, detailed summary of the following passage in modern English.

INSTRUCTIONS:
- Preserve ALL key themes, character motivations, and plot points
- Maintain the emotional tone and literary significance
- Include specific details about relationships, conflicts, and imagery
- Explain archaic references and wordplay in modern terms
- Keep approximately 60-70% of the original length while making it accessible
- Use clear, engaging modern prose
- Only give the summary, do not include any additional commentary or analysis

Passage Title/Context: {heading}

Shakespearean Text:
{content}

Detailed Modern Summary:"""
        )
        
        # Condensing prompt for subsequent iterations (summary input)
        self.condensing_prompt = PromptTemplate(
            input_variables=["content", "target_reduction", "heading"],
            template="""Condense the following summary while preserving essential information.

INSTRUCTIONS:
- Reduce length by approximately {target_reduction}%
- Keep the most important themes, plot points, and character details
- Maintain clarity and readability
- Preserve the narrative flow and key insights
- Focus on the core story elements and character development

Section: {heading}

Current Summary:
{content}

Condensed Summary:"""
        )
    
    def summarize_chunk(self, content: str, heading: str = "") -> str:
        """Create initial detailed summary from Shakespearean text."""
        prompt = self.detailed_prompt.format(content=content, heading=heading)
        return self.llm.invoke(prompt).strip()
    
    def condense_summary(self, summary: str, heading: str = "", target_reduction: int = 25) -> str:
        """Condense an existing summary."""
        prompt = self.condensing_prompt.format(
            content=summary, 
            heading=heading,
            target_reduction=target_reduction
        )
        return self.llm.invoke(prompt).strip()
    
    def meets_quality_thresholds(self, evaluations: Dict[str, float], 
                                thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if evaluations meet quality thresholds."""
        failed_metrics = []
        
        for metric, threshold in thresholds.items():
            if metric in evaluations and evaluations[metric] < threshold:
                failed_metrics.append(f"{metric}: {evaluations[metric]:.3f} < {threshold}")
        
        return len(failed_metrics) == 0, failed_metrics
    
    def process_chunk_with_evaluations(self, content: str, heading: str = "", 
                                     thresholds: Dict[str, float] = None,
                                     max_attempts: int = 3,
                                     is_first_iteration: bool = True,
                                     original_content: str = None) -> Dict[str, any]:
        """Process a single chunk with quality evaluations."""
        if thresholds is None:
            thresholds = {
                'semantic_similarity': 0.7,
                'content_word_overlap': 0.6,
                'bigram_overlap': 0.4,
                'entity_preservation': 0.8,
                'word_count_ratio': 0.5,
                'tfidf_similarity': 0.5,
                'information_density': 0.6,
                'readability_improvement': 0.5,
                'sentence_structure_score': 0.4,
                'trigram_overlap': 0.3
            }
        
        # For evaluation, use original content if provided, otherwise use current content
        evaluation_reference = original_content if original_content is not None else content
        
        results = {
            'input_content': content,
            'heading': heading,
            'input_word_count': len(content.split()),
            'original_content': evaluation_reference,
            'original_word_count': len(evaluation_reference.split()),
            'attempts': [],
            'final_summary': '',
            'final_evaluations': {},
            'success': False,
            'is_first_iteration': is_first_iteration
        }
        
        for attempt in range(max_attempts):
            print(f"    Attempt {attempt + 1}/{max_attempts}")
            
            # Generate summary based on iteration type
            if is_first_iteration:
                summary = self.summarize_chunk(content, heading)
            else:
                summary = self.condense_summary(content, heading, 25)
            
            # Evaluate summary against the original content (not the immediate input)
            evaluations = self.evaluator.evaluate_all_metrics(evaluation_reference, summary)
            
            # Check if meets thresholds (only for first iteration)
            if is_first_iteration:
                meets_quality, failed_metrics = self.meets_quality_thresholds(evaluations, thresholds)
            else:
                # For subsequent iterations, accept the result (quality naturally degrades)
                meets_quality = True
                failed_metrics = []
            
            attempt_result = {
                'attempt': attempt + 1,
                'summary': summary,
                'word_count': len(summary.split()),
                'evaluations': evaluations,
                'meets_quality': meets_quality,
                'failed_metrics': failed_metrics
            }
            results['attempts'].append(attempt_result)
            
            if meets_quality:
                results['final_summary'] = summary
                results['final_evaluations'] = evaluations
                results['success'] = True
                if is_first_iteration:
                    print(f"    ✅ Quality thresholds met!")
                else:
                    print(f"    ✅ Summary condensed!")
                break
            else:
                print(f"    ❌ Failed metrics: {', '.join(failed_metrics)}")
        
        if not results['success']:
            # Use best attempt if none meet thresholds
            best_attempt = max(results['attempts'], key=lambda x: x['evaluations']['overall_score'])
            results['final_summary'] = best_attempt['summary']
            results['final_evaluations'] = best_attempt['evaluations']
            print(f"    ⚠️  Using best attempt (overall score: {best_attempt['evaluations']['overall_score']:.3f})")
        
        return results

def main():
    # Configuration
    N_ITERATIONS = 3  # Total number of iterations
    MAX_ATTEMPTS_PER_CHUNK = 3  # Max attempts per chunk before accepting best result
    
    # Quality thresholds - only applied to first iteration
    QUALITY_THRESHOLDS = {
        'semantic_similarity': 0.7,      # Semantic meaning preservation
        'content_word_overlap': 0.6,     # Important words preserved
        'bigram_overlap': 0.4,           # Phrase structure preservation
        'entity_preservation': 0.8,      # Named entities preserved
        'word_count_ratio': 0.5,         # Appropriate length reduction
        'tfidf_similarity': 0.5,         # Term importance preservation
        'information_density': 0.6,      # Information per word
        'readability_improvement': 0.5,   # Readability enhancement
        'sentence_structure_score': 0.4, # Sentence structure preservation
        'trigram_overlap': 0.3           # Complex phrase preservation
    }
    
    # Initialize summarizer
    summarizer = ShakespeareanSummarizer()
    
    # Load chunks
    print("Loading chunks...")
    chunks = load_chunks("ranked_chunks.json")
    
    # Store original chunks for evaluation reference
    original_chunks = [(chunk['content'], chunk.get('heading', f'Chunk {i+1}')) 
                      for i, chunk in enumerate(chunks)]
    
    # Store results across all iterations
    all_iterations_results = []
    current_summaries = original_chunks.copy()  # Start with original content
    
    # Process N iterations
    for iteration in range(N_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{N_ITERATIONS}")
        print(f"{'='*50}")
        
        iteration_results = []
        next_summaries = []  # Store summaries for next iteration
        
        for i, (content, heading) in enumerate(current_summaries):
            print(f"\nProcessing chunk {i+1}/{len(current_summaries)}: {heading}")
            
            # Process the chunk
            result = summarizer.process_chunk_with_evaluations(
                content=content,
                heading=heading,
                thresholds=QUALITY_THRESHOLDS,
                max_attempts=MAX_ATTEMPTS_PER_CHUNK,
                is_first_iteration=(iteration == 0),
                original_content=original_chunks[i][0]  # Always reference original for evaluation
            )
            
            iteration_results.append(result)
            
            # Prepare input for next iteration
            next_summaries.append((result['final_summary'], heading))
            
            # Print summary stats
            print(f"  Words: {result['input_word_count']} → {len(result['final_summary'].split())}")
            print(f"  Overall Score: {result['final_evaluations']['overall_score']:.3f}")
        
        # Update current summaries for next iteration
        current_summaries = next_summaries
        
        all_iterations_results.append({
            'iteration': iteration + 1,
            'results': iteration_results
        })
    
    # Save final summaries
    print(f"\n{'='*50}")
    print("SAVING RESULTS")
    print(f"{'='*50}")
    
    final_summaries = []
    final_iteration = all_iterations_results[-1]['results']
    
    for i, result in enumerate(final_iteration):
        final_summaries.append(f"Summary {i+1} - {result['heading']}:\n{result['final_summary']}\n")
    
    # Save combined summaries
    full_summary = "\n".join(final_summaries)
    with open("combined_summary.txt", "w", encoding="utf-8") as f:
        f.write(full_summary)
    
    # Save detailed results
    with open("detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(all_iterations_results, f, indent=2, ensure_ascii=False)
    
    # Generate evaluation report
    generate_evaluation_report(all_iterations_results, QUALITY_THRESHOLDS)
    
    print("✅ Done!")
    print(f"📁 Files created:")
    print(f"  - combined_summary.txt (final summaries)")
    print(f"  - detailed_results.json (full results with evaluations)")
    print(f"  - evaluation_report.txt (comprehensive analysis)")

def generate_evaluation_report(all_iterations_results: List[Dict], 
                              thresholds: Dict[str, float]) -> None:
    """Generate a comprehensive evaluation report."""
    report_lines = []
    report_lines.append("OBJECTIVE EVALUATION REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Thresholds used
    report_lines.append("QUALITY THRESHOLDS USED (First Iteration Only):")
    for metric, threshold in thresholds.items():
        report_lines.append(f"  {metric}: {threshold}")
    report_lines.append("")
    
    # Iteration-by-iteration analysis
    for iteration_data in all_iterations_results:
        iteration_num = iteration_data['iteration']
        results = iteration_data['results']
        
        report_lines.append(f"ITERATION {iteration_num} RESULTS:")
        report_lines.append("-" * 30)
        
        # Calculate statistics
        total_chunks = len(results)
        successful_chunks = sum(1 for r in results if r['success'])
        avg_attempts = np.mean([len(r['attempts']) for r in results])
        
        # Metric averages (always compared to original)
        metric_averages = {}
        for metric in thresholds.keys():
            scores = [r['final_evaluations'][metric] for r in results]
            metric_averages[metric] = np.mean(scores)
        
        overall_avg = np.mean([r['final_evaluations']['overall_score'] for r in results])
        
        # Word count statistics - show iteration progression
        input_words = [r['input_word_count'] for r in results]
        final_words = [len(r['final_summary'].split()) for r in results]
        original_words = [r['original_word_count'] for r in results]
        
        # Reduction from input (step-by-step)
        step_reduction = np.mean([(inp - final) / inp * 100 
                                for inp, final in zip(input_words, final_words)])
        
        # Total reduction from original
        total_reduction = np.mean([(orig - final) / orig * 100 
                                 for orig, final in zip(original_words, final_words)])
        
        report_lines.append(f"Success Rate: {successful_chunks}/{total_chunks} ({successful_chunks/total_chunks*100:.1f}%)")
        report_lines.append(f"Average Attempts: {avg_attempts:.1f}")
        report_lines.append(f"Step Reduction: {step_reduction:.1f}% (from previous iteration)")
        report_lines.append(f"Total Reduction: {total_reduction:.1f}% (from original)")
        report_lines.append(f"Overall Quality Score: {overall_avg:.3f}")
        report_lines.append("")
        
        report_lines.append("Metric Averages (vs Original):")
        for metric, avg_score in metric_averages.items():
            threshold = thresholds[metric]
            if iteration_num == 1:  # Only show threshold status for first iteration
                status = "✅" if avg_score >= threshold else "❌"
                report_lines.append(f"  {status} {metric}: {avg_score:.3f} (threshold: {threshold})")
            else:
                report_lines.append(f"  📊 {metric}: {avg_score:.3f}")
        report_lines.append("")
    
    # Progression tracking
    report_lines.append("COMPRESSION PROGRESSION:")
    report_lines.append("-" * 30)
    for i, iteration_data in enumerate(all_iterations_results):
        results = iteration_data['results']
        avg_words = np.mean([len(r['final_summary'].split()) for r in results])
        if i == 0:
            original_avg = np.mean([r['original_word_count'] for r in results])
            report_lines.append(f"Original Average: {original_avg:.0f} words")
        
        iteration_num = iteration_data['iteration']
        report_lines.append(f"Iteration {iteration_num}: {avg_words:.0f} words")
    
    report_lines.append("")
    
    # Final iteration detailed results
    final_iteration = all_iterations_results[-1]['results']
    report_lines.append("FINAL RESULTS BY CHUNK:")
    report_lines.append("-" * 30)
    
    for i, result in enumerate(final_iteration):
        report_lines.append(f"\nChunk {i+1}: {result['heading']}")
        report_lines.append(f"  Success: {'✅' if result['success'] else '❌'}")
        report_lines.append(f"  Attempts: {len(result['attempts'])}")
        report_lines.append(f"  Original → Final: {result['original_word_count']} → {len(result['final_summary'].split())}")
        total_reduction = (result['original_word_count'] - len(result['final_summary'].split())) / result['original_word_count'] * 100
        report_lines.append(f"  Total Reduction: {total_reduction:.1f}%")
        report_lines.append(f"  Overall Score: {result['final_evaluations']['overall_score']:.3f}")
        
        # Show failed metrics if any (only for first iteration)
        if not result['success'] and result['attempts'] and result['is_first_iteration']:
            last_attempt = result['attempts'][-1]
            if last_attempt['failed_metrics']:
                report_lines.append(f"  Failed Metrics: {', '.join(last_attempt['failed_metrics'])}")
    
    # Save report
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    main()