#!/usr/bin/env python3
"""Debug script to test LLM content generation with specific suggestions"""

from bioguider.generation.suggestion_extractor import SuggestionExtractor
from bioguider.generation.report_loader import EvaluationReportLoader
from bioguider.generation.llm_content_generator import LLMContentGenerator
from bioguider.generation.style_analyzer import StyleAnalyzer
from bioguider.generation.repo_reader import RepoReader
import json

def debug_llm_generation():
    # Load report and extract suggestions
    loader = EvaluationReportLoader()
    report, _ = loader.load("logs/evaluation_report_github_satijalab_seurat.json")
    
    extractor = SuggestionExtractor()
    all_suggestions = extractor.extract(report)
    
    # Get first suggestion (README dependencies)
    first_suggestion = all_suggestions[0]  # readme-dependencies-README.md
    print("=== FIRST SUGGESTION ===")
    print(f"ID: {first_suggestion.id}")
    print(f"Category: {first_suggestion.category}")
    print(f"Action: {first_suggestion.action}")
    print(f"Content guidance: {first_suggestion.content_guidance}")
    print(f"Target files: {first_suggestion.target_files}")
    print()
    
    # Read README file
    reader = RepoReader("data/.adalflow/repos/satijalab_seurat")
    files, missing = reader.read_files(["README.md"])
    
    if "README.md" in files:
        readme_content = files["README.md"]
        print("=== README CONTENT (first 500 chars) ===")
        print(readme_content[:500])
        print()
        
        # Analyze style
        style_analyzer = StyleAnalyzer()
        style = style_analyzer.analyze(files)
        
        print("=== STYLE PROFILE ===")
        print(f"Tone markers: {style.tone_markers}")
        print(f"Heading style: {style.heading_style}")
        print(f"List style: {style.list_style}")
        print(f"Link style: {style.link_style}")
        print()
        
        # Test LLM generation (mock LLM for testing)
        class MockLLM:
            def generate(self, messages, **kwargs):
                # Return the prompt to see what's being sent
                system_msg = messages[0][0].content
                print("=== LLM PROMPT SENT ===")
                print(system_msg)
                print()
                
                # Return a mock response
                return type('MockResult', (), {
                    'generations': [[type('MockGeneration', (), {
                        'text': '## Dependencies\n\n- ggplot2\n- dplyr\n- tidyr\n- Matrix\n- SeuratObject\n\n### Installation Guide\n\n1. Install CRAN dependencies\n2. Install Bioconductor dependencies\n3. Verify installation'
                    })()]],
                    'llm_output': {'token_usage': {'total_tokens': 100}}
                })()
        
        mock_llm = MockLLM()
        llm_gen = LLMContentGenerator(mock_llm)
        
        print("=== TESTING LLM GENERATION ===")
        gen_section, gen_usage = llm_gen.generate_section(
            suggestion=first_suggestion,
            style=style,
            context=readme_content,
        )
        
        print("=== LLM OUTPUT ===")
        print(gen_section)
        print()
        
        print("=== TOKEN USAGE ===")
        print(gen_usage)
        
    else:
        print("README.md not found in repository")

if __name__ == "__main__":
    debug_llm_generation()
