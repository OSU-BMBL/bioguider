#!/usr/bin/env python3
"""Debug script to test LLM generation for Overview section specifically"""

from bioguider.generation.suggestion_extractor import SuggestionExtractor
from bioguider.generation.report_loader import EvaluationReportLoader
from bioguider.generation.llm_content_generator import LLMContentGenerator
from bioguider.generation.style_analyzer import StyleAnalyzer
from bioguider.generation.repo_reader import RepoReader
from bioguider.agents.common_conversation import CommonConversation
from langchain_core.messages import SystemMessage, HumanMessage
import json

def debug_overview_generation():
    # Load report and extract suggestions
    loader = EvaluationReportLoader()
    report, _ = loader.load("logs/evaluation_report_github_satijalab_seurat.json")
    
    extractor = SuggestionExtractor()
    all_suggestions = extractor.extract(report)
    
    # Find the overview suggestion
    overview_suggestion = None
    for s in all_suggestions:
        if 'purpose' in s.id.lower() and 'overview' in s.action.lower():
            overview_suggestion = s
            break
    
    if not overview_suggestion:
        print("No overview suggestion found")
        return
        
    print("=== OVERVIEW SUGGESTION ===")
    print(f"ID: {overview_suggestion.id}")
    print(f"Category: {overview_suggestion.category}")
    print(f"Action: {overview_suggestion.action}")
    print(f"Content guidance: {overview_suggestion.content_guidance}")
    print(f"Target files: {overview_suggestion.target_files}")
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
        
        # Test LLM generation with real LLM
        from system_tests.conftest import get_azure_openai
        real_llm = get_azure_openai()
        
        llm_gen = LLMContentGenerator(real_llm)
        
        print("=== TESTING REAL LLM GENERATION FOR OVERVIEW ===")
        gen_section, gen_usage = llm_gen.generate_section(
            suggestion=overview_suggestion,
            style=style,
            context=readme_content,
        )
        
        print("=== LLM OUTPUT ===")
        print(gen_section)
        print()
        
        print("=== TOKEN USAGE ===")
        print(gen_usage)
        
        # Check if it contains placeholder content
        if "Clear 2–3 sentence summary" in gen_section:
            print("❌ ERROR: Generated content contains placeholder text!")
        else:
            print("✅ SUCCESS: Generated content does not contain placeholder text")
            
        # Check if it follows the guidance
        if "comparative statement" in gen_section.lower() or "beneficial" in gen_section.lower():
            print("✅ SUCCESS: Generated content follows the guidance")
        else:
            print("❌ WARNING: Generated content may not follow the guidance")
        
    else:
        print("README.md not found in repository")

if __name__ == "__main__":
    debug_overview_generation()
