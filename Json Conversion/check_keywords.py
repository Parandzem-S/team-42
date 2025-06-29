import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
import requests
from dataclasses import dataclass
import re
import torch  # Added for Phi4 transformers implementation
import os  # Added for environment variables

@dataclass
class AnalysisResult:
    """
    📊 Data structure to hold financial distress analysis results
    
    Attributes:
        prompt_name: Name of the financial distress indicator being tested
        category: Category of the distress indicator (e.g., "Management and Governance Issues")
        keywords_found: List of keywords detected in the text
        context_matches: List of contextual phrases that match the indicator
        confidence_score: Float 0-100 indicating confidence in the finding
        reasoning: Detailed explanation of why this indicator was flagged
        source_file: Path to the file where this indicator was found
    """
    prompt_name: str
    category: str
    keywords_found: List[str]
    context_matches: List[str]
    confidence_score: float
    reasoning: str
    source_file: str

class LLMProvider(ABC):
    """
    🤖 Abstract base class for Large Language Model providers
    
    This enables switching between different LLMs (ChatGPT, Phi4, etc.)
    while maintaining the same interface for financial analysis.
    """
    
    @abstractmethod
    def analyze_text(self, prompt: str, text: str) -> str:
        """
        🔍 Analyze financial text using the specific LLM implementation
        
        Args:
            prompt: The structured analysis prompt containing distress indicators
            text: The financial document text to analyze
            
        Returns:
            LLM's analysis response as a string
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        📛 Return the name of the LLM provider for reporting purposes
        """
        pass

class ChatGPTProvider(LLMProvider):
    """
    🤖 ChatGPT implementation for financial distress analysis
    
    Uses OpenAI's API to analyze financial documents for distress indicators.
    Optimized for accurate, structured financial analysis.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        🔧 Initialize ChatGPT provider with API credentials
        
        Args:
            api_key: OpenAI API key for authentication
            model: Model to use (default: gpt-4 for best analysis quality)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def analyze_text(self, prompt: str, text: str) -> str:
        """
        💬 Analyze financial text using ChatGPT's advanced reasoning capabilities
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in detecting signs of financial distress in banking institutions. Provide structured, evidence-based analysis."},
                    {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent, factual analysis
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error analyzing with ChatGPT: {str(e)}"
    
    def get_provider_name(self) -> str:
        return "🤖 ChatGPT"

class Phi4Provider(LLMProvider):
    """
    🧠 Microsoft Phi4 implementation for financial analysis using Hugging Face Transformers
    
    Uses local inference via Hugging Face Transformers library for privacy and control.
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-4"):
        """
        ⚙️ Initialize Phi4 provider with Hugging Face Transformers
        
        Args:
            model_name: Hugging Face model identifier (default: microsoft/Phi-4)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # 📦 Load Hugging Face transformers for local inference
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"🔄 Loading Phi4 model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print(f"✅ Successfully loaded Phi4 model: {model_name}")
        except ImportError:
            raise ImportError("❌ transformers library not found. Install with: pip install transformers torch")
    
    def analyze_text(self, prompt: str, text: str) -> str:
        """
        🔍 Analyze financial text using Phi4's reasoning capabilities
        """
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"
        
        try:
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", max_length=2048, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            return response[len(full_prompt):].strip()
        except Exception as e:
            return f"❌ Error analyzing with Phi4: {str(e)}"
    
    def get_provider_name(self) -> str:
        return f"🧠 Phi4 ({self.model_name})"

class FinancialDistressAnalyzer:
    """
    🏦 Main analyzer class for detecting financial distress signals in banking documents
    
    This class orchestrates the entire analysis process:
    - 📝 Loads financial distress indicator prompts
    - 📄 Processes text and Excel files 
    - 🔍 Applies intelligent text chunking for large documents
    - 🤖 Uses LLMs to analyze content for distress indicators
    - 📊 Generates comprehensive risk assessment reports
    """
    
    def __init__(self, llm_provider: LLMProvider, max_tokens: int = 8000):
        """
        🚀 Initialize the financial distress analyzer
        
        Args:
            llm_provider: The LLM provider instance (ChatGPT, Phi4, etc.)
            max_tokens: Maximum tokens per analysis chunk (default: 8000 for safety)
        """
        self.llm_provider = llm_provider
        self.prompts = []
        self.results = []
        self.max_tokens = max_tokens  # Safe token limit to prevent API errors
    
    def load_prompts(self, prompts_file_path: str):
        """
        📋 Load financial distress interrogation prompts from JSON file
        
        The prompts file contains structured indicators across categories like:
        - 🏛️ Management and Governance Issues
        - 📈 Market and External Perceptions  
        - 🏢 Business Model and Strategic Shifts
        
        Args:
            prompts_file_path: Path to the JSON file containing distress indicators
        """
        try:
            with open(prompts_file_path, 'r') as f:
                data = json.load(f)
                self.prompts = data.get('prompts', [])
            print(f"✅ Loaded {len(self.prompts)} financial distress indicators successfully")
        except Exception as e:
            print(f"❌ Error loading prompts: {str(e)}")
    
    def chunk_text(self, text: str, max_chars: int = 6000) -> List[str]:
        """
        ✂️ Intelligently split large documents into manageable chunks
        
        Uses a sophisticated chunking strategy that:
        1. 📖 Preserves paragraph boundaries when possible
        2. 📝 Falls back to sentence boundaries if needed
        3. 🔗 Maintains context and readability across chunks
        
        Args:
            text: The input text to be chunked
            max_chars: Maximum characters per chunk (≈1,500 tokens)
            
        Returns:
            List of text chunks ready for LLM analysis
        """
        if len(text) <= max_chars:
            return [text]  # No chunking needed
        
        chunks = []
        # Try to split at natural boundaries (paragraphs first, then sentences)
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed limit, save current chunk
            if len(current_chunk) + len(paragraph) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # If even a single paragraph is too long, split it at sentence boundaries
            if len(current_chunk) > max_chars:
                sentences = current_chunk.split('. ')
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > max_chars and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        if temp_chunk:
                            temp_chunk += ". " + sentence
                        else:
                            temp_chunk = sentence
                
                current_chunk = temp_chunk
        
        # Add the final chunk if it contains content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def analyze_text_chunks(self, text: str, prompt_data: Dict[str, Any], source_file: str) -> AnalysisResult:
        """
        🔍 Analyze text by breaking it into chunks and combining results
        
        For large documents that exceed token limits, this method:
        1. 📊 Splits text into manageable chunks
        2. 🤖 Analyzes each chunk with the LLM
        3. 🔗 Intelligently combines findings across all chunks
        4. 📈 Takes the highest confidence score found
        
        Args:
            text: Full text content to analyze
            prompt_data: Financial distress indicator configuration
            source_file: Name of the source file being analyzed
            
        Returns:
            Combined analysis result from all chunks
        """
        chunks = self.chunk_text(text)
        
        if len(chunks) == 1:
            # Single chunk - analyze normally without chunking overhead
            analysis_prompt = self.create_analysis_prompt(prompt_data)
            llm_response = self.llm_provider.analyze_text(analysis_prompt, text)
            return self._parse_llm_response(llm_response, prompt_data, source_file)
        
        # Multiple chunks - analyze each and intelligently combine results
        print(f"  📄 Document split into {len(chunks)} chunks for analysis")
        
        all_keywords_found = []
        all_context_matches = []
        max_confidence = 0.0
        combined_reasoning = []
        
        for i, chunk in enumerate(chunks):
            print(f"    🔍 Analyzing chunk {i+1}/{len(chunks)}")
            
            analysis_prompt = self.create_analysis_prompt(prompt_data)
            # Add context about chunking to help LLM understand the analysis scope
            chunk_prompt = f"{analysis_prompt}\n\nNote: This is chunk {i+1} of {len(chunks)} from a larger document."
            
            llm_response = self.llm_provider.analyze_text(chunk_prompt, chunk)
            chunk_result = self._parse_llm_response(llm_response, prompt_data, f"{source_file} (chunk {i+1})")
            
            # Combine results across chunks
            all_keywords_found.extend(chunk_result.keywords_found)
            all_context_matches.extend(chunk_result.context_matches)
            max_confidence = max(max_confidence, chunk_result.confidence_score)
            
            # Only include reasoning from chunks with significant findings
            if chunk_result.reasoning and chunk_result.confidence_score > 20:
                combined_reasoning.append(f"Chunk {i+1}: {chunk_result.reasoning}")
        
        # Remove duplicates while preserving order
        all_keywords_found = list(set(all_keywords_found))
        all_context_matches = list(set(all_context_matches))
        
        # Combine reasoning from all significant findings
        final_reasoning = "; ".join(combined_reasoning) if combined_reasoning else "No significant indicators found across document chunks"
        
        return AnalysisResult(
            prompt_name=prompt_data['name'],
            category=prompt_data['category'],
            keywords_found=all_keywords_found,
            context_matches=all_context_matches,
            confidence_score=max_confidence,
            reasoning=final_reasoning,
            source_file=source_file
        )
    
    def create_analysis_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        📝 Create a structured, comprehensive prompt for LLM financial analysis
        
        This method builds a detailed prompt that:
        - 🎯 Focuses the LLM on specific financial distress indicators
        - 📊 Provides context about what to look for
        - 📋 Requests structured output for consistent parsing
        - 🔍 Guides the LLM to provide evidence-based analysis
        
        Args:
            prompt_data: Dictionary containing the financial distress indicator configuration
            
        Returns:
            Formatted prompt string ready for LLM analysis
        """
        prompt_template = f"""
FINANCIAL DISTRESS ANALYSIS TASK

You are analyzing financial documents for signs of banking distress. Focus on this specific indicator:

**Category**: {prompt_data['category']}
**Indicator**: {prompt_data['name']}

**Keywords to search for**: {', '.join(prompt_data['keywords'])}

**Context**: {prompt_data.get('context', 'General financial distress indicator - look for any signs of this issue in the document')}

**ANALYSIS INSTRUCTIONS**:
1. Carefully search the provided text for any of the specified keywords or related concepts
2. Identify specific phrases, sentences, or passages that suggest this type of financial distress
3. Provide a confidence score (0-100) indicating how strongly the text suggests this distress indicator
4. Explain your reasoning, citing specific text passages as evidence
5. If no clear evidence is found, state this explicitly with reasoning

**REQUIRED RESPONSE FORMAT**:
- KEYWORDS_FOUND: [list any keywords or related terms found]
- CONFIDENCE_SCORE: [0-100 where 0=no evidence, 100=strong evidence]
- EVIDENCE: [specific text passages that support your assessment]
- REASONING: [detailed explanation of your analysis and why you assigned this confidence score]
- CONCLUSION: [clear summary of whether this distress indicator is present and at what level]

Focus on accuracy and evidence-based analysis. Be specific about what you found and why it matters.
"""
        return prompt_template
    
    def analyze_text_file(self, file_path: str) -> List[AnalysisResult]:
        """
        📄 Analyze a text file for financial distress indicators
        
        Processes financial documents (like earnings presentations, reports) to detect
        signs of banking distress across all loaded indicator categories.
        
        Args:
            file_path: Path to the text file to analyze
            
        Returns:
            List of analysis results for each financial distress indicator
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            print(f"📄 Text file size: {len(text_content)} characters")
            
            results = []
            
            for i, prompt_data in enumerate(self.prompts):
                print(f"🔍 Processing prompt {i+1}/{len(self.prompts)}: {prompt_data['name']}")
                
                # Use intelligent chunked analysis to handle large documents
                result = self.analyze_text_chunks(text_content, prompt_data, file_path)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing text file: {str(e)}")
            return []
    
    def analyze_excel_file(self, file_path: str) -> List[AnalysisResult]:
        """
        📊 Analyze an Excel file for financial distress indicators
        
        Converts Excel data to text format and processes it through the same
        analysis pipeline as text files, handling multiple sheets.
        
        Args:
            file_path: Path to the Excel file to analyze
            
        Returns:
            List of analysis results for each financial distress indicator
        """
        try:
            # Read all sheets from Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # Combine all sheets into text for analysis
            combined_text = ""
            for sheet_name, df in excel_data.items():
                combined_text += f"\n--- Sheet: {sheet_name} ---\n"
                combined_text += df.to_string(index=False)
                combined_text += "\n"
            
            print(f"📊 Excel file size: {len(combined_text)} characters")
            
            results = []
            
            for i, prompt_data in enumerate(self.prompts):
                print(f"🔍 Processing Excel prompt {i+1}/{len(self.prompts)}: {prompt_data['name']}")
                
                # Use chunked analysis for Excel data as well
                result = self.analyze_text_chunks(combined_text, prompt_data, file_path)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing Excel file: {str(e)}")
            return []
    
    def _parse_llm_response(self, response: str, prompt_data: Dict[str, Any], source_file: str) -> AnalysisResult:
        """
        🔧 Parse LLM response into structured AnalysisResult object
        
        Extracts structured information from the LLM's text response using regex patterns
        to identify keywords found, confidence scores, and reasoning.
        
        Args:
            response: Raw text response from the LLM
            prompt_data: Original prompt configuration data
            source_file: Source file being analyzed
            
        Returns:
            Structured AnalysisResult object with parsed information
        """
        try:
            # Extract information using regex patterns
            keywords_found = self._extract_list_from_response(response, "KEYWORDS_FOUND")
            confidence_score = self._extract_confidence_score(response)
            reasoning = self._extract_section_from_response(response, "REASONING")
            
            # Find context matches in the response
            context_matches = []
            for keyword in prompt_data['keywords']:
                if keyword.lower() in response.lower():
                    context_matches.append(keyword)
            
            return AnalysisResult(
                prompt_name=prompt_data['name'],
                category=prompt_data['category'],
                keywords_found=keywords_found,
                context_matches=context_matches,
                confidence_score=confidence_score,
                reasoning=reasoning,
                source_file=source_file
            )
            
        except Exception as e:
            return AnalysisResult(
                prompt_name=prompt_data['name'],
                category=prompt_data['category'],
                keywords_found=[],
                context_matches=[],
                confidence_score=0.0,
                reasoning=f"Error parsing response: {str(e)}",
                source_file=source_file
            )
    
    def _extract_list_from_response(self, response: str, section_name: str) -> List[str]:
        """
        📝 Extract list items from structured LLM response sections
        
        Uses regex to find and parse list items from sections like KEYWORDS_FOUND.
        
        Args:
            response: LLM response text
            section_name: Name of the section to extract (e.g., "KEYWORDS_FOUND")
            
        Returns:
            List of extracted items, cleaned and filtered
        """
        pattern = f"{section_name}:\\s*(.*?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            items_text = match.group(1).strip()
            # Parse list items, handling various formats
            items = [item.strip().strip('[]') for item in items_text.split(',')]
            return [item for item in items if item and item != 'None']
        return []
    
    def _extract_confidence_score(self, response: str) -> float:
        """
        📊 Extract confidence score from LLM response
        
        Searches for CONFIDENCE_SCORE pattern and extracts the numeric value.
        
        Args:
            response: LLM response text
            
        Returns:
            Confidence score as float (0.0-100.0)
        """
        pattern = r"CONFIDENCE_SCORE:\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _extract_section_from_response(self, response: str, section_name: str) -> str:
        """
        📖 Extract a specific section's content from LLM response
        
        Uses regex to find and extract content from named sections like REASONING.
        
        Args:
            response: LLM response text
            section_name: Name of the section to extract
            
        Returns:
            Section content as cleaned string
        """
        pattern = f"{section_name}:\\s*(.*?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def generate_report(self, results: List[AnalysisResult], output_file: str = "financial_distress_report.txt"):
        """
        📋 Generate a comprehensive financial distress analysis report
        
        Creates a detailed report categorizing findings by risk level and providing
        executive summary, detailed findings, and actionable insights.
        
        Args:
            results: List of analysis results from all indicators
            output_file: Path where the report should be saved
            
        Returns:
            Generated report text (also saved to file)
        """
        high_risk_indicators = [r for r in results if r.confidence_score >= 70]
        medium_risk_indicators = [r for r in results if 30 <= r.confidence_score < 70]
        
        report = f"""
=== FINANCIAL DISTRESS ANALYSIS REPORT ===
LLM Provider: {self.llm_provider.get_provider_name()}
Total Indicators Analyzed: {len(results)}

=== EXECUTIVE SUMMARY ===
🔴 High Risk Indicators (70-100): {len(high_risk_indicators)}
🟡 Medium Risk Indicators (30-69): {len(medium_risk_indicators)}
🟢 Low Risk Indicators (0-29): {len(results) - len(high_risk_indicators) - len(medium_risk_indicators)}

=== HIGH RISK INDICATORS ===
"""
        
        for result in high_risk_indicators:
            report += f"""
🔴 Indicator: {result.prompt_name}
📊 Category: {result.category}
📈 Confidence Score: {result.confidence_score}
📁 Source: {result.source_file}
🔍 Keywords Found: {', '.join(result.keywords_found) if result.keywords_found else 'None'}
💭 Reasoning: {result.reasoning}
---
"""
        
        report += "\n=== MEDIUM RISK INDICATORS ===\n"
        
        for result in medium_risk_indicators:
            report += f"""
🟡 Indicator: {result.prompt_name}
📊 Category: {result.category}
📈 Confidence Score: {result.confidence_score}
📁 Source: {result.source_file}
🔍 Keywords Found: {', '.join(result.keywords_found) if result.keywords_found else 'None'}
---
"""
        
        # Save report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 Report saved to {output_file}")
        return report

def main():
    """
    🚀 Main execution function that orchestrates the entire analysis process
    
    Handles user interaction, LLM selection, file processing, and report generation.
    Provides a complete command-line interface for financial distress analysis.
    """
    print("=== 🏦 Financial Distress Analysis Tool ===")
    
    # Choose LLM provider
    print("\n🤖 Available LLM Providers:")
    print("1. ChatGPT")
    print("2. Phi4")
    
    choice = input("Select LLM provider (1 or 2): ").strip()
    
    if choice == "1":
        # 🔑 Get OpenAI API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable not found.")
            print("💡 Please set it with: export OPENAI_API_KEY='your-api-key'")
            return
        
        model = input("📋 Enter model name (default: gpt-4): ").strip() or "gpt-4"
        llm_provider = ChatGPTProvider(api_key, model)
        
    elif choice == "2":
        model_name = input("📋 Enter model name (default: microsoft/Phi-4): ").strip() or "microsoft/Phi-4"
        llm_provider = Phi4Provider(model_name=model_name)
        
    else:
        print("❌ Invalid choice. Exiting.")
        return
    
    # Initialize analyzer with token limit
    analyzer = FinancialDistressAnalyzer(llm_provider, max_tokens=8000)
    
    # Load prompts
    prompts_file = "derived interrogation prompts linked to various qualitative metrics.json"
    analyzer.load_prompts(prompts_file)
    
    if not analyzer.prompts:
        print("❌ No prompts loaded. Exiting.")
        return
    
    # Analyze files
    all_results = []
    
    # Analyze text file
    text_file = os.path.join("output", "JPM_2025_Q1_presentation.txt")
    if text_file:
        print(f"\n🔍 Analyzing text file: {text_file}")
        text_results = analyzer.analyze_text_file(text_file)
        all_results.extend(text_results)
    
    # Analyze Excel file
    excel_file = os.path.join("output", "JPM_2025_Q1_qa_data.xlsx")
    if excel_file:
        print(f"\n🔍 Analyzing Excel file: {excel_file}")
        excel_results = analyzer.analyze_excel_file(excel_file)
        all_results.extend(excel_results)
    
    # Generate report
    if all_results:
        print(f"\n📋 Generating analysis report...")
        report = analyzer.generate_report(all_results)
        print("\n" + "="*50)
        print("✅ ANALYSIS COMPLETE")
        print("="*50)
        print(f"📊 Total indicators analyzed: {len(all_results)}")
        print(f"🔴 High risk indicators found: {len([r for r in all_results if r.confidence_score >= 70])}")
        print("📋 Report saved to: financial_distress_report.txt")
    else:
        print("❌ No results to analyze.")

if __name__ == "__main__":
    main()