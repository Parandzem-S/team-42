import torch, json, os, re
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import openai
import requests
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """Data structure to hold analysis results"""
    prompt_name: str
    category: str
    keywords_found: List[str]
    context_matches: List[str]
    confidence_score: float
    reasoning: str
    source_file: str

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def analyze_text(self, prompt: str, text: str) -> str:
        """Analyze text using the specific LLM"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider"""
        pass

class ChatGPTProvider(LLMProvider):
    """ChatGPT implementation of LLM provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def analyze_text(self, prompt: str, text: str) -> str:
        """Analyze text using ChatGPT"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert in detecting signs of financial distress in banking institutions."},
                    {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing with ChatGPT: {str(e)}"
    
    def get_provider_name(self) -> str:
        return "ChatGPT"

class Phi4Provider(LLMProvider):
    """Phi4 implementation of LLM provider"""
    
    def __init__(self, method, model_name, device: Optional[str] = None):
                # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.method = method
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            **model_kwargs
        )
        print(f"Loaded Phi4 model: {model_name}")

    def analyze_text(self, prompt: str, text: str) -> str:
        """Analyze text using Phi4"""
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"
        """Analyze using Hugging Face transformers"""
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
        return response[len(prompt):].strip()
    
    def get_provider_name(self) -> str:
        return f"Phi4 ({self.method})"
    
class FinancialDistressAnalyzer:
    """Main analyzer class that processes files and detects financial distress signals"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.prompts = []
        self.results = []
    
    def load_prompts(self, prompts_file_path: str):
        """Load interrogation prompts from JSON file"""
        try:
            with open(prompts_file_path, 'r') as f:
                data = json.load(f)
                self.prompts = data.get('prompts', [])
            print(f"Loaded {len(self.prompts)} prompts successfully")
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")
    
    def create_analysis_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """Create a structured prompt for LLM analysis"""
        prompt_template = f"""
You are analyzing financial documents for signs of distress. Focus on the following financial distress indicator:

**Category**: {prompt_data['category']}
**Indicator**: {prompt_data['name']}

**Keywords to look for**: {', '.join(prompt_data['keywords'])}

**Context**: {prompt_data.get('context', 'General financial distress indicator')}

**Instructions**:
1. Carefully search the provided text for any of the specified keywords or related concepts
2. Identify specific phrases, sentences, or passages that suggest this type of financial distress
3. Provide a confidence score (0-100) indicating how strongly the text suggests this distress indicator
4. Explain your reasoning, citing specific text passages as evidence
5. If no clear evidence is found, state this explicitly

**Required Response Format**:
- KEYWORDS_FOUND: [list any keywords found]
- CONFIDENCE_SCORE: [0-100]
- EVIDENCE: [specific text passages that support your assessment]
- REASONING: [detailed explanation of your analysis]
- CONCLUSION: [summary of whether this distress indicator is present]
"""
        return prompt_template
    
    def analyze_text_file(self, file_path: str) -> List[AnalysisResult]:
        """Analyze a text file for financial distress indicators"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            results = []
            
            for i, prompt_data in enumerate(self.prompts):
                print(f"Processing prompt {i+1}/{len(self.prompts)}: {prompt_data['name']}")
                
                analysis_prompt = self.create_analysis_prompt(prompt_data)
                llm_response = self.llm_provider.analyze_text(analysis_prompt, text_content)
                
                # Parse LLM response
                result = self._parse_llm_response(llm_response, prompt_data, file_path)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing text file: {str(e)}")
            return []
    
    def analyze_excel_file(self, file_path: str) -> List[AnalysisResult]:
        """Analyze an Excel file for financial distress indicators"""
        try:
            # Read all sheets from Excel file
            excel_data = pd.read_excel(file_path)

            combined_text = excel_data.to_string(index=False, header=False)
            results = []
            
            for i, prompt_data in enumerate(self.prompts):
                print(f"Processing Excel prompt {i+1}/{len(self.prompts)}: {prompt_data['name']}")
                
                analysis_prompt = self.create_analysis_prompt(prompt_data)
                llm_response = self.llm_provider.analyze_text(analysis_prompt, combined_text)
                
                # Parse LLM response
                result = self._parse_llm_response(llm_response, prompt_data, file_path)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing Excel file: {str(e)}")
            return []
    
    def _parse_llm_response(self, response: str, prompt_data: Dict[str, Any], source_file: str) -> AnalysisResult:
        """Parse LLM response into structured result"""
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
        """Extract list items from LLM response"""
        pattern = f"{section_name}:\\s*(.*?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            items_text = match.group(1).strip()
            # Parse list items
            items = [item.strip().strip('[]') for item in items_text.split(',')]
            return [item for item in items if item and item != 'None']
        return []
    
    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from LLM response"""
        pattern = r"CONFIDENCE_SCORE:\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 0.0
    
    def _extract_section_from_response(self, response: str, section_name: str) -> str:
        """Extract a specific section from LLM response"""
        pattern = f"{section_name}:\\s*(.*?)(?=\\n[A-Z_]+:|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    def generate_report(self, results: List[AnalysisResult], output_file: str = "financial_distress_report.txt"):
        """Generate a comprehensive analysis report"""
        high_risk_indicators = [r for r in results if r.confidence_score >= 70]
        medium_risk_indicators = [r for r in results if 30 <= r.confidence_score < 70]
        
        report = f"""
=== FINANCIAL DISTRESS ANALYSIS REPORT ===
LLM Provider: {self.llm_provider.get_provider_name()}
Total Indicators Analyzed: {len(results)}

=== EXECUTIVE SUMMARY ===
High Risk Indicators (70-100): {len(high_risk_indicators)}
Medium Risk Indicators (30-69): {len(medium_risk_indicators)}
Low Risk Indicators (0-29): {len(results) - len(high_risk_indicators) - len(medium_risk_indicators)}

=== HIGH RISK INDICATORS ===
"""
        
        for result in high_risk_indicators:
            report += f"""
Indicator: {result.prompt_name}
Category: {result.category}
Confidence Score: {result.confidence_score}
Source: {result.source_file}
Keywords Found: {', '.join(result.keywords_found) if result.keywords_found else 'None'}
Reasoning: {result.reasoning}
---
"""
        
        report += "\n=== MEDIUM RISK INDICATORS ===\n"
        
        for result in medium_risk_indicators:
            report += f"""
Indicator: {result.prompt_name}
Category: {result.category}
Confidence Score: {result.confidence_score}
Source: {result.source_file}
Keywords Found: {', '.join(result.keywords_found) if result.keywords_found else 'None'}
---
"""
        
        # Save report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {output_file}")
        return report

def main():
    """Main execution function"""
    print("=== Financial Distress Analysis Tool ===")
    
    # Choose LLM provider
    print("\nAvailable LLM Providers:")
    print("1. ChatGPT")
    print("2. Phi4")
    
    choice = input("Select LLM provider (1 or 2): ").strip()
    
    if choice == "1":
        api_key = os.environ.get("OPENAI_API_KEY")  # Use environment variable if available
        model = "gpt-4"
        llm_provider = ChatGPTProvider(api_key, model)
    elif choice == "2":
        model_name = "microsoft/phi-4"
        method = "transformers"       
        llm_provider = Phi4Provider(model_name, method)
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Initialize analyzer
    analyzer = FinancialDistressAnalyzer(llm_provider)
    
    # Load prompts
    prompts_file = "derived interrogation prompts linked to various qualitative metrics.json"
    analyzer.load_prompts(prompts_file)
    
    if not analyzer.prompts:
        print("No prompts loaded. Exiting.")
        return
    
    # Analyze files
    all_results = []
    
    # Analyze text file
    text_file = "JPM_2025_Q1_presentation.txt"
    text_file = os.path.join("output", text_file)
    if text_file:
        print(f"\nAnalyzing text file: {text_file}")
        text_results = analyzer.analyze_text_file(text_file)
        all_results.extend(text_results)
    
    # Analyze Excel file
    excel_file = "JPM_2025_Q1_qa_data.xlsx"
    excel_file = os.path.join("output", excel_file)
    if excel_file:
        print(f"\nAnalyzing Excel file: {excel_file}")
        excel_results = analyzer.analyze_excel_file(excel_file)
        all_results.extend(excel_results)
    
    # Generate report
    if all_results:
        print(f"\nGenerating analysis report...")
        report = analyzer.generate_report(all_results)
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Total indicators analyzed: {len(all_results)}")
        print(f"High risk indicators found: {len([r for r in all_results if r.confidence_score >= 70])}")
        print("Report saved to: financial_distress_report.txt")
    else:
        print("No results to analyze.")

if __name__ == "__main__":
    main()